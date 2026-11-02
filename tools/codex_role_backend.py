"""Scheduler-only, fresh Codex account calls for the existing research controller.

The harness supplies role history and memory explicitly. Native Codex sessions,
repository files, other arms and local model weights are not mounted in calls.
"""
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import time

from private_inference import AllocationDeadlineExceeded, ContextCapacityError, RuntimeIsolationError

CODEX_INPUT_CHARACTER_LIMIT = 1048576
ACTION_TRANSPORT_INSTRUCTION = '\nTransport format: return the requested action object inside the required response field.'

def wire_schema(original):
    """Broaden unsupported wire constructs; validate the original schema afterwards."""
    if isinstance(original, list):
        return [wire_schema(item) for item in original]
    if not isinstance(original, dict):
        return original
    schema = {key: wire_schema(value) for key, value in original.items()
              if key not in ("prefixItems", "uniqueItems")}
    if "const" in schema:
        schema["enum"] = [schema.pop("const")]
    if "enum" in schema and "type" not in schema:
        kinds = {"null" if value is None else "boolean" if isinstance(value, bool) else
                 "string" if isinstance(value, str) else "number" for value in schema["enum"]}
        schema["type"] = next(iter(kinds)) if len(kinds) == 1 else sorted(kinds)
    if "prefixItems" in original:
        options = wire_schema(original["prefixItems"])
        schema["items"] = {"anyOf": options} if options else {}
    return schema


def root_wire_schema(original):
    schema = wire_schema(original)
    if "anyOf" in schema and "type" not in schema:
        branches = schema["anyOf"]
        if not all(item.get("type") == "object" for item in branches):
            raise ValueError("Codex root schema must describe object variants")
        # Keep variant fields coupled; flattening their unions permits invalid mixed actions.
        schema = {"type": "object", "required": ["response"], "additionalProperties": False,
                  "properties": {"response": schema}}
    return schema


def normalize_schema_sets(value, schema, path=()):
    """Normalize set-valued syntax only; never add evidence or alter a scalar/model term."""
    from jsonschema import Draft202012Validator
    if not isinstance(schema, dict):
        return value, []
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            candidate, edits = normalize_schema_sets(value, branch, path)
            if Draft202012Validator(branch).is_valid(candidate):
                return candidate, edits
        return value, []
    edits = []
    if isinstance(value, dict):
        result = dict(value)
        for key, child in value.items():
            result[key], changes = normalize_schema_sets(child, schema.get("properties", {}).get(key, {}), path + (key,))
            edits.extend(changes)
        return result, edits
    if isinstance(value, list):
        result = list(value)
        citation = bool(path and path[-1] == "evidence_ids" and schema.get("prefixItems"))
        if schema.get("uniqueItems") or citation:
            result = list({json.dumps(item, sort_keys=True): item for item in result}.values())
        if citation:
            current = schema["prefixItems"][0].get("const")
            if current in result:
                result = [current] + [item for item in result if item != current]
        if result != value:
            edits.append({"path": list(path), "requested": value, "effective": result})
        return result, edits
    return value, edits


def failure_category(text):
    diagnostic = text.lower()
    if "input_too_large" in diagnostic:
        return "cli_input_capacity"
    if "context_length_exceeded" in diagnostic or "context window" in diagnostic:
        return "context_capacity"
    if any(word in diagnostic for word in ("rate_limit", "usage limit", "usage_limit", "quota")):
        return "account_usage_limit"
    if "auth.json" in diagnostic and any(word in diagnostic for word in ("read-only", "permission", "rename")):
        return "auth_refresh_write_blocked"
    if any(word in diagnostic for word in ("unauthorized", "authentication", "refresh token", "401")):
        return "account_authentication"
    if any(word in diagnostic for word in ("invalid schema", "invalid_json_schema", "output schema")):
        return "output_schema"
    if "model" in diagnostic and any(word in diagnostic for word in ("not supported", "not found", "unavailable")):
        return "model_unavailable"
    if any(word in diagnostic for word in ("connection", "dns", "tls", "network", "timed out")):
        return "network"
    return "cli_failure"


def safe_cli_diagnostic(text):
    """Keep bounded CLI errors without recording authentication or webhook secrets."""
    text = re.sub(r'https://hooks\.slack\.com/services/[^\s"\'<>]+', '[REDACTED_WEBHOOK]', text)
    text = re.sub(r'\bsk-[A-Za-z0-9_-]+', '[REDACTED_KEY]', text)
    text = re.sub(r'(?i)(bearer\s+)[^\s"\']+', r'\1[REDACTED]', text)
    text = re.sub(r'(?i)((?:access_token|refresh_token|id_token|api_key|authorization)["\']?\s*[:=]\s*["\']?)[^"\'\s,}]+', r'\1[REDACTED]', text)
    return text[-2500:]


class CodexRoleBackend:
    def __init__(self, directory, deadline, *, model, codex_binary=None, auth_home=None,
                 call_timeout=None, reasoning_effort="medium"):
        if not os.environ.get("JOB_ID"):
            raise RuntimeIsolationError("Reviewed SGE allocation required")
        if os.environ.get("RESEARCH_AGENT_ALLOW_CHILD_CLEANUP") != "1":
            raise RuntimeIsolationError("Owned child cleanup opt-in required")
        if not model or (call_timeout is not None and call_timeout <= 0):
            raise ValueError("Explicit model and positive call timeout required")
        from jsonschema import Draft202012Validator
        self.validator_class = Draft202012Validator
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.home = self.directory / "codex-home"
        self.home.mkdir(mode=0o700)
        self.deadline, self.call_timeout = deadline, call_timeout
        self.model, self.reasoning_effort = model, reasoning_effort
        executable = codex_binary or shutil.which("codex")
        if not executable or not shutil.which("bwrap"):
            raise RuntimeIsolationError("Codex CLI and bwrap are required")
        self.binary = Path(executable).resolve(strict=True)
        self.auth = (Path(auth_home or os.environ.get("CODEX_HOME", Path.home() / ".codex")) /
                     "auth.json").resolve(strict=True)
        self.process = None
        self.last_metadata, self.cleanup_metadata = {}, {}
        self.last_content = ""
        self.startup_seconds = 0
        self.device_report = {"device": "hosted_codex", "gpu": False,
                              "native_home_policy": "run_scoped_assets_ephemeral_native_state",
                              "provider": "codex-account", "isolation_verified": False}
        self.calls = 0

    def _folder(self, name):
        folder = self.directory / name
        folder.mkdir(mode=0o700)
        (folder / "work").mkdir(mode=0o700)
        return folder

    def _sandbox(self, folder):
        command = [shutil.which("bwrap"), "--die-with-parent", "--new-session", "--unshare-pid",
                   "--cap-drop", "ALL", "--proc", "/proc", "--dev", "/dev",
                   "--tmpfs", "/tmp", "--tmpfs", "/home", "--dir", "/home/agent"]
        for path in ("/usr", "/bin", "/lib", "/lib64", "/etc/ssl", "/etc/resolv.conf",
                     "/etc/hosts", "/etc/nsswitch.conf"):
            if Path(path).exists():
                command += ["--ro-bind", path, path]
        command += ["--ro-bind", str(self.binary), "/codex",
                    "--bind", str(folder / "work"), "/work",
                    "--bind", str(self.home), "/codex-home",
                    # Reuse only this run's assets; native dynamic state stays ephemeral.
                    "--tmpfs", "/codex-home/tmp", "--tmpfs", "/codex-home/memories",
                    "--ro-bind", str(self.auth), "/codex-home/auth.json",
                    "--setenv", "HOME", "/home/agent", "--setenv", "CODEX_HOME", "/codex-home",
                    "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8",
                    "--setenv", "RUST_LOG", "off", "--chdir", "/work", "--"]
        # Network is needed for official Codex authentication and model inference.
        return command

    def _cli(self):
        settings = ["forced_login_method=\"chatgpt\"", "cli_auth_credentials_store=\"file\"",
                    "model_provider=\"openai\"", "features.shell_tool=false",
                    "features.unified_exec=false", "features.apply_patch_freeform=false",
                    "features.multi_agent=false", "features.apps=false",
                    "features.shell_snapshot=false", "web_search=\"disabled\"",
                    "history.persistence=\"none\"",
                    "memories.use_memories=false", "memories.generate_memories=false",
                    "memories.dedicated_tools=false", "sqlite_home=\"/tmp/codex-state\"",
                    "log_dir=\"/tmp/codex-logs\"",
                    "model_reasoning_effort=" + json.dumps(self.reasoning_effort)]
        command = ["/codex", "--ask-for-approval", "never"]
        for setting in settings:
            command += ["-c", setting]
        return command

    def _run(self, command, prompt=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise AllocationDeadlineExceeded("Codex allocation deadline reached")
        environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, env=environment,
                                        start_new_session=True)
        try:
            stdout, stderr = self.process.communicate(
                prompt, timeout=remaining if self.call_timeout is None else min(remaining, self.call_timeout))
        except BaseException as error:
            if self.process.poll() is None:
                # This group belongs only to the bounded child created immediately above.
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.communicate()
            self.cleanup_metadata = {"method": "owned_process_group_cleanup", "verified": True,
                                     "returncode": self.process.returncode}
            if isinstance(error, subprocess.TimeoutExpired):
                self.last_metadata["timeout"] = True
                if time.monotonic() >= self.deadline:
                    raise AllocationDeadlineExceeded("Codex allocation deadline reached") from error
                raise RuntimeIsolationError("Codex call timed out; no automatic inference retry") from error
            raise
        self.cleanup_metadata = {"method": "bounded_cli_exit", "verified": self.process.poll() is not None,
                                 "returncode": self.process.returncode}
        return self.process.returncode, stdout, stderr

    def __enter__(self):
        started = time.monotonic()
        folder = self._folder("preflight")
        code, _, stderr = self._run(self._sandbox(folder) + ["/bin/sh", "-c",
            "test ! -e /user && test ! -e /proc/1/root/user && test ! -e /root && "
            "test ! -e /dev/nvidia0 && test -f /codex-home/auth.json"])
        if code:
            # This command performs visibility tests only and cannot print credentials.
            diagnostic = stderr.strip()[:1000]
            self.last_metadata.update(failure_category="filesystem_preflight", returncode=code,
                                      preflight_diagnostic=diagnostic)
            raise RuntimeIsolationError("Codex filesystem isolation probe failed: " + diagnostic)
        code, stdout, stderr = self._run(self._sandbox(folder) + self._cli() + ["login", "status"])
        if code or "chatgpt" not in (stdout + stderr).lower():
            raise RuntimeIsolationError("Codex ChatGPT account authentication is unavailable")
        self.startup_seconds = round(time.monotonic() - started, 3)
        self.device_report.update(isolation_verified=True, auth_method="chatgpt",
            account_auth_read_only=True, native_tools_requested_disabled=True,
            isolation_scope="research filesystem absent; only system files, call directory and auth are mounted")
        return self

    def complete(self, prompt, schema, tokens, seed):
        started = time.monotonic()
        self.calls += 1
        self.last_content = ""
        self.last_metadata = {"provider": "codex-account", "model_requested": self.model,
            "call_index": self.calls, "completion_submitted": False,
            "session_policy": "fresh_ephemeral_harness_supplied_context",
            "native_home_policy": "run_scoped_assets_ephemeral_native_state",
            "legacy_generation_tokens_hint": tokens, "generation_token_cap_enforced": False,
            "legacy_seed_hint": seed, "seed_supported": False,
            "context_truncated_by_adapter": False}
        folder = self._folder("call-%04d" % self.calls)
        schema_path = folder / "work/schema.json"
        schema_path.write_text(json.dumps(root_wire_schema(schema)))
        schema_path.chmod(0o600)
        wrapped = "anyOf" in schema and "type" not in schema
        if wrapped:
            prompt += ACTION_TRANSPORT_INSTRUCTION
        self.last_metadata.update(prompt_bytes=len(prompt.encode('utf-8')), prompt_chars=len(prompt),
                                  schema_bytes=schema_path.stat().st_size)
        # Verified on this pinned CLI: turn/start rejects text above this character count.
        if len(prompt) > CODEX_INPUT_CHARACTER_LIMIT:
            self.last_metadata.update(failure_category='cli_input_capacity',
                                      transport_input_char_limit=CODEX_INPUT_CHARACTER_LIMIT,
                                      failure_boundary='input_preflight')
            raise ContextCapacityError('Codex turn/start input exceeds 1048576 characters; no evidence removed')
        command = self._sandbox(folder) + self._cli() + ["exec", "--ephemeral", "--ignore-user-config",
            "--ignore-rules", "--sandbox", "read-only",
            "--skip-git-repo-check", "--json", "--model", self.model,
            "--output-schema", "/work/schema.json", "--output-last-message", "/work/result.json", "-"]
        self.last_metadata["completion_submitted"] = True
        try:
            code, stdout, stderr = self._run(command, prompt)
            self.last_metadata.update(seconds=round(time.monotonic() - started, 3), returncode=code)
            events = []
            for line in stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
            self.last_metadata["usage"] = [event["usage"] for event in events
                if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict)]
            self.last_metadata["event_types"] = sorted({str(event.get("type")) for event in events})
            self.last_metadata["error_event_count"] = sum(
                1 for event in events if (event.get("item") or {}).get("type") == "error")
            tool_events = [event for event in events if event.get("type") in ("item.started", "item.completed")
                           and (event.get("item") or {}).get("type") not in
                           ("agent_message", "reasoning", "todo_list", "error", None)]
            if tool_events:
                self.last_metadata["unexpected_item_types"] = sorted({
                    str((event.get("item") or {}).get("type")) for event in tool_events})
                raise RuntimeIsolationError("Unexpected native Codex tool event; inference-only contract violated")
            if code:
                category = failure_category(stdout + stderr)
                self.last_metadata["failure_category"] = category
                self.last_metadata['failure_boundary'] = 'model_execution'
                if category == "context_capacity":
                    # Capacity failures must not echo potentially sensitive prompt text.
                    raise ContextCapacityError("Codex context capacity exceeded; no evidence removed")
                errors = [event for event in events if event.get('type') == 'error' or
                          (event.get('item') or {}).get('type') == 'error']
                diagnostic = safe_cli_diagnostic(stderr + '\n' + json.dumps(errors))
                self.last_metadata.update(cli_diagnostic=diagnostic, failure_boundary='model_execution')
                error_path = folder / 'work/cli-error.json'
                error_path.write_text(json.dumps({'category': category, 'diagnostic': diagnostic}))
                error_path.chmod(0o600)
                if category == 'cli_input_capacity':
                    raise ContextCapacityError('Codex turn/start character capacity exceeded; no evidence removed')
                raise RuntimeIsolationError("Codex CLI request failed: " + category)
            output = folder / "work/result.json"
            if not output.is_file() or output.stat().st_size > 1_000_000:
                raise RuntimeIsolationError("Codex structured result missing or oversized")
            content = output.read_text()
            try:
                value = json.loads(content)
            except json.JSONDecodeError as error:
                raise RuntimeIsolationError("Codex result is not complete JSON") from error
            if not isinstance(value, dict):
                raise RuntimeIsolationError("Codex structured response must be an object")
            if wrapped:
                if set(value) != {"response"} or not isinstance(value["response"], dict):
                    raise RuntimeIsolationError("Codex action transport envelope invalid")
                value = value["response"]
                content = json.dumps(value, ensure_ascii=False)
            value, normalizations = normalize_schema_sets(value, schema)
            if normalizations:
                self.last_metadata["schema_set_normalizations"] = normalizations
                content = json.dumps(value, ensure_ascii=False)
            violation = next(self.validator_class(schema).iter_errors(value), None)
            if violation is not None:
                self.last_metadata["failure_category"] = "original_schema_validation"
                raise RuntimeIsolationError("Codex output violates original schema: " + str(violation.validator))
            output.chmod(0o600)
            self.last_content = content
            self.last_metadata["parse_status"] = "succeeded"
            return value, dict(self.last_metadata), content
        except Exception as error:
            self.last_metadata.update(seconds=round(time.monotonic() - started, 3),
                                      failure_type=type(error).__name__)
            raise

    def __exit__(self, *args):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.communicate()
        self.cleanup_metadata["verified"] = self.process is None or self.process.poll() is not None
