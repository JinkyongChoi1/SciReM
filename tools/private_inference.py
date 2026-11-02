"""Job-local CPU model server with fixed-only KV restoration and no TCP listener."""
import http.client
import json
import os
import re
import resource
from pathlib import Path
import socket
import subprocess
import time

from validate_local_synthetic import sandbox, parse_object
from research_memory.gpu_device import choose_gpu


class RuntimeIsolationError(RuntimeError):
    pass


class AllocationDeadlineExceeded(RuntimeIsolationError):
    """The reviewed allocation's inference deadline has actually expired."""

    deadline_exhausted = True


class ContextCapacityError(ValueError):
    """Full evidence cannot be delivered safely; callers must escalate, never trim."""


class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, address, timeout):
        super().__init__("localhost", timeout=timeout)
        self.address = address

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.address)


class PrivateInference:
    def __init__(self, directory, runtime, prefix, deadline, *, device="cpu", cuda_toolkit=None):
        if not os.environ.get("JOB_ID"):
            raise RuntimeIsolationError("Reviewed SGE allocation required")
        self.directory, self.runtime = Path(directory), Path(runtime)
        if device not in ("cpu", "gpu") or (device == "gpu" and cuda_toolkit is None):
            raise RuntimeIsolationError("Reviewed CUDA toolkit and one-GPU mode are required for GPU inference")
        self.device = device
        self.cuda_toolkit = Path(cuda_toolkit).resolve() if cuda_toolkit is not None else None
        self.device_report = {"device": device, "exclusive": False, "worker_visibility_verification": "pending"}
        self.prefix, self.deadline = prefix, deadline
        self.process = None
        self.stdout = self.stderr = None
        self.directory_fd = None
        self.failed = False
        self.closed = False
        self.startup_seconds = 0
        self.last_content = ""
        self.last_metadata = {}
        self.context_limit = 32768
        self.safety_margin_tokens = 32
        self.cleanup_metadata = {}

    def launch_configuration(self):
        environment = {"PATH": "/usr/bin:/bin", "HOME": "/work", "LANG": "C.UTF-8",
                       "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "", "LLAMA_SERVER_ROUTER_PORT": "0"}
        command = sandbox(self.directory, self.runtime, weights=True)[:-1]
        if self.cuda_toolkit is not None:
            if not (self.cuda_toolkit / "lib64").is_dir():
                raise RuntimeIsolationError("Reviewed CUDA toolkit library directory is missing")
            command += ["--ro-bind", str(self.cuda_toolkit), "/cuda", "--setenv", "LD_LIBRARY_PATH", "/runtime:/cuda/lib64"]
        if self.device == "gpu":
            self.device_report = {"device": "gpu", **choose_gpu()}
            self.device_report["address_space_limit_bytes"] = list(resource.getrlimit(resource.RLIMIT_AS))
            for path in self.device_report["mount_paths"]:
                command += ["--dev-bind", path, path]
            environment["CUDA_VISIBLE_DEVICES"] = self.device_report["selected_uuid"]
        command += ["--", "/runtime/llama-server", "-m", "/model.gguf", "-ngl", "99" if self.device == "gpu" else "0",
            "-sm", "none", "-mg", "0", "-t", "4", "-tb", "4", "-c", "32768", "-np", "1",
            "-b", "256", "-ub", "256",
            "--host", "/work/model.sock", "--no-webui", "--slots", "--slot-save-path", "/work/cache",
            "--cache-ram", "0", "--no-cache-idle-slots", "--no-warmup", "--no-context-shift", "--reasoning", "off",
            "-lv", "4"]
        return command, environment

    def verify_acceleration(self):
        if self.device == "gpu":
            log = (self.directory / "server.stderr").read_text(errors="replace")
            counts = re.findall(r"found\s+(\d+)\s+CUDA\s+devices", log, re.IGNORECASE)
            # Pinned llama-server prints device_info instead of the older init count.
            devices = set(re.findall(r"common_param:\s+-\s+CUDA(\d+)\s*:", log))
            offload = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", log)
            if (not counts and not devices or any(int(count) != 1 for count in counts) or
                    devices and devices != {"0"} or not offload or
                    not 0 < int(offload[-1][0]) <= int(offload[-1][1])):
                raise RuntimeIsolationError("One CUDA device and actual model-layer offload were not verified")
            self.device_report.update(device_count=1, cuda_offloaded_layers=int(offload[-1][0]),
                worker_visibility_verification="native CUDA initialization and positive layer offload verified")
        else:
            self.device_report.update(device_count=0, cuda_offloaded_layers=0,
                worker_visibility_verification="CPU mode: no GPU device mounts, empty CUDA visibility, zero offload requested")

    def request(self, path, payload=None, timeout=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise AllocationDeadlineExceeded("Reviewed allocation deadline reached")
        request_timeout = min(300 if timeout is None else timeout, remaining)
        if request_timeout <= 0:
            raise AllocationDeadlineExceeded("Reviewed allocation deadline reached")
        connection = UnixHTTP(f"/proc/self/fd/{self.directory_fd}/model.sock", request_timeout)
        try:
            body = None if payload is None else json.dumps(payload)
            connection.request("GET" if payload is None else "POST", path, body,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            value = json.loads(response.read())
            if response.status != 200:
                raise RuntimeIsolationError("Private inference endpoint failed: " + str(response.status))
            return value
        finally:
            connection.close()

    def render(self, content):
        return self.request("/apply-template", {"messages": [{"role": "user", "content": content}],
            "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]

    def render_prefix(self):
        # Cache before dynamic user content, not after a closed chat turn/assistant header.
        marker = "__PRIVATE_STATIC_BOUNDARY_928314__"
        if marker in self.prefix:
            raise RuntimeIsolationError("Static boundary marker collision")
        rendered = self.render(self.prefix + marker)
        if rendered.count(marker) != 1:
            raise RuntimeIsolationError("Cannot identify exact fixed-only chat prefix")
        return rendered.split(marker, 1)[0]

    def capacity_metadata(self, rendered, generation_tokens):
        if type(generation_tokens) is not int or generation_tokens < 0:
            raise ValueError("A finite nonnegative generation token budget is required")
        tokenized = self.request("/tokenize", {"content": rendered, "add_special": True,
                                             "parse_special": True, "with_pieces": False})
        tokens = tokenized.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
            raise RuntimeIsolationError("Native tokenizer did not return token IDs")
        required = len(tokens) + generation_tokens + self.safety_margin_tokens
        metadata = {"capacity_scope": "actual_rendered_chat_prompt", "rendered_prompt_tokens": len(tokens),
                    "requested_generation_tokens": generation_tokens, "context_limit": self.context_limit,
                    "safety_margin_tokens": self.safety_margin_tokens, "required_tokens": required,
                    "capacity_status": "fits" if required <= self.context_limit else "exceeded",
                    "completion_submitted": False,
                    "context_shift_enabled": False, "tokenizer_add_special": True, "tokenizer_parse_special": True}
        self.last_metadata = metadata
        if required > self.context_limit:
            raise ContextCapacityError("Full rendered prompt plus generation exceeds context capacity; no evidence was trimmed")
        return metadata

    def __enter__(self):
        if not (self.runtime / "llama-server").is_file():
            raise RuntimeIsolationError("Reviewed pinned llama-server build is missing")
        self.directory.mkdir(mode=0o700)
        (self.directory / "cache").mkdir(mode=0o700)
        self.directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        self.stdout = (self.directory / "server.stdout").open("x")
        self.stderr = (self.directory / "server.stderr").open("x")
        start = time.monotonic()
        try:
            command, environment = self.launch_configuration()
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=self.stdout, stderr=self.stderr,
                                            text=True, env=environment)
            readiness_deadline = min(self.deadline, start + 180)
            while True:
                now = time.monotonic()
                if now >= self.deadline:
                    raise AllocationDeadlineExceeded("Reviewed allocation deadline reached during startup")
                if self.process.poll() is not None or now >= readiness_deadline:
                    raise RuntimeIsolationError("Private runtime startup failed or timed out")
                try:
                    self.request("/health", timeout=1)
                    break
                except (OSError, http.client.HTTPException, RuntimeIsolationError):
                    time.sleep(0.2)
            os.chmod(self.directory / "model.sock", 0o600)
            actual_limit = self.request("/props").get("default_generation_settings", {}).get("n_ctx")
            if type(actual_limit) is not int or actual_limit <= 0 or actual_limit > self.context_limit:
                raise RuntimeIsolationError("Cannot verify native per-slot context capacity")
            self.context_limit = actual_limit
            self.rendered_fixed_prefix = self.render_prefix()
            self.capacity_metadata(self.rendered_fixed_prefix, 0)
            self.request("/completion", {"prompt": self.rendered_fixed_prefix, "n_predict": 0,
                         "cache_prompt": True, "id_slot": 0, "seed": 42})
            saved = self.request("/slots/0?action=save", {"filename": "fixed-prefix.bin"})
            if saved.get("n_saved", 0) <= 0:
                raise RuntimeIsolationError("Fixed-prefix cache not saved")
            self.fixed_tokens = saved["n_saved"]
            self.request("/slots/0?action=erase", {})
            self.verify_acceleration()
            self.startup_seconds = round(time.monotonic() - start, 3)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def complete(self, dynamic, schema, tokens, seed, reuse_prefix=True):
        self.last_content, self.last_metadata = "", {}
        if self.failed:
            self.last_metadata.update(failure_boundary="preflight", failure_cause="RuntimePreviouslyFailed",
                                      deadline_exhausted=False)
            raise RuntimeIsolationError("Runtime failed closed after a previous transport/cache error")
        start = time.monotonic()
        boundary = "cache_restore" if reuse_prefix else "render"
        try:
            if reuse_prefix:
                restored = self.request("/slots/0?action=restore", {"filename": "fixed-prefix.bin"})
                if restored.get("n_restored") != self.fixed_tokens:
                    raise RuntimeIsolationError("Unexpected fixed-prefix cache token count")
            boundary = "prompt_render"
            rendered = self.render(self.prefix + dynamic)
            if not rendered.startswith(self.rendered_fixed_prefix):
                raise RuntimeIsolationError("Request is not an extension of the fixed-only prefix")
            boundary = "tokenize_and_capacity"
            capacity = self.capacity_metadata(rendered, tokens)
            self.last_metadata["completion_submitted"] = True
            boundary = "completion_request"
            response = self.request("/completion", {"prompt": rendered,
                "json_schema": schema, "n_predict": tokens, "temperature": 0.6, "seed": seed,
                "cache_prompt": reuse_prefix, "id_slot": 0, "stream": False},
                timeout=self.deadline - time.monotonic())
            # Erase dynamic KV before returning or parsing model text, on both experimental arms.
            boundary = "dynamic_cache_erase"
            self.request("/slots/0?action=erase", {})
        except ContextCapacityError:
            try:
                self.request("/slots/0?action=erase", {})
            except Exception as error:
                self.failed = True
                raise RuntimeIsolationError("Cannot erase restored slot after capacity rejection") from error
            raise
        except Exception as error:
            self.failed = True
            self.last_metadata.update(failure_boundary=boundary, failure_cause=type(error).__name__,
                deadline_exhausted=isinstance(error, AllocationDeadlineExceeded) or
                    isinstance(error, TimeoutError) and time.monotonic() >= self.deadline)
            raise RuntimeIsolationError("Private runtime transport/cache boundary failed") from error
        metadata = {**capacity, "completion_submitted": True, "seconds": round(time.monotonic() - start, 3), "timings": response.get("timings", {}),
                    "stop_type": response.get("stop_type"), "stopped_limit": response.get("stopped_limit"),
                    "stopped_eos": response.get("stopped_eos"), "truncated": response.get("truncated"),
                    "tokens_evaluated": response.get("tokens_evaluated"),
                    "tokens_cached": response.get("tokens_cached"), "fixed_prefix_tokens": self.fixed_tokens,
                    "cache_policy": "restore_fixed_only_then_erase_dynamic" if reuse_prefix else "no_prefix_cache_dynamic_erased",
                    "runtime_pid": self.process.pid}
        content = response.get("content", "")
        self.last_content = content
        self.last_metadata = metadata
        if response.get("truncated"):
            self.last_metadata["capacity_status"] = "server_reported_truncation"
            raise ContextCapacityError("Server reported context truncation; response rejected without scientific interpretation")
        return parse_object(content), metadata, content

    def __exit__(self, *_):
        if self.closed:
            return
        self.closed = True
        cleanup_started = time.monotonic()
        self.cleanup_metadata = {"method": "owned_worker_native_stdin_exit", "native_command_sent": False,
                                 "wait_limit_seconds": 30, "batch_tokens": 256, "microbatch_tokens": 256}
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    # Pinned server's native router-child protocol, not a process signal.
                    self.process.stdin.write("cmd_router_to_child:exit\n")
                    self.process.stdin.flush()
                    self.cleanup_metadata["native_command_sent"] = True
                    self.process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Fail closed; scheduler remains the resource-limit authority. No broad cleanup/signals.
                self.failed = True
        self.cleanup_metadata.update(seconds=round(time.monotonic()-cleanup_started, 3),
                                     returncode=self.process.poll() if self.process is not None else None)
        for stream in (self.stdout, self.stderr):
            if stream is not None:
                stream.close()
        if self.directory_fd is not None:
            os.close(self.directory_fd)
            self.directory_fd = None
