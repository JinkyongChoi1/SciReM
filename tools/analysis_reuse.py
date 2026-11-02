"""Reuse only exact successful own-job analyses against frozen data and Runner."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


class VerifiedAnalysisCache:
    def __init__(self, data_identity):
        self.data_identity = deepcopy(data_identity)
        self.entries = {}

    def run(self, session, action, runner):
        key = json.dumps([self.data_identity, action], sort_keys=True, separators=(",", ":"))
        if key in self.entries:
            path, digest = self.entries[key]
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Prior own-job analysis source changed; refuse reuse")
            result = json.loads(content)
            result["execution_reuse"] = {"source_attempt": path.parent.name,
                "source_sha256": digest,
                "reason": "Identical action and frozen input; not independent evidence"}
            return result
        result = runner(session, action)
        if result.get("returncode") == 0:
            path = Path(session) / ("executed-analysis-" + hashlib.sha256(key.encode()).hexdigest()[:16] + ".json")
            path.write_text(json.dumps(result, sort_keys=True))
            path.chmod(0o600)
            self.entries[key] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
        return result
