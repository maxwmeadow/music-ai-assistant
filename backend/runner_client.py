import json
import pathlib
import requests
from typing import Optional, Dict, Any
from .schemas import RunnerEvalResponse

class RunnerClient:
    def __init__(self, ingest_url: Optional[str], inbox_path: Optional[str], timeout_s: float):
        self.ingest_url = ingest_url
        self.inbox_path = inbox_path
        self.timeout_s = timeout_s

    def eval(self, payload: Dict[str, Any]) -> RunnerEvalResponse:
        """
        Runner expects:
          HTTP: POST /eval with body {"musicData": {...}}
          Response JSON: { "dsl_code": "...", "executable_code": "...", "parsed_data": {...} }

        If inbox_path is used (file-drop mode), we write the payload as-is and return a queued meta.
        """
        # HTTP runner mode
        if self.ingest_url:
            # Prefer IR for the runner; it expects "musicData"
            body: Dict[str, Any]
            if "ir" in payload and payload["ir"] is not None:
                body = {"musicData": payload["ir"]}
            elif "code" in payload and payload["code"] is not None:
                # Runner doesn’t accept raw DSL in current server.js;
                # Plan for later support
                body = {"musicData": {"__dsl_passthrough": payload["code"]}}
            else:
                body = {"musicData": {}}

            r = requests.post(self.ingest_url, json=body, timeout=self.timeout_s)
            r.raise_for_status()

            # Expect JSON back
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}

            # Map runner fields → public API
            dsl = data.get("dsl_code", "")
            meta = {
                "executable_code": data.get("executable_code"),
                "parsed_data": data.get("parsed_data"),
                "runner_response": data,  # keep full payload for debugging
            }
            return RunnerEvalResponse(dsl=dsl, meta=meta)

        # File-drop mode (legacy / optional)
        if self.inbox_path:
            p = pathlib.Path(self.inbox_path)
            p.write_text(json.dumps(payload), encoding="utf-8")
            return RunnerEvalResponse(dsl="", meta={"queued": True, "inbox": str(p)})

        # No runner configured
        raise RuntimeError("Runner is not configured (set RUNNER_INGEST_URL or RUNNER_INBOX_PATH)")