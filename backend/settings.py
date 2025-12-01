import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    allowed_origins: list[str]
    runner_ingest_url: str | None
    runner_inbox_path: str | None
    request_timeout_s: float

def load_settings () -> Settings:
    return Settings(
        allowed_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
        runner_ingest_url=os.getenv("RUNNER_INGEST_URL"),           # example http://localhost:5001/_____
        runner_inbox_path=os.getenv("RUNNER_INBOX_PATH"),           # alternative to ingest url by file path
        request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "3"))
    )