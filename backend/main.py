from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from .settings import load_settings
from .schemas import RunBody, RunnerEvalResponse, IR
from .runner_client import RunnerClient
from . import compiler_stub

settings = load_settings()

app = FastAPI(title="Music Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

runner = RunnerClient(
    ingest_url=settings.runner_ingest_url,
    inbox_path=settings.runner_inbox_path,
    timeout_s=settings.request_timeout_s
)

# Helper function to protect against non-configured IR
def require_ir(ir: IR | None) -> IR:
    if ir is None:
        raise HTTPException(status_code=400, detail="Missing 'ir'")
    return ir

@app.get("/health")
def health():
    return {"ok": True, "version": app.version}

@app.get("/test", response_class=PlainTextResponse, summary="Return sample DSL for quick frontend playback")
def test():
    # Temporary: use stub DSL so the frontend can build Tone.js playback now.
    return compiler_stub.compile_scale_to_dsl()

@app.post("/run", response_model=RunnerEvalResponse)
def run(body: RunBody):
    provided = sum(1 for v in [body.code, body.ir] if v is not None)
    if provided != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one of 'code' or 'ir'.")

    # If runner configured, forward (unchanged)
    try:
        if settings.runner_ingest_url or settings.runner_inbox_path:
            if body.code is not None:
                return RunnerEvalResponse(dsl=body.code, meta={"source":"echo"})
            else:
                # IR path
                payload = {"ir": require_ir(body.ir).model_dump()}
                return runner.eval(payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Runner error: {e}")

    # Local fallback
    if body.code:
        return RunnerEvalResponse(dsl=body.code, meta={"source": "echo"})
    else:
        dsl = compiler_stub.json_ir_to_dsl(require_ir(body.ir))
        return RunnerEvalResponse(dsl=dsl, meta={"source": "local-stub"})

