from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from .settings import load_settings
from .schemas import RunBody, RunnerEvalResponse, IR
from .runner_client import RunnerClient
from . import compiler_stub

settings = load_settings()

# ADD THIS DEBUG LOGGING
print("=" * 50)
print("BACKEND STARTUP DEBUG")
print(f"RUNNER_INGEST_URL: {settings.runner_ingest_url}")
print(f"RUNNER_INBOX_PATH: {settings.runner_inbox_path}")
print(f"ALLOWED_ORIGINS: {settings.allowed_origins}")
print(f"Runner configured: {bool(settings.runner_ingest_url or settings.runner_inbox_path)}")
print("=" * 50)

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


def require_ir(ir: IR | None) -> IR:
    if ir is None:
        raise HTTPException(status_code=400, detail="Missing 'ir'")
    return ir


@app.get("/health")
def health():
    return {"ok": True, "version": app.version}


@app.get("/test", response_class=PlainTextResponse)
def test():
    print("TEST ENDPOINT CALLED")
    return compiler_stub.compile_scale_to_dsl()


@app.post("/run", response_model=RunnerEvalResponse)
def run(body: RunBody):
    print("RUN ENDPOINT CALLED")
    print(f"Request body: {body}")
    print(f"Code provided: {body.code is not None}")
    print(f"IR provided: {body.ir is not None}")

    provided = sum(1 for v in [body.code, body.ir] if v is not None)
    if provided != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one of 'code' or 'ir'.")

    # Check if runner is configured
    runner_configured = bool(settings.runner_ingest_url or settings.runner_inbox_path)
    print(f"Runner configured: {runner_configured}")
    print(f"Runner URL: {settings.runner_ingest_url}")

    try:
        if runner_configured:
            print("FORWARDING TO RUNNER")

            if body.code is not None:
                print("Processing DSL code")
                # Create a simple IR for DSL passthrough
                payload = {
                    "ir": {
                        "metadata": {"tempo": 120},
                        "tracks": [],
                        "__dsl_passthrough": body.code
                    }
                }
            else:
                print("Processing IR data")
                payload = {"ir": require_ir(body.ir).model_dump()}

            print(f"Payload to runner: {payload}")
            result = runner.eval(payload)
            print(f"Runner response: {result}")
            return result

    except Exception as e:
        print(f"Runner error: {e}")
        print(f"Error type: {type(e)}")
        raise HTTPException(status_code=502, detail=f"Runner error: {e}")

    # Fallback
    print("Using local fallback")
    if body.code:
        return RunnerEvalResponse(dsl=body.code, meta={"source": "echo"})
    else:
        dsl = compiler_stub.json_ir_to_dsl(require_ir(body.ir))
        return RunnerEvalResponse(dsl=dsl, meta={"source": "local-stub"})