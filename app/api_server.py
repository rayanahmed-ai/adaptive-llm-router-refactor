from fastapi import FastAPI

from app.api.v8_routes import router


app = FastAPI(
    title="Adaptive LLM Router",
    version="8.3.0",
    description=(
        "Adaptive multi-model LLM routing "
        "service using Amazon Bedrock."
    ),
)

app.include_router(
    router,
    prefix="/api",
)


@app.get("/")
def root():

    return {
        "service":
            "adaptive-llm-routing",

        "version":
            "V8.3",

        "status":
            "running",
    }