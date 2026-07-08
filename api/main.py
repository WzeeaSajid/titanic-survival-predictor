import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.schemas import (
    PassengerInput,
    PredictionOutput,
    HealthResponse,
    ErrorResponse,
)
from src.predict import predict_survival, _pipeline, MODEL_PATH

load_dotenv()

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# LIFESPAN — runs on startup and shutdown
# Replaces deprecated @app.on_event("startup")
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("  Titanic Survival Prediction API — Starting")
    logger.info("=" * 50)

    if _pipeline is None:
        logger.error("CRITICAL: Model not loaded. Run src/train.py first.")
    else:
        logger.info(f"Model loaded from: {MODEL_PATH}")
        logger.info("API is ready to serve predictions")

    yield  # API is live and serving requests here

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("API shutting down")


# ============================================================
# APP INSTANCE
# ============================================================
app = FastAPI(
    title="Titanic Survival Prediction API",
    description=(
        "Predicts survival probability for Titanic passengers "
        "using a Random Forest model trained on the Kaggle Titanic dataset. "
        "Submit passenger details and receive a survival prediction with probability."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc UI
    openapi_url="/openapi.json",
)


# ============================================================
# MIDDLEWARE
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten this in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL EXCEPTION HANDLER
# Returns clean JSON errors instead of raw Python tracebacks
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


# ============================================================
# ROUTES
# ============================================================

# ── Health Check ──────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and the model is loaded.",
    tags=["System"],
)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None,
        model_path=MODEL_PATH,
    )


# ── Predict ───────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionOutput,
    responses={
        200: {"description": "Successful prediction",       "model": PredictionOutput},
        422: {"description": "Validation error",            "model": ErrorResponse},
        500: {"description": "Model not loaded or crashed", "model": ErrorResponse},
    },
    summary="Predict Survival",
    description=(
        "Submit passenger details to receive a survival prediction. "
        "Returns survived (0 or 1), probability (0.0–1.0), "
        "and a human-readable interpretation."
    ),
    tags=["Prediction"],
)
def predict(passenger: PassengerInput):

    # Guard: model must be loaded
    if _pipeline is None:
        logger.error("Prediction attempted but model is not loaded")
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Contact the administrator.",
        )

    logger.info(
        f"Prediction request — "
        f"Pclass: {passenger.pclass}, Sex: {passenger.sex}, "
        f"Age: {passenger.age}, Fare: {passenger.fare}, "
        f"Embarked: {passenger.embarked}"
    )

    try:
        result = predict_survival(
            pclass=passenger.pclass,
            sex=passenger.sex,
            age=passenger.age,
            fare=passenger.fare,
            sibsp=passenger.sibsp,
            parch=passenger.parch,
            embarked=passenger.embarked,
            cabin=passenger.cabin,
            name=passenger.name,
            ticket=passenger.ticket,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    logger.info(
        f"Prediction result — "
        f"Survived: {result['survived']}, "
        f"Probability: {result['probability']}, "
        f"Interpretation: {result['interpretation']}"
    )

    return PredictionOutput(
        survived=result["survived"],
        probability=result["probability"],
        interpretation=result["interpretation"],
    )


# ── Model Info ────────────────────────────────────────────────
@app.get(
    "/model-info",
    summary="Model Information",
    description="Returns model metadata including training metrics and parameters.",
    tags=["System"],
)
def model_info():
    import json

    meta_path = os.getenv("META_PATH", "models/model_meta.json")

    if not os.path.exists(meta_path):
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found. Run src/train.py first.",
        )

    with open(meta_path) as f:
        meta = json.load(f)

    return meta


# ============================================================
# MOUNT STATIC FILES
# Custom HTML/CSS/JS frontend served at "/"
# ============================================================
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_home():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


# ============================================================
# RUN DIRECTLY (development only)
# For production always use: uvicorn api.main:app
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,    # auto-reload on code changes (dev only)
        log_level="info",
    )
