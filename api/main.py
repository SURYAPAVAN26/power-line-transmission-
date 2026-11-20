"""
FastAPI application — Insulator Defect Detection API.

Endpoints
---------
POST /detect          Single image inference
POST /detect/batch    Batch image inference
GET  /health          Model status
POST /annotated       Returns annotated PNG
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .inference import InferenceEngine
from .schemas import (
    BatchDetectionResponse,
    DetectionResponse,
    HealthResponse,
    Prediction,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()  # loads .env if present

MODEL_PATH          = os.getenv("MODEL_PATH", "models/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
LOG_LEVEL           = os.getenv("LOG_LEVEL", "info").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global engine (populated at startup)
# ---------------------------------------------------------------------------

engine: InferenceEngine | None = None

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    global engine
    logger.info("Loading inference engine (model: %s) …", MODEL_PATH)
    engine = InferenceEngine(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    if engine.is_loaded:
        logger.info("Model ready. Classes: %s", engine.class_names)
    else:
        logger.warning("Model NOT loaded — /detect will return empty predictions.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Insulator Defect Detection API",
    description=(
        "YOLOv8-powered REST API for detecting defects and insulators in aerial images. "
        "Upload an image and receive bounding-box predictions in JSON or as an annotated PNG."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _read_and_validate(file: UploadFile) -> bytes:
    """Read upload bytes; raise 400/413 on bad input."""
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {sorted(ALLOWED_MIME_TYPES)}",
        )
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_IMAGE_BYTES // (1024*1024)} MB.",
        )
    return data


def _build_detection_response(
    filename: str,
    detections: list,
    inference_ms: float,
    width: int,
    height: int,
) -> DetectionResponse:
    predictions = [
        Prediction(
            class_id=d["class_id"],
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox=d["bbox"],
        )
        for d in detections
    ]
    defect_count    = sum(1 for p in predictions if p.class_name == "defect")
    insulator_count = sum(1 for p in predictions if p.class_name == "insulator")

    return DetectionResponse(
        filename=filename,
        predictions=predictions,
        inference_time_ms=inference_ms,
        image_width=width,
        image_height=height,
        total_detections=len(predictions),
        defect_count=defect_count,
        insulator_count=insulator_count,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model health check",
    tags=["Utility"],
)
async def health() -> HealthResponse:
    """Return current model status and class map."""
    loaded = engine is not None and engine.is_loaded
    classes = engine.class_names if engine else {}
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_path=MODEL_PATH,
        classes=classes,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )


@app.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect insulators / defects in a single image",
    tags=["Inference"],
)
async def detect(file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, WEBP)")) -> DetectionResponse:
    """
    Upload a single image and receive a list of bounding-box predictions.

    - **file**: multipart image upload
    """
    image_bytes = await _read_and_validate(file)
    filename    = file.filename or "upload.jpg"

    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference engine not initialised.")

    try:
        detections, inference_ms, width, height = engine.predict_timed(image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference error for file '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    return _build_detection_response(filename, detections, inference_ms, width, height)


@app.post(
    "/detect/batch",
    response_model=BatchDetectionResponse,
    summary="Detect insulators / defects in multiple images",
    tags=["Inference"],
)
async def detect_batch(
    files: List[UploadFile] = File(..., description="One or more image files"),
) -> BatchDetectionResponse:
    """
    Upload multiple images at once; each is processed independently.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference engine not initialised.")

    results: list[DetectionResponse] = []
    total_detections = 0

    for file in files:
        filename = file.filename or "upload.jpg"
        try:
            image_bytes = await _read_and_validate(file)
            detections, inference_ms, width, height = engine.predict_timed(image_bytes)
            resp = _build_detection_response(filename, detections, inference_ms, width, height)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Batch inference error for '%s': %s", filename, exc)
            # Return a zero-detection entry rather than aborting the whole batch
            resp = DetectionResponse(
                filename=filename,
                predictions=[],
                inference_time_ms=0.0,
                image_width=0,
                image_height=0,
                total_detections=0,
                defect_count=0,
                insulator_count=0,
            )

        results.append(resp)
        total_detections += resp.total_detections

    return BatchDetectionResponse(
        results=results,
        total_images=len(results),
        total_detections=total_detections,
    )


@app.post(
    "/annotated",
    response_class=Response,
    summary="Return an annotated PNG with bounding boxes drawn",
    tags=["Inference"],
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Annotated image as PNG",
        }
    },
)
async def annotated(
    file: UploadFile = File(..., description="Image file to annotate"),
) -> Response:
    """
    Upload an image; receive a PNG with bounding boxes and labels drawn on it.
    """
    image_bytes = await _read_and_validate(file)

    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference engine not initialised.")

    try:
        detections, _ms, _w, _h = engine.predict_timed(image_bytes)
        png_bytes = engine.draw_boxes(image_bytes, detections)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Annotation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Annotation failed: {exc}",
        ) from exc

    return Response(content=png_bytes, media_type="image/png")
