"""
Pydantic schemas for the Insulator Defect Detection API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box in [x1, y1, x2, y2] format (absolute pixel coordinates)."""
    x1: float = Field(..., description="Left edge")
    y1: float = Field(..., description="Top edge")
    x2: float = Field(..., description="Right edge")
    y2: float = Field(..., description="Bottom edge")


class Prediction(BaseModel):
    """Single object detection result."""
    class_name: str = Field(..., description="Detected class label")
    class_id: int = Field(..., description="Numeric class index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0–1")
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] in absolute pixels",
    )


class DetectionResponse(BaseModel):
    """Response for a single /detect call."""
    filename: str = Field(..., description="Original uploaded filename")
    predictions: List[Prediction] = Field(default_factory=list)
    inference_time_ms: float = Field(..., description="Model inference time in ms")
    image_width: int = Field(..., description="Input image width in pixels")
    image_height: int = Field(..., description="Input image height in pixels")
    total_detections: int = Field(..., description="Total number of detections")
    defect_count: int = Field(..., description="Number of defect detections")
    insulator_count: int = Field(..., description="Number of insulator detections")


class BatchDetectionResponse(BaseModel):
    """Response for a /detect/batch call."""
    results: List[DetectionResponse]
    total_images: int
    total_detections: int


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = Field(..., description="'ok' or 'degraded'")
    model_loaded: bool
    model_path: str
    classes: dict = Field(default_factory=dict, description="Class id -> name mapping")
    confidence_threshold: float


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    detail: str
    filename: Optional[str] = None
