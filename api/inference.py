"""
InferenceEngine — wraps a YOLOv8 model for prediction and annotation.
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default class map (overridden by model.names when available)
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "defect",
    1: "insulator",
}

# Colours for bounding-box drawing (BGR)
BOX_COLOURS: Dict[str, tuple] = {
    "defect":    (0,   0,   255),  # red
    "insulator": (0,   200,  0),   # green
}
DEFAULT_COLOUR = (255, 165, 0)     # orange fallback


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR NumPy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes — unsupported format or corrupted data.")
    return img


def _bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    """Encode a BGR NumPy array to PNG bytes."""
    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("cv2.imencode failed while converting annotated image to PNG.")
    return buf.tobytes()


class InferenceEngine:
    """
    Thin wrapper around an ultralytics YOLO model.

    Parameters
    ----------
    model_path : str
        Path to the YOLOv8 weights file (e.g. ``models/best.pt``).
    confidence_threshold : float
        Minimum confidence score to include a detection (default 0.25).
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.25) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names: Dict[int, str] = DEFAULT_CLASS_NAMES.copy()
        self._load(model_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, model_path: str) -> None:
        """Attempt to load the YOLO model; sets self.model on success."""
        try:
            from ultralytics import YOLO  # deferred so the module is importable even without ultralytics installed at schema-only time

            logger.info("Loading YOLO model from: %s", model_path)
            self.model = YOLO(model_path)

            # Prefer names embedded in the checkpoint
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = {int(k): str(v) for k, v in self.model.names.items()}
            logger.info("Model loaded successfully. Classes: %s", self.class_names)
        except FileNotFoundError:
            model_url = os.getenv("MODEL_URL")
            if model_url:
                try:
                    logger.info("Model file not found at '%s'. Downloading from MODEL_URL: %s", model_path, model_url)
                    import urllib.request
                    from pathlib import Path
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(model_url, model_path)
                    logger.info("Downloaded model weights to %s. Reloading...", model_path)
                    self.model = YOLO(model_path)
                    if hasattr(self.model, "names") and self.model.names:
                        self.class_names = {int(k): str(v) for k, v in self.model.names.items()}
                    logger.info("Model loaded successfully. Classes: %s", self.class_names)
                    return
                except Exception as dl_err:
                    logger.error("Failed to download model weights from %s: %s", model_url, dl_err)
            logger.warning("Model file not found at '%s'. Running in fallback mode (no inference).", model_path)
            self.model = None
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load model: %s", exc)
            self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 inference on raw image bytes.

        Returns
        -------
        list of dict
            Each dict has keys: class_id, class_name, confidence, bbox ([x1,y1,x2,y2]).
        """
        if not self.is_loaded:
            return []

        bgr = _bytes_to_bgr(image_bytes)

        t0 = time.perf_counter()
        results = self.model.predict(
            source=bgr,
            conf=self.confidence_threshold,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf   = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(
                    {
                        "class_id":   cls_id,
                        "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(conf, 4),
                        "bbox":       [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "_inference_ms": elapsed_ms,  # only on first box; stripped below
                    }
                )

        # Attach timing only to the first entry so callers can extract it
        return detections

    def get_inference_time_ms(self, image_bytes: bytes) -> float:
        """Return raw inference time without full prediction (used internally)."""
        if not self.is_loaded:
            return 0.0
        bgr = _bytes_to_bgr(image_bytes)
        t0 = time.perf_counter()
        self.model.predict(source=bgr, conf=self.confidence_threshold, verbose=False)
        return (time.perf_counter() - t0) * 1000.0

    def predict_timed(self, image_bytes: bytes):
        """
        Returns (detections, inference_time_ms, image_width, image_height).
        Single inference call — avoids running the model twice.
        """
        if not self.is_loaded:
            bgr = _bytes_to_bgr(image_bytes)
            h, w = bgr.shape[:2]
            return [], 0.0, w, h

        bgr = _bytes_to_bgr(image_bytes)
        h, w = bgr.shape[:2]

        t0 = time.perf_counter()
        results = self.model.predict(
            source=bgr,
            conf=self.confidence_threshold,
            verbose=False,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        detections: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf   = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(
                    {
                        "class_id":   cls_id,
                        "class_name": self.class_names.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(conf, 4),
                        "bbox":       [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    }
                )

        return detections, elapsed_ms, w, h

    def draw_boxes(self, image_bytes: bytes, predictions: List[Dict[str, Any]]) -> bytes:
        """
        Draw bounding boxes and labels on the image.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data (JPEG / PNG / BMP …).
        predictions : list of dict
            Output of :meth:`predict_timed` or :meth:`predict`.

        Returns
        -------
        bytes
            PNG-encoded annotated image.
        """
        bgr = _bytes_to_bgr(image_bytes)

        for det in predictions:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            class_name = det.get("class_name", "unknown")
            confidence = det.get("confidence", 0.0)

            colour = BOX_COLOURS.get(class_name, DEFAULT_COLOUR)
            thickness = 2

            # Draw rectangle
            cv2.rectangle(bgr, (x1, y1), (x2, y2), colour, thickness)

            # Prepare label
            label = f"{class_name} {confidence:.2f}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thick = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

            # Label background
            lx1, ly1 = x1, max(y1 - th - baseline - 4, 0)
            lx2, ly2 = x1 + tw + 4, y1
            cv2.rectangle(bgr, (lx1, ly1), (lx2, ly2), colour, cv2.FILLED)

            # Label text (white)
            cv2.putText(
                bgr, label,
                (x1 + 2, y1 - baseline - 2),
                font, font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA,
            )

        return _bgr_to_png_bytes(bgr)
