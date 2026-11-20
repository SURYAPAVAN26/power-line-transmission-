"""
tests/test_api.py — pytest suite for Insulator Defect Detection API

Tests:
  - test_health          : GET /health returns 200 with expected fields
  - test_detect_dummy    : POST /detect with a valid JPEG returns detection JSON
  - test_invalid_file    : POST /detect with a non-image returns 422 / error
  - test_batch           : POST /detect/batch with two images returns two results

Run:
    pytest tests/ -v
"""

import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient


# ── Import app ────────────────────────────────────────────────────────────────
# The app must be importable; ensure PYTHONPATH includes project root.
def _get_client() -> "TestClient":
    from api.main import app  # noqa: PLC0415
    return TestClient(app)


@pytest.fixture(scope="module")
def client():
    return _get_client()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return a minimal valid JPEG (single grey pixel, stretched via PIL)."""
    try:
        from PIL import Image  # pillow
        img = Image.new("RGB", (width, height), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        # Fallback: minimal 1×1 JPEG bytes (hardcoded, always valid)
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
            b"\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05"
            b"\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A"
            b"\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1"
            b"\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJ"
            b"STUVWXYZ\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00"
            b"\x00\x00\xff\xd9"
        )


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return a minimal valid solid-grey PNG."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(100, 149, 237))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Minimal 1x1 red PNG
        def chunk(tag, data):
            c = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)

        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        raw = b"\x00\xff\x00\x00"  # filter byte + RGB
        idat = zlib.compress(raw)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat)
            + chunk(b"IEND", b"")
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_path" in data
        assert "classes" in data

    def test_health_status_value(self, client):
        data = client.get("/health").json()
        assert data["status"] in ("ok", "degraded")


class TestDetect:
    def test_detect_dummy_jpeg(self, client):
        """POST a valid JPEG; expect 200 with the required response fields."""
        img_bytes = _make_jpeg_bytes()
        resp = client.post(
            "/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "filename" in data
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert "inference_time_ms" in data
        assert "image_width" in data
        assert "image_height" in data

    def test_detect_dummy_png(self, client):
        """POST a valid PNG; expect 200."""
        img_bytes = _make_png_bytes()
        resp = client.post(
            "/detect",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "predictions" in data

    def test_detect_predictions_schema(self, client):
        """Each prediction must have class_name, confidence, bbox."""
        img_bytes = _make_jpeg_bytes()
        resp = client.post(
            "/detect",
            files={"file": ("schema_test.jpg", img_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        for pred in resp.json()["predictions"]:
            assert "class_name" in pred
            assert "confidence" in pred
            assert "bbox" in pred
            assert len(pred["bbox"]) == 4


class TestInvalidFile:
    def test_invalid_file_txt(self, client):
        """Uploading a plain text file should return 400 or 422."""
        resp = client.post(
            "/detect",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code in (400, 415, 422), resp.text

    def test_invalid_file_pdf(self, client):
        """Uploading a PDF should return 400 or 422."""
        fake_pdf = b"%PDF-1.4 fake content"
        resp = client.post(
            "/detect",
            files={"file": ("bad.pdf", fake_pdf, "application/pdf")},
        )
        assert resp.status_code in (400, 415, 422), resp.text

    def test_invalid_file_empty(self, client):
        """Uploading an empty file should return 400 or 422."""
        resp = client.post(
            "/detect",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code in (400, 422), resp.text


class TestBatch:
    def test_batch_two_images(self, client):
        """POST /detect/batch with two images; expect a list of two results."""
        img1 = _make_jpeg_bytes(64, 64)
        img2 = _make_png_bytes(64, 64)
        resp = client.post(
            "/detect/batch",
            files=[
                ("files", ("img1.jpg", img1, "image/jpeg")),
                ("files", ("img2.png", img2, "image/png")),
            ],
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_batch_result_shape(self, client):
        """Each item in the batch result must have the standard fields."""
        img = _make_jpeg_bytes()
        resp = client.post(
            "/detect/batch",
            files=[
                ("files", ("a.jpg", img, "image/jpeg")),
                ("files", ("b.jpg", img, "image/jpeg")),
            ],
        )
        assert resp.status_code == 200
        for item in resp.json():
            assert "filename" in item
            assert "predictions" in item
            assert "inference_time_ms" in item

    def test_batch_single_image(self, client):
        """Batch endpoint should handle a single file without error."""
        img = _make_jpeg_bytes()
        resp = client.post(
            "/detect/batch",
            files=[("files", ("single.jpg", img, "image/jpeg"))],
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 1
