# ⚡ Insulator Defect Detection System

> End-to-end YOLOv8-powered pipeline for detecting defects on power-line insulators from aerial imagery, with a FastAPI inference backend and React dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Insulator Defect Detector                    │
│                                                                 │
│  ┌──────────────┐     HTTP/JSON     ┌───────────────────────┐  │
│  │              │ ────────────────► │                       │  │
│  │  React + Vite│                   │  FastAPI (uvicorn)    │  │
│  │  Frontend    │ ◄──────────────── │  /detect             │  │
│  │  :5173 / :80 │   predictions +   │  /detect/batch        │  │
│  │              │   annotated img   │  /health              │  │
│  └──────────────┘                   └──────────┬────────────┘  │
│                                                │               │
│                                     ┌──────────▼────────────┐  │
│                                     │  InferenceEngine      │  │
│                                     │  ultralytics YOLO     │  │
│                                     │  models/best.pt       │  │
│                                     └───────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Training Pipeline                                        │  │
│  │  datasets/CPLID_Final  →  train.py  →  models/best.pt   │  │
│  │  augment_config.yaml   →  evaluate.py  →  metrics        │  │
│  │                        →  export.py   →  ONNX + TS       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

Docker Compose Layout
─────────────────────
  [frontend :80]  ──nginx proxy /api/──►  [api :8000]
                                              │
                                        models volume
                                        (./models/)
```

---

## Project Structure

```
insulator-defect-detector/
├── api/                        # FastAPI backend
│   ├── main.py                 # Routes: /detect, /detect/batch, /health
│   ├── inference.py            # InferenceEngine (YOLOv8 wrapper)
│   ├── schemas.py              # Pydantic request/response models
│   ├── requirements.txt
│   └── .env.example
├── frontend/                   # React + Vite + Tailwind UI
│   ├── src/
│   │   ├── App.jsx             # Main app (upload, results, batch)
│   │   ├── components/
│   │   │   ├── UploadZone.jsx
│   │   │   ├── DetectionResult.jsx
│   │   │   ├── HealthBadge.jsx
│   │   │   └── BatchResults.jsx
│   │   └── api/client.js       # Axios instance
│   └── package.json
├── training/                   # Training pipeline
│   ├── train.py                # YOLOv8 training script
│   ├── evaluate.py             # Evaluation + confusion matrix + PR curve
│   ├── export.py               # ONNX + TorchScript export + benchmark
│   ├── augment_config.yaml     # Augmentation hyperparameters
│   ├── dataset.yaml            # Dataset config (CPLID_Final)
│   └── requirements.txt
├── docker/
│   ├── Dockerfile.api          # python:3.11-slim + uvicorn
│   ├── Dockerfile.frontend     # node:20-alpine build + nginx:alpine
│   └── nginx.conf              # SPA routing + /api/ proxy
├── tests/
│   └── test_api.py             # pytest: health, detect, batch, invalid
├── models/                     # Trained weights (gitignored)
├── datasets/                   # Raw dataset (gitignored)
├── docker-compose.yml
├── Makefile
└── .gitignore
```

---

## Quickstart

### 1 — Clone & Install

```bash
git clone https://github.com/your-org/insulator-defect-detector.git
cd insulator-defect-detector
make install
```

### 2 — Prepare Dataset

```bash
make data
# Then place CPLID images + YOLO labels in:
#   datasets/CPLID_Final/images/{train,val,test}/
#   datasets/CPLID_Final/labels/{train,val,test}/
```

### 3 — Train

```bash
# Default: YOLOv8n, 100 epochs, imgsz=640
make train

# Custom
make train MODEL=yolov8m.pt EPOCHS=200 IMGSZ=1280 DEVICE=0
```

### 4 — Evaluate

```bash
make evaluate
# Outputs mAP50/95, confusion matrix PNG, PR curve PNG → runs/evaluate/
```

### 5 — Export

```bash
make export
# Exports ONNX + TorchScript → models/
# Prints latency benchmark table
```

### 6 — Run API (dev)

```bash
make api
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 7 — Run Frontend (dev)

```bash
make frontend
# UI: http://localhost:5173
```

---

## Dataset — CPLID

| Property | Value |
|---|---|
| Name | Chinese Power Line Insulator Dataset (CPLID) |
| Classes | `defect` (0), `insulator` (1) |
| Format | YOLO (`.txt` per image) |
| Source | Aerial drone imagery of 220 kV transmission lines |
| Split | 70 % train / 15 % val / 15 % test |

Place the dataset at `datasets/CPLID_Final/` following the directory structure shown above.

---

## API Reference

### `GET /health`

Returns model status.

| Field | Type | Description |
|---|---|---|
| `status` | `string` | `"ok"` or `"degraded"` |
| `model_loaded` | `bool` | Whether weights are loaded |
| `model_path` | `string` | Path to weights file |
| `classes` | `object` | `{0: "defect", 1: "insulator"}` |

---

### `POST /detect`

Detect defects in a single image.

**Request:** `multipart/form-data`  
- `file` — JPEG or PNG image

**Response:**

```json
{
  "filename": "pylon_east.jpg",
  "predictions": [
    {
      "class_name": "defect",
      "confidence": 0.87,
      "bbox": [120, 45, 210, 130]
    }
  ],
  "inference_time_ms": 23.4,
  "image_width": 1920,
  "image_height": 1080
}
```

---

### `POST /detect/batch`

Detect defects in multiple images in one call.

**Request:** `multipart/form-data`  
- `files[]` — one or more JPEG/PNG images

**Response:** Array of detection results (same schema as `/detect`).

---

### `POST /annotated`

Returns the input image annotated with bounding boxes as a PNG.

**Request:** `multipart/form-data`  
- `file` — JPEG or PNG image

**Response:** `image/png` binary stream.

---

## Docker Deploy

```bash
# 1. Build images
make docker-build

# 2. Place model weights
mkdir -p models
cp /path/to/best.pt models/

# 3. Start stack
make docker-up
# Frontend : http://localhost
# API      : http://localhost:8000
# API Docs : http://localhost:8000/docs

# 4. Stop stack
make docker-down
```

> [!NOTE]  
> The `models/` directory is bind-mounted into the API container as a read-only volume.  
> You do not need to rebuild the image when swapping weights.

---

## Training Results *(placeholder — update after training)*

| Model | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall | FPS (CPU) |
|---|---|---|---|---|---|
| YOLOv8n | — | — | — | — | — |
| YOLOv8s | — | — | — | — | — |
| YOLOv8m | — | — | — | — | — |

> [!TIP]  
> Run `make evaluate` after training to populate this table automatically.

---

## Running Tests

```bash
make test
# or directly:
pytest tests/ -v
```

Test cases:

| Test | Description |
|---|---|
| `test_health_status_200` | Health endpoint returns HTTP 200 |
| `test_health_response_shape` | Health JSON contains required fields |
| `test_detect_dummy_jpeg` | JPEG upload returns valid detection JSON |
| `test_detect_dummy_png` | PNG upload returns valid detection JSON |
| `test_detect_predictions_schema` | Each prediction has `class_name`, `confidence`, `bbox` |
| `test_invalid_file_txt` | Text file upload returns 400/422 |
| `test_invalid_file_empty` | Empty file returns 400/422 |
| `test_batch_two_images` | Batch endpoint returns list of two results |
| `test_batch_result_shape` | Each batch result has the standard fields |

---

## Augmentation Strategy

Configured in `training/augment_config.yaml`:

| Technique | Value | Purpose |
|---|---|---|
| `mosaic` | 1.0 | Multi-image context |
| `mixup` | 0.15 | Label smoothing |
| `copy_paste` | 0.3 | Small-object synthesis |
| `degrees` | ±15° | Rotation invariance |
| `hsv_s / hsv_v` | 0.7 / 0.4 | Lighting variation |
| `flipud` | 0.3 | Aerial orientation |
| `erasing` | 0.4 | Occlusion robustness |

---

## Cloud Deployment (Render & Vercel)

This project is configured to deploy with zero hassle on free tiers:
- **Backend (FastAPI + YOLOv8)**: Deploy on **Render** (as a Docker Web Service).
- **Frontend (React + Vite)**: Deploy on **Vercel** (static SPA with dynamic API route).

### Step 1: Deploy Backend to Render

1. Push your repository to GitHub.
2. Sign in to [render.com](https://render.com) and click **New + > Web Service**.
3. Select your repository.
4. Choose **Docker** as the environment:
   - **Dockerfile Path**: `docker/Dockerfile.api`
   - **Docker Context**: `.`
   - *(Or click **New + > Blueprint** and select `render.yaml` for automatic setup)*
5. Under **Environment Variables**:
   - `MODEL_PATH`: `models/best.pt`
   - *(Optional)* `MODEL_URL`: Direct link to download weights (e.g. from GitHub Releases or HuggingFace) if not committed to git.
6. Click **Create Web Service**. Once deployed, copy your Render URL (e.g., `https://insulator-defect-api.onrender.com`).

### Step 2: Deploy Frontend to Vercel

1. Sign in to [vercel.com](https://vercel.com) and click **Add New > Project**.
2. Import your GitHub repository.
3. Configure the project:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Expand **Environment Variables** and add:
   - `VITE_API_URL`: Your Render backend URL (e.g. `https://insulator-defect-api.onrender.com`)
5. Click **Deploy**.

---

## License

MIT © 2024 — See `LICENSE` for details.
