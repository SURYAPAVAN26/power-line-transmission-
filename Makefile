# ============================================================
# Makefile — Insulator Defect Detection
# ============================================================
.PHONY: install data train api frontend docker-build docker-up docker-down test export help

PYTHON     ?= python
PIP        ?= pip
MODEL      ?= yolov8n.pt
EPOCHS     ?= 100
IMGSZ      ?= 640
BATCH      ?= 16
DEVICE     ?=
WEIGHTS    ?= models/best.pt
DATA       ?= training/dataset.yaml

# ── Default target ────────────────────────────────────────────
help:
	@echo ""
	@echo "  Insulator Defect Detection — Makefile Targets"
	@echo "  ─────────────────────────────────────────────"
	@echo "  install       Install Python + Node dependencies"
	@echo "  data          Prepare / download the CPLID dataset"
	@echo "  train         Train YOLOv8 model"
	@echo "  evaluate      Evaluate model on test split"
	@echo "  export        Export model to ONNX + TorchScript"
	@echo "  api           Run FastAPI dev server (port 8000)"
	@echo "  frontend      Run Vite dev server  (port 5173)"
	@echo "  docker-build  Build Docker images"
	@echo "  docker-up     Start all services via docker-compose"
	@echo "  docker-down   Stop all services"
	@echo "  test          Run pytest test suite"
	@echo "  lint          Run ruff + mypy"
	@echo ""

# ── Install ───────────────────────────────────────────────────
install:
	@echo "[install] Python deps …"
	$(PIP) install -r training/requirements.txt
	$(PIP) install -r api/requirements.txt
	$(PIP) install pytest httpx Pillow
	@echo "[install] Node deps …"
	cd frontend && npm install
	@echo "[install] Done."

# ── Data preparation ──────────────────────────────────────────
data:
	@echo "[data] Preparing CPLID dataset …"
	mkdir -p datasets/CPLID_Final/images/train \
	         datasets/CPLID_Final/images/val \
	         datasets/CPLID_Final/images/test \
	         datasets/CPLID_Final/labels/train \
	         datasets/CPLID_Final/labels/val \
	         datasets/CPLID_Final/labels/test
	@echo "[data] Dataset directory structure created."
	@echo "[data] Place your images + YOLO labels in datasets/CPLID_Final/"

# ── Train ─────────────────────────────────────────────────────
train:
	@echo "[train] Starting YOLOv8 training …"
	mkdir -p models
	$(PYTHON) training/train.py \
		--data $(DATA) \
		--model $(MODEL) \
		--epochs $(EPOCHS) \
		--imgsz $(IMGSZ) \
		--batch $(BATCH) \
		--device "$(DEVICE)" \
		--augment-config training/augment_config.yaml

# ── Evaluate ──────────────────────────────────────────────────
evaluate:
	@echo "[evaluate] Running model evaluation …"
	$(PYTHON) training/evaluate.py \
		--weights $(WEIGHTS) \
		--data $(DATA) \
		--split test

# ── Export ────────────────────────────────────────────────────
export:
	@echo "[export] Exporting model …"
	$(PYTHON) training/export.py \
		--weights $(WEIGHTS) \
		--imgsz $(IMGSZ)

# ── API dev server ────────────────────────────────────────────
api:
	@echo "[api] Starting FastAPI dev server on :8000 …"
	MODEL_PATH=$(WEIGHTS) uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ── Frontend dev server ───────────────────────────────────────
frontend:
	@echo "[frontend] Starting Vite dev server on :5173 …"
	cd frontend && npm run dev

# ── Docker ────────────────────────────────────────────────────
docker-build:
	@echo "[docker] Building images …"
	docker compose build

docker-up:
	@echo "[docker] Starting services …"
	mkdir -p models
	docker compose up -d
	@echo "[docker] Services running:"
	@echo "  Frontend : http://localhost"
	@echo "  API      : http://localhost:8000"
	@echo "  API docs : http://localhost:8000/docs"

docker-down:
	@echo "[docker] Stopping services …"
	docker compose down

docker-logs:
	docker compose logs -f

# ── Test ──────────────────────────────────────────────────────
test:
	@echo "[test] Running pytest …"
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Lint ──────────────────────────────────────────────────────
lint:
	@echo "[lint] Running ruff …"
	ruff check api/ training/ tests/
	@echo "[lint] Running mypy …"
	mypy api/ --ignore-missing-imports

# ── Clean ─────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	@echo "[clean] Done."
