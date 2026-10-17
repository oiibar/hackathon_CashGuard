# One container: React build + FastAPI + YOLOv8 (CPU).
#   docker build -t cashguard .
#   docker run -p 8000:8000 -e API_KEY=secret cashguard
# Weights are baked in at build time, so the container runs with no network.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
# CPU-only torch keeps the image ~2 GB smaller. For NVIDIA GPU use a CUDA
# base image and drop the --index-url line.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ ./
RUN python scripts/download_weights.py
COPY --from=web /web/dist /app/frontend/dist

ENV DATA_DIR=/app/data FRONTEND_DIST=/app/frontend/dist
VOLUME /app/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
