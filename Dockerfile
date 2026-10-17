FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/scripts/download_weights.py backend/scripts/build_world_model.py backend/scripts/download_clip.py ./scripts/
RUN python scripts/download_weights.py
RUN python scripts/build_world_model.py || echo "WARNING: cash_world.pt not built, cash rule disabled"
RUN python scripts/download_clip.py
COPY backend/ ./
COPY --from=web /web/dist /app/frontend/dist

ENV YOLO_OFFLINE=1 DATA_DIR=/app/data FRONTEND_DIST=/app/frontend/dist
VOLUME /app/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
