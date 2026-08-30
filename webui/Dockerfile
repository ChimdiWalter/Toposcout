# toposcout-ui — the only public TopoScout service.
# Build from the repo root:
#   docker build -f webui/Dockerfile -t toposcout-ui .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /workspace

COPY webui/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY toposcout_core ./toposcout_core
COPY scientific_worker ./scientific_worker
COPY webui ./webui
# frozen cross-domain pilot evidence (small: inputs/masks/overlays/JSON)
COPY artifacts/pilots ./artifacts/pilots

EXPOSE 8080

CMD ["sh", "-c", "uvicorn webui.app:app --host 0.0.0.0 --port ${PORT}"]
