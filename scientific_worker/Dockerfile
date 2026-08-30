# TopoScout scientific worker (Cloud Run, CPU).
#
# Build context = build/worker_context produced by
#     python scripts/vendor_worker_sources.py
# (vendored research sources actually imported by the strict loader, the
# worker package, requirements.txt, and the fixed 335 MB T0 checkpoint).
#
#     docker build -f scientific_worker/Dockerfile -t toposcout-worker build/worker_context
#
# The request surface cannot choose model/checkpoint/thresholds; the paths
# below are baked in and only overridable at deploy time by trusted config.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    TOPOSCOUT_SW_RESEARCH_ROOT=/workspace/research \
    TOPOSCOUT_SW_CHECKPOINT=/models/best_checkpoint.pth \
    TOPOSCOUT_REAL_THREADS=2

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY research ./research
COPY scientific_worker ./scientific_worker
COPY vendor_manifest.json ./
COPY models/best_checkpoint.pth /models/best_checkpoint.pth

EXPOSE 8080

CMD ["sh", "-c", "uvicorn scientific_worker.app:app --host 0.0.0.0 --port ${PORT}"]
