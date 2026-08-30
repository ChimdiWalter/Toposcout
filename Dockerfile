FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /workspace

COPY pyproject.toml ./
COPY app ./app
COPY toposcout_core ./toposcout_core
COPY scientific_worker ./scientific_worker
COPY demo_inputs ./demo_inputs

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["sh", "-c", "adk api_server --host 0.0.0.0 --port ${PORT} ."]
