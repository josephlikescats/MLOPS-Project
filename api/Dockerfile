# Inference service for the churn + repeat-purchase models.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a code change does not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ml/ is needed at runtime, not just at training time: the API resolves the
# model through ml.registry and builds its request frame from ml.config's
# feature order, so the serving contract comes from the same code that trained.
COPY api ./api
COPY ml ./ml
COPY scripts ./scripts
COPY artifacts ./artifacts

# Run as a non-root user; nothing in the image needs write access to /app.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
