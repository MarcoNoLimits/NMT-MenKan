FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY scripts /app/scripts
COPY nllb_int8 /app/nllb_int8

ENV MODEL_DIR=/app/nllb_int8
ENV SPM_PATH=/app/nllb_int8/sentencepiece.bpe.model
ENV MAX_INPUT_CHARS=2000
ENV TRANSLATION_TIMEOUT_MS=10000
ENV REQUIRE_API_KEY=0

EXPOSE 7860

CMD ["sh", "-c", "uvicorn scripts.nmt_http_api:app --host 0.0.0.0 --port ${PORT:-7860}"]
