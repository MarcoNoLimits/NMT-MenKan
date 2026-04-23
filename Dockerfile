FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY scripts /app/scripts

# Download the CTranslate2 model from HF Hub (avoids LFS quota on the Space repo)
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('marconolimits/en-it-nmt-ct2', local_dir='/app/model')"

ENV MODEL_DIR=/app/model
ENV SPM_PATH=/app/model/sentencepiece.bpe.model
ENV MAX_INPUT_CHARS=2000
ENV TRANSLATION_TIMEOUT_MS=30000
ENV REQUIRE_API_KEY=0
# --- CPU threading (2 vCPU HF Spaces) ---
# One translation slot, both cores devoted to it.
ENV NMT_INTER_THREADS=1
ENV NMT_INTRA_THREADS=2
# Allow BLAS/OpenMP to use all available cores.
ENV OPENBLAS_NUM_THREADS=2
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

EXPOSE 7860

CMD ["sh", "-c", "uvicorn scripts.nmt_http_api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 75"]
