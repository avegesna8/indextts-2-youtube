FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 build-essential pkg-config && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
ARG INDEXTTS_COMMIT=ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c
RUN git clone https://github.com/index-tts/index-tts.git /opt/index-tts && cd /opt/index-tts && git checkout "$INDEXTTS_COMMIT"
WORKDIR /opt/index-tts
RUN uv sync --frozen --no-dev
RUN uv pip install runpod
ARG MODEL_REVISION=c39ce5ba981572cb187443877ff559dfb246ce63
RUN .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2.5', revision='$MODEL_REVISION', local_dir='/models/indextts25')"
ENV PATH="/opt/index-tts/.venv/bin:$PATH" PYTHONPATH=/opt/index-tts MODEL_DIR=/models/indextts25 PYTHONUNBUFFERED=1
WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "handler.py"]
