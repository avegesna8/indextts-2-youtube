FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (torch/torchaudio/etc. go in requirements.txt)
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Your code (repo + install as package)
COPY index-tts /app/index-tts
RUN pip install --no-cache-dir -e /app/index-tts

# --- Download model weights (IndexTTS-2) ---
# If the repo is gated, uncomment ARG/ENV + login:
# ARG HF_TOKEN
# ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}
RUN mkdir -p /app/index-tts/checkpoints && \
    pip install --no-cache-dir huggingface_hub && \
    huggingface-cli download IndexTeam/IndexTTS-2 \
      --local-dir /app/index-tts/checkpoints \
      --local-dir-use-symlinks False

# Dockerfile (add near the bottom, before COPY . . if you want)
RUN mkdir -p /app/app/assets && \
    ffmpeg -hide_banner -loglevel error -f lavfi -i sine=frequency=440:duration=1 \
      -ar 22050 -ac 1 -sample_fmt s16 /app/app/assets/ref.wav     

# App code last (handler.py, app/)
COPY . .

# Runtime env
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/index-tts
ENV MODEL_DIR=/app/index-tts/checkpoints
ENV DEFAULT_MODE=tts

CMD ["python", "handler.py"]