FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (torch/torchaudio/etc. go in requirements.txt)
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# --- Copy code ---
# 1) copy the repo code (without heavy checkpoints)
COPY index-tts /app/index-tts
# 2) ensure checkpoints are included even if .dockerignore later changes
#    (safe even if they were already copied above)
COPY index-tts/checkpoints /app/index-tts/checkpoints
# 3) your app code (handler.py, app/)
COPY . .

# Runtime env
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/index-tts
# set to "echo" for quick sanity, "tts" for real synthesis
ENV DEFAULT_MODE=tts

CMD ["python", "handler.py"]
