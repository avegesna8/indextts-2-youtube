FROM python:3.10-slim

# --- System deps (audio I/O + encoding) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---
# Put torch/torchaudio/etc. in requirements.txt (CUDA wheels if you have GPU)
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# --- Your code (vendored repo + app code) ---
# This copies your *local* clone of index-tts into the image
COPY index-tts /app/index-tts
# This copies handler.py and app/ (model_runner.py, etc.)
COPY . .

# --- Runtime env ---
# Make both your app/ and the repo’s package importable
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/index-tts

# Where the weights live (matches README)
ENV MODEL_DIR=/app/index-tts/checkpoints

# Start RunPod serverless loop (handler.py calls serverless.start)
# Tip: set DEFAULT_MODE=echo for first deploy; switch to "tts" later or send {"mode":"tts"} in payload.
ENV DEFAULT_MODE=echo
CMD ["python", "handler.py"]