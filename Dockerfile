FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (make sure requirements.txt includes torch with CUDA cu121 wheels)
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Copy the vendored index-tts code (already in your repo)
COPY index-tts /app/index-tts

# Copy your own serverless app
COPY . .

# RunPod entry
ENV RUNPOD_HANDLER=handler.py
CMD ["python", "-m", "runpod.serverless"]
