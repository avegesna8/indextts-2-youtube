FROM python:3.10-slim

# Small set of system deps (git for cloning; libsndfile if you use soundfile; ffmpeg optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libsndfile1 ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

WORKDIR /app

# Install Python deps (torch CUDA wheels via cu121 index)
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Clone inference code (shallow to speed up)
RUN git clone --depth=1 https://github.com/IndexTeam/index-tts.git /app/index-tts

# Copy your serverless app
COPY . .

# Serverless entry
ENV RUNPOD_HANDLER=handler.py
CMD ["python", "-m", "runpod.serverless"]
