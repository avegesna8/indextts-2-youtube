FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libsndfile1 python3 python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# Python
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# Install your Python deps first (cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone the IndexTTS repo with inference code
RUN git clone https://github.com/IndexTeam/index-tts.git /app/index-tts

# Copy your app last
COPY . .

# Serverless entry
ENV RUNPOD_HANDLER=handler.py
CMD ["python3", "-m", "runpod.serverless"]
