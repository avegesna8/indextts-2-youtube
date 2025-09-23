FROM python:3.10-slim

# System tools some pip installs need (git). Keep it tiny.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (keep these pure-Python to avoid system libs)
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Tell RunPod which handler to run (your handler.py in repo root)
ENV RUNPOD_HANDLER=handler.py

# Start the serverless runtime
CMD ["python", "-m", "runpod.serverless"]
