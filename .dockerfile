FROM runpod/serverless:latest

WORKDIR /app
# Install Python deps
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Tell RunPod which handler to run
ENV RUNPOD_HANDLER=handler.py
CMD ["python", "-m", "runpod.serverless"]
