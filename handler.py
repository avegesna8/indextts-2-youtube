from runpod import serverless

def handler(event: dict):
    body = event.get("input", {}) or {}
    text = body.get("text", "")
    return {"echo": text, "message": f"You said: {text}"}

serverless.start({"handler": handler})