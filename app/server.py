# app/server.py
# Local dev server: uvicorn app.server:app --host 0.0.0.0 --port 8000
import base64
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from app.model_runner import load, synthesize_to_wav_bytes

app = FastAPI(title="IndexTTS Local API", version="0.1.0")

@app.on_event("startup")
async def _startup():
    load()

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/tts")
async def tts(
    text: str = Form(...),
    ref_audio: UploadFile | None = None
):
    ref_path = None
    if ref_audio:
        data = await ref_audio.read()
        tmp_path = f"/tmp/{ref_audio.filename}"
        with open(tmp_path, "wb") as f:
            f.write(data)
        ref_path = tmp_path

    wav_bytes = synthesize_to_wav_bytes(text=text, ref_audio_path=ref_path)
    return JSONResponse({
        "format": "wav",
        "sr": 22050,
        "audio_base64": base64.b64encode(wav_bytes).decode("utf-8")
    })