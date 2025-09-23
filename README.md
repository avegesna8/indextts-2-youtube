# IndexTTS-2 on RunPod Serverless (Starter)

This repo gives you a **serverless HTTPS endpoint** on RunPod that returns a WAV (base64) for a given text.  
It ships with a **placeholder** TTS (1s silence) so deployment works immediately; swap in real IndexTTS-2 calls in `app/model_runner.py`.

## Deploy (RunPod Serverless)

1. Push this repo to GitHub.
2. In RunPod → **Serverless → Deploy a New Endpoint → Connect GitHub** → pick this repo.
3. Choose a GPU template (L4/L40S/A40 are fine for TTS).  
4. (Optional) Set environment variables:
   - `HF_MODEL_REPO=IndexTeam/IndexTTS-2`
   - `MODEL_DIR=/models/indextts2`
   - `HF_TOKEN=<your_hf_token>` if the HF repo needs auth
5. Deploy. You’ll receive a public URL like:
