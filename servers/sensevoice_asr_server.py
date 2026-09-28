"""SenseVoice ASR server on AX650N via pyaxengine — works both on-chip (AX650) and
on an AXCL PCIe card (x86/aarch64 host).

Uses the python runtime shipped in https://huggingface.co/AXERA-TECH/SenseVoice :

    SENSEVOICE_DIR=/path/to/SenseVoice   # contains python/ and sensevoice_ax650/
    AXCL_DEVICE_ID=0                     # AXCL card index (ignored on-chip)
    python servers/sensevoice_asr_server.py --port 8013

API (same as servers/sherpa_asr_server.py):
    POST /asr/file   multipart "file" (any format ffmpeg/librosa can read) -> {"text": ...}
    GET  /health
"""
import argparse
import contextlib
import os
import re
import sys
import tempfile
import threading

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

SENSEVOICE_DIR = os.path.abspath(os.getenv("SENSEVOICE_DIR", "./SenseVoice"))
sys.path.insert(0, os.path.join(SENSEVOICE_DIR, "python"))

app = FastAPI(title="SenseVoice ASR (AX650N)")
_model = None
_lock = threading.Lock()  # one NPU session, serialize requests
_TAG_RE = re.compile(r"<\|[^|]*\|>")


def _load(language: str):
    import axengine as axe
    from SenseVoiceAx import SenseVoiceAx

    device = os.getenv("AXCL_DEVICE_ID")
    if device is not None:
        # pyaxengine only honours the card index via provider_options (not the providers tuple)
        orig = axe.InferenceSession

        def session(path, *a, **kw):
            kw.setdefault("provider_options", [{"device_id": int(device)}])
            return orig(path, *a, **kw)

        axe.InferenceSession = session

    root = os.path.join(SENSEVOICE_DIR, "sensevoice_ax650")
    with contextlib.redirect_stdout(sys.stderr):
        model = SenseVoiceAx(
            os.path.join(root, "sensevoice.axmodel"),
            os.path.join(root, "am.mvn"),
            os.path.join(root, "tokens.txt"),
            os.path.join(root, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
            max_seq_len=256,
            beam_size=1,
            streaming=False,
        )
    model.default_language = language
    return model


@app.post("/asr/file")
async def asr_file(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "model not loaded")
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        with _lock, contextlib.redirect_stdout(sys.stderr):
            text = _model.infer(path, language=_model.default_language)
    except Exception as e:
        raise HTTPException(500, f"ASR failed: {e}")
    finally:
        os.remove(path)
    return {"text": re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.getenv("ASR_API_PORT", 8013)))
    ap.add_argument("--language", default="auto", choices=["auto", "zh", "en", "yue", "ja", "ko"])
    args = ap.parse_args()
    _model = _load(args.language)
    uvicorn.run(app, host=args.host, port=args.port)
