"""SenseVoice ASR server on AX650N via pyaxengine — on-chip (AX650 board) or AXCL card.

Only needs numpy + pyaxengine + kaldi_native_fbank (+ ffmpeg for non-wav input); the HTTP
server is the standard library, so it runs on a stock AX650 board image.

Uses the model + python runtime from https://huggingface.co/AXERA-TECH/SenseVoice :

    SENSEVOICE_DIR=/path/to/SenseVoice   # contains python/ and sensevoice_ax650/
    AXCL_DEVICE_ID=0                     # AXCL card index (ignored on-chip)
    python3 servers/sensevoice_asr_server.py --port 8013

API:
    POST /asr/file   multipart field "file" (any audio ffmpeg can read), or a raw audio body
                     -> {"text": "..."}
    GET  /health
"""
import argparse
import contextlib
import email.parser
import email.policy
import io
import json
import os
import re
import subprocess
import sys
import threading
import types
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

SENSEVOICE_DIR = os.path.abspath(os.getenv("SENSEVOICE_DIR", "./SenseVoice"))
SR = 16000
_TAG_RE = re.compile(r"<\|[^|]*\|>")
_model = None
_lock = threading.Lock()  # one NPU session: serialise requests


def _load(language: str):
    # SenseVoiceAx.py imports torch/librosa only for streaming and file loading, neither is used here
    for name in ("torch", "librosa"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.path.insert(0, os.path.join(SENSEVOICE_DIR, "python"))
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
    with contextlib.redirect_stdout(sys.stderr):  # axengine prints to stdout
        model = SenseVoiceAx(os.path.join(root, "sensevoice.axmodel"), os.path.join(root, "am.mvn"),
                             os.path.join(root, "tokens.txt"),
                             os.path.join(root, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
                             max_seq_len=256, beam_size=1, streaming=False)
    model.default_language = language
    return model


def _decode(data: bytes) -> np.ndarray:
    """Audio bytes -> mono float32 @16 kHz. 16 kHz mono PCM wav is read directly, anything else via ffmpeg."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2:
                return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    except (wave.Error, EOFError):
        pass
    p = subprocess.run([os.getenv("FFMPEG_BIN", "ffmpeg"), "-loglevel", "error", "-i", "pipe:0",
                        "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"], input=data, capture_output=True)
    if p.returncode != 0:
        raise ValueError(f"cannot decode audio: {p.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


def transcribe(audio: np.ndarray) -> str:
    if audio.size < SR // 10:
        return ""
    with _lock, contextlib.redirect_stdout(sys.stderr):
        text = _model.infer_waveform(audio, _model.default_language)
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok", "model_loaded": _model is not None})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/asr/file", "/asr"):
            return self._json(404, {"detail": "not found"})
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        ctype = self.headers.get("Content-Type", "")
        try:
            if ctype.startswith("multipart/form-data"):
                msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                    b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
                part = next((p for p in msg.iter_parts() if p.get_param("name", header="content-disposition")
                             in ("file", "audio")), None)
                if part is None:
                    return self._json(400, {"detail": 'multipart field "file" missing'})
                body = part.get_payload(decode=True)
            self._json(200, {"text": transcribe(_decode(body))})
        except ValueError as e:
            self._json(400, {"detail": str(e)})
        except Exception as e:
            self._json(500, {"detail": f"ASR failed: {e}"})

    def log_message(self, fmt, *args):  # keep the console quiet
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.getenv("ASR_API_PORT", 8013)))
    ap.add_argument("--language", default="auto", choices=["auto", "zh", "en", "yue", "ja", "ko"])
    args = ap.parse_args()
    _model = _load(args.language)
    print(f"SenseVoice ASR on http://{args.host}:{args.port}  (POST /asr/file, GET /health)", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
