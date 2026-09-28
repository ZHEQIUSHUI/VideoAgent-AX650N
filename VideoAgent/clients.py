"""Thin HTTP clients for the axllm OpenAI-compatible services and the ASR server.

Every model client knows its own limits (max prefill tokens / context length).
They are read from `/v1/models` (axllm reports `prefill_max_token_num` and
`max_token_len`); older servers that do not report them are measured once with
a binary search and the result is cached on disk.
"""
import base64
import io
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
import requests

from .config import Endpoint
from .tokens import TokenCounter

log = logging.getLogger("videoagent")

# chat-template tokens around each message (<|im_start|>role\n ... <|im_end|>\n) + generation prompt
_MSG_OVERHEAD = 6
_PROMPT_OVERHEAD = 8
_SAFETY = 0.94


class ModelError(RuntimeError):
    pass


class ContextOverflow(ModelError):
    """The prompt does not fit into the model's prefill / context window."""


_OVERFLOW_HINTS = ("长度上限", "context", "too long", "exceed", "prefill", "max_token")


def _raise_for_error(err, status: int = 0):
    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
    if any(h in msg.lower() for h in _OVERFLOW_HINTS):
        raise ContextOverflow(msg)
    raise ModelError(f"HTTP {status}: {msg}" if status else msg)


@dataclass
class ModelLimits:
    model_id: str
    max_prefill: int
    max_context: int
    source: str             # "server" | "env" | "probe" | "default"
    image_tokens: int = 0   # tokens one image costs (vision models only, 0 = unknown)


class _LimitCache:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def get(self, key: str):
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f).get(key)
        except Exception:
            return None

    def put(self, key: str, value: dict):
        with self._lock:
            data = {}
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
            data[key] = value
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


def jpeg_b64(img, quality: int = 90) -> str:
    """PIL image -> base64 jpeg."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _digits(n: int) -> str:
    # Qwen tokenizers split numbers into single digits -> exactly n tokens
    return ("0123456789" * (n // 10 + 1))[:n]


class _BaseModel:
    kind = "model"

    def __init__(self, ep: Endpoint, cache: _LimitCache, timeout: float = 600):
        self.ep = ep
        self.base = ep.base_url.rstrip("/")
        self.cache = cache
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {ep.api_key or 'EMPTY'}"
        self._limits: ModelLimits | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- limits
    @property
    def limits(self) -> ModelLimits:
        if self._limits is None:
            with self._lock:
                if self._limits is None:
                    self._limits = self._resolve_limits()
                    log.info("%s limits: %s", self.kind, self._limits)
        return self._limits

    @property
    def model(self) -> str:
        return self.limits.model_id

    def refresh_limits(self, force_probe: bool = False) -> ModelLimits:
        with self._lock:
            self._limits = self._resolve_limits(force_probe=force_probe)
        return self._limits

    def _post(self, path: str, body: dict, stream: bool = False) -> requests.Response:
        """POST that waits while the server is busy (axllm serve answers 429 when its queue is full)."""
        deadline = time.time() + self.timeout
        delay = 0.5
        while True:
            r = self.session.post(f"{self.base}{path}", json=body, stream=stream, timeout=self.timeout)
            if r.status_code not in (429, 503) or time.time() > deadline:
                return r
            r.close()
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)

    def _list_models(self) -> list[dict]:
        r = self.session.get(f"{self.base}/models", timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])

    def _resolve_limits(self, force_probe: bool = False) -> ModelLimits:
        try:
            models = self._list_models()
        except Exception as e:
            raise ModelError(f"{self.kind} service {self.base} unreachable: {e}") from e
        if not models:
            raise ModelError(f"{self.kind} service {self.base} reports no model")
        entry = next((m for m in models if m.get("id") == self.ep.model), None)
        if entry is None:
            if self.ep.model:
                log.warning("%s model %r not served at %s, using %r", self.kind, self.ep.model,
                            self.base, models[0].get("id"))
            entry = models[0]
        model_id = entry["id"]
        key = f"{self.base}|{model_id}"

        prefill = int(entry.get("prefill_max_token_num") or 0)
        context = int(entry.get("max_token_len") or 0)
        source = "server" if prefill else ""
        cached = self.cache.get(key) or {}

        if self.ep.max_prefill:
            prefill, source = self.ep.max_prefill, "env"
        if self.ep.max_context:
            context = self.ep.max_context
        if not prefill and not force_probe and cached.get("max_prefill"):
            prefill, source = cached["max_prefill"], cached.get("source", "probe")
            context = context or cached.get("max_context", 0)
        if not prefill:
            log.warning("%s server does not report its prefill limit, measuring it once...", self.kind)
            prefill, measured_ctx = self._probe_prefill(model_id)
            source = "probe"
            context = context or measured_ctx
        context = context or prefill

        image_tokens = cached.get("image_tokens", 0) if cached.get("max_prefill") == prefill else 0
        limits = ModelLimits(model_id, prefill, max(context, prefill), source, image_tokens)
        if self.supports_images and not limits.image_tokens:
            try:
                limits.image_tokens = self._probe_image_tokens(model_id, prefill)
            except Exception as e:
                log.warning("%s image token probe failed (%s), assuming 256 for now", self.kind, e)
                self.cache.put(key, asdict(limits))  # cache without the guess, re-probe next start
                limits.image_tokens = 256
                return limits
        self.cache.put(key, asdict(limits))
        return limits

    supports_images = False

    def _try_prompt(self, model_id: str, n_tokens: int) -> int:
        """Send a prompt of ~n tokens. Returns prompt tokens used, raises on failure."""
        raise NotImplementedError

    def _probe_prefill(self, model_id: str) -> tuple[int, int]:
        lo, hi = 0, 0
        n = 512
        # grow until failure (failures are cheap: the server rejects before prefill)
        while n <= 65536:
            try:
                self._try_prompt(model_id, n)
                lo = n
                n *= 2
            except ModelError:
                hi = n
                break
        if not hi:
            return lo, lo
        if not lo:
            n = 64
            while n < hi:
                try:
                    self._try_prompt(model_id, n)
                    lo = n
                    break
                except ModelError:
                    n *= 2
            if not lo:
                raise ModelError(f"{self.kind}: even a {n}-token prompt fails")
        while hi - lo > 32:
            mid = (lo + hi) // 2
            try:
                self._try_prompt(model_id, mid)
                lo = mid
            except ModelError:
                hi = mid
        # lo counts digits only; the whole prompt is a bit longer
        return lo + _PROMPT_OVERHEAD, lo + _PROMPT_OVERHEAD

    def _probe_image_tokens(self, model_id: str, max_prefill: int) -> int:
        return 0


class ChatModel(_BaseModel):
    """LLM or VLM behind an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, ep: Endpoint, cache: _LimitCache, counter: TokenCounter,
                 kind: str = "llm", vision: bool = False, timeout: float = 600, thinking_switch: bool = False):
        super().__init__(ep, cache, timeout)
        self.kind = kind
        self.thinking_switch = thinking_switch
        self.supports_images = vision
        self.counter = counter

    # ------------------------------------------------------------ budgeting
    def prompt_budget(self, max_new_tokens: int, n_messages: int = 2) -> int:
        """Tokens available for message *content* so that prefill and context both fit."""
        lim = self.limits
        room = min(lim.max_prefill, lim.max_context - max_new_tokens)
        room -= _PROMPT_OVERHEAD + _MSG_OVERHEAD * n_messages
        return max(0, int(room * _SAFETY))

    def count(self, text: str) -> int:
        return self.counter.count(text)

    # -------------------------------------------------------------- requests
    def _body(self, messages, max_tokens, stream, model_id=None, **extra):
        body = {
            "model": model_id or self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "stream": stream,
        }
        if self.thinking_switch:
            # Qwen3 hybrid LLMs: skip the <think> phase. Never send this to Instruct-only models
            # (e.g. Qwen3-VL-2B-Instruct): axllm then injects an empty think block the model was not
            # trained on, and it answers in English or stops at the first token.
            body["enable_thinking"] = False
        body.update(extra)
        return body

    def _calibrate(self, messages, usage):
        if not usage:
            return
        text = "".join(m["content"] if isinstance(m["content"], str) else
                       "".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")
                       for m in messages)
        n_img = sum(1 for m in messages if not isinstance(m["content"], str)
                    for p in m["content"] if p.get("type") in ("image_url", "image"))
        img = n_img * (self._limits.image_tokens if self._limits else 0)
        overhead = _PROMPT_OVERHEAD + _MSG_OVERHEAD * len(messages) + img
        self.counter.calibrate(text, int(usage.get("prompt_tokens") or 0), overhead)

    def chat(self, messages, max_tokens: int = 256, model_id: str | None = None, **extra) -> str:
        r = self._post("/chat/completions", self._body(messages, max_tokens, False, model_id, **extra))
        try:
            data = r.json()
        except ValueError:
            raise ModelError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400 or "error" in data:
            _raise_for_error(data.get("error", data), r.status_code)
        if model_id is None:
            self._calibrate(messages, data.get("usage"))
        return (data["choices"][0]["message"].get("content") or "").strip()

    def stream(self, messages, max_tokens: int = 512, **extra):
        """Yield text deltas. Raises ContextOverflow / ModelError (also mid-stream)."""
        with self._post("/chat/completions", self._body(messages, max_tokens, True, **extra), stream=True) as r:
            if r.status_code >= 400:
                try:
                    _raise_for_error(r.json().get("error", r.text), r.status_code)
                except ValueError:
                    raise ModelError(f"HTTP {r.status_code}: {r.text[:300]}")
            for raw in r.iter_lines(decode_unicode=False):
                if not raw or not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].strip()
                if payload == b"[DONE]":
                    break
                data = json.loads(payload)
                if "error" in data:
                    _raise_for_error(data["error"])
                if data.get("usage"):
                    self._calibrate(messages, data["usage"])
                choices = data.get("choices") or []
                if not choices:
                    continue
                text = (choices[0].get("delta") or {}).get("content")
                if text:
                    yield text
                if choices[0].get("finish_reason"):
                    if data.get("usage"):
                        self._calibrate(messages, data["usage"])
                    break

    # ---------------------------------------------------------------- probes
    def _try_prompt(self, model_id: str, n_tokens: int) -> int:
        msgs = [{"role": "user", "content": _digits(n_tokens)}]
        r = self._post("/chat/completions", self._body(msgs, 1, False, model_id))
        data = r.json()
        if r.status_code >= 400 or "error" in data:
            _raise_for_error(data.get("error", data), r.status_code)
        return int((data.get("usage") or {}).get("prompt_tokens") or n_tokens)

    def _probe_image_tokens(self, model_id: str, max_prefill: int) -> int:
        from PIL import Image
        img = jpeg_b64(Image.new("RGB", (64, 64), (128, 128, 128)))

        def usage(content):
            r = self._post("/chat/completions", self._body([{"role": "user", "content": content}], 1, False, model_id))
            data = r.json()
            if r.status_code >= 400 or "error" in data:
                _raise_for_error(data.get("error", data), r.status_code)
            return int(data["usage"]["prompt_tokens"])

        text_only = usage([{"type": "text", "text": "hi"}])
        with_img = usage([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                          {"type": "text", "text": "hi"}])
        return max(1, with_img - text_only)


class EmbeddingModel(_BaseModel):
    """Qwen3-VL-Embedding behind axllm's /embeddings (accepts chat-style `messages`)."""

    kind = "embedding"
    supports_images = True
    DOC_INSTRUCTION = "Represent the user's input."

    def _embed_request(self, messages, model_id=None) -> dict:
        body = {"model": model_id or self.model, "messages": messages, "encoding_format": "float",
                "add_special_tokens": True}
        r = self._post("/embeddings", body)
        try:
            data = r.json()
        except ValueError:
            raise ModelError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400 or "error" in data:
            _raise_for_error(data.get("error", data), r.status_code)
        return data

    @staticmethod
    def _messages(content, instruction=DOC_INSTRUCTION):
        return [{"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content}]

    @staticmethod
    def _normalize(vec) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def text_budget(self) -> int:
        return max(16, int((self.limits.max_prefill - 32) * _SAFETY))

    def embed_text(self, text: str, counter: TokenCounter | None = None) -> np.ndarray:
        if counter is not None:
            text = counter.truncate(text, self.text_budget())
        for _ in range(4):
            try:
                data = self._embed_request(self._messages([{"type": "text", "text": text}]))
                return self._normalize(data["data"][0]["embedding"])
            except ContextOverflow:
                text = text[: int(len(text) * 0.7)]
        raise ContextOverflow("embedding input still too long after truncation")

    def max_images(self, wanted: int) -> int:
        per = self.limits.image_tokens or 256
        return max(1, min(wanted, int((self.limits.max_prefill - 32) * _SAFETY) // per))

    def embed_images(self, images_b64: list[str]) -> np.ndarray:
        imgs = images_b64[: self.max_images(len(images_b64))]
        while True:
            content = [{"type": "image", "image": f"data:image/jpeg;base64,{b}"} for b in imgs]
            try:
                return self._normalize(self._embed_request(self._messages(content))["data"][0]["embedding"])
            except ContextOverflow:
                if len(imgs) == 1:
                    raise
                imgs = imgs[:: 2] if len(imgs) > 2 else imgs[:1]

    def _try_prompt(self, model_id: str, n_tokens: int) -> int:
        self._embed_request(self._messages([{"type": "text", "text": _digits(n_tokens)}]), model_id)
        return n_tokens

    def _probe_image_tokens(self, model_id: str, max_prefill: int) -> int:
        from PIL import Image
        item = {"type": "image", "image": "data:image/jpeg;base64," +
                jpeg_b64(Image.new("RGB", (64, 64), (128, 128, 128)))}
        data = self._embed_request(self._messages([item]), model_id)
        used = int((data.get("usage") or {}).get("prompt_tokens") or 0)
        if used:
            text_only = int((self._embed_request(self._messages([{"type": "text", "text": "hi"}]), model_id)
                             .get("usage") or {}).get("prompt_tokens") or 0)
            if 0 < text_only < used:
                return used - text_only
        # no usage reported: find how many images fit, the per-image cost follows from that
        lo, hi = 1, 64
        while lo < hi:
            mid = (lo + hi + 1) // 2
            try:
                self._embed_request(self._messages([item] * mid), model_id)
                lo = mid
            except ModelError:
                hi = mid - 1
        return max(1, (max_prefill - 32) // lo)


class ASRClient:
    """SenseVoice ASR server: POST /asr/file (multipart) -> {"text": ...}."""

    _TAG_RE = re.compile(r"<\|[^|]*\|>")

    def __init__(self, base_url: str, timeout: float = 120):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.base)

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base}/health", timeout=5).ok
        except Exception:
            return False

    def transcribe(self, wav_path: str) -> str:
        with open(wav_path, "rb") as f:
            r = requests.post(f"{self.base}/asr/file", files={"file": f}, timeout=self.timeout)
        if r.status_code != 200:
            raise ModelError(f"ASR HTTP {r.status_code}: {r.text[:200]}")
        text = self._TAG_RE.sub("", r.json().get("text", ""))
        return re.sub(r"\s+", " ", text).strip()


def wait_ready(models, timeout: float = 0):
    """Resolve limits of all models (blocks until the services answer or timeout)."""
    t0 = time.time()
    while True:
        try:
            return [m.limits for m in models]
        except ModelError:
            if time.time() - t0 > timeout:
                raise
            time.sleep(3)
