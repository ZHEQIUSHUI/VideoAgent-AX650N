"""VideoAgent facade: wires settings, model clients, index store, indexer and query engine."""
import logging
import os
from dataclasses import asdict

from .clients import ASRClient, ChatModel, EmbeddingModel, _LimitCache
from .config import Settings
from .indexer import Indexer
from .query import QueryEngine
from .store import Store
from .tokens import TokenCounter

log = logging.getLogger("videoagent")


class VideoAgent:
    def __init__(self, working_dir: str | None = None, settings: Settings | None = None):
        self.settings = settings or Settings()
        if working_dir:
            self.settings.working_dir = working_dir
        s = self.settings
        self.store = Store(s.working_dir)
        cache = _LimitCache(os.path.join(self.store.dir, "model_limits.json"))
        self.counter = TokenCounter(s.tokenizer_json)
        self.llm = ChatModel(s.llm, cache, self.counter, "llm", vision=False, timeout=s.request_timeout,
                             thinking_switch=s.llm_disable_thinking)
        self.vlm = ChatModel(s.vlm, cache, self.counter, "vlm", vision=True, timeout=s.request_timeout)
        self.embedder = EmbeddingModel(s.embedding, cache, timeout=s.request_timeout)
        self.asr = ASRClient(s.asr_url)
        self.indexer = Indexer(self)
        self.engine = QueryEngine(self)

    # ------------------------------------------------------------- public
    def index(self, paths: list[str] | str, progress=None) -> list[str]:
        """Index videos (already indexed files are skipped). Returns their names."""
        if isinstance(paths, str):
            paths = [paths]
        return [self.indexer.index(p, progress)[0] for p in paths]

    def ask(self, query: str, videos: list[str] | None = None):
        """Stream answer events, see QueryEngine.ask."""
        return self.engine.ask(query, videos)

    def query(self, query: str, videos: list[str] | None = None) -> dict:
        """Blocking helper: {"answer", "refs", "mode"}; raises on error."""
        for ev in self.ask(query, videos):
            if ev["type"] == "error":
                raise RuntimeError(ev["text"])
            if ev["type"] == "done":
                return ev
        raise RuntimeError("no answer")

    def status(self) -> dict:
        out = {}
        for name, m in (("llm", self.llm), ("vlm", self.vlm), ("embedding", self.embedder)):
            try:
                out[name] = {"url": m.base, **asdict(m.limits)}
            except Exception as e:
                out[name] = {"url": m.base, "error": str(e)}
        out["asr"] = {"url": self.asr.base or "(disabled)",
                      "ok": self.asr.health() if self.asr.enabled else None}
        out["token_counter"] = "exact (tokenizer.json)" if self.counter.exact else "estimate + auto-calibration"
        out["videos"] = self.store.names()
        return out
