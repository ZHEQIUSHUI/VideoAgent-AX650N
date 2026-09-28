"""Gradio web UI:  python webui.py [--port 7869] [--working-dir ./working_dir]

Both indexing and answering are visualised step by step: segment cards appear as
they are captioned, and every query shows its token budget, the chosen strategy,
retrieval hits, VLM re-watching, map-reduce summaries and the final context.
"""
import argparse
import hashlib
import html
import json
import logging
import os
import queue
import threading
import urllib.parse

import gradio as gr

from VideoAgent import VideoAgent
from VideoAgent.media import export_clip, fmt_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

EXAMPLES = ["用简体中文描述这段画面", "视频里出现了哪些人物？他们在做什么？", "有人在宣读告示吗？在什么时候？"]

CSS = """
.va-steps{display:flex;flex-direction:column;gap:10px;font-size:14px}
.va-step{border:1px solid var(--border-color-primary);border-radius:10px;padding:10px 12px;
  background:var(--block-background-fill)}
.va-step h4{margin:0 0 6px;font-size:14px;display:flex;align-items:center;gap:8px}
.va-dot{width:10px;height:10px;border-radius:50%;background:rgba(127,127,127,.5);flex:none}
.va-run .va-dot{background:#f59e0b;animation:va-pulse 1s infinite}
.va-done .va-dot{background:#10b981}
.va-skip{opacity:.55}
@keyframes va-pulse{50%{opacity:.3}}
.va-muted{color:var(--body-text-color-subdued);font-size:12px}
.va-bar{position:relative;height:18px;border-radius:9px;background:rgba(127,127,127,.2);overflow:hidden;margin:6px 0}
.va-bar>span{position:absolute;left:0;top:0;bottom:0;border-radius:9px}
.va-bar>i{position:absolute;top:-2px;bottom:-2px;width:2px;background:#ef4444}
.va-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px}
.va-legend b{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.va-badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;
  background:var(--color-accent-soft);color:var(--color-accent)}
.va-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px}
.va-card{border:1px solid var(--border-color-primary);border-radius:8px;overflow:hidden;
  background:var(--background-fill-secondary);font-size:12px;line-height:1.45}
.va-card img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#000}
.va-card .va-body{padding:6px 8px}
.va-card .va-t{font-weight:600;display:flex;justify-content:space-between;gap:6px}
.va-card .va-txt{display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical;overflow:hidden}
.va-card .va-asr{color:var(--body-text-color-subdued);margin-top:3px;display:-webkit-box;
  -webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.va-card.va-hit{outline:2px solid #10b981}
.va-score{height:4px;background:rgba(127,127,127,.2);border-radius:2px;margin-top:4px}
.va-score>span{display:block;height:100%;border-radius:2px;background:#10b981}
.va-list{display:flex;flex-direction:column;gap:6px}
.va-row{display:flex;gap:8px;align-items:flex-start}
.va-row img{width:96px;aspect-ratio:16/9;object-fit:cover;border-radius:4px;flex:none}
.va-chip{display:inline-block;margin:2px 4px 2px 0;padding:1px 6px;border-radius:4px;font-size:12px;
  border:1px solid var(--border-color-primary)}
.va-prog{height:8px;border-radius:4px;background:rgba(127,127,127,.2);overflow:hidden}
.va-prog>span{display:block;height:100%;background:var(--color-accent);transition:width .3s}

/* ---- "watch along" indexing view ---- */
.va-watch{display:grid;grid-template-columns:minmax(260px,1.1fr) 1fr;gap:12px;align-items:stretch}
@media (max-width:900px){.va-watch{grid-template-columns:1fr}}
.va-screen{position:relative;aspect-ratio:16/9;background:#000;border-radius:10px;overflow:hidden}
.va-screen img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:0}
.va-screen img:only-of-type{opacity:1}
.va-scan{position:absolute;left:0;right:0;height:30%;top:-30%;pointer-events:none;
  background:linear-gradient(to bottom,transparent,rgba(16,185,129,.28),transparent);animation:va-scan 1.8s linear infinite}
@keyframes va-scan{to{top:100%}}
.va-screen .va-hud{position:absolute;left:8px;top:6px;color:#fff;font:600 12px/1.4 monospace;
  text-shadow:0 1px 2px #000}
.va-screen .va-rec{position:absolute;right:8px;top:6px;color:#ef4444;font:700 12px monospace;animation:va-pulse 1s infinite}
.va-screen .va-corners{position:absolute;inset:8px;pointer-events:none;
  background:linear-gradient(#10b981,#10b981) left top/18px 2px no-repeat,linear-gradient(#10b981,#10b981) left top/2px 18px no-repeat,
  linear-gradient(#10b981,#10b981) right top/18px 2px no-repeat,linear-gradient(#10b981,#10b981) right top/2px 18px no-repeat,
  linear-gradient(#10b981,#10b981) left bottom/18px 2px no-repeat,linear-gradient(#10b981,#10b981) left bottom/2px 18px no-repeat,
  linear-gradient(#10b981,#10b981) right bottom/18px 2px no-repeat,linear-gradient(#10b981,#10b981) right bottom/2px 18px no-repeat}
.va-think{border:1px solid var(--border-color-primary);border-radius:10px;padding:10px 12px;
  background:var(--block-background-fill);font-size:13px;line-height:1.6;overflow:auto;max-height:320px}
.va-think .va-typing{animation:va-type 2.4s steps(40,end) both}
@keyframes va-type{from{clip-path:inset(0 100% 0 0)}to{clip-path:inset(0 0 0 0)}}
.va-cursor::after{content:"▋";animation:va-pulse .8s infinite;color:var(--color-accent)}
.va-tl{position:relative;margin:12px 0 4px}
.va-tl-row{display:flex;gap:2px;height:34px}
.va-tl-seg{flex:1;border-radius:3px;background-color:rgba(127,127,127,.2);background-size:cover;background-position:center;
  position:relative;overflow:hidden}
.va-tl-seg.va-cur{outline:2px solid #f59e0b;animation:va-pulse 1s infinite}
.va-tl-seg.va-ok::after{content:"";position:absolute;inset:0;box-shadow:inset 0 -3px #10b981}
.va-tl-seg.va-ctx{outline:2px solid #10b981}
.va-tl-seg.va-hitm::after{content:"";position:absolute;inset:0;box-shadow:inset 0 -3px #6366f1}
.va-tl-seg.va-dim{opacity:.35}
.va-tl-chap{display:flex;gap:2px;height:16px;margin-top:3px}
.va-tl-chap>div{border-radius:3px;background:var(--color-accent-soft);color:var(--color-accent);font-size:10px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:0 4px;line-height:16px;animation:va-fade .6s}
.va-tl-head{position:absolute;top:-4px;bottom:-4px;width:2px;background:#ef4444;transition:left .5s;
  box-shadow:0 0 6px #ef4444}
.va-tl-axis{display:flex;justify-content:space-between;font:11px monospace;color:var(--body-text-color-subdued)}
.va-new{animation:va-fade .7s}
@keyframes va-fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.va-cyc2 img{animation:va-cyc2 1.4s infinite}@keyframes va-cyc2{0%,49.99%{opacity:1}50.00%,100%{opacity:0}}
.va-cyc3 img{animation:va-cyc3 2.1s infinite}@keyframes va-cyc3{0%,33.32%{opacity:1}33.33%,100%{opacity:0}}
.va-cyc4 img{animation:va-cyc4 2.8s infinite}@keyframes va-cyc4{0%,24.99%{opacity:1}25.00%,100%{opacity:0}}
.va-cyc5 img{animation:va-cyc5 3.5s infinite}@keyframes va-cyc5{0%,19.99%{opacity:1}20.00%,100%{opacity:0}}
.va-cyc6 img{animation:va-cyc6 4.2s infinite}@keyframes va-cyc6{0%,16.66%{opacity:1}16.67%,100%{opacity:0}}
.va-cyc7 img{animation:va-cyc7 4.9s infinite}@keyframes va-cyc7{0%,14.28%{opacity:1}14.29%,100%{opacity:0}}
.va-cyc8 img{animation:va-cyc8 5.6s infinite}@keyframes va-cyc8{0%,12.49%{opacity:1}12.50%,100%{opacity:0}}
.va-cyc9 img{animation:va-cyc9 6.3s infinite}@keyframes va-cyc9{0%,11.10%{opacity:1}11.11%,100%{opacity:0}}
.va-cyc10 img{animation:va-cyc10 7.0s infinite}@keyframes va-cyc10{0%,9.99%{opacity:1}10.00%,100%{opacity:0}}
"""


def esc(s) -> str:
    return html.escape(str(s or ""))


def thumb(path: str, width: int = 0) -> str:
    """URL of a frame served by Gradio (launch(allowed_paths=[working_dir])) — keeps updates tiny."""
    if not path or not os.path.exists(path):
        return ""
    return "/gradio_api/file=" + urllib.parse.quote(os.path.abspath(path))


def img_tag(path: str) -> str:
    src = thumb(path)
    return f'<img src="{src}" loading="lazy">' if src else ""


def span(start, end) -> str:
    return f"{fmt_time(start)}–{fmt_time(end)}"


def card(item: dict, hit: bool = False, show_score: bool = False) -> str:
    score = ""
    if show_score:
        pct = max(0, min(100, int(item.get("score", 0) * 100)))
        score = f'<div class="va-score"><span style="width:{pct}%"></span></div>'
    asr = f'<div class="va-asr">🎤 {esc(item["transcript"])}</div>' if item.get("transcript") else ""
    right = f'{item["score"]:.2f}' if show_score else (f'#{item["n"]}' if item.get("n") else "")
    if show_score and item.get("video"):
        right = f'{item["video"]} · {right}'
    return (f'<div class="va-card{" va-hit" if hit else ""}">{img_tag(item.get("thumb", ""))}'
            f'<div class="va-body"><div class="va-t"><span>{span(item["start"], item["end"])}</span>'
            f'<span class="va-muted">{esc(right)}</span></div>{score}'
            f'<div class="va-txt">{esc(item.get("text") or item.get("caption"))}</div>{asr}</div></div>')


def timeline(duration: float, bounds, states: dict, thumbs: dict, head: float | None = None,
             chapters=(), extra_cls: dict | None = None) -> str:
    """Segment strip of one video. states: idx -> "ok"|"cur"; thumbs: idx -> frame path."""
    segs = []
    for i, (a, b) in enumerate(bounds):
        cls = {"ok": "va-ok", "cur": "va-cur"}.get(states.get(i), "")
        if extra_cls and i in extra_cls:
            cls += " " + extra_cls[i]
        bg = thumb(thumbs[i], 160) if i in thumbs else ""
        style = f"flex:{max(b - a, 0.1):.2f};" + (f"background-image:url({bg})" if bg else "")
        segs.append(f'<div class="va-tl-seg {cls}" style="{style}" title="{span(a, b)}"></div>')
    chap = ""
    if chapters:
        cells, t = [], 0.0
        for c in chapters:
            if c["start"] > t:
                cells.append(f'<div style="flex:{c["start"] - t:.2f};background:none"></div>')
            cells.append(f'<div style="flex:{max(c["end"] - c["start"], 0.1):.2f}" title="{esc(c["text"])}">'
                         f'📖 {esc(c["text"][:30])}</div>')
            t = c["end"]
        chap = f'<div class="va-tl-chap">{"".join(cells)}</div>'
    hd = f'<div class="va-tl-head" style="left:{100 * head / max(duration, 1):.2f}%"></div>' if head is not None else ""
    axis = "".join(f"<span>{fmt_time(duration * k / 4)}</span>" for k in range(5))
    return (f'<div class="va-tl"><div class="va-tl-row">{"".join(segs)}</div>{hd}{chap}</div>'
            f'<div class="va-tl-axis">{axis}</div>')


class WatchView:
    """Live state of one indexing run: 'watching' screen + timeline + cards."""

    def __init__(self):
        self.name, self.duration, self.bounds = "", 0.0, []
        self.states, self.thumbs, self.cards, self.chapters = {}, {}, [], []
        self.current = None     # Segment being captioned
        self.last = None        # last finished Segment
        self.frac, self.msg = 0.0, ""

    def feed(self, frac, msg, data, stage):
        self.frac, self.msg = frac, msg
        if stage == "plan":
            self.__init__()
            self.name, self.duration, self.bounds = data["name"], data["duration"], data["bounds"]
            self.frac, self.msg = frac, msg
        elif stage == "watching":
            self.current = data
            self.states[data.index] = "cur"
        elif stage == "segment":
            self.states[data.index] = "ok"
            if data.frames:
                self.thumbs[data.index] = data.frames[len(data.frames) // 2]
            self.last = data
            self.cards.insert(0, card({"start": data.start, "end": data.end, "text": data.caption,
                                       "transcript": data.transcript,
                                       "thumb": self.thumbs.get(data.index, "")}))
        elif stage == "chapter" and data.get("level") == 1:
            self.chapters.append(data)
            self.chapters.sort(key=lambda c: c["start"])

    def render(self) -> str:
        prog = (f'<div class="va-prog"><span style="width:{self.frac * 100:.0f}%"></span></div>'
                f'<div class="va-muted">{esc(self.msg)}　{self.frac * 100:.0f}%</div>')
        if not self.bounds:
            return prog
        cur = self.current if self.current is not None and self.states.get(self.current.index) == "cur" else None
        if cur is not None and cur.frames:
            imgs = "".join(img_tag_full(f, i * 0.7) for i, f in enumerate(cur.frames[:10]))
            screen = (f'<div class="va-screen va-cyc{min(len(cur.frames), 10)}">{imgs}<div class="va-scan"></div>'
                      f'<div class="va-corners"></div><div class="va-hud">AI 正在观看 {span(cur.start, cur.end)}<br>'
                      f'{len(cur.frames)} 帧 · 「{esc(self.name)}」</div><div class="va-rec">● ANALYZING</div></div>')
        elif self.last is not None and self.last.frames:
            screen = (f'<div class="va-screen">{img_tag_full(self.last.frames[len(self.last.frames) // 2])}'
                      f'<div class="va-hud">{span(self.last.start, self.last.end)}</div></div>')
        else:
            screen = '<div class="va-screen"><div class="va-scan"></div><div class="va-hud">准备中…</div></div>'
        if self.last is not None:
            asr = f'<div class="va-muted">🎤 {esc(self.last.transcript)}</div>' if self.last.transcript else ""
            think = (f'<div class="va-think"><b>📝 {span(self.last.start, self.last.end)} 看到了：</b>'
                     f'<div class="va-typing">{esc(self.last.caption)}</div>{asr}'
                     f'{"<div class=va-cursor>继续观看下一段</div>" if cur is not None else ""}</div>')
        else:
            think = '<div class="va-think va-cursor">等待第一段分析结果 </div>'
        head = cur.start if cur is not None else (self.last.end if self.last is not None else 0)
        tl = timeline(self.duration, self.bounds, self.states, self.thumbs, head, self.chapters)
        grid = "".join(c if i else c.replace('class="va-card', 'class="va-new va-card', 1)
                       for i, c in enumerate(self.cards))
        return (f'{prog}<div class="va-watch">{screen}{think}</div>{tl}'
                f'<div class="va-grid" style="margin-top:10px">{grid}</div>')


def img_tag_full(path: str, delay: float = 0.0) -> str:
    src = thumb(path)
    return f'<img src="{src}" style="animation-delay:{delay:.1f}s">' if src else ""


MODE_TEXT = {
    "full": ("全量", "全部片段都放得进上下文，直接交给 LLM 通读。"),
    "retrieve": ("检索", "内容超出上下文，针对问题检索最相关的片段。"),
    "summarize": ("分段整合", "概括类问题且内容超出上下文：分组浓缩（map），再合并回答（reduce）。"),
}


class Process:
    """State of one query's reasoning, rendered as a vertical stepper."""

    def __init__(self, meta: dict | None = None):
        self.meta = meta or {}  # video -> (duration, bounds, thumbs)
        self.plan = None
        self.hits = None
        self.refined = []
        self.condensed = {}
        self.context = None
        self.status = "准备中..."
        self.answering = False
        self.done = False
        self.error = None

    def feed(self, ev: dict) -> bool:
        t = ev["type"]
        if t == "status":
            self.status = ev["text"]
        elif t == "plan":
            self.plan = ev
        elif t == "hits":
            self.hits = ev
        elif t == "refined":
            self.refined.append(ev["item"])
        elif t == "condensed":
            g = self.condensed.setdefault(ev["level"], {"total": ev["total"], "items": [],
                                                        "pre": ev.get("precomputed", False)})
            g["items"].append(ev["item"])
        elif t == "context":
            self.context = ev
            self.answering = True
        elif t == "done":
            self.done = True
        elif t == "error":
            self.error = ev["text"]
        else:
            return False
        return True

    @staticmethod
    def _step(title, state, body="", extra=""):
        cls = {"run": "va-run", "done": "va-done", "skip": "va-skip"}.get(state, "")
        return f'<div class="va-step {cls}"><h4><span class="va-dot"></span>{title}{extra}</h4>{body}</div>'

    def render(self) -> str:
        out = []
        p = self.plan
        if p is None:
            out.append(self._step("分析上下文预算", "run", f'<div class="va-muted">{esc(self.status)}</div>'))
            return f'<div class="va-steps">{"".join(out)}</div>'
        mode = p["mode"]
        scale = max(p["total_tokens"], p["max_prefill"]) * 1.05
        w_total = 100 * p["total_tokens"] / scale
        w_budget = 100 * p["budget"] / scale
        w_prefill = 100 * p["max_prefill"] / scale
        color = "#10b981" if p["total_tokens"] <= p["budget"] else "#f59e0b"
        body = (f'<div class="va-bar"><span style="width:{w_total:.1f}%;background:{color}"></span>'
                f'<i style="left:{w_budget:.1f}%"></i><i style="left:{w_prefill:.1f}%;background:#6366f1"></i></div>'
                f'<div class="va-legend"><span><b style="background:{color}"></b>{p["segments"]} 个片段 ≈ '
                f'{p["total_tokens"]} token</span><span><b style="background:#ef4444"></b>可用预算 {p["budget"]}</span>'
                f'<span><b style="background:#6366f1"></b>prefill 上限 {p["max_prefill"]}</span>'
                f'<span class="va-muted">上下文 {p["max_context"]}，预留回答 {p["answer_tokens"]}</span></div>'
                f'<div style="margin-top:6px"><span class="va-badge">{MODE_TEXT[mode][0]}</span> '
                f'{MODE_TEXT[mode][1]}</div>')
        out.append(self._step("上下文预算（自动读取模型 prefill 上限）", "done", body))

        if self.hits is not None:
            thr = self.hits["threshold"]
            cards = "".join(card(it, hit=self.hits["used"] and it["score"] >= thr, show_score=True)
                            for it in self.hits["items"])
            note = ("绿框 = 进入回答的片段" if self.hits["used"]
                    else "概括类问题 / 无高相关片段 → 改为整合全部片段")
            out.append(self._step("多模态检索（文本 + 画面向量）", "done",
                                  f'<div class="va-muted">阈值 {thr}，{note}</div><div class="va-grid">{cards}</div>'))
        if mode == "retrieve" and self.refined:
            cards = "".join(card(it) for it in self.refined)
            out.append(self._step("VLM 带着问题重新观察", "done", f'<div class="va-grid">{cards}</div>'))
        if mode == "summarize":
            body = ""
            for level, g in sorted(self.condensed.items()):
                rows = "".join(
                    f'<div class="va-row">{img_tag(it.get("thumb", ""))}'
                    f'<div><b>{span(it["start"], it["end"])}</b><div class="va-muted">{esc(it["text"])}</div></div></div>'
                    for it in g["items"])
                title = f"第 {level} 层章节（索引时预先整合）" if g["pre"] else f"第 {level} 轮整合"
                body += (f'<div style="margin:4px 0"><b>{title}</b> '
                         f'<span class="va-muted">{len(g["items"])}/{g["total"]} 组</span></div>'
                         f'<div class="va-list">{rows}</div>')
            state = "done" if self.context else "run"
            out.append(self._step("分段整合：章节摘要（map-reduce）", state,
                                  body or f'<div class="va-muted">{esc(self.status)}</div>'))
        if self.context:
            c = self.context
            chips = "".join(f'<span class="va-chip">[{it["n"]}] {span(it["start"], it["end"])}</span>'
                            for it in c["items"])
            pct = min(100, 100 * c["tokens"] / max(1, c["budget"]))
            body = (f'<div class="va-prog"><span style="width:{pct:.0f}%"></span></div>'
                    f'<div class="va-muted">{c["tokens"]} / {c["budget"]} token，{len(c["items"])} 条资料</div>'
                    f'<div>{chips}</div>')
            for video, (duration, bounds, thumbs) in self.meta.items():
                used = [it for it in c["items"] if it["video"] == video]
                hits = [it for it in (self.hits or {}).get("items", []) if it["video"] == video]
                if not used and not hits:
                    continue
                extra = {}
                for i, (a, b) in enumerate(bounds):
                    if any(it["start"] < b and it["end"] > a for it in used):
                        extra[i] = "va-ctx"
                    elif any(it["start"] < b and it["end"] > a for it in hits):
                        extra[i] = "va-hitm"
                    else:
                        extra[i] = "va-dim"
                body += (f'<div class="va-muted" style="margin-top:8px">「{esc(video)}」绿框 = 进入上下文，'
                         f'紫线 = 检索命中</div>' + timeline(duration, bounds, {}, thumbs, None, (), extra))
            out.append(self._step("组装回答上下文", "done", body))
        if self.error:
            out.append(self._step("出错", "done", f"<div>❌ {esc(self.error)}</div>"))
        elif self.answering:
            out.append(self._step("LLM 生成回答", "done" if self.done else "run",
                                  f'<div class="va-muted">{esc("完成" if self.done else self.status)}</div>'))
        elif not self.done:
            out.append(self._step(esc(self.status), "run"))
        return f'<div class="va-steps">{"".join(out)}</div>'


def build(agent: VideoAgent) -> gr.Blocks:
    def video_choices():
        return agent.store.names()

    def video_table():
        return [[n, fmt_time(v["duration"]), len(v["segments"]), v["path"]]
                for n, v in ((n, agent.store.videos[n]) for n in agent.store.names())]

    def segments_html(name: str | None) -> str:
        if not name or name not in agent.store.videos:
            return '<div class="va-muted">选择一个视频查看它的片段时间线</div>'
        v = agent.store.videos[name]
        segs = agent.store.segments([name])
        thumbs = {x.index: x.frames[len(x.frames) // 2] for x in segs if x.frames}
        chapters = (agent.store.levels(name) or [[]])[0]
        tl = timeline(v["duration"], [(x.start, x.end) for x in segs], {x.index: "ok" for x in segs}, thumbs,
                      None, chapters)
        chap = "".join(f'<div class="va-row"><b style="white-space:nowrap">📖 {span(c["start"], c["end"])}</b>'
                       f'<div>{esc(c["text"])}</div></div>' for c in chapters)
        cards = "".join(card({"start": x.start, "end": x.end, "text": x.caption, "transcript": x.transcript,
                              "thumb": thumbs.get(x.index, "")}) for x in segs)
        return (f'{tl}<h4>章节摘要</h4><div class="va-list">{chap or "（无）"}</div>'
                f'<h4>片段</h4><div class="va-grid">{cards}</div>')

    # ------------------------------------------------------------ indexing
    def run_index(files):
        if not files:
            yield "请先选择视频文件", gr.skip(), gr.skip(), gr.skip(), gr.skip()
            return
        paths = [f if isinstance(f, str) else f.name for f in files]
        q = queue.Queue()
        result = {}

        def work():
            try:
                agent.index(paths, progress=lambda *ev: q.put(ev))
            except Exception as e:
                logging.exception("indexing failed")
                result["error"] = e
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()
        view, lines = WatchView(), []
        while True:
            ev = q.get()
            if ev is None:
                break
            frac, msg, data, stage = (list(ev) + [None, ""])[:4]
            view.feed(frac, msg, data, stage)
            if stage in ("", "plan", "chapter", "done"):
                lines.append(msg)
            yield "\n".join(lines[-50:]), view.render(), gr.skip(), gr.skip(), gr.skip()
        lines.append(f"❌ 索引失败：{result['error']}" if "error" in result else "✅ 完成")
        choices = video_choices()
        yield "\n".join(lines[-50:]), view.render(), gr.Dropdown(choices=choices, value=[]), video_table(), \
            gr.Dropdown(choices=choices)

    def remove_video(name):
        if name:
            agent.store.remove_video(name)
        choices = video_choices()
        return gr.Dropdown(choices=choices, value=None), gr.Dropdown(choices=choices, value=[]), video_table(), \
            segments_html(None)

    # --------------------------------------------------------------- query
    def make_clips(refs):
        clips = []
        for r in refs[:8]:
            path = agent.store.video_path(r["video"])
            if not path or not os.path.exists(path):
                continue
            key = hashlib.md5(f"{path}|{r['start']}|{r['end']}".encode()).hexdigest()[:10]
            out = os.path.join(agent.store.dir, "clips", f"{r['video']}_{int(r['start'])}_{key}.mp4")
            try:
                clips.append((export_clip(path, r["start"], r["end"], out),
                              f"[{r['n']}] {r['video']} {span(r['start'], r['end'])}"))
            except Exception as e:
                logging.warning("clip export failed: %s", e)
        return clips

    def video_meta(names):
        meta = {}
        for n in names or agent.store.names():
            if n in agent.store.videos:
                segs = agent.store.segments([n])
                meta[n] = (agent.store.videos[n]["duration"], [(x.start, x.end) for x in segs],
                           {x.index: x.frames[len(x.frames) // 2] for x in segs if x.frames})
        return meta

    def run_query(question, videos):
        proc = Process(video_meta(videos))
        yield "", proc.render(), []
        answer = ""
        for ev in agent.ask(question, videos or None):
            changed = proc.feed(ev)
            if ev["type"] == "delta":
                answer = ev["answer"]
                yield answer, gr.skip(), gr.skip()
            elif ev["type"] == "error":
                yield f"❌ {ev['text']}", proc.render(), []
                return
            elif ev["type"] == "done":
                yield ev["answer"], proc.render(), make_clips(ev["refs"])
            elif changed:
                yield answer, proc.render(), gr.skip()

    def status_json(force=False):
        if force:
            for m in (agent.llm, agent.vlm, agent.embedder):
                try:
                    m.refresh_limits()
                except Exception as e:
                    logging.warning("refresh failed: %s", e)
        return json.dumps(agent.status(), ensure_ascii=False, indent=2)

    with gr.Blocks(title="VideoAgent · AX650N") as demo:
        gr.Markdown("## 🎬 VideoAgent · AX650N 视频理解与问答\n"
                    "上传视频 → 自动分段、语音识别、画面描述、向量化；提问时按模型 prefill 上限自适应组织上下文，"
                    "全过程可视化。")
        with gr.Tab("智能问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        question = gr.Textbox(label="问题", placeholder="例如：用简体中文描述这段画面", scale=4)
                        videos = gr.Dropdown(choices=video_choices(), multiselect=True,
                                             label="限定视频（不选=全部）", scale=2)
                    with gr.Row():
                        ask_btn = gr.Button("提问", variant="primary")
                        clear_btn = gr.Button("清空")
                    gr.Examples(EXAMPLES, inputs=question)
                    answer = gr.Markdown(label="回答", min_height=120)
                    clips = gr.Gallery(label="参考片段", columns=3, height="auto")
                with gr.Column(scale=2):
                    gr.Markdown("#### 推理过程")
                    process = gr.HTML('<div class="va-muted">提问后这里会逐步展示：上下文预算 → 策略 → 检索/整合 → 回答</div>')
        with gr.Tab("视频索引"):
            with gr.Row():
                with gr.Column(scale=1):
                    files = gr.File(label="上传视频", file_count="multiple", file_types=["video"])
                    index_btn = gr.Button("开始索引", variant="primary")
                    index_log = gr.Textbox(label="日志", lines=8, max_lines=20, autoscroll=True)
                with gr.Column(scale=2):
                    index_view = gr.HTML('<div class="va-muted">索引时，每个片段完成后会在这里出现：'
                                         '缩略图 / 时间段 / 画面描述 / 语音转写</div>')
        with gr.Tab("已索引视频"):
            table = gr.Dataframe(headers=["视频", "时长", "片段数", "路径"], value=video_table(),
                                 interactive=False)
            with gr.Row():
                browse = gr.Dropdown(choices=video_choices(), label="查看片段时间线", scale=3)
                remove_btn = gr.Button("删除该视频索引", variant="stop", scale=1)
            timeline = gr.HTML(segments_html(None))
        with gr.Tab("系统状态"):
            status_box = gr.Code(language="json", label="模型服务与上下文上限", value=status_json)
            with gr.Row():
                refresh_btn = gr.Button("刷新")
                reprobe_btn = gr.Button("重新读取模型上限")

        ask_btn.click(run_query, [question, videos], [answer, process, clips])
        question.submit(run_query, [question, videos], [answer, process, clips])
        clear_btn.click(lambda: ("", "", "", []), None, [question, answer, process, clips])
        index_btn.click(run_index, [files], [index_log, index_view, videos, table, browse])
        browse.change(segments_html, [browse], [timeline])
        remove_btn.click(remove_video, [browse], [browse, videos, table, timeline])
        refresh_btn.click(lambda: status_json(False), None, status_box)
        reprobe_btn.click(lambda: status_json(True), None, status_box)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.getenv("WEBUI_PORT", 7869)))
    ap.add_argument("--working-dir", default=None)
    args = ap.parse_args()
    agent = VideoAgent(args.working_dir)
    demo = build(agent)
    demo.queue(default_concurrency_limit=2).launch(server_name=args.host, server_port=args.port, show_error=True,
                                                   css=CSS, allowed_paths=[agent.store.dir])
