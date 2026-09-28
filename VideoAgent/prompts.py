"""Prompt templates (simplified Chinese, sized for 1.7B/2B models)."""

CAPTION_FALLBACK = "请用简体中文描述这些画面。"


def caption_prompt(start: str, end: str, n_frames: int, transcript: str) -> str:
    speech = f"这段时间的语音转写为：「{transcript}」。\n" if transcript else ""
    return (
        f"这是视频 {start} 到 {end} 之间按时间顺序抽取的 {n_frames} 帧画面。\n{speech}"
        "请用简体中文客观、具体地描述这段画面：场景与环境、人物（外貌、衣着、动作、表情）、"
        "重要物体和画面中的文字，以及发生了什么事。只描述看到的内容，不要编造，150 字以内。"
    )


def refine_prompt(start: str, end: str, n_frames: int, transcript: str, query: str) -> str:
    speech = f"这段时间的语音转写为：「{transcript}」。\n" if transcript else ""
    return (
        f"这是视频 {start} 到 {end} 之间按时间顺序抽取的 {n_frames} 帧画面。\n{speech}"
        f"用户的问题是：「{query}」。\n"
        "请用简体中文详细描述画面中与这个问题相关的内容（人物、动作、物体、文字、事件）；"
        "如果画面与问题无关，就简要描述画面。只描述看到的内容，不要编造，200 字以内。"
    )


ANSWER_SYSTEM = (
    "你是视频内容问答助手。下面是从视频中检索到的片段资料，每条以 [编号] 开头并注明视频名和时间段，"
    "包含画面描述和语音转写。\n"
    "请只根据这些资料，用简体中文回答用户的问题：\n"
    "- 引用资料时在句末标注对应编号，如 [1]、[2][3]；\n"
    "- 按时间顺序组织内容；资料不足以回答时直接说明，不要编造；\n"
    "- 直接给出答案，不要输出思考过程。\n\n"
    "资料：\n{context}"
)


def map_system(query: str | None, max_chars: int) -> str:
    focus = f"围绕用户的问题「{query}」，" if query else ""
    return (
        "你是视频内容整理助手。下面是一段视频中按时间顺序排列的片段资料（画面描述和语音转写）。\n"
        f"请{focus}用简体中文把这些片段浓缩成一段按时间顺序的连贯叙述（不要分条、不要写时间），"
        "保留人物、动作、事件和关键台词，"
        f"请控制在{max_chars}字以内。只依据资料，不要编造，不要输出思考过程。"
    )


def global_hint(spans: list[tuple[float, float]]) -> str:
    """Answer skeleton for 'describe the whole video' questions, so a small model covers every part."""
    def t(x):
        x = int(round(x))
        return f"{x // 3600:02d}:{x % 3600 // 60:02d}:{x % 60:02d}"
    if len(spans) > 12:
        return "\n（请先用一句话总述，再按时间顺序分段概括整个视频，每段标注资料编号。）"
    lines = "\n".join(f"[{i}] {t(a)}-{t(b)}：……" for i, (a, b) in enumerate(spans, 1))
    return f"\n请按下面的格式回答，覆盖每一条资料：\n总述：……\n{lines}"


NO_VIDEO = "还没有已索引的视频，请先在「视频索引」页上传并索引视频。"
