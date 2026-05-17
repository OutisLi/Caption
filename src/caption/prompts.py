"""Prompt builders for subtitle translation and optimization."""

import json
from collections.abc import Sequence

from caption.types import SubtitleCue, WordSpan


def build_translation_prompt(cues: Sequence[SubtitleCue], target_language: str) -> str:
    """
    Build a JSON translation prompt.

    Parameters
    ----------
    cues : Sequence[SubtitleCue]
        Cues to translate.
    target_language : str
        Translation target language.

    Returns
    -------
    str
        Prompt text.
    """
    payload = [{"id": cue.index, "text": cue.source_text} for cue in cues]
    return (
        "You are a professional subtitle translation expert.\n"
        "\n"
        "[STRICT SUBTITLE TRANSLATION TASK]\n"
        f"Translate each item into {target_language}.\n"
        "The input may contain ASR fragments, sentence continuations, names, course numbers, or imperfect punctuation.\n"
        "Use neighboring items only to understand referents and terminology. Do not use context to invent missing content.\n"
        "\n"
        "Critical rules:\n"
        "1. Keep ids unchanged. Keep item count unchanged. Preserve item order.\n"
        "2. Do not infer, explain, merge, split, summarize, or add causal connections.\n"
        "3. If an item is fragmentary, keep the translation fragmentary instead of completing the sentence.\n"
        "4. Translate every item, including informal speech, repeated words, and incomplete clauses.\n"
        "5. Use natural subtitle wording in the target language while preserving the source meaning.\n"
        'Return JSON exactly like: {"translations":[{"id":1,"text":"..."}]}\n'
        f"Input JSON:\n{json.dumps({'items': payload}, ensure_ascii=False)}"
    )


def build_optimization_prompt(
    cues: Sequence[SubtitleCue],
    tokens: Sequence[WordSpan],
    target_language: str,
    retry_feedback: str | None = None,
    max_segment_seconds: float = 5.0,
    max_target_chars: int = 22,
    min_segment_seconds: float = 2.0,
    pause_seconds: float = 1.0,
) -> str:
    """
    Build a subtitle optimization prompt.

    Parameters
    ----------
    cues : Sequence[SubtitleCue]
        Translated cues to optimize.
    tokens : Sequence[WordSpan]
        Timestamped source tokens to segment.
    target_language : str
        Translation target language.
    retry_feedback : str | None
        Optional validation error from the previous attempt.
    max_segment_seconds : float
        Maximum duration for each optimized cue.
    max_target_chars : int
        Maximum target-language characters for each optimized cue.
    min_segment_seconds : float
        Preferred minimum duration for each optimized cue.
    pause_seconds : float
        Pause length that allows a shorter standalone cue.

    Returns
    -------
    str
        Prompt text.
    """
    raw_cues = [
        {
            "id": cue.index,
            "start": round(cue.start, 3),
            "end": round(cue.end, 3),
            "source_text": cue.source_text,
            "target_text": cue.target_text,
        }
        for cue in cues
    ]
    token_items = [
        {
            "id": index,
            "text": token.text,
            "start": round(token.start, 3),
            "end": round(token.end, 3),
        }
        for index, token in enumerate(tokens, start=1)
    ]
    target_label = target_language or "the target language when present"
    feedback = (
        f"\nPrevious attempt failed validation: {retry_feedback}\nReturn a corrected JSON response.\n"
        if retry_feedback
        else ""
    )
    return (
        "This is a subtitle generation and semantic segmentation task.\n"
        "You are given timestamped source tokens plus raw bilingual cues produced by local ASR and a first-pass translator.\n"
        "The raw cues are context only. Use token ids to choose natural subtitle boundaries.\n"
        "\n"
        "Goal:\n"
        "Create subtitle-ready cues that read naturally while staying aligned to the audio.\n"
        "You may split or merge raw cues, but every output boundary must align to source token ids.\n"
        "\n"
        "Critical rules:\n"
        "1. Do not output timestamps. The program will compute timestamps from token ids.\n"
        "2. Each output item must contain start_token_id, end_token_id, source_text, and target_text.\n"
        "3. Token ranges must be adjacent and cover every token exactly once in order.\n"
        f"4. Each output cue must be no longer than {max_segment_seconds:.1f} seconds.\n"
        f"5. Each target_text should be no longer than {max_target_chars} visible characters when target_text is present.\n"
        f"6. Avoid cues shorter than {min_segment_seconds:.1f} seconds unless there is a pause of at least {pause_seconds:.1f} seconds before or after the cue.\n"
        "7. Do not create word-by-word or dangling phrase subtitles; keep complete readable subtitle units whenever the limits allow it.\n"
        "8. Preserve meaning. Do not add facts, explanations, or inferred logical connections.\n"
        "9. Improve source_text only for punctuation, casing, spacing, obvious ASR formatting, and minor readability.\n"
        f"10. Make target_text natural in {target_label} while preserving source meaning.\n"
        "11. Prefer complete semantic units, but keep timing plausible and concise.\n"
        "Keep source_text in the source language and target_text in the requested target language when target_text is present.\n"
        'Return JSON exactly like: {"items":[{"start_token_id":1,"end_token_id":4,"source_text":"...","target_text":"..."}]}\n'
        f"{feedback}"
        f"Input JSON:\n{json.dumps({'tokens': token_items, 'raw_cues': raw_cues}, ensure_ascii=False)}"
    )
