"""Parse and apply structured LLM subtitle responses."""

import json
from collections.abc import Sequence
from dataclasses import replace

from caption.segment import validate_word_spans
from caption.types import SubtitleCue, WordSpan


class TranslationError(RuntimeError):
    """Raised when the LLM response cannot be trusted."""


def parse_translation_response(response_text: str, expected_ids: Sequence[int]) -> dict[int, str]:
    """
    Parse and validate LLM translation JSON.

    Parameters
    ----------
    response_text : str
        Raw LLM response.
    expected_ids : Sequence[int]
        Cue ids expected in the response.

    Returns
    -------
    dict[int, str]
        Translation text by cue id.

    Raises
    ------
    TranslationError
        If JSON is invalid or ids do not match.
    """
    try:
        data = json.loads(strip_markdown_json(response_text))
    except json.JSONDecodeError as exc:
        raise TranslationError("translation response is not valid JSON") from exc

    items = data.get("translations")
    if not isinstance(items, list):
        raise TranslationError("translation response must contain a translations list")

    translations: dict[int, str] = {}
    ordered_texts: list[str] = []
    for item in items:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise TranslationError("translation item must contain id and text")
        text = str(item["text"]).strip()
        translations[_required_int(item["id"], "translation id")] = text
        ordered_texts.append(text)

    if set(translations) != set(expected_ids):
        if len(ordered_texts) == len(expected_ids):
            return dict(zip(expected_ids, ordered_texts))
        raise TranslationError("translation ids do not match input cue ids")

    return translations


def parse_optimized_segments_response(
    response_text: str,
    source_tokens: Sequence[WordSpan],
    max_segment_seconds: float | None = None,
    max_target_chars: int | None = None,
    min_segment_seconds: float | None = None,
    pause_seconds: float = 0.6,
) -> list[SubtitleCue]:
    """
    Parse LLM-optimized subtitle segments.

    Parameters
    ----------
    response_text : str
        Raw LLM response.
    source_tokens : Sequence[WordSpan]
        Source tokens used to compute timestamps.
    max_segment_seconds : float | None
        Optional maximum optimized cue duration.
    max_target_chars : int | None
        Optional maximum target-language cue length.
    min_segment_seconds : float | None
        Optional minimum optimized cue duration unless a nearby pause supports it.
    pause_seconds : float
        Pause length that allows a short standalone cue.

    Returns
    -------
    list[SubtitleCue]
        Optimized subtitle cues with temporary indices.

    Raises
    ------
    TranslationError
        If JSON is invalid or ids are missing, duplicated, skipped, or out of order.
    """
    try:
        data = json.loads(strip_markdown_json(response_text))
    except json.JSONDecodeError as exc:
        raise TranslationError("optimized segments response is not valid JSON") from exc

    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise TranslationError("optimized segments response must contain a non-empty items list")

    return _parse_token_optimized_segments(
        data, source_tokens, max_segment_seconds, max_target_chars, min_segment_seconds, pause_seconds
    )


def _parse_token_optimized_segments(
    data: dict,
    source_tokens: Sequence[WordSpan],
    max_segment_seconds: float | None,
    max_target_chars: int | None,
    min_segment_seconds: float | None,
    pause_seconds: float,
) -> list[SubtitleCue]:
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise TranslationError("optimized segments response must contain a non-empty items list")
    if not source_tokens:
        raise TranslationError("source tokens must not be empty")
    try:
        validate_word_spans(source_tokens)
    except ValueError as exc:
        raise TranslationError(str(exc)) from exc

    cursor = 1
    cues: list[SubtitleCue] = []
    for index, item in enumerate(items, start=1):
        if (
            not isinstance(item, dict)
            or "start_token_id" not in item
            or "end_token_id" not in item
            or "source_text" not in item
            or "target_text" not in item
        ):
            raise TranslationError(
                "optimized segment must contain start_token_id, end_token_id, source_text, and target_text"
            )
        start_token_id = _required_int(item["start_token_id"], "start_token_id")
        end_token_id = _required_int(item["end_token_id"], "end_token_id")
        if start_token_id != cursor or end_token_id < start_token_id or end_token_id > len(source_tokens):
            raise TranslationError("optimized token ranges must cover every input token exactly once in order")

        first_token = source_tokens[start_token_id - 1]
        last_token = source_tokens[end_token_id - 1]
        duration = last_token.end - first_token.start
        if max_segment_seconds is not None and duration > max_segment_seconds:
            raise TranslationError("optimized segment duration exceeds maximum")
        if (
            len(items) > 1
            and min_segment_seconds is not None
            and duration < min_segment_seconds
            and not _has_boundary_pause(source_tokens, start_token_id, end_token_id, pause_seconds)
        ):
            raise TranslationError("optimized segment is too short without a supporting pause")

        target_text = str(item["target_text"]).strip()
        if max_target_chars is not None and target_text and len(target_text) > max_target_chars:
            raise TranslationError("optimized segment target text exceeds maximum")

        cues.append(
            SubtitleCue(
                index=index,
                start=first_token.start,
                end=last_token.end,
                source_text=str(item["source_text"]).strip(),
                target_text=target_text,
            )
        )
        cursor = end_token_id + 1

    if cursor != len(source_tokens) + 1:
        raise TranslationError("optimized token ranges did not cover all input tokens")
    return cues


def _has_boundary_pause(
    source_tokens: Sequence[WordSpan], start_token_id: int, end_token_id: int, pause_seconds: float
) -> bool:
    previous_pause = (
        start_token_id > 1
        and source_tokens[start_token_id - 1].start - source_tokens[start_token_id - 2].end >= pause_seconds
    )
    next_pause = (
        end_token_id < len(source_tokens)
        and source_tokens[end_token_id].start - source_tokens[end_token_id - 1].end >= pause_seconds
    )
    return previous_pause or next_pause


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TranslationError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TranslationError(f"{field_name} must be an integer") from exc


def apply_translations(cues: Sequence[SubtitleCue], translations: dict[int, str]) -> list[SubtitleCue]:
    """
    Apply translated text to cues.

    Parameters
    ----------
    cues : Sequence[SubtitleCue]
        Source cues.
    translations : dict[int, str]
        Translation text by cue id.

    Returns
    -------
    list[SubtitleCue]
        New cues with target_text filled.

    Raises
    ------
    TranslationError
        If any cue has no translation.
    """
    result: list[SubtitleCue] = []
    for cue in cues:
        if cue.index not in translations:
            raise TranslationError(f"missing translation for cue {cue.index}")
        result.append(replace(cue, target_text=translations[cue.index]))
    return result


def strip_markdown_json(text: str) -> str:
    """
    Remove a Markdown JSON code fence if the model returned one.

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    str
        JSON text.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    return "\n".join(lines[1:-1]).strip()
