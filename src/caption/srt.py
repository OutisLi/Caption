"""Render subtitle cues as SRT text."""

from collections.abc import Iterable
from typing import Literal

from caption.types import SubtitleCue

SubtitleLanguage = Literal["source", "target"]
TranslationPosition = Literal["top", "bottom"]


def format_srt_timestamp(seconds: float) -> str:
    """
    Format seconds as an SRT timestamp.

    Parameters
    ----------
    seconds : float
        Timestamp in seconds.

    Returns
    -------
    str
        Timestamp in ``HH:MM:SS,mmm`` format.

    Raises
    ------
    ValueError
        If seconds is negative.
    """
    if seconds < 0:
        raise ValueError("seconds must be non-negative")

    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def render_srt(cues: Iterable[SubtitleCue], language: SubtitleLanguage) -> str:
    """
    Render single-language subtitles.

    Parameters
    ----------
    cues : Iterable[SubtitleCue]
        Subtitle cues to render.
    language : {"source", "target"}
        Text field to render.

    Returns
    -------
    str
        SRT content.
    """
    blocks: list[str] = []
    for cue in cues:
        text = cue.source_text if language == "source" else cue.target_text
        blocks.append(_render_block(cue.index, cue.start, cue.end, [text]))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_bilingual_srt(cues: Iterable[SubtitleCue], translation_position: TranslationPosition = "bottom") -> str:
    """
    Render bilingual subtitles.

    Parameters
    ----------
    cues : Iterable[SubtitleCue]
        Subtitle cues to render.
    translation_position : {"top", "bottom"}
        Whether translated text appears above or below source text.

    Returns
    -------
    str
        SRT content.

    Raises
    ------
    ValueError
        If translation_position is not supported.
    """
    if translation_position not in {"top", "bottom"}:
        raise ValueError("translation_position must be 'top' or 'bottom'")

    blocks: list[str] = []
    for cue in cues:
        lines = (
            [cue.target_text, cue.source_text] if translation_position == "top" else [cue.source_text, cue.target_text]
        )
        blocks.append(_render_block(cue.index, cue.start, cue.end, lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _render_block(index: int, start: float, end: float, lines: list[str]) -> str:
    if end < start:
        raise ValueError("cue end must be greater than or equal to start")
    time_range = f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}"
    return "\n".join([str(index), time_range, *lines])
