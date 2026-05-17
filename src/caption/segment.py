"""Build subtitle cues from timestamped ASR words."""

from collections.abc import Iterable

from caption.types import SubtitleCue, WordSpan

END_PUNCTUATION = ".!?。！？"
SPACE_LANG_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def build_cues(words: Iterable[WordSpan], max_chars_per_cue: int, max_seconds_per_cue: float) -> list[SubtitleCue]:
    """
    Build subtitle cues from word timestamps.

    Parameters
    ----------
    words : Iterable[WordSpan]
        Timestamped words or characters.
    max_chars_per_cue : int
        Maximum text length per cue before a split is preferred.
    max_seconds_per_cue : float
        Maximum cue duration in seconds before a split is preferred.

    Returns
    -------
    list[SubtitleCue]
        Subtitle cues with source text and empty target text.
    """
    source_words = [word for word in words if word.text.strip()]
    validate_word_spans(source_words)
    cues: list[SubtitleCue] = []
    current: list[WordSpan] = []

    for word in source_words:
        if current and _would_exceed_chars(current, word, max_chars_per_cue):
            cues.append(_cue_from_words(len(cues) + 1, current))
            current = []

        current.append(word)
        if _should_close_cue(current, max_seconds_per_cue):
            cues.append(_cue_from_words(len(cues) + 1, current))
            current = []

    if current:
        cues.append(_cue_from_words(len(cues) + 1, current))

    return cues


def validate_word_spans(words: Iterable[WordSpan]) -> None:
    """
    Validate timestamped tokens.

    Parameters
    ----------
    words : Iterable[WordSpan]
        Timestamped words or characters.

    Raises
    ------
    ValueError
        If a timestamp is negative, reversed, or non-monotonic.
    """
    previous_end = 0.0
    for word in words:
        if word.start < 0 or word.end < word.start:
            raise ValueError("invalid timestamp: word spans must satisfy 0 <= start <= end")
        if word.start < previous_end:
            raise ValueError("word timestamps must be monotonic")
        previous_end = word.end


def _would_exceed_chars(current: list[WordSpan], word: WordSpan, max_chars: int) -> bool:
    text = _join_words([*current, word])
    return len(text) > max_chars


def _should_close_cue(words: list[WordSpan], max_seconds: float) -> bool:
    if not words:
        return False
    duration = words[-1].end - words[0].start
    return _ends_sentence(words[-1].text) or duration >= max_seconds


def _cue_from_words(index: int, words: list[WordSpan]) -> SubtitleCue:
    return SubtitleCue(
        index=index,
        start=words[0].start,
        end=words[-1].end,
        source_text=_join_words(words),
    )


def _join_words(words: list[WordSpan]) -> str:
    result = ""
    for word in words:
        text = word.text.strip()
        if not result:
            result = text
            continue
        if _needs_space(result[-1], text[0]):
            result += " "
        result += text
    return result


def _needs_space(left: str, right: str) -> bool:
    return left in SPACE_LANG_CHARS and right in SPACE_LANG_CHARS


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(tuple(END_PUNCTUATION))
