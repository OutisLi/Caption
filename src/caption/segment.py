"""Build subtitle cues from timestamped ASR words or restored sentences."""

import math
from collections.abc import Iterable, Sequence

from caption.sentence import join_words, rendered_lengths
from caption.types import Sentence, SentenceLayout, SubtitleCue, SubtitleLine, WordSpan

END_PUNCTUATION = ".!?。！？"

# Fraction of one line slot a cut may stray from the even split. The window bounds how
# unbalanced two neighbouring lines can become before line quality is considered at all.
_CUT_SEARCH_FRACTION = 0.35

# Seconds of silence a break hint from the segmentation stage is worth. Set above the 95th
# percentile of inter-word gaps in continuous speech, so a hint outranks an ordinary pause
# while a genuinely long pause can still win where the model reported no hint at all.
_BREAK_HINT_BONUS = 0.7

# Smallest share of the character budget worth putting on a line of its own. Expressed as
# a fraction so that the floor follows the configured budget instead of contradicting it.
_MIN_LINE_CHARS_FRACTION = 0.25


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


def layout_sentence(sentence: Sentence, max_chars_per_line: int, max_seconds_per_line: float) -> SentenceLayout:
    """
    Split one sentence into display lines that fit the on-screen limits.

    The program owns this decision because it holds everything the decision needs: exact
    word timestamps and an exact word-to-text mapping. It picks the fewest lines the
    limits allow, spaces the cuts evenly in time so no line is left with a stray word,
    and prefers a cut that falls on clause punctuation whenever one is close enough to
    the even split.

    Parameters
    ----------
    sentence : Sentence
        Sentence to lay out.
    max_chars_per_line : int
        Source-language character budget for one line.
    max_seconds_per_line : float
        Longest time one line may stay on screen.

    Returns
    -------
    SentenceLayout
        Sentence with untranslated display lines covering it in order.
    """
    cuts = [0, *_line_cuts(sentence, max_chars_per_line, max_seconds_per_line), len(sentence.words)]
    lines = tuple(
        SubtitleLine(
            start=sentence.words[cuts[position]].start,
            end=sentence.words[cuts[position + 1] - 1].end,
            source_text=join_words(sentence.words[cuts[position] : cuts[position + 1]]),
        )
        for position in range(len(cuts) - 1)
    )
    return SentenceLayout(sentence=sentence, lines=lines)


def build_cues_from_layouts(layouts: Sequence[SentenceLayout]) -> list[SubtitleCue]:
    """
    Flatten laid-out sentences into numbered subtitle cues.

    Parameters
    ----------
    layouts : Sequence[SentenceLayout]
        Laid-out sentences in transcript order.

    Returns
    -------
    list[SubtitleCue]
        Subtitle cues with sequential indices.
    """
    lines = [line for layout in layouts for line in layout.lines]
    return [
        SubtitleCue(
            index=index,
            start=line.start,
            end=line.end,
            source_text=line.source_text,
            target_text=line.target_text,
        )
        for index, line in enumerate(lines, start=1)
    ]


def _line_count(sentence: Sentence, max_chars_per_line: int, max_seconds_per_line: float) -> int:
    """
    Return how many display lines a sentence needs.

    The limits set the floor, but the amount of text sets the ceiling. Duration on its own
    must never force a split: a silence inside a sentence stretches its span without adding
    anything to read, and spreading two words over two lines only makes the subtitle flicker.
    Capping by available text also keeps the layout honest towards the translation stage,
    which is asked for one line of prose per display line and cannot supply what is not there.
    """
    by_duration = math.ceil(sentence.duration / max_seconds_per_line) if max_seconds_per_line > 0 else 1
    by_length = math.ceil(len(sentence.text) / max_chars_per_line) if max_chars_per_line > 0 else 1
    minimum_line_chars = max(1, int(max_chars_per_line * _MIN_LINE_CHARS_FRACTION))
    affordable = len(sentence.text) // minimum_line_chars
    return max(1, min(len(sentence.words), affordable, max(by_duration, by_length)))


def _line_cuts(sentence: Sentence, max_chars_per_line: int, max_seconds_per_line: float) -> list[int]:
    """Return the interior word indices that separate consecutive display lines."""
    count = _line_count(sentence, max_chars_per_line, max_seconds_per_line)
    if count < 2:
        return []
    prefix = rendered_lengths(sentence.words)
    slot = prefix[-1] / count
    window = _cut_window(slot, max_chars_per_line)
    cuts: list[int] = []
    for position in range(1, count):
        # Every remaining line still needs at least one word, which keeps the candidate
        # range non-empty and the cuts strictly increasing.
        candidates = range((cuts[-1] if cuts else 0) + 1, len(sentence.words) - (count - position) + 1)
        cuts.append(_choose_cut(sentence, prefix, candidates, slot * position, window))
    return cuts


def _cut_window(slot: float, max_chars_per_line: int) -> float:
    """
    Return how far a cut may stray from the even split, in characters.

    A line grows by at most the window on each side, so bounding the window by half the
    remaining headroom bounds the line itself and no line can overflow the budget. Where
    the text nearly fills every line there is no headroom and the window correctly
    collapses to an even split.

    Only the upper bound is enforced. Overflow is a hard defect, since the line no longer
    fits the screen, while a line coming out short is merely untidy; and the ranking
    already leans towards the even split, so a cut only moves this far when a real pause
    or a break hint argues for it.
    """
    return max(0.0, min(slot * _CUT_SEARCH_FRACTION, (max_chars_per_line - slot) / 2))


def _choose_cut(
    sentence: Sentence, prefix: list[int], candidates: range, target_chars: float, window: float
) -> int:
    """
    Choose where one display line ends.

    Two signals agree on where a break belongs. A speaker pauses at syntactic boundaries and
    runs compound nouns together, so the silence before a word estimates break quality
    without knowing the language at all; a break hint marks a boundary the segmentation stage
    identified grammatically. They add, so a hinted pause outranks either alone.

    Those signals only get a say inside a window around the even split, measured in the same
    characters the display limits use. Outside it the limits are already at risk, so the cut
    closest to the split wins regardless of how natural a break elsewhere would read.
    """
    words = sentence.words

    def rank(index: int) -> tuple[bool, float, float]:
        distance = abs(prefix[index] - target_chars)
        if distance > window:
            return False, 0.0, -distance
        pause = words[index].start - words[index - 1].end
        hint = _BREAK_HINT_BONUS if index in sentence.breaks else 0.0
        return True, pause + hint, -distance

    return max(candidates, key=rank)


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
    return len(join_words([*current, word])) > max_chars


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
        source_text=join_words(words),
    )


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(tuple(END_PUNCTUATION))
