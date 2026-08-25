"""Group an ASR word stream into sentences.

The model never reproduces the transcript. It reports boundaries by word id, and this
module turns those ids into sentences. The distinction matters: asking a model to repeat
four hundred words in order to convey twenty boundary positions makes every word a chance
to invalidate the whole answer, and speech transcripts invite exactly that failure, since
their stutters and repetitions read as typos a helpful model will quietly clean up.

Ids cost nothing to verify and degrade gracefully. A boundary that cannot be honoured is
dropped, which merges two sentences and costs a little translation context; nothing about
timing, coverage, or text depends on the model getting them all right.
"""

from collections.abc import Iterable, Sequence
from dataclasses import replace

from caption.types import Sentence, WordSpan

_BATCH_CUT_TAIL_FRACTION = 0.25
_SPACE_LANG_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def speech_words(words: Sequence[WordSpan]) -> list[WordSpan]:
    """
    Drop tokens that carry no pronounceable content.

    Parameters
    ----------
    words : Sequence[WordSpan]
        Timestamped ASR tokens.

    Returns
    -------
    list[WordSpan]
        Tokens with at least one alphanumeric character.
    """
    return [word for word in words if any(character.isalnum() for character in word.text)]


def join_words(words: Iterable[WordSpan]) -> str:
    """
    Render a run of words as display text.

    Parameters
    ----------
    words : Iterable[WordSpan]
        Timestamped tokens in order.

    Returns
    -------
    str
        Words joined with a space only where both sides need one, so that scripts written
        without spaces stay unspaced.
    """
    rendered = ""
    for word in words:
        rendered = _append_word(rendered, word.text)
    return rendered


def rendered_lengths(words: Sequence[WordSpan]) -> list[int]:
    """
    Return the rendered length of every word prefix.

    Display limits are written in characters, so the layout stage has to measure candidate
    cuts in characters too. Prefix lengths make that a lookup rather than a re-render.

    Parameters
    ----------
    words : Sequence[WordSpan]
        Timestamped tokens in order.

    Returns
    -------
    list[int]
        Length of ``join_words(words[:index])`` for every index from zero to the word count.
    """
    lengths = [0]
    rendered = ""
    for word in words:
        rendered = _append_word(rendered, word.text)
        lengths.append(len(rendered))
    return lengths


def _append_word(rendered: str, text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return rendered
    if rendered and rendered[-1] in _SPACE_LANG_CHARS and stripped[0] in _SPACE_LANG_CHARS:
        rendered += " "
    return rendered + stripped


def split_word_batches(words: Sequence[WordSpan], max_words: int) -> list[list[WordSpan]]:
    """
    Split the word stream into batches that are unlikely to end mid-sentence.

    Each cut is placed at the longest silence in the tail of the batch, so batches align
    with natural pauses and a sentence rarely straddles two of them.

    Parameters
    ----------
    words : Sequence[WordSpan]
        Timestamped ASR tokens.
    max_words : int
        Maximum number of tokens per batch.

    Returns
    -------
    list[list[WordSpan]]
        Consecutive batches covering every token exactly once.

    Raises
    ------
    ValueError
        If max_words is not positive.
    """
    if max_words < 1:
        raise ValueError("max_words must be positive")
    batches: list[list[WordSpan]] = []
    start = 0
    while len(words) - start > max_words:
        cut = _pause_cut(words, start, start + max_words)
        batches.append(list(words[start:cut]))
        start = cut
    if start < len(words):
        batches.append(list(words[start:]))
    return batches


def sentences_from_boundaries(
    words: Sequence[WordSpan], sentence_ends: Iterable[int], line_breaks: Iterable[int]
) -> list[Sentence]:
    """
    Cut a batch of words into sentences at the reported boundaries.

    Parameters
    ----------
    words : Sequence[WordSpan]
        Timestamped ASR tokens of one batch.
    sentence_ends : Iterable[int]
        One-based ids of the last word of each sentence. Ids outside the batch are
        ignored, order and duplicates do not matter, and a missing final boundary is
        supplied so that the batch is always covered exactly once.
    line_breaks : Iterable[int]
        One-based ids of words that may start a new display line. Ids that fall outside a
        sentence interior are ignored.

    Returns
    -------
    list[Sentence]
        Sentences indexed from one, together covering every word of the batch.
    """
    if not words:
        return []
    cuts = sorted({end for end in sentence_ends if 0 < end < len(words)}) + [len(words)]
    hints = {position - 1 for position in line_breaks if 1 < position <= len(words)}

    sentences: list[Sentence] = []
    start = 0
    for index, end in enumerate(cuts, start=1):
        span = words[start:end]
        sentences.append(
            Sentence(
                index=index,
                text=join_words(span),
                words=tuple(span),
                breaks=frozenset(hint - start for hint in hints if start < hint < end),
            )
        )
        start = end
    return sentences


def renumber(sentences: Sequence[Sentence]) -> list[Sentence]:
    """
    Reindex sentences from one after concatenating per-batch results.

    Parameters
    ----------
    sentences : Sequence[Sentence]
        Sentences in transcript order.

    Returns
    -------
    list[Sentence]
        Sentences with sequential indices.
    """
    return [replace(sentence, index=index) for index, sentence in enumerate(sentences, start=1)]


def _pause_cut(words: Sequence[WordSpan], start: int, limit: int) -> int:
    """Return the index after the longest silence in the tail of the candidate batch."""
    tail_start = max(start + 1, limit - int((limit - start) * _BATCH_CUT_TAIL_FRACTION))
    best_index = limit
    best_gap = -1.0
    for index in range(tail_start, limit):
        gap = words[index].start - words[index - 1].end
        if gap > best_gap:
            best_gap = gap
            best_index = index
    return best_index
