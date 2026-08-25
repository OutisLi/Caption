import pytest

from caption.sentence import (
    join_words,
    renumber,
    sentences_from_boundaries,
    speech_words,
    split_word_batches,
)
from caption.types import WordSpan

WORDS = [
    WordSpan("welcome", 0.0, 0.5),
    WordSpan("everyone", 0.5, 1.1),
    WordSpan("to", 1.1, 1.3),
    WordSpan("cs", 1.3, 1.6),
    WordSpan("thirtythree", 1.6, 2.3),
    WordSpan("this", 3.0, 3.3),
    WordSpan("is", 3.3, 3.5),
    WordSpan("tatsu", 3.5, 4.0),
]


def test_boundaries_cut_the_batch_and_carry_their_timestamps() -> None:
    sentences = sentences_from_boundaries(WORDS, sentence_ends=[5, 8], line_breaks=[3, 7])

    assert [sentence.text for sentence in sentences] == ["welcome everyone to cs thirtythree", "this is tatsu"]
    assert [sentence.index for sentence in sentences] == [1, 2]
    assert (sentences[0].start, sentences[0].end) == (0.0, 2.3)
    assert (sentences[1].start, sentences[1].end) == (3.0, 4.0)
    assert sentences[1].duration == pytest.approx(1.0)


def test_line_breaks_become_sentence_relative_indices() -> None:
    sentences = sentences_from_boundaries(WORDS, sentence_ends=[5, 8], line_breaks=[3, 7])

    # Word id 3 is the third word of the first sentence, so index 2 inside it.
    assert sentences[0].breaks == frozenset({2})
    # Word id 7 is the second word of the second sentence, which starts at word id 6.
    assert sentences[1].breaks == frozenset({1})


def test_a_missing_final_boundary_is_supplied_so_the_batch_stays_covered() -> None:
    sentences = sentences_from_boundaries(WORDS, sentence_ends=[5], line_breaks=[])

    assert len(sentences) == 2
    assert sum(len(sentence.words) for sentence in sentences) == len(WORDS)
    assert sentences[-1].words[-1] is WORDS[-1]


def test_unusable_ids_are_dropped_instead_of_failing() -> None:
    """Boundaries are advisory: a bad id merges two sentences and breaks nothing."""
    sentences = sentences_from_boundaries(
        WORDS, sentence_ends=[0, 5, 5, 99, -3], line_breaks=[1, 0, 99, 6]
    )

    assert [len(sentence.words) for sentence in sentences] == [5, 3]
    # Id 1 would start a line at the very first word, and id 6 at the first word of the
    # second sentence; both are already sentence starts, so neither survives.
    assert all(sentence.breaks == frozenset() for sentence in sentences)


def test_no_boundaries_at_all_still_yields_one_covering_sentence() -> None:
    sentences = sentences_from_boundaries(WORDS, sentence_ends=[], line_breaks=[])

    assert len(sentences) == 1
    assert len(sentences[0].words) == len(WORDS)
    assert sentences_from_boundaries([], sentence_ends=[3], line_breaks=[]) == []


def test_join_words_spaces_only_where_both_sides_need_it() -> None:
    assert join_words([WordSpan("hello", 0.0, 1.0), WordSpan("world", 1.0, 2.0)]) == "hello world"
    assert join_words([WordSpan("你好", 0.0, 1.0), WordSpan("世界", 1.0, 2.0)]) == "你好世界"
    assert join_words([WordSpan("wait", 0.0, 1.0), WordSpan(",", 1.0, 1.1), WordSpan("no", 1.1, 2.0)]) == "wait,no"


def test_batches_cut_at_the_longest_pause_and_cover_every_word() -> None:
    batches = split_word_batches(WORDS, max_words=6)

    assert [len(batch) for batch in batches] == [5, 3]
    assert [word.text for batch in batches for word in batch] == [word.text for word in WORDS]
    assert batches[1][0].text == "this"


def test_batches_stay_whole_when_the_stream_fits() -> None:
    assert [len(batch) for batch in split_word_batches(WORDS, max_words=100)] == [8]
    assert split_word_batches([], max_words=10) == []
    with pytest.raises(ValueError, match="max_words must be positive"):
        split_word_batches(WORDS, max_words=0)


def test_speech_words_drop_tokens_without_pronounceable_content() -> None:
    words = [WordSpan("hello", 0.0, 0.5), WordSpan("  ", 0.5, 0.6), WordSpan(",", 0.6, 0.7)]

    assert [word.text for word in speech_words(words)] == ["hello"]


def test_renumber_reindexes_concatenated_batches() -> None:
    first = sentences_from_boundaries(WORDS[:5], sentence_ends=[5], line_breaks=[])
    second = sentences_from_boundaries(WORDS[5:], sentence_ends=[3], line_breaks=[])

    assert [sentence.index for sentence in renumber([*first, *second])] == [1, 2]
