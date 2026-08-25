import pytest

from caption.segment import build_cues, build_cues_from_layouts, layout_sentence
from caption.sentence import join_words
from caption.types import Sentence, SubtitleLine, WordSpan


def _sentence(words: list[WordSpan], breaks: frozenset[int] = frozenset()) -> Sentence:
    """Build a sentence the way the segmentation stage does, from its words alone."""
    return Sentence(index=1, text=join_words(words), words=tuple(words), breaks=breaks)


def test_build_cues_splits_at_punctuation_duration_and_filters_empty_words() -> None:
    words = [
        WordSpan("", 0.0, 0.1),
        WordSpan("Hello", 0.0, 0.3),
        WordSpan("world.", 0.3, 0.8),
        WordSpan("Next", 0.9, 1.2),
        WordSpan("line.", 1.2, 1.6),
    ]

    cues = build_cues(words, max_chars_per_cue=60, max_seconds_per_cue=6.0)

    assert [cue.source_text for cue in cues] == ["Hello world.", "Next line."]
    assert [(cue.index, cue.start, cue.end) for cue in cues] == [(1, 0.0, 0.8), (2, 0.9, 1.6)]

    duration_cues = build_cues(
        [
            WordSpan("one", 0.0, 1.0),
            WordSpan("two", 1.0, 2.1),
            WordSpan("three", 2.1, 3.0),
        ],
        max_chars_per_cue=60,
        max_seconds_per_cue=2.0,
    )

    assert [cue.source_text for cue in duration_cues] == ["one two", "three"]
    assert [(cue.start, cue.end) for cue in duration_cues] == [(0.0, 2.1), (2.1, 3.0)]


def test_short_sentence_stays_on_one_line() -> None:
    sentence = _sentence([WordSpan("this", 3.0, 3.3), WordSpan("is", 3.3, 3.5), WordSpan("tatsu", 3.5, 4.0)])

    layout = layout_sentence(sentence, max_chars_per_line=60, max_seconds_per_line=6.0)

    assert layout.lines == (SubtitleLine(start=3.0, end=4.0, source_text="this is tatsu"),)


def test_a_long_sentence_splits_at_the_pause_between_clauses() -> None:
    words = [
        WordSpan("when", 0.0, 0.6),
        WordSpan("tatsu", 0.6, 1.4),
        WordSpan("and", 1.4, 2.0),
        WordSpan("i", 2.0, 2.6),
        WordSpan("started", 2.6, 3.6),
        WordSpan("we", 3.8, 4.4),
        WordSpan("were", 4.4, 5.2),
        WordSpan("unsure", 5.2, 6.4),
    ]
    layout = layout_sentence(_sentence(words), max_chars_per_line=60, max_seconds_per_line=4.0)

    # The clause boundary carries the only pause and sits close enough to the even split
    # to win it.
    assert [line.source_text for line in layout.lines] == ["when tatsu and i started", "we were unsure"]
    assert [(line.start, line.end) for line in layout.lines] == [(0.0, 3.6), (3.8, 6.4)]


def test_line_break_follows_the_pause_rather_than_the_even_split() -> None:
    """A compound noun is spoken without a pause, so the break belongs at the clause edge.

    The timestamps are taken verbatim from a lecture transcript in which the even split
    falls inside "scaling laws".
    """
    words = [
        WordSpan("So", 7.920, 8.400),
        WordSpan("last", 8.880, 9.120),
        WordSpan("time", 9.120, 9.440),
        WordSpan("Tatsu", 9.440, 9.760),
        WordSpan("talked", 9.760, 10.000),
        WordSpan("about", 10.000, 10.240),
        WordSpan("scaling", 10.240, 10.560),
        WordSpan("laws", 10.560, 11.040),
        WordSpan("and", 11.200, 11.760),
        WordSpan("we're", 11.760, 11.840),
        WordSpan("going", 11.840, 12.000),
        WordSpan("to", 12.000, 12.000),
        WordSpan("take", 12.000, 12.240),
        WordSpan("a", 12.240, 12.320),
        WordSpan("little", 12.320, 12.560),
        WordSpan("bit", 12.560, 12.720),
        WordSpan("break", 12.720, 12.960),
        WordSpan("from", 12.960, 13.200),
        WordSpan("that", 13.200, 13.520),
    ]
    layout = layout_sentence(_sentence(words), max_chars_per_line=60, max_seconds_per_line=4.0)

    assert layout.lines[0].source_text.endswith("scaling laws")
    assert layout.lines[1].source_text.startswith("and we're")


def test_a_break_hint_moves_the_cut_within_the_window() -> None:
    words = [WordSpan("word", index * 0.5, index * 0.5 + 0.45) for index in range(20)]

    without_hint = layout_sentence(_sentence(words), 60, 6.0)
    with_hint = layout_sentence(_sentence(words, frozenset({11})), 60, 6.0)

    # Nothing distinguishes the candidates on their own, so the cut lands on the even
    # split; a hint one word later still sits inside the window and outweighs it.
    assert [len(line.source_text) for line in without_hint.lines] == [49, 49]
    assert [len(line.source_text) for line in with_hint.lines] == [54, 44]


def test_line_count_follows_whichever_display_limit_binds() -> None:
    words = [WordSpan(f"word{index}", index * 2.0, index * 2.0 + 1.5) for index in range(6)]
    sentence = _sentence(words)

    by_duration = layout_sentence(sentence, max_chars_per_line=40, max_seconds_per_line=4.0)
    by_length = layout_sentence(sentence, max_chars_per_line=15, max_seconds_per_line=100.0)
    unsplit = layout_sentence(sentence, max_chars_per_line=60, max_seconds_per_line=100.0)

    assert len(by_duration.lines) == 3
    assert len(by_length.lines) == 3
    assert len(unsplit.lines) == 1
    assert "".join(line.source_text for line in by_duration.lines).replace(" ", "") == "word0word1word2word3word4word5"
    assert by_duration.lines[0].start == 0.0
    assert by_duration.lines[-1].end == 11.5


def test_a_sentence_never_splits_into_more_lines_than_words() -> None:
    sentence = _sentence([WordSpan("yes", 0.0, 30.0)])

    layout = layout_sentence(sentence, max_chars_per_line=1, max_seconds_per_line=1.0)

    assert len(layout.lines) == 1


def test_a_silence_inside_a_short_sentence_does_not_force_a_split() -> None:
    """A pause stretches the span without adding anything to read, so one line still fits."""
    sentence = _sentence([WordSpan("right", 0.0, 0.5), WordSpan("so", 8.0, 8.4)])

    layout = layout_sentence(sentence, max_chars_per_line=60, max_seconds_per_line=6.0)

    assert [line.source_text for line in layout.lines] == ["right so"]
    assert (layout.lines[0].start, layout.lines[0].end) == (0.0, 8.4)


def test_cuts_are_measured_in_characters_so_a_line_cannot_overflow() -> None:
    """Speech rate varies, so an even split in time is a lopsided split in text."""
    slow = [WordSpan("word", float(index), float(index) + 0.9) for index in range(10)]
    fast = [WordSpan("word", 10.0 + index * 0.2, 10.0 + index * 0.2 + 0.15) for index in range(10)]

    layout = layout_sentence(_sentence([*slow, *fast]), max_chars_per_line=60, max_seconds_per_line=6.0)

    # Splitting this evenly in time would cut around the sixth word, leaving 29 and 69
    # characters, so the second line would overflow the 60-character budget.
    assert [len(line.source_text) for line in layout.lines] == [49, 49]


def test_an_unsplittable_run_still_lands_near_the_even_split() -> None:
    """With no headroom the window collapses, and closeness to the split must still win."""
    words = [WordSpan("a", float(index), float(index) + 0.5) for index in range(8)]
    words += [WordSpan("encyclopedia", 8.0 + index * 0.1, 8.0 + index * 0.1 + 0.09) for index in range(8)]

    layout = layout_sentence(_sentence(words), max_chars_per_line=60, max_seconds_per_line=6.0)

    # The only real pause sits after the short words, where a naive ranking would cut and
    # leave 15 characters against 104.
    assert min(len(line.source_text) for line in layout.lines) > 40


def test_a_long_sentence_still_splits_when_there_is_text_to_fill_the_lines() -> None:
    words = [WordSpan(f"word{index}", index * 0.9, index * 0.9 + 0.8) for index in range(12)]
    sentence = _sentence(words)

    layout = layout_sentence(sentence, max_chars_per_line=40, max_seconds_per_line=6.0)

    assert len(layout.lines) == 2
    assert all(len(line.source_text) >= 10 for line in layout.lines)


def test_cues_are_numbered_across_sentences() -> None:
    first = layout_sentence(
        _sentence([WordSpan("one", 0.0, 1.0), WordSpan("two", 1.0, 2.0)]),
        max_chars_per_line=4,
        max_seconds_per_line=6.0,
    )
    second = layout_sentence(_sentence([WordSpan("three", 2.0, 3.0)]), 60, 6.0)

    cues = build_cues_from_layouts([first, second])

    assert [cue.index for cue in cues] == [1, 2, 3]
    assert [cue.source_text for cue in cues] == ["one", "two", "three"]
    assert [(cue.start, cue.end) for cue in cues] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


def test_build_cues_rejects_invalid_timestamps() -> None:
    with pytest.raises(ValueError, match="invalid timestamp"):
        build_cues([WordSpan("bad", 1.0, 0.5)], max_chars_per_cue=60, max_seconds_per_cue=6.0)

    with pytest.raises(ValueError, match="monotonic"):
        build_cues(
            [WordSpan("one", 0.0, 1.0), WordSpan("two", 0.5, 1.5)],
            max_chars_per_cue=60,
            max_seconds_per_cue=6.0,
        )
