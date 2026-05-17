import pytest

from caption.segment import build_cues
from caption.types import WordSpan


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


def test_build_cues_rejects_invalid_timestamps() -> None:
    with pytest.raises(ValueError, match="invalid timestamp"):
        build_cues([WordSpan("bad", 1.0, 0.5)], max_chars_per_cue=60, max_seconds_per_cue=6.0)

    with pytest.raises(ValueError, match="monotonic"):
        build_cues(
            [WordSpan("one", 0.0, 1.0), WordSpan("two", 0.5, 1.5)],
            max_chars_per_cue=60,
            max_seconds_per_cue=6.0,
        )
