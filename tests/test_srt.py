from caption.srt import format_srt_timestamp, render_bilingual_srt, render_srt
from caption.types import SubtitleCue


def test_format_srt_timestamp_uses_comma_milliseconds() -> None:
    assert format_srt_timestamp(3723.4567) == "01:02:03,457"


def test_render_srt_outputs_single_language_and_bilingual_variants() -> None:
    cues = [
        SubtitleCue(index=1, start=0.0, end=1.25, source_text="Hello.", target_text="你好。"),
        SubtitleCue(index=2, start=1.25, end=3.0, source_text="World.", target_text="世界。"),
    ]

    assert render_srt(cues, language="source") == (
        "1\n00:00:00,000 --> 00:00:01,250\nHello.\n\n2\n00:00:01,250 --> 00:00:03,000\nWorld.\n"
    )
    assert render_bilingual_srt(cues[:1], translation_position="top") == (
        "1\n00:00:00,000 --> 00:00:01,250\n你好。\nHello.\n"
    )
