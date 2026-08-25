from pathlib import Path
import threading

import pytest

from caption.pipeline import process_job, run_pipeline
from caption.translator import TranslationDraft
from caption.types import (
    AsrResult,
    CaptionConfig,
    MediaJob,
    Sentence,
    SentenceLayout,
    SubtitleLine,
    TranscriptGlossary,
    WordSpan,
)

ASR_WORDS = [WordSpan("Hello", 0.0, 0.4), WordSpan("world.", 0.4, 1.0)]


class FakeAsr:
    def transcribe(self, audio_path: Path, language: str | None = None) -> AsrResult:
        assert audio_path.name == "clip.wav"
        assert language == "English"
        return AsrResult(
            text="Hello world.",
            language="English",
            words=list(ASR_WORDS),
            chunks=[
                {"text": "Hello chunk.", "start": 0.0, "end": 0.5},
                {"text": "Second chunk.", "start": 0.5, "end": 1.0},
            ],
        )


def _layout(words: list[WordSpan], source_text: str, target_text: str = "") -> SentenceLayout:
    sentence = Sentence(index=1, text="Hello world.", words=tuple(words))
    line = SubtitleLine(
        start=words[0].start, end=words[-1].end, source_text=source_text, target_text=target_text
    )
    return SentenceLayout(sentence=sentence, lines=(line,))


def _draft(words: list[WordSpan], source_text: str, target_text: str) -> TranslationDraft:
    return TranslationDraft(
        sentences=(_layout(words, source_text, target_text),),
        glossary=TranscriptGlossary(topic="A greeting."),
    )


class FakeTranslator:
    """Translator fake whose review pass is textually distinguishable from its draft."""

    def __init__(self) -> None:
        self.segmented = False
        self.translated = False
        self.reviewed = False

    def segment(self, words: list[WordSpan]) -> list[SentenceLayout]:
        self.segmented = True
        return [_layout(words, "Hello world.")]

    def translate(self, words: list[WordSpan]) -> TranslationDraft:
        self.translated = True
        assert [word.text for word in words] == ["Hello", "world."]
        return _draft(words, "Hello, world.", "你好，世界。")

    def review(self, draft: TranslationDraft) -> list[SentenceLayout]:
        self.reviewed = True
        return [
            SentenceLayout(
                sentence=layout.sentence,
                lines=tuple(
                    SubtitleLine(line.start, line.end, "Hello, world!", "你好，世界！") for line in layout.lines
                ),
            )
            for layout in draft.sentences
        ]


class FailingTranslator:
    def segment(self, words: list[WordSpan]) -> list[SentenceLayout]:
        raise RuntimeError("segmentation failed")

    def translate(self, words: list[WordSpan]) -> TranslationDraft:
        raise RuntimeError("translation failed")

    def review(self, draft: TranslationDraft) -> list[SentenceLayout]:
        raise AssertionError("review must not run after a failed translation")


def test_process_job_writes_draft_before_review_and_review_wins(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(
        input_path=input_path, output_dir=tmp_path / "out", relative_output_dir=Path("course/week1"), stem="clip"
    )
    config = CaptionConfig(source_language="English", target_language="zh", write_text=True, review=True)
    translator = FakeTranslator()

    paths = process_job(job, config, asr=FakeAsr(), translator=translator, save_asr_json=True)

    assert translator.translated is True
    assert translator.reviewed is True
    assert paths.asr_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"
    assert paths.asr_txt.read_text(encoding="utf-8") == "Hello chunk.\n\nSecond chunk.\n"
    assert paths.raw_source_srt is not None
    assert paths.raw_target_srt is not None
    assert paths.raw_bilingual_srt is not None
    assert paths.raw_source_txt is not None
    assert paths.raw_source_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello, world.\n"
    assert paths.raw_target_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\n你好，世界。\n"
    assert paths.raw_source_txt.read_text(encoding="utf-8") == "Hello, world.\n"
    assert paths.source_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello, world!\n"
    assert paths.target_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\n你好，世界！\n"
    assert (
        paths.bilingual_srt.read_text(encoding="utf-8")
        == "1\n00:00:00,000 --> 00:00:01,000\nHello, world!\n你好，世界！\n"
    )
    assert paths.asr_json is not None
    assert '"text": "Hello world."' in paths.asr_json.read_text(encoding="utf-8")
    assert paths.asr_srt == tmp_path / "out" / "asr" / "course" / "week1" / "clip.asr.srt"
    assert paths.raw_bilingual_srt == tmp_path / "out" / "raw" / "course" / "week1" / "clip.raw.bilingual.srt"
    assert paths.bilingual_srt == tmp_path / "out" / "final" / "course" / "week1" / "clip.bilingual.srt"


def test_process_job_without_review_publishes_the_draft(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(input_path=input_path, output_dir=tmp_path / "out", stem="clip")
    config = CaptionConfig(source_language="English", target_language="zh", review=False)
    translator = FakeTranslator()

    paths = process_job(job, config, asr=FakeAsr(), translator=translator, save_asr_json=True)

    assert translator.reviewed is False
    assert paths.source_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello, world.\n"
    assert paths.raw_source_srt is not None
    assert not paths.raw_source_srt.exists()


def test_process_job_without_translator_falls_back_to_asr_segmentation(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(input_path=input_path, output_dir=tmp_path / "out", stem="clip")
    config = CaptionConfig(source_language="English", target_language=None)

    paths = process_job(job, config, asr=FakeAsr(), translator=None, save_asr_json=False)

    assert paths.source_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"
    assert not paths.target_srt.exists()
    assert not paths.bilingual_srt.exists()


def test_process_job_segments_with_the_llm_without_translating(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(input_path=input_path, output_dir=tmp_path / "out", stem="clip")
    config = CaptionConfig(source_language="English", target_language=None, review=True)
    translator = FakeTranslator()

    paths = process_job(job, config, asr=FakeAsr(), translator=translator, save_asr_json=False)

    assert translator.segmented is True
    assert translator.translated is False
    assert translator.reviewed is False
    assert paths.source_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"
    assert not paths.target_srt.exists()


def test_process_job_skips_txt_outputs_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(input_path=input_path, output_dir=tmp_path / "out", stem="clip")
    config = CaptionConfig(source_language="English", target_language="zh", review=True)

    paths = process_job(job, config, asr=FakeAsr(), translator=FakeTranslator(), save_asr_json=True)

    assert paths.asr_srt.exists()
    assert paths.asr_json is not None
    assert paths.asr_json.exists()
    assert paths.bilingual_srt.exists()
    assert paths.raw_bilingual_srt is not None
    assert paths.raw_bilingual_srt.exists()
    assert not paths.asr_txt.exists()
    assert not paths.source_txt.exists()
    assert not paths.target_txt.exists()
    assert paths.raw_source_txt is not None
    assert paths.raw_target_txt is not None
    assert not paths.raw_source_txt.exists()
    assert not paths.raw_target_txt.exists()
    assert paths.asr_txt not in paths.written_paths
    assert paths.source_txt not in paths.written_paths
    assert paths.target_txt not in paths.written_paths


def test_plain_text_mode_stops_after_asr_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(
        input_path=input_path, output_dir=tmp_path / "out", relative_output_dir=Path("course/week1"), stem="clip"
    )
    config = CaptionConfig(source_language="English", target_language=None, plain_text=True)
    stale_bilingual = job.output_dir / "final" / "course" / "week1" / "clip.bilingual.srt"
    stale_bilingual.parent.mkdir(parents=True)
    stale_bilingual.write_text("old", encoding="utf-8")

    paths = process_job(job, config, asr=FakeAsr(), translator=FailingTranslator(), save_asr_json=True)

    assert paths.asr_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"
    assert paths.asr_txt.read_text(encoding="utf-8") == "Hello chunk.\n\nSecond chunk.\n"
    assert not paths.source_srt.exists()
    assert not paths.source_txt.exists()
    assert not paths.target_srt.exists()
    assert paths.bilingual_srt.exists()
    assert paths.asr_json is not None
    assert paths.asr_json.exists()
    assert paths.written_paths == (paths.asr_json, paths.asr_srt, paths.asr_txt)


def test_asr_outputs_are_saved_before_translation_failure(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(input_path=input_path, output_dir=tmp_path / "out", stem="clip")
    config = CaptionConfig(source_language="English", target_language="zh")

    with pytest.raises(RuntimeError, match="translation failed"):
        process_job(job, config, asr=FakeAsr(), translator=FailingTranslator(), save_asr_json=True)

    assert (tmp_path / "out" / "asr" / "clip.asr.json").exists()
    assert (tmp_path / "out" / "asr" / "clip.asr.srt").read_text(
        encoding="utf-8"
    ) == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"
    assert not (tmp_path / "out" / "asr" / "clip.asr.txt").exists()


class ExplodingAsr:
    def transcribe(self, audio_path: Path, language: str | None = None) -> AsrResult:
        raise AssertionError("ASR must not run when a cached result exists")


def test_process_job_reuses_cached_asr_json(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(input_path=input_path, output_dir=tmp_path / "out", stem="clip")
    config = CaptionConfig(source_language="English", target_language="zh")

    first = process_job(job, config, asr=FakeAsr(), translator=FakeTranslator(), save_asr_json=True)
    assert first.asr_json is not None and first.asr_json.exists()

    paths = process_job(job, config, asr=ExplodingAsr(), translator=FakeTranslator(), save_asr_json=True)

    assert paths.asr_json is not None
    assert paths.asr_json not in paths.written_paths
    assert paths.source_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello, world.\n"
    assert paths.target_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\n你好，世界。\n"


def test_process_job_rejects_invalid_asr_cache(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(input_path=input_path, output_dir=tmp_path / "out", stem="clip")
    config = CaptionConfig(source_language="English", target_language="zh")
    cache_path = tmp_path / "out" / "asr" / "clip.asr.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"text": "oops"}', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid ASR cache file"):
        process_job(job, config, asr=ExplodingAsr(), translator=FakeTranslator(), save_asr_json=True)


class RecordingAsr:
    """ASR fake that records call order and optionally blocks on the second call."""

    def __init__(self, on_second_call: threading.Event | None = None, release: threading.Event | None = None) -> None:
        self.calls: list[str] = []
        self.on_second_call = on_second_call
        self.release = release

    def transcribe(self, audio_path: Path, language: str | None = None) -> AsrResult:
        self.calls.append(audio_path.name)
        if len(self.calls) == 2:
            if self.on_second_call is not None:
                self.on_second_call.set()
            if self.release is not None:
                self.release.wait(timeout=1.0)
        return AsrResult(text="Hello world.", language="English", words=list(ASR_WORDS), chunks=[])


def _write_fake_media(tmp_path: Path, names: tuple[str, ...]) -> None:
    for name in names:
        (tmp_path / name).write_bytes(b"fake")


def test_run_pipeline_overlaps_asr_with_translation(tmp_path: Path) -> None:
    _write_fake_media(tmp_path, ("a.wav", "b.wav"))
    second_asr_started = threading.Event()

    class GatedTranslator:
        def segment(self, words: list[WordSpan]) -> list[SentenceLayout]:
            raise AssertionError("segmentation must not run when a translation target is set")

        def translate(self, words: list[WordSpan]) -> TranslationDraft:
            assert second_asr_started.wait(timeout=10), "translation must overlap with the next file's ASR"
            return _draft(words, "Hello, world.", "你好，世界。")

        def review(self, draft: TranslationDraft) -> list[SentenceLayout]:
            return list(draft.sentences)

    asr = RecordingAsr(on_second_call=second_asr_started)
    outputs = run_pipeline(
        tmp_path,
        tmp_path / "out",
        CaptionConfig(source_language="English", target_language="zh"),
        asr=asr,
        translator=GatedTranslator(),
    )

    assert asr.calls == ["a.wav", "b.wav"]
    assert [output.bilingual_srt.name for output in outputs] == ["a.bilingual.srt", "b.bilingual.srt"]
    assert all(output.bilingual_srt.exists() for output in outputs)


def test_run_pipeline_stops_asr_after_translation_failure(tmp_path: Path) -> None:
    _write_fake_media(tmp_path, ("a.wav", "b.wav", "c.wav"))
    asr = RecordingAsr(release=threading.Event())

    with pytest.raises(RuntimeError, match="translation failed"):
        run_pipeline(
            tmp_path,
            tmp_path / "out",
            CaptionConfig(source_language="English", target_language="zh"),
            asr=asr,
            translator=FailingTranslator(),
        )

    assert asr.calls == ["a.wav", "b.wav"]


def test_run_pipeline_without_llm_stays_sequential(tmp_path: Path) -> None:
    _write_fake_media(tmp_path, ("a.wav", "b.wav"))
    asr = RecordingAsr()

    outputs = run_pipeline(
        tmp_path,
        tmp_path / "out",
        CaptionConfig(source_language="English", target_language=None, plain_text=True),
        asr=asr,
        translator=None,
    )

    assert asr.calls == ["a.wav", "b.wav"]
    assert [output.asr_srt.name for output in outputs] == ["a.asr.srt", "b.asr.srt"]
    assert all(output.asr_srt.exists() for output in outputs)
