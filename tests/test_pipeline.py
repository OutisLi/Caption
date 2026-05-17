from pathlib import Path

from caption.pipeline import process_job
from caption.types import AsrResult, CaptionConfig, MediaJob, SubtitleCue, WordSpan


class FakeAsr:
    def transcribe(self, audio_path: Path, language: str | None = None) -> AsrResult:
        assert audio_path.name == "clip.wav"
        assert language == "English"
        return AsrResult(
            text="Hello world.",
            language="English",
            words=[WordSpan("Hello", 0.0, 0.4), WordSpan("world.", 0.4, 1.0)],
            chunks=[
                {"text": "Hello chunk.", "start": 0.0, "end": 0.5},
                {"text": "Second chunk.", "start": 0.5, "end": 1.0},
            ],
        )


class FakeTranslator:
    def __init__(self) -> None:
        self.called = False

    def translate(self, cues: list[SubtitleCue]) -> list[SubtitleCue]:
        self.called = True
        return [
            SubtitleCue(
                index=cue.index,
                start=cue.start,
                end=cue.end,
                source_text=cue.source_text,
                target_text="你好，世界。",
            )
            for cue in cues
        ]


class FakeOptimizer:
    def __init__(self) -> None:
        self.called = False

    def optimize(self, cues: list[SubtitleCue], tokens: list[WordSpan]) -> list[SubtitleCue]:
        self.called = True
        assert [token.text for token in tokens] == ["Hello", "world."]
        return [
            SubtitleCue(
                index=cue.index,
                start=cue.start,
                end=cue.end,
                source_text="Hello, world.",
                target_text="你好，世界。",
            )
            for cue in cues
        ]


class FailingTranslator:
    def translate(self, cues: list[SubtitleCue]) -> list[SubtitleCue]:
        raise RuntimeError("translation failed")


def test_process_job_writes_incremental_outputs_for_translated_optimized_flow(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(
        input_path=input_path, output_dir=tmp_path / "out", relative_output_dir=Path("course/week1"), stem="clip"
    )
    config = CaptionConfig(source_language="English", target_language="zh", write_text=True)
    translator = FakeTranslator()
    optimizer = FakeOptimizer()

    paths = process_job(job, config, asr=FakeAsr(), translator=translator, optimizer=optimizer, save_asr_json=True)

    assert translator.called is True
    assert optimizer.called is True
    assert paths.asr_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"
    assert paths.asr_txt.read_text(encoding="utf-8") == "Hello chunk.\n\nSecond chunk.\n"
    assert paths.source_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello, world.\n"
    assert paths.target_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\n你好，世界。\n"
    assert (
        paths.bilingual_srt.read_text(encoding="utf-8")
        == "1\n00:00:00,000 --> 00:00:01,000\nHello, world.\n你好，世界。\n"
    )
    assert paths.source_txt.read_text(encoding="utf-8") == "Hello, world.\n"
    assert paths.raw_source_srt is not None
    assert paths.raw_target_srt is not None
    assert paths.raw_bilingual_srt is not None
    assert paths.raw_source_txt is not None
    assert paths.raw_source_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"
    assert paths.raw_target_srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\n你好，世界。\n"
    assert (
        paths.raw_bilingual_srt.read_text(encoding="utf-8")
        == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n你好，世界。\n"
    )
    assert paths.raw_source_txt.read_text(encoding="utf-8") == "Hello world.\n"
    assert paths.asr_json is not None
    assert '"text": "Hello world."' in paths.asr_json.read_text(encoding="utf-8")
    assert paths.asr_srt == tmp_path / "out" / "asr" / "course" / "week1" / "clip.asr.srt"
    assert paths.raw_bilingual_srt == tmp_path / "out" / "raw" / "course" / "week1" / "clip.raw.bilingual.srt"
    assert paths.bilingual_srt == tmp_path / "out" / "final" / "course" / "week1" / "clip.bilingual.srt"


def test_process_job_skips_txt_outputs_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(input_path=input_path, output_dir=tmp_path / "out", stem="clip")
    config = CaptionConfig(source_language="English", target_language="zh")

    paths = process_job(
        job, config, asr=FakeAsr(), translator=FakeTranslator(), optimizer=FakeOptimizer(), save_asr_json=True
    )

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


def test_plain_text_mode_stops_after_raw_source_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    job = MediaJob(
        input_path=input_path, output_dir=tmp_path / "out", relative_output_dir=Path("course/week1"), stem="clip"
    )
    config = CaptionConfig(source_language="English", target_language=None, plain_text=True)
    stale_bilingual = job.output_dir / "final" / "course" / "week1" / "clip.bilingual.srt"
    stale_bilingual.parent.mkdir(parents=True)
    stale_bilingual.write_text("old", encoding="utf-8")

    paths = process_job(job, config, asr=FakeAsr(), translator=FailingTranslator(), optimizer=None, save_asr_json=True)

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

    try:
        process_job(job, config, asr=FakeAsr(), translator=FailingTranslator(), optimizer=None, save_asr_json=True)
    except RuntimeError:
        pass

    assert (tmp_path / "out" / "asr" / "clip.asr.json").exists()
    assert (tmp_path / "out" / "asr" / "clip.asr.srt").read_text(
        encoding="utf-8"
    ) == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"
    assert not (tmp_path / "out" / "asr" / "clip.asr.txt").exists()
