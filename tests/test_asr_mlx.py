from pathlib import Path

from caption.asr_mlx import DEFAULT_ALIGNER_MODEL, DEFAULT_ASR_MODEL, LocalMlxAsr


class FakeResult:
    text = "Hello world."
    language = "English"
    chunks = [{"start": 0.0, "end": 1.0, "text": "Hello world."}]
    segments = [
        {"text": "Hello", "start": 0.0, "end": 0.4},
        {"text": "world.", "start": 0.4, "end": 1.0},
    ]


class FakeSession:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict] = []

    def transcribe(self, audio: Path, **kwargs: object) -> FakeResult:
        self.calls.append({"audio": audio, **kwargs})
        return FakeResult()


class SmallOverlapResult:
    text = "one two"
    language = "English"
    chunks = [{"start": 0.0, "end": 1.0, "text": "one two"}]
    segments = [
        {"text": "one", "start": 0.0, "end": 1.0},
        {"text": "two", "start": 0.9, "end": 1.5},
    ]


class SmallOverlapSession:
    def transcribe(self, audio: Path, **kwargs: object) -> SmallOverlapResult:
        return SmallOverlapResult()


def test_local_mlx_asr_uses_default_models_and_normalizes_words(tmp_path: Path) -> None:
    sessions: list[FakeSession] = []
    aligner_models: list[str] = []

    def session_factory(model: str) -> FakeSession:
        session = FakeSession(model)
        sessions.append(session)
        return session

    def aligner_factory(model: str) -> str:
        aligner_models.append(model)
        return f"aligner:{model}"

    asr = LocalMlxAsr(session_factory=session_factory, aligner_factory=aligner_factory)
    audio_path = tmp_path / "clip.wav"

    result = asr.transcribe(audio_path, language="English")

    assert sessions[0].model == DEFAULT_ASR_MODEL
    assert aligner_models == [DEFAULT_ALIGNER_MODEL]
    assert result.text == "Hello world."
    assert result.language == "English"
    assert [(word.text, word.start, word.end) for word in result.words] == [("Hello", 0.0, 0.4), ("world.", 0.4, 1.0)]
    assert sessions[0].calls[0]["return_timestamps"] is True
    assert sessions[0].calls[0]["return_chunks"] is True
    assert sessions[0].calls[0]["forced_aligner"] == f"aligner:{DEFAULT_ALIGNER_MODEL}"


def test_local_mlx_asr_clamps_small_timestamp_overlaps(tmp_path: Path) -> None:
    asr = LocalMlxAsr(
        session_factory=lambda model: SmallOverlapSession(),
        aligner_factory=lambda model: f"aligner:{model}",
    )

    result = asr.transcribe(tmp_path / "clip.wav", language="English")

    assert [(word.text, word.start, word.end) for word in result.words] == [("one", 0.0, 1.0), ("two", 1.0, 1.5)]
