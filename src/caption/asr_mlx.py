"""MLX Qwen3-ASR adapter."""

from pathlib import Path
from typing import Any, Callable

from caption.types import AsrResult, WordSpan
from caption.progress import AsrProgress

DEFAULT_ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
MAX_TIMESTAMP_OVERLAP_SECONDS = 0.2

SessionFactory = Callable[[str], Any]
AlignerFactory = Callable[[str], Any]


class LocalMlxAsr:
    """Transcribe media with MLX Qwen3-ASR and a forced aligner."""

    def __init__(
        self,
        model: str = DEFAULT_ASR_MODEL,
        aligner_model: str = DEFAULT_ALIGNER_MODEL,
        session_factory: SessionFactory | None = None,
        aligner_factory: AlignerFactory | None = None,
    ) -> None:
        """
        Create an ASR adapter.

        Parameters
        ----------
        model : str
            ASR model id or local path.
        aligner_model : str
            Forced aligner model id or local path.
        session_factory : SessionFactory | None
            Optional factory for tests.
        aligner_factory : AlignerFactory | None
            Optional factory for tests.
        """
        self.model = model
        self.aligner_model = aligner_model
        self._session_factory = session_factory or _create_session
        self._aligner_factory = aligner_factory or _create_aligner
        self._session: Any | None = None
        self._aligner: Any | None = None

    def transcribe(self, audio_path: Path, language: str | None = None) -> AsrResult:
        """
        Transcribe an audio or video file.

        Parameters
        ----------
        audio_path : Path
            Input media path.
        language : str | None
            Optional forced source language.

        Returns
        -------
        AsrResult
            Normalized transcription result.
        """
        progress = AsrProgress()
        result = self._get_session().transcribe(
            audio_path,
            language=language,
            return_timestamps=True,
            return_chunks=True,
            forced_aligner=self._get_aligner(),
            on_progress=progress,
        )
        return AsrResult(
            text=str(getattr(result, "text", "")),
            language=str(getattr(result, "language", "")),
            words=_normalize_words(getattr(result, "segments", None)),
            chunks=list(getattr(result, "chunks", []) or []),
        )

    def _get_session(self) -> Any:
        if self._session is None:
            self._session = self._session_factory(self.model)
        return self._session

    def _get_aligner(self) -> Any:
        if self._aligner is None:
            self._aligner = self._aligner_factory(self.aligner_model)
        return self._aligner


def _normalize_words(segments: Any) -> list[WordSpan]:
    words: list[WordSpan] = []
    previous_end = 0.0
    for segment in segments or []:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        if start < previous_end and previous_end - start <= MAX_TIMESTAMP_OVERLAP_SECONDS:
            start = previous_end
        words.append(WordSpan(text=text, start=start, end=end))
        previous_end = max(previous_end, end)
    return words


def _create_session(model: str) -> Any:
    from mlx_qwen3_asr import Session

    return Session(model=model)


def _create_aligner(model: str) -> Any:
    from mlx_qwen3_asr import ForcedAligner

    return ForcedAligner(model_path=model)
