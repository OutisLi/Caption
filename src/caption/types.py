"""Shared data types for caption generation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaJob:
    """Describe one media file and its output location.

    Parameters
    ----------
    input_path : Path
        Source audio or video file.
    output_dir : Path
        Base directory where generated subtitle files are written.
    stem : str
        Output filename stem.
    relative_output_dir : Path
        Input-root-relative directory preserved under each output stage.
    """

    input_path: Path
    output_dir: Path
    stem: str
    relative_output_dir: Path = Path()


@dataclass(frozen=True)
class OutputPaths:
    """Store generated subtitle output paths.

    Parameters
    ----------
    asr_srt : Path
        ASR-only source-language subtitle path.
    asr_txt : Path
        ASR-only source-language plain text path.
    source_srt : Path
        Source-language subtitle path.
    source_txt : Path
        Source-language plain text path.
    target_srt : Path
        Target-language subtitle path.
    target_txt : Path
        Target-language plain text path.
    bilingual_srt : Path
        Bilingual subtitle path.
    raw_source_srt : Path | None
        Unoptimized source-language subtitle path.
    raw_source_txt : Path | None
        Unoptimized source-language plain text path.
    raw_target_srt : Path | None
        Unoptimized target-language subtitle path.
    raw_target_txt : Path | None
        Unoptimized target-language plain text path.
    raw_bilingual_srt : Path | None
        Unoptimized bilingual subtitle path.
    asr_json : Path | None
        Optional ASR debug JSON path.
    written_paths : tuple[Path, ...]
        Files written by the current run.
    """

    asr_srt: Path
    asr_txt: Path
    source_srt: Path
    source_txt: Path
    target_srt: Path
    target_txt: Path
    bilingual_srt: Path
    raw_source_srt: Path | None = None
    raw_source_txt: Path | None = None
    raw_target_srt: Path | None = None
    raw_target_txt: Path | None = None
    raw_bilingual_srt: Path | None = None
    asr_json: Path | None = None
    written_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class WordSpan:
    """Represent one recognized word or character with timestamps.

    Parameters
    ----------
    text : str
        Recognized text.
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    """

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SubtitleCue:
    """Represent one subtitle cue.

    Parameters
    ----------
    index : int
        One-based subtitle cue index.
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    source_text : str
        Source-language text.
    target_text : str
        Target-language text.
    """

    index: int
    start: float
    end: float
    source_text: str
    target_text: str = ""


@dataclass(frozen=True)
class AsrResult:
    """Store normalized ASR output.

    Parameters
    ----------
    text : str
        Full recognized text.
    language : str
        Detected or forced language name.
    words : list[WordSpan]
        Word or character timestamps.
    chunks : list[dict]
        Optional backend chunk metadata.
    """

    text: str
    language: str
    words: list[WordSpan]
    chunks: list[dict]


@dataclass(frozen=True)
class CaptionConfig:
    """Configure caption generation.

    Parameters
    ----------
    target_language : str | None
        Translation target language. None means source-only optimization.
    source_language : str | None
        Optional forced source language for ASR.
    translation_position : str
        Where translated text appears in bilingual subtitles.
    max_chars_per_cue : int
        Maximum visual characters per cue.
    max_seconds_per_cue : float
        Maximum cue duration in seconds.
    plain_text : bool
        Whether to stop after raw source SRT/TXT generation.
    write_text : bool
        Whether to write TXT sidecar files outside plain-text mode.
    """

    target_language: str | None
    source_language: str | None = None
    translation_position: str = "bottom"
    max_chars_per_cue: int = 60
    max_seconds_per_cue: float = 6.0
    plain_text: bool = False
    write_text: bool = False
