"""End-to-end caption generation pipeline."""

import json
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Protocol

from caption.media import build_output_paths, discover_media_jobs
from caption.progress import log_step
from caption.types import AsrResult, CaptionConfig, MediaJob, OutputPaths, SubtitleCue, WordSpan
from caption.segment import build_cues
from caption.srt import render_bilingual_srt, render_srt


class AsrEngine(Protocol):
    """Protocol for local ASR engines."""

    def transcribe(self, audio_path: Path, language: str | None = None) -> AsrResult:
        """
        Transcribe a media file.

        Parameters
        ----------
        audio_path : Path
            Audio or video path.
        language : str | None
            Optional source language.

        Returns
        -------
        AsrResult
            Transcription with timestamps.
        """


class Translator(Protocol):
    """Protocol for subtitle translators."""

    def translate(self, cues: list[SubtitleCue]) -> list[SubtitleCue]:
        """
        Translate subtitle cues.

        Parameters
        ----------
        cues : list[SubtitleCue]
            Source subtitle cues.

        Returns
        -------
        list[SubtitleCue]
            Translated subtitle cues.
        """


class CaptionOptimizer(Protocol):
    """Protocol for LLM subtitle text optimizers."""

    def optimize(self, cues: list[SubtitleCue], tokens: list[WordSpan]) -> list[SubtitleCue]:
        """
        Optimize source and target subtitle text and boundaries.

        Parameters
        ----------
        cues : list[SubtitleCue]
            Translated subtitle cues.
        tokens : list[WordSpan]
            Timestamped ASR tokens used for subtitle boundary decisions.

        Returns
        -------
        list[SubtitleCue]
            Optimized subtitle cues.
        """


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    config: CaptionConfig,
    asr: AsrEngine,
    translator: Translator | None,
    optimizer: CaptionOptimizer | None = None,
    save_asr_json: bool = False,
) -> list[OutputPaths]:
    """
    Process a file or folder.

    Parameters
    ----------
    input_path : Path
        Input media file or folder.
    output_dir : Path
        Output directory.
    config : CaptionConfig
        Caption generation configuration.
    asr : AsrEngine
        ASR engine.
    translator : Translator | None
        Optional translator. Required only for target-language translation.
    optimizer : CaptionOptimizer | None
        Optional subtitle text optimizer.
    save_asr_json : bool
        Whether to save ASR debug JSON.

    Returns
    -------
    list[OutputPaths]
        Generated output paths.

    Raises
    ------
    ValueError
        If no media files are discovered.
    """
    jobs = discover_media_jobs(input_path, output_dir)
    if not jobs:
        raise ValueError(f"no media files found under {input_path}")
    log_step(f"Discovered {len(jobs)} media file(s)", icon="🔎")
    outputs: list[OutputPaths] = []
    for index, job in enumerate(jobs, start=1):
        log_step(f"Processing file {index}/{len(jobs)}: {job.input_path}", icon="📄")
        outputs.append(process_job(job, config, asr, translator, optimizer, save_asr_json))
    return outputs


def process_job(
    job: MediaJob,
    config: CaptionConfig,
    asr: AsrEngine,
    translator: Translator | None,
    optimizer: CaptionOptimizer | None = None,
    save_asr_json: bool = False,
) -> OutputPaths:
    """
    Process one media job.

    Parameters
    ----------
    job : MediaJob
        Media job.
    config : CaptionConfig
        Caption generation configuration.
    asr : AsrEngine
        ASR engine.
    translator : Translator | None
        Optional translator. Required only for target-language translation.
    optimizer : CaptionOptimizer | None
        Optional subtitle text optimizer.
    save_asr_json : bool
        Whether to save ASR debug JSON.

    Returns
    -------
    OutputPaths
        Generated output paths.

    Raises
    ------
    ValueError
        If ASR returns no timestamped words.
    """
    paths = build_output_paths(job.output_dir, job.relative_output_dir, job.stem, save_asr_json)
    written_paths: list[Path] = []

    log_step(f"ASR started: {job.input_path}", icon="🎙️")
    asr_result = asr.transcribe(job.input_path, language=config.source_language)
    if paths.asr_json is not None:
        paths.asr_json.parent.mkdir(parents=True, exist_ok=True)
        paths.asr_json.write_text(json.dumps(asdict(asr_result), ensure_ascii=False, indent=2), encoding="utf-8")
        written_paths.append(paths.asr_json)
        log_step(f"ASR JSON saved: {paths.asr_json}", icon="💾")
    if not asr_result.words:
        raise ValueError(f"ASR returned no timestamped words for {job.input_path}")

    write_txt = config.plain_text or config.write_text
    cues = build_cues(asr_result.words, config.max_chars_per_cue, config.max_seconds_per_cue)
    asr_text = _asr_plain_text(asr_result, cues)
    written_paths.extend(_write_source_outputs(paths.asr_srt, paths.asr_txt, cues, asr_text, write_txt))
    log_step(f"ASR text outputs saved: {_format_paths(written_paths)}", icon="💾")
    if config.plain_text:
        log_step("Plain-text mode enabled; skipped LLM steps", icon="⏭️")
        return replace(paths, written_paths=tuple(written_paths))

    if config.target_language:
        if translator is None:
            raise ValueError("translator is required when target_language is set")
        log_step(f"Translation started: target={config.target_language}", icon="🌐")
    translated_cues = translator.translate(cues) if config.target_language else cues
    if optimizer is not None:
        if (
            paths.raw_source_srt is None
            or paths.raw_source_txt is None
            or paths.raw_target_srt is None
            or paths.raw_target_txt is None
            or paths.raw_bilingual_srt is None
        ):
            raise ValueError("raw subtitle paths are required when optimizer is enabled")
        written_paths.extend(
            _write_source_outputs(
                paths.raw_source_srt,
                paths.raw_source_txt,
                translated_cues,
                _plain_text(cue.source_text for cue in translated_cues),
                write_txt,
            )
        )
        if config.target_language:
            written_paths.extend(
                _write_target_outputs(
                    paths.raw_target_srt,
                    paths.raw_target_txt,
                    paths.raw_bilingual_srt,
                    translated_cues,
                    config.translation_position,
                    write_txt,
                )
            )
            log_step(f"Raw translated outputs saved: {paths.raw_bilingual_srt}", icon="💾")
    if optimizer is not None:
        log_step("LLM subtitle optimization started", icon="🧠")
        translated_cues = optimizer.optimize(translated_cues, asr_result.words)
        log_step("LLM subtitle optimization completed", icon="✅")

    written_paths.extend(
        _write_source_outputs(
            paths.source_srt,
            paths.source_txt,
            translated_cues,
            _plain_text(cue.source_text for cue in translated_cues),
            write_txt,
        )
    )
    if config.target_language:
        written_paths.extend(
            _write_target_outputs(
                paths.target_srt,
                paths.target_txt,
                paths.bilingual_srt,
                translated_cues,
                config.translation_position,
                write_txt,
            )
        )
        log_step(f"Final bilingual output saved: {paths.bilingual_srt}", icon="✅")
    else:
        log_step(f"Final source output saved: {paths.source_srt}", icon="✅")

    return replace(paths, written_paths=tuple(written_paths))


def _write_source_outputs(
    srt_path: Path, txt_path: Path, cues: list[SubtitleCue], plain_text: str, write_txt: bool
) -> list[Path]:
    _write_text(srt_path, render_srt(cues, language="source"))
    written_paths = [srt_path]
    if write_txt:
        _write_text(txt_path, plain_text)
        written_paths.append(txt_path)
    return written_paths


def _write_target_outputs(
    srt_path: Path,
    txt_path: Path,
    bilingual_path: Path,
    cues: list[SubtitleCue],
    translation_position: str,
    write_txt: bool,
) -> list[Path]:
    _write_text(srt_path, render_srt(cues, language="target"))
    _write_text(bilingual_path, render_bilingual_srt(cues, translation_position=translation_position))
    written_paths = [srt_path, bilingual_path]
    if write_txt:
        _write_text(txt_path, _plain_text(cue.target_text for cue in cues))
        written_paths.insert(1, txt_path)
    return written_paths


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _plain_text(lines: Iterable[str]) -> str:
    return "\n".join(str(line).strip() for line in lines if str(line).strip()) + "\n"


def _format_paths(paths: list[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def _asr_plain_text(asr_result: AsrResult, fallback_cues: list[SubtitleCue]) -> str:
    chunk_lines = [
        str(chunk.get("text", "")).strip() for chunk in asr_result.chunks if str(chunk.get("text", "")).strip()
    ]
    if chunk_lines:
        return "\n\n".join(chunk_lines) + "\n"
    return _plain_text(cue.source_text for cue in fallback_cues)
