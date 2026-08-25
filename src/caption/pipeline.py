"""End-to-end caption generation pipeline."""

import json
import queue
import threading
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Protocol

from caption.language import Language, resolve_language
from caption.media import build_output_paths, discover_media_jobs
from caption.mux import SubtitleTrack, mux_subtitles
from caption.progress import log_step
from caption.segment import build_cues, build_cues_from_layouts
from caption.srt import render_bilingual_srt, render_srt
from caption.translator import TranslationDraft
from caption.types import (
    AsrResult,
    CaptionConfig,
    MediaJob,
    OutputPaths,
    SentenceLayout,
    SubtitleCue,
    WordSpan,
)


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
    """Protocol for the LLM subtitle stages."""

    def segment(self, words: list[WordSpan]) -> list[SentenceLayout]:
        """
        Restore sentences and lay them out as display lines, without translating.

        Parameters
        ----------
        words : list[WordSpan]
            Timestamped ASR tokens.

        Returns
        -------
        list[SentenceLayout]
            Sentences with untranslated display lines.
        """

    def translate(self, words: list[WordSpan]) -> TranslationDraft:
        """
        Restore sentences, build a glossary, and translate the transcript.

        Parameters
        ----------
        words : list[WordSpan]
            Timestamped ASR tokens.

        Returns
        -------
        TranslationDraft
            First-pass translation and the transcript context behind it.
        """

    def review(self, draft: TranslationDraft) -> list[SentenceLayout]:
        """
        Refine a first-pass translation through review rounds.

        Parameters
        ----------
        draft : TranslationDraft
            First-pass translation.

        Returns
        -------
        list[SentenceLayout]
            Sentences with revised text.
        """


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    config: CaptionConfig,
    asr: AsrEngine,
    translator: Translator | None,
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
    if _can_overlap_stages(config, translator):
        return _run_overlapped(jobs, config, asr, translator, save_asr_json)
    outputs: list[OutputPaths] = []
    for index, job in enumerate(jobs, start=1):
        log_step(f"Processing file {index}/{len(jobs)}: {job.input_path}", icon="📄")
        outputs.append(process_job(job, config, asr, translator, save_asr_json))
    return outputs


def _can_overlap_stages(config: CaptionConfig, translator: Translator | None) -> bool:
    """Return whether ASR and LLM stages can overlap across files."""
    return not config.plain_text and translator is not None


_PIPELINE_SENTINEL = object()


def _run_overlapped(
    jobs: list[MediaJob],
    config: CaptionConfig,
    asr: AsrEngine,
    translator: Translator | None,
    save_asr_json: bool,
) -> list[OutputPaths]:
    """
    Overlap ASR with LLM stages across files.

    A producer thread runs ASR one file at a time and hands results to the
    consumer (translation/optimization) through a queue, so file N+1 is
    transcribed while file N is still being translated. Failures in either
    stage abort the run; the producer finishes its in-flight file first so
    its ASR artifacts stay usable as cache on the next run.
    """
    work_queue: queue.Queue[object] = queue.Queue()
    stop = threading.Event()

    def produce() -> None:
        try:
            for index, job in enumerate(jobs, start=1):
                if stop.is_set():
                    break
                log_step(f"Processing file {index}/{len(jobs)}: {job.input_path}", icon="📄")
                paths = _job_output_paths(job, config, save_asr_json)
                asr_result, written_paths = _run_asr_stage(job, config, asr, paths)
                work_queue.put((job, paths, asr_result, written_paths))
        except Exception as exc:
            work_queue.put(exc)
        finally:
            work_queue.put(_PIPELINE_SENTINEL)

    producer = threading.Thread(target=produce, name="caption-asr-producer", daemon=True)
    producer.start()

    outputs: list[OutputPaths] = []
    try:
        while True:
            item = work_queue.get()
            if item is _PIPELINE_SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            job, paths, asr_result, written_paths = item  # type: ignore[misc]
            outputs.append(_run_llm_stage(job, config, translator, paths, asr_result, written_paths))
    except Exception:
        stop.set()
        producer.join()
        raise
    producer.join()
    return outputs


def process_job(
    job: MediaJob,
    config: CaptionConfig,
    asr: AsrEngine,
    translator: Translator | None,
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
    MuxError
        If subtitle embedding is enabled and muxing fails.
    """
    paths = _job_output_paths(job, config, save_asr_json)
    asr_result, written_paths = _run_asr_stage(job, config, asr, paths)
    return _run_llm_stage(job, config, translator, paths, asr_result, written_paths)


def _run_asr_stage(
    job: MediaJob,
    config: CaptionConfig,
    asr: AsrEngine,
    paths: OutputPaths,
) -> tuple[AsrResult, list[Path]]:
    """
    Run the ASR stage for one job: transcribe (or reuse the cache) and persist ASR artifacts.

    Returns
    -------
    tuple[AsrResult, list[Path]]
        ASR result and the paths written by this stage.

    Raises
    ------
    ValueError
        If ASR returns no timestamped words.
    """
    written_paths: list[Path] = []

    asr_result = _load_cached_asr_result(paths.asr_json)
    if asr_result is not None:
        log_step(f"ASR cache reused: {paths.asr_json}", icon="♻️")
    else:
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
    return asr_result, written_paths


def _run_llm_stage(
    job: MediaJob,
    config: CaptionConfig,
    translator: Translator | None,
    paths: OutputPaths,
    asr_result: AsrResult,
    written_paths: list[Path],
) -> OutputPaths:
    """
    Run the LLM stage for one job: translation, optional review, raw and final writes.

    Parameters
    ----------
    job : MediaJob
        Media job. The input path is required when embedding an MKV.
    config : CaptionConfig
        Caption generation configuration.
    translator : Translator | None
        Optional translator. Required only for target-language translation.
    paths : OutputPaths
        Output paths for this job.
    asr_result : AsrResult
        ASR result from the ASR stage.
    written_paths : list[Path]
        Paths already written by the ASR stage.

    Returns
    -------
    OutputPaths
        Generated output paths.

    Raises
    ------
    ValueError
        If translation is requested without a translator, or a subtitle cache is incomplete.
    MuxError
        If subtitle embedding is enabled and muxing fails.
    """
    paths = _job_output_paths(job, config, paths.asr_json is not None, asr_result)
    if config.plain_text:
        log_step("Plain-text mode enabled; skipped LLM steps", icon="⏭️")
    elif _reuse_final_subtitles(config, paths):
        log_step(f"Subtitle cache reused: {_cache_label(config, paths)}", icon="♻️")
    elif translator is None:
        if config.target_language:
            raise ValueError("translator is required when target_language is set")
        cues = build_cues(asr_result.words, config.max_chars_per_cue, config.max_seconds_per_cue)
        written_paths.extend(_write_final_outputs(config, paths, cues))
        log_step(f"Final source output saved: {paths.source_srt}", icon="✅")
    elif not config.target_language:
        log_step("Sentence segmentation started", icon="✂️")
        cues = build_cues_from_layouts(translator.segment(asr_result.words))
        written_paths.extend(_write_final_outputs(config, paths, cues))
        log_step(f"Final source output saved: {paths.source_srt}", icon="✅")
    else:
        log_step(f"Translation started: target={config.target_language}", icon="🌐")
        draft = translator.translate(asr_result.words)
        sentences: Sequence[SentenceLayout] = draft.sentences
        if config.review:
            written_paths.extend(_write_raw_outputs(config, paths, build_cues_from_layouts(sentences)))
            log_step(f"Raw translated outputs saved: {paths.raw_bilingual_srt}", icon="💾")
            log_step("LLM subtitle review started", icon="🧠")
            sentences = translator.review(draft)
            log_step("LLM subtitle review completed", icon="✅")

        written_paths.extend(_write_final_outputs(config, paths, build_cues_from_layouts(sentences)))
        log_step(f"Final bilingual output saved: {paths.bilingual_srt}", icon="✅")

    if config.embed:
        written_paths.append(_embed_mkv(job, config, paths, asr_result))
    return replace(paths, written_paths=tuple(written_paths))


def _embed_mkv(job: MediaJob, config: CaptionConfig, paths: OutputPaths, asr_result: AsrResult) -> Path:
    """Mux finished subtitle tracks into an MKV of the source media."""
    tracks = _subtitle_tracks(config, paths, asr_result)
    log_step(f"MKV mux started: {paths.mkv}", icon="🎬")
    mux_subtitles(job.input_path, paths.mkv, tracks)
    log_step(f"MKV mux completed: {paths.mkv}", icon="✅")
    return paths.mkv


def _subtitle_tracks(config: CaptionConfig, paths: OutputPaths, asr_result: AsrResult) -> list[SubtitleTrack]:
    """
    Choose which SRT files to embed and how players should label them.

    A translated job carries three tracks: source, target, and bilingual. The
    bilingual track is the default. Source-only jobs embed a single track.
    """
    source = _source_language(config, asr_result)
    if config.plain_text:
        return [SubtitleTrack(paths.asr_srt, source.tag, source.title, default=True)]
    if not config.target_language:
        return [SubtitleTrack(paths.source_srt, source.tag, source.title, default=True)]
    target = resolve_language(config.target_language)
    return [
        SubtitleTrack(paths.source_srt, source.tag, source.title, default=False),
        SubtitleTrack(paths.target_srt, target.tag, target.title, default=False),
        SubtitleTrack(paths.bilingual_srt, "mul", "Bilingual", default=True),
    ]


def _expected_final_srts(config: CaptionConfig, paths: OutputPaths) -> tuple[Path, ...]:
    """Return the final SRT files that constitute a reusable subtitle cache."""
    if config.plain_text:
        return ()
    if config.target_language:
        return (paths.source_srt, paths.target_srt, paths.bilingual_srt)
    return (paths.source_srt,)


def _reuse_final_subtitles(config: CaptionConfig, paths: OutputPaths) -> bool:
    """
    Reuse finished final SRTs when the complete set is already on disk.

    Missing files mean there is no cache. A partial or empty set is invalid and
    must fail, matching the ASR JSON cache: never continue as if the work is done,
    and never start a translation that would mix old and new files.

    Returns
    -------
    bool
        True when the expected final SRTs exist and can be reused.

    Raises
    ------
    ValueError
        If some but not all expected files exist, or a file is empty.
    """
    expected = _expected_final_srts(config, paths)
    if not expected:
        return False
    existing = tuple(path for path in expected if path.exists())
    if not existing:
        return False
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise ValueError(f"incomplete subtitle cache under {expected[0].parent}: missing {missing[0].name}")
    for path in expected:
        if not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"invalid subtitle cache file {path}: empty")
    return True


def _cache_label(config: CaptionConfig, paths: OutputPaths) -> Path:
    """Return the cache path to mention in logs."""
    return paths.bilingual_srt if config.target_language else paths.source_srt


def _job_output_paths(
    job: MediaJob,
    config: CaptionConfig,
    save_asr_json: bool,
    asr_result: AsrResult | None = None,
) -> OutputPaths:
    """Build output paths, using the ASR language once it is known."""
    source_lang = config.source_language or ""
    if asr_result is not None:
        source_lang = source_lang or asr_result.language
    return build_output_paths(
        job.output_dir,
        job.relative_output_dir,
        job.stem,
        save_asr_json,
        source_lang=source_lang,
        target_lang=config.target_language or "",
    )


def _source_language(config: CaptionConfig, asr_result: AsrResult) -> Language:
    """Prefer the forced source language, then the ASR report, then undefined."""
    return resolve_language(config.source_language or asr_result.language or "und")


def _write_raw_outputs(config: CaptionConfig, paths: OutputPaths, cues: list[SubtitleCue]) -> list[Path]:
    """
    Persist the pre-review translation so that review always starts from a saved state.

    Raises
    ------
    ValueError
        If the output layout carries no raw subtitle paths.
    """
    if (
        paths.raw_source_srt is None
        or paths.raw_source_txt is None
        or paths.raw_target_srt is None
        or paths.raw_target_txt is None
        or paths.raw_bilingual_srt is None
    ):
        raise ValueError("raw subtitle paths are required when review is enabled")
    written_paths = _write_source_outputs(
        paths.raw_source_srt,
        paths.raw_source_txt,
        cues,
        _plain_text(cue.source_text for cue in cues),
        config.write_text,
    )
    written_paths.extend(
        _write_target_outputs(
            paths.raw_target_srt,
            paths.raw_target_txt,
            paths.raw_bilingual_srt,
            cues,
            config.translation_position,
            config.write_text,
        )
    )
    return written_paths


def _write_final_outputs(config: CaptionConfig, paths: OutputPaths, cues: list[SubtitleCue]) -> list[Path]:
    """Persist the final source outputs, plus target outputs when translation ran."""
    written_paths = _write_source_outputs(
        paths.source_srt,
        paths.source_txt,
        cues,
        _plain_text(cue.source_text for cue in cues),
        config.write_text,
    )
    if config.target_language:
        written_paths.extend(
            _write_target_outputs(
                paths.target_srt,
                paths.target_txt,
                paths.bilingual_srt,
                cues,
                config.translation_position,
                config.write_text,
            )
        )
    return written_paths


def _load_cached_asr_result(path: Path | None) -> AsrResult | None:
    """
    Load a previously persisted ASR result when available.

    Parameters
    ----------
    path : Path | None
        ASR JSON path. None or a missing file means no cache.

    Returns
    -------
    AsrResult | None
        Cached ASR result, or None when no cache exists.

    Raises
    ------
    ValueError
        If the cache file exists but is not a valid ASR result.
    """
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        words = [
            WordSpan(text=str(word["text"]), start=float(word["start"]), end=float(word["end"]))
            for word in data["words"]
        ]
        chunks = list(data.get("chunks", []))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid ASR cache file {path}: {exc}") from exc
    return AsrResult(text=str(data.get("text", "")), language=str(data.get("language", "")), words=words, chunks=chunks)


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
