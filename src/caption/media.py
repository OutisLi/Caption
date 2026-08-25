"""Discover media inputs and derive output paths."""

from pathlib import Path

from caption.language import filename_language_code
from caption.types import MediaJob, OutputPaths

MEDIA_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def discover_media_jobs(input_path: Path, output_dir: Path) -> list[MediaJob]:
    """
    Discover media files from a file or directory.

    Parameters
    ----------
    input_path : Path
        Input media file or directory.
    output_dir : Path
        Directory where outputs should be written.

    Returns
    -------
    list[MediaJob]
        Sorted media jobs.

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    ValueError
        If a file input is not a supported media type.
    """
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.is_file():
        if not is_media_file(input_path):
            raise ValueError(f"unsupported media file: {input_path}")
        return [MediaJob(input_path=input_path, output_dir=output_dir, stem=input_path.stem)]

    files = [path for path in input_path.rglob("*") if path.is_file() and is_media_file(path)]
    return [
        MediaJob(
            input_path=path,
            output_dir=output_dir,
            stem=path.stem,
            relative_output_dir=path.relative_to(input_path).parent,
        )
        for path in sorted(files)
    ]


def build_output_paths(
    output_dir: Path,
    relative_output_dir: Path,
    stem: str,
    save_asr_json: bool,
    source_lang: str = "",
    target_lang: str = "",
) -> OutputPaths:
    """
    Build output subtitle and MKV paths.

    Final monolingual files take a language suffix (``name_zh.srt``). The bilingual
    file is the unsuffixed ``name.srt``.

    Parameters
    ----------
    output_dir : Path
        Base output directory.
    relative_output_dir : Path
        Input-root-relative directory preserved under each output stage.
    stem : str
        Output filename stem.
    save_asr_json : bool
        Whether to include an ASR debug JSON path.
    source_lang : str
        Source-language label used for the final source filename.
    target_lang : str
        Target-language label used for the final target filename.

    Returns
    -------
    OutputPaths
        Generated output paths.
    """
    asr_dir = output_dir / "asr" / relative_output_dir
    raw_dir = output_dir / "srt" / "raw" / relative_output_dir
    final_dir = output_dir / "srt" / "final" / relative_output_dir
    mkv_dir = output_dir / "mkv" / relative_output_dir
    asr_json = asr_dir / f"{stem}.asr.json" if save_asr_json else None
    source_code = filename_language_code(source_lang)
    target_code = filename_language_code(target_lang)
    return OutputPaths(
        asr_srt=asr_dir / f"{stem}.asr.srt",
        asr_txt=asr_dir / f"{stem}.asr.txt",
        source_srt=final_dir / f"{stem}_{source_code}.srt",
        source_txt=final_dir / f"{stem}_{source_code}.txt",
        target_srt=final_dir / f"{stem}_{target_code}.srt",
        target_txt=final_dir / f"{stem}_{target_code}.txt",
        bilingual_srt=final_dir / f"{stem}.srt",
        mkv=mkv_dir / f"{stem}.mkv",
        raw_source_srt=raw_dir / f"{stem}.raw.source.srt",
        raw_source_txt=raw_dir / f"{stem}.raw.source.txt",
        raw_target_srt=raw_dir / f"{stem}.raw.target.srt",
        raw_target_txt=raw_dir / f"{stem}.raw.target.txt",
        raw_bilingual_srt=raw_dir / f"{stem}.raw.bilingual.srt",
        asr_json=asr_json,
    )


def is_media_file(path: Path) -> bool:
    """
    Return whether a path has a supported media extension.

    Parameters
    ----------
    path : Path
        Path to check.

    Returns
    -------
    bool
        True when the suffix is supported.
    """
    return path.suffix.lower() in MEDIA_EXTENSIONS
