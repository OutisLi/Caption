"""Mux generated subtitles into an MKV beside the source media."""

from dataclasses import dataclass
from pathlib import Path
import subprocess


class MuxError(RuntimeError):
    """Raised when subtitle embedding cannot be completed."""


@dataclass(frozen=True)
class SubtitleTrack:
    """One subtitle stream to mux.

    Parameters
    ----------
    path : Path
        SRT file to attach.
    language : str
        ISO 639-2/B language tag.
    title : str
        Track title shown by players.
    default : bool
        Whether players should select this track first.
    """

    path: Path
    language: str
    title: str
    default: bool = False


def build_mux_command(media_path: Path, output_path: Path, tracks: list[SubtitleTrack]) -> list[str]:
    """
    Build the ffmpeg command that copies media streams and attaches SRT tracks.

    Video and audio are copied from the source. Existing subtitle streams in the
    source are dropped so the generated tracks are the only ones present.

    Parameters
    ----------
    media_path : Path
        Source audio or video file.
    output_path : Path
        Destination MKV path.
    tracks : list[SubtitleTrack]
        Subtitle streams, in mux order.

    Returns
    -------
    list[str]
        argv for ``ffmpeg``.

    Raises
    ------
    MuxError
        If ``tracks`` is empty.
    """
    if not tracks:
        raise MuxError("no subtitle tracks to embed")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media_path),
    ]
    for track in tracks:
        command.extend(["-i", str(track.path)])
    command.extend(["-map", "0:v?", "-map", "0:a?"])
    for index in range(len(tracks)):
        command.extend(["-map", f"{index + 1}:0"])
    command.extend(["-c", "copy"])
    for index, track in enumerate(tracks):
        # FFmpeg 7+ rejects -disposition:s:s:N ("stream type specified multiple
        # times"). Metadata still uses the per-stream form -metadata:s:s:N;
        # disposition is scoped by type already, so the specifier is :s:N.
        command.extend(
            [
                f"-metadata:s:s:{index}",
                f"language={track.language}",
                f"-metadata:s:s:{index}",
                f"title={track.title}",
                f"-disposition:s:{index}",
                "default" if track.default else "0",
            ]
        )
    command.append(str(output_path))
    return command


def mux_subtitles(media_path: Path, output_path: Path, tracks: list[SubtitleTrack]) -> None:
    """
    Embed subtitle tracks into a stream-copied MKV.

    The destination is written through a sibling ``.part.mkv`` file and replaced only
    after ffmpeg exits successfully, so a failed mux never leaves a truncated MKV.

    Parameters
    ----------
    media_path : Path
        Source audio or video file.
    output_path : Path
        Destination MKV path.
    tracks : list[SubtitleTrack]
        Subtitle streams, in mux order.

    Raises
    ------
    MuxError
        If a subtitle file is missing, ffmpeg is not on PATH, or muxing fails.
    """
    for track in tracks:
        if not track.path.is_file():
            raise MuxError(f"subtitle file does not exist: {track.path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".part.mkv")
    command = build_mux_command(media_path, partial, tracks)
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise MuxError("ffmpeg is not on PATH; install ffmpeg to embed subtitles") from exc
    if completed.returncode != 0:
        partial.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "").strip() or f"exit {completed.returncode}"
        raise MuxError(f"ffmpeg mux failed for {media_path}: {detail}")
    partial.replace(output_path)
