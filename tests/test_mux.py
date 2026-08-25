from pathlib import Path
import shutil
import subprocess

import pytest

from caption.language import LanguageError, filename_language_code, resolve_language
from caption.mux import MuxError, SubtitleTrack, build_mux_command, mux_subtitles


def test_resolve_language_accepts_codes_and_names() -> None:
    chinese = resolve_language("zh")
    assert chinese == resolve_language("Chinese")
    assert chinese == resolve_language("zh-CN")
    assert chinese.tag == "chi"
    assert chinese.code == "zh"
    assert chinese.title == "Chinese"

    english = resolve_language("English")
    assert english == resolve_language("en")
    assert english.tag == "eng"
    assert english.code == "en"

    assert resolve_language("").tag == "und"
    assert resolve_language("yue").tag == "yue"
    assert resolve_language("nav").tag == "nav"
    assert filename_language_code("English") == "en"
    assert filename_language_code("zh") == "zh"


def test_resolve_language_rejects_an_unknown_label() -> None:
    with pytest.raises(LanguageError, match="unrecognized language"):
        resolve_language("not-a-language")


def test_build_mux_command_copies_media_and_marks_the_default_track(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    source = tmp_path / "clip.source.srt"
    target = tmp_path / "clip.target.srt"
    bilingual = tmp_path / "clip.bilingual.srt"
    dest = tmp_path / "clip.mkv"
    tracks = [
        SubtitleTrack(source, "eng", "English", default=False),
        SubtitleTrack(target, "chi", "Chinese", default=False),
        SubtitleTrack(bilingual, "mul", "Bilingual", default=True),
    ]

    command = build_mux_command(media, dest, tracks)

    assert command[:8] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media), "-i"]
    assert command[command.index("-map") : command.index("-c")] == [
        "-map",
        "0:v?",
        "-map",
        "0:a?",
        "-map",
        "1:0",
        "-map",
        "2:0",
        "-map",
        "3:0",
    ]
    assert command[command.index("-c") : command.index("-c") + 2] == ["-c", "copy"]
    assert "-c:s" not in command
    assert "-metadata:s:s:0" in command
    assert command[command.index("-metadata:s:s:0") + 1] == "language=eng"
    assert command[command.index("-metadata:s:s:1") + 1] == "language=chi"
    assert command[command.index("-metadata:s:s:2") + 1] == "language=mul"
    assert "-disposition:s:s:0" not in command
    assert command[command.index("-disposition:s:0") + 1] == "0"
    assert command[command.index("-disposition:s:1") + 1] == "0"
    assert command[command.index("-disposition:s:2") + 1] == "default"
    assert command[-1] == str(dest)


def test_mux_subtitles_replaces_the_destination_only_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    source = tmp_path / "clip.source.srt"
    source.write_text("1\n", encoding="utf-8")
    dest = tmp_path / "clip.mkv"
    dest.write_bytes(b"stale")
    tracks = [SubtitleTrack(source, "eng", "English", default=True)]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"muxed")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    mux_subtitles(media, dest, tracks)

    assert dest.read_bytes() == b"muxed"
    assert not dest.with_suffix(".part.mkv").exists()


def test_mux_subtitles_cleans_a_partial_file_when_ffmpeg_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    source = tmp_path / "clip.source.srt"
    source.write_text("1\n", encoding="utf-8")
    dest = tmp_path / "clip.mkv"
    tracks = [SubtitleTrack(source, "eng", "English", default=True)]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"truncated")
        return subprocess.CompletedProcess(command, 1, "", "no video stream")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(MuxError, match="no video stream"):
        mux_subtitles(media, dest, tracks)

    assert not dest.exists()
    assert not dest.with_suffix(".part.mkv").exists()


def test_mux_subtitles_fails_when_ffmpeg_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    source = tmp_path / "clip.source.srt"
    source.write_text("1\n", encoding="utf-8")

    def missing_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing_run)

    with pytest.raises(MuxError, match="ffmpeg is not on PATH"):
        mux_subtitles(media, tmp_path / "clip.mkv", [SubtitleTrack(source, "eng", "English", default=True)])


def test_mux_subtitles_writes_language_tagged_tracks_with_ffmpeg(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for this check")

    media = tmp_path / "clip.mp4"
    generated = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=1",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(media),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if generated.returncode != 0:
        pytest.skip(f"cannot generate test media: {generated.stderr.strip()}")

    source = tmp_path / "en.srt"
    target = tmp_path / "zh.srt"
    bilingual = tmp_path / "bi.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    target.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    bilingual.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n你好\n", encoding="utf-8")
    dest = tmp_path / "clip.mkv"

    mux_subtitles(
        media,
        dest,
        [
            SubtitleTrack(source, "eng", "English", default=False),
            SubtitleTrack(target, "chi", "Chinese", default=False),
            SubtitleTrack(bilingual, "mul", "Bilingual", default=True),
        ],
    )

    probed = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index:stream_tags=language,title:stream_disposition=default",
            "-of",
            "compact",
            str(dest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in probed.stdout.splitlines() if line.startswith("stream|")]
    assert "tag:language=eng" in lines[0] and "disposition:default=0" in lines[0]
    assert "tag:language=chi" in lines[1] and "disposition:default=0" in lines[1]
    assert "tag:language=mul" in lines[2] and "disposition:default=1" in lines[2]
