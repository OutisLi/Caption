from pathlib import Path

from caption.media import build_output_paths, discover_media_jobs


def test_discover_media_jobs_handles_file_directory_and_outputs(tmp_path: Path) -> None:
    media_file = tmp_path / "clip.MP4"
    media_file.write_bytes(b"fake")
    output_dir = tmp_path / "out"

    jobs = discover_media_jobs(media_file, output_dir)

    assert len(jobs) == 1
    assert jobs[0].input_path == media_file
    assert jobs[0].output_dir == output_dir
    assert jobs[0].stem == "clip"
    assert jobs[0].relative_output_dir == Path()

    (tmp_path / "a.mp3").write_bytes(b"fake")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.wav").write_bytes(b"fake")
    (nested / "notes.txt").write_text("ignore", encoding="utf-8")

    jobs = discover_media_jobs(tmp_path, tmp_path / "out")

    assert [job.input_path.name for job in jobs] == ["a.mp3", "clip.MP4", "b.wav"]
    assert [job.stem for job in jobs] == ["a", "clip", "b"]
    assert [job.relative_output_dir for job in jobs] == [Path(), Path(), Path("nested")]

    paths = build_output_paths(tmp_path / "out", Path("course/week1"), "clip", save_asr_json=True)

    assert paths.asr_srt == tmp_path / "out" / "asr" / "course" / "week1" / "clip.asr.srt"
    assert paths.source_srt == tmp_path / "out" / "final" / "course" / "week1" / "clip.source.srt"
    assert paths.target_srt == tmp_path / "out" / "final" / "course" / "week1" / "clip.target.srt"
    assert paths.bilingual_srt == tmp_path / "out" / "final" / "course" / "week1" / "clip.bilingual.srt"
    assert paths.raw_bilingual_srt == tmp_path / "out" / "raw" / "course" / "week1" / "clip.raw.bilingual.srt"
    assert paths.asr_json == tmp_path / "out" / "asr" / "course" / "week1" / "clip.asr.json"


def test_directory_media_jobs_preserve_input_root_internal_layout(tmp_path: Path) -> None:
    input_root = tmp_path / "media"
    first_dir = input_root / "first"
    second_dir = input_root / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir()
    (first_dir / "clip.mp4").write_bytes(b"fake")
    (second_dir / "clip.wav").write_bytes(b"fake")

    jobs = discover_media_jobs(input_root, tmp_path / "out")

    assert [job.stem for job in jobs] == ["clip", "clip"]
    assert [job.relative_output_dir for job in jobs] == [Path("first"), Path("second")]
