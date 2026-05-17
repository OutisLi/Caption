import pytest

import caption.cli as cli
from caption.cli import apply_model_cache_dir, require_llm_settings
from caption.config import load_runtime_config
from caption.types import CaptionConfig, OutputPaths
from caption.asr_mlx import DEFAULT_ALIGNER_MODEL, DEFAULT_ASR_MODEL


def test_config_file_drives_runtime_settings(tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
api_key = "key"
base_url = "https://api.example.com"
model = "some-model"
enable_thinking = true
reasoning_effort = "high"
concurrency = 4
optimization_retries = 3

[asr]
model = "/models/asr"
aligner_model = "/models/aligner"
model_cache_dir = "/models/cache"

[output]
dir = "my_outputs"
save_asr_json = true

[subtitle]
translation_position = "bottom"
max_chars_per_cue = 42
max_seconds_per_cue = 5.0
optimization_window_seconds = 30.0
max_optimized_seconds = 5.0
max_optimized_target_chars = 15
min_optimized_seconds = 1.2
optimization_pause_seconds = 0.6
optimize = true
""",
        encoding="utf-8",
    )
    config = load_runtime_config(config_path)

    assert config.llm.model == "some-model"
    assert config.output_dir == cli.Path("my_outputs")
    assert config.optimize_subtitles is True
    assert config.llm.concurrency == 4
    assert config.optimization_window_seconds == 30.0
    assert config.max_optimized_seconds == 5.0
    assert config.max_optimized_target_chars == 15
    assert config.min_optimized_seconds == 1.2
    assert config.optimization_pause_seconds == 0.6
    assert config.llm.optimization_retries == 3
    assert require_llm_settings(config) == ("key", "https://api.example.com", "some-model")
    monkeypatch.delenv("HF_HOME", raising=False)

    apply_model_cache_dir(config)

    assert cli.os.environ["HF_HOME"] == "/models/cache"


def test_config_allows_local_asr_without_llm_settings(tmp_path: cli.Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[output]
dir = "outputs"
""",
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.asr_model == DEFAULT_ASR_MODEL
    assert config.aligner_model == DEFAULT_ALIGNER_MODEL
    assert config.max_optimized_seconds == 5.0
    assert config.max_optimized_target_chars == 22
    assert config.min_optimized_seconds == 2.0
    assert config.optimization_pause_seconds == 1.0
    with pytest.raises(ValueError, match="llm.api_key"):
        require_llm_settings(config)


def test_plain_text_cli_does_not_create_llm_client(
    tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.toml").write_text(
        """
[output]
dir = "outputs"
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    written_path = tmp_path / "outputs" / "clip.asr.srt"

    def fail_create_openai_client(**kwargs: object) -> object:
        raise AssertionError("plain-text mode must not create an LLM client")

    def fake_run_pipeline(
        input_path: cli.Path,
        output_dir: cli.Path,
        config: CaptionConfig,
        **kwargs: object,
    ) -> list[OutputPaths]:
        assert config.plain_text is True
        assert config.write_text is True
        assert kwargs["translator"] is None
        assert kwargs["optimizer"] is None
        return [
            OutputPaths(
                asr_srt=written_path,
                asr_txt=tmp_path / "outputs" / "clip.asr.txt",
                source_srt=tmp_path / "outputs" / "clip.source.srt",
                source_txt=tmp_path / "outputs" / "clip.source.txt",
                target_srt=tmp_path / "outputs" / "clip.target.srt",
                target_txt=tmp_path / "outputs" / "clip.target.txt",
                bilingual_srt=tmp_path / "outputs" / "clip.bilingual.srt",
                written_paths=(written_path,),
            )
        ]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "create_openai_client", fail_create_openai_client)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path), "--plain-text", "--text"]) == 0
    assert capsys.readouterr().out == f"{written_path}\n"


def test_text_option_enables_txt_outputs(tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[llm]
api_key = "key"
model = "model"

[subtitle]
optimize = false
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")

    def fake_run_pipeline(
        input_path: cli.Path,
        output_dir: cli.Path,
        config: CaptionConfig,
        **kwargs: object,
    ) -> list[OutputPaths]:
        assert config.write_text is True
        assert config.plain_text is False
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "create_openai_client", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path), "--target-lang", "zh", "--text"]) == 0


def test_target_language_defaults_to_zh(tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[llm]
api_key = "key"
model = "model"

[subtitle]
optimize = false
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")

    def fake_run_pipeline(
        input_path: cli.Path,
        output_dir: cli.Path,
        config: CaptionConfig,
        **kwargs: object,
    ) -> list[OutputPaths]:
        assert config.target_language == "zh"
        assert kwargs["translator"] is not None
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "create_openai_client", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path)]) == 0


def test_empty_target_language_disables_translation(tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[subtitle]
optimize = false
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")

    def fake_run_pipeline(
        input_path: cli.Path,
        output_dir: cli.Path,
        config: CaptionConfig,
        **kwargs: object,
    ) -> list[OutputPaths]:
        assert config.target_language is None
        assert kwargs["translator"] is None
        assert kwargs["optimizer"] is None
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "create_openai_client", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path), "--target-lang", ""]) == 0
