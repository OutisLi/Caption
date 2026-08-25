import pytest

import caption.cli as cli
from caption.cli import apply_model_cache_dir, require_llm_settings
from caption.config import DEFAULTS_PATH, load_runtime_config
from caption.types import CaptionConfig, OutputPaths


def test_config_file_drives_runtime_settings(tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "anthropic"
api_key = "key"
base_url = "https://api.example.com"
model = "some-model"
enable_thinking = true
reasoning_effort = "high"
concurrency = 4
retries = 3

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
review_rounds = 4
review_pass_score = 5
""",
        encoding="utf-8",
    )
    config = load_runtime_config(config_path)

    assert config.llm.provider == "anthropic"
    assert config.llm.model == "some-model"
    assert config.llm.concurrency == 4
    assert config.llm.retries == 3
    assert config.output_dir == cli.Path("my_outputs")
    assert config.subtitle.segmentation == "llm"
    assert config.subtitle.max_chars_per_cue == 42
    assert config.subtitle.max_seconds_per_cue == 5.0
    assert config.subtitle.review_rounds == 4
    assert config.subtitle.review_pass_score == 5
    assert require_llm_settings(config) == config.llm
    monkeypatch.delenv("HF_HOME", raising=False)

    apply_model_cache_dir(config)

    assert cli.os.environ["HF_HOME"] == "/models/cache"


def test_an_empty_config_resolves_entirely_from_the_packaged_defaults(tmp_path: cli.Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")

    config = load_runtime_config(config_path)

    assert DEFAULTS_PATH.exists()
    assert config.asr_model == "mlx-community/Qwen3-ASR-1.7B-8bit"
    assert config.aligner_model == "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
    assert config.model_cache_dir is None
    assert config.output_dir == cli.Path("outputs")
    assert config.save_asr_json is True
    assert config.embed is True
    assert config.llm.provider == "openai"
    assert config.llm.base_url is None
    assert config.llm.concurrency == 4
    assert config.llm.retries == 3
    assert config.llm.request_timeout == 30.0
    assert config.llm.enable_thinking is True
    assert config.llm.reasoning_effort == "high"
    assert config.subtitle.segmentation == "llm"
    assert config.subtitle.target_lang == "zh"
    assert config.subtitle.translation_position == "bottom"
    assert config.subtitle.max_chars_per_cue == 60
    assert config.subtitle.max_seconds_per_cue == 6.0
    assert config.subtitle.review_rounds == 2
    assert config.subtitle.review_pass_score == 4


def test_credentials_have_no_default_and_fail_locally_when_omitted(tmp_path: cli.Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[llm]\nmodel = "some-model"\n', encoding="utf-8")

    config = load_runtime_config(config_path)

    assert config.llm.api_key == ""
    with pytest.raises(ValueError, match="llm.api_key"):
        require_llm_settings(config)


def test_a_single_overridden_key_leaves_the_rest_of_its_section_on_defaults(tmp_path: cli.Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[llm]\nconcurrency = 20\n", encoding="utf-8")

    config = load_runtime_config(config_path)

    assert config.llm.concurrency == 20
    assert config.llm.retries == 3
    assert config.llm.provider == "openai"


def test_overriding_one_sampling_key_keeps_the_rest_of_both_modes(tmp_path: cli.Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[llm.instruct]\npresence_penalty = 0.0\n", encoding="utf-8")

    config = load_runtime_config(config_path)

    assert config.llm.instruct_sampling.presence_penalty == 0.0
    assert config.llm.instruct_sampling.temperature == 0.7
    assert config.llm.instruct_sampling.top_k == 20
    assert config.llm.thinking_sampling.temperature == 1.0
    assert config.llm.thinking_sampling.top_p == 0.95


def test_config_rejects_a_top_level_key_outside_any_table(tmp_path: cli.Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('concurrency = 20\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must sit inside a table"):
        load_runtime_config(config_path)


def test_config_rejects_review_pass_score_that_can_never_reject(tmp_path: cli.Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[subtitle]
review_pass_score = 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="subtitle.review_pass_score"):
        load_runtime_config(config_path)


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

    def fail_create_llm_completion_client(**kwargs: object) -> object:
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
        return [
            OutputPaths(
                asr_srt=written_path,
                asr_txt=tmp_path / "outputs" / "clip.asr.txt",
                source_srt=tmp_path / "outputs" / "clip.source.srt",
                source_txt=tmp_path / "outputs" / "clip.source.txt",
                target_srt=tmp_path / "outputs" / "clip.target.srt",
                target_txt=tmp_path / "outputs" / "clip.target.txt",
                bilingual_srt=tmp_path / "outputs" / "clip.bilingual.srt",
                mkv=tmp_path / "outputs" / "clip.mkv",
                written_paths=(written_path,),
            )
        ]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "create_llm_completion_client", fail_create_llm_completion_client)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path), "--plain-text", "--text"]) == 0
    assert capsys.readouterr().out == f"{written_path}\n"


def test_cli_stops_before_asr_when_llm_preflight_fails(
    tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.toml").write_text(
        """
[llm]
api_key = "key"
model = "model"
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")

    def fail_validate_llm_completion_client(client: object) -> None:
        raise cli.TranslationError("LLM preflight failed: bad key")

    def fail_local_mlx_asr(**kwargs: object) -> object:
        raise AssertionError("ASR must not start when LLM preflight fails")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "create_llm_completion_client", lambda **kwargs: object())
    monkeypatch.setattr(cli, "validate_llm_completion_client", fail_validate_llm_completion_client)
    monkeypatch.setattr(cli, "LocalMlxAsr", fail_local_mlx_asr)

    assert cli.main([str(input_path)]) == 1
    assert "caption: LLM preflight failed: bad key" in capsys.readouterr().err


def test_cli_passes_user_output_options_to_pipeline(tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[llm]
api_key = "key"
model = "model"

[subtitle]
review_rounds = 0
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
        assert config.review is False
        assert config.embed is True
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "create_llm_completion_client", lambda **kwargs: object())
    monkeypatch.setattr(cli, "validate_llm_completion_client", lambda client: None)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path), "--target-lang", "zh", "--text"]) == 0


def test_cli_disables_mkv_embed_from_output_config(
    tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.toml").write_text(
        """
[llm]
api_key = "key"
model = "model"

[output]
embed = false
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    seen: list[bool] = []

    def fake_run_pipeline(
        input_path: cli.Path,
        output_dir: cli.Path,
        config: CaptionConfig,
        **kwargs: object,
    ) -> list[OutputPaths]:
        seen.append(config.embed)
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "create_llm_completion_client", lambda **kwargs: object())
    monkeypatch.setattr(cli, "validate_llm_completion_client", lambda client: None)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path), "--target-lang", ""]) == 0
    assert seen == [False]


def test_asr_segmentation_never_reaches_the_llm(tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[subtitle]
segmentation = "asr"
target_lang = ""
review_rounds = 2
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")

    def fail_create_llm_completion_client(**kwargs: object) -> object:
        raise AssertionError("ASR segmentation must not reach the LLM")

    def fake_run_pipeline(
        input_path: cli.Path,
        output_dir: cli.Path,
        config: CaptionConfig,
        **kwargs: object,
    ) -> list[OutputPaths]:
        assert config.target_language is None
        assert kwargs["translator"] is None
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "create_llm_completion_client", fail_create_llm_completion_client)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path)]) == 0


def test_llm_segmentation_runs_without_a_translation_target(
    tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.toml").write_text(
        """
[llm]
api_key = "key"
model = "model"

[subtitle]
target_lang = ""
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
        assert kwargs["translator"] is not None
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "create_llm_completion_client", lambda **kwargs: object())
    monkeypatch.setattr(cli, "validate_llm_completion_client", lambda client: None)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path)]) == 0


def test_config_rejects_asr_segmentation_with_a_translation_target(tmp_path: cli.Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[subtitle]
segmentation = "asr"
target_lang = "zh"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="subtitle.segmentation"):
        load_runtime_config(config_path)


def test_config_target_lang_drives_default_and_cli_overrides(
    tmp_path: cli.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.toml").write_text(
        """
[llm]
api_key = "key"
model = "model"

[subtitle]
target_lang = "ja"
review_rounds = 0
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "clip.wav"
    input_path.write_bytes(b"fake")
    seen_languages: list[str | None] = []

    def fake_run_pipeline(
        input_path: cli.Path,
        output_dir: cli.Path,
        config: CaptionConfig,
        **kwargs: object,
    ) -> list[OutputPaths]:
        seen_languages.append(config.target_language)
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "LocalMlxAsr", lambda **kwargs: object())
    monkeypatch.setattr(cli, "create_llm_completion_client", lambda **kwargs: object())
    monkeypatch.setattr(cli, "validate_llm_completion_client", lambda client: None)
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main([str(input_path)]) == 0
    assert cli.main([str(input_path), "--target-lang", ""]) == 0

    assert seen_languages == ["ja", None]
