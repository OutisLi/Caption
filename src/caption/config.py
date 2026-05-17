"""Load runtime configuration from TOML."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

from caption.asr_mlx import DEFAULT_ALIGNER_MODEL, DEFAULT_ASR_MODEL


@dataclass(frozen=True)
class LlmSettings:
    """LLM API settings."""

    api_key: str
    base_url: str | None
    model: str
    enable_thinking: bool
    reasoning_effort: str
    concurrency: int
    optimization_retries: int


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings loaded from config.toml."""

    output_dir: Path
    asr_model: str
    aligner_model: str
    model_cache_dir: Path | None
    translation_position: str
    max_chars_per_cue: int
    max_seconds_per_cue: float
    optimization_window_seconds: float
    max_optimized_seconds: float
    max_optimized_target_chars: int
    min_optimized_seconds: float
    optimization_pause_seconds: float
    save_asr_json: bool
    optimize_subtitles: bool
    llm: LlmSettings


def load_runtime_config(path: Path) -> RuntimeConfig:
    """
    Load runtime configuration from a TOML file.

    Parameters
    ----------
    path : Path
        TOML configuration path.

    Returns
    -------
    RuntimeConfig
        Parsed runtime configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required non-LLM settings are invalid.
    """
    with path.open("rb") as file:
        data = tomllib.load(file)

    llm = data.get("llm", {})
    asr = data.get("asr", {})
    output = data.get("output", {})
    subtitle = data.get("subtitle", {})

    api_key = str(llm.get("api_key", "")).strip()
    model = str(llm.get("model", "")).strip()
    translation_position = str(subtitle.get("translation_position", "bottom"))
    if translation_position not in {"top", "bottom"}:
        raise ValueError("subtitle.translation_position must be 'top' or 'bottom'")

    model_cache_dir = str(asr.get("model_cache_dir", "")).strip()
    return RuntimeConfig(
        output_dir=Path(str(output.get("dir", "outputs"))),
        asr_model=str(asr.get("model", DEFAULT_ASR_MODEL)).strip() or DEFAULT_ASR_MODEL,
        aligner_model=str(asr.get("aligner_model", DEFAULT_ALIGNER_MODEL)).strip() or DEFAULT_ALIGNER_MODEL,
        model_cache_dir=Path(model_cache_dir) if model_cache_dir else None,
        translation_position=translation_position,
        max_chars_per_cue=int(subtitle.get("max_chars_per_cue", 60)),
        max_seconds_per_cue=float(subtitle.get("max_seconds_per_cue", 6.0)),
        optimization_window_seconds=float(subtitle.get("optimization_window_seconds", 30.0)),
        max_optimized_seconds=float(subtitle.get("max_optimized_seconds", 5.0)),
        max_optimized_target_chars=int(subtitle.get("max_optimized_target_chars", 22)),
        min_optimized_seconds=float(subtitle.get("min_optimized_seconds", 2.0)),
        optimization_pause_seconds=float(subtitle.get("optimization_pause_seconds", 1.0)),
        save_asr_json=bool(output.get("save_asr_json", True)),
        optimize_subtitles=bool(subtitle.get("optimize", True)),
        llm=LlmSettings(
            api_key=api_key,
            base_url=str(llm.get("base_url", "")).strip() or None,
            model=model,
            enable_thinking=bool(llm.get("enable_thinking", True)),
            reasoning_effort=str(llm.get("reasoning_effort", "high")),
            concurrency=int(llm.get("concurrency", 4)),
            optimization_retries=int(llm.get("optimization_retries", 3)),
        ),
    )
