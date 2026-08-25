"""Load runtime configuration from TOML.

Defaults live in a single place: the packaged ``defaults.toml``. A user's ``config.toml``
is overlaid on top of it, so omitting a key yields the packaged value and the two files
can never disagree about what a default is. Keys with no sensible default - credentials,
the chat model name, the base URL - are absent from ``defaults.toml`` on purpose, so that
omitting them still fails with a clear local error.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib

from caption.llm_json import MAX_REVIEW_SCORE, MIN_REVIEW_SCORE
from caption.types import SamplingParams

DEFAULTS_PATH = Path(__file__).with_name("defaults.toml")


@dataclass(frozen=True)
class LlmSettings:
    """LLM endpoint and request settings."""

    provider: str
    api_key: str
    base_url: str | None
    model: str
    enable_thinking: bool
    reasoning_effort: str
    concurrency: int
    retries: int
    request_timeout: float
    thinking_sampling: SamplingParams
    instruct_sampling: SamplingParams


@dataclass(frozen=True)
class SubtitleSettings:
    """Subtitle content and refinement settings."""

    segmentation: str
    target_lang: str
    translation_position: str
    max_chars_per_cue: int
    max_seconds_per_cue: float
    review_rounds: int
    review_pass_score: int


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings loaded from config.toml."""

    output_dir: Path
    asr_model: str
    aligner_model: str
    model_cache_dir: Path | None
    save_asr_json: bool
    llm: LlmSettings
    subtitle: SubtitleSettings


def load_runtime_config(path: Path) -> RuntimeConfig:
    """
    Load runtime configuration, filling omitted keys from the packaged defaults.

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
        If a top-level entry is not a table, or if the provider, segmentation mode,
        translation position, or review pass score is invalid, or if ASR segmentation is
        combined with a translation target.
    """
    overrides = _load_toml(path)
    _require_tables(overrides)
    data = _overlay(_load_toml(DEFAULTS_PATH), overrides)
    llm = data["llm"]
    asr = data["asr"]
    output = data["output"]
    subtitle = data["subtitle"]

    llm_provider = str(llm["provider"]).strip().lower()
    if llm_provider not in {"openai", "anthropic"}:
        raise ValueError("llm.provider must be 'openai' or 'anthropic'")
    translation_position = str(subtitle["translation_position"])
    if translation_position not in {"top", "bottom"}:
        raise ValueError("subtitle.translation_position must be 'top' or 'bottom'")

    segmentation = str(subtitle["segmentation"]).strip().lower()
    if segmentation not in {"llm", "asr"}:
        raise ValueError("subtitle.segmentation must be 'llm' or 'asr'")
    target_lang = str(subtitle["target_lang"]).strip()
    if segmentation == "asr" and target_lang:
        raise ValueError("subtitle.segmentation = 'asr' cannot translate; set subtitle.target_lang to \"\"")

    model_cache_dir = str(asr.get("model_cache_dir", "")).strip()
    return RuntimeConfig(
        output_dir=Path(str(output["dir"])),
        asr_model=str(asr["model"]).strip(),
        aligner_model=str(asr["aligner_model"]).strip(),
        model_cache_dir=Path(model_cache_dir) if model_cache_dir else None,
        save_asr_json=bool(output["save_asr_json"]),
        llm=LlmSettings(
            provider=llm_provider,
            api_key=str(llm.get("api_key", "")).strip(),
            base_url=str(llm.get("base_url", "")).strip() or None,
            model=str(llm.get("model", "")).strip(),
            enable_thinking=bool(llm["enable_thinking"]),
            reasoning_effort=str(llm["reasoning_effort"]),
            concurrency=int(llm["concurrency"]),
            retries=int(llm["retries"]),
            request_timeout=float(llm["request_timeout"]),
            thinking_sampling=_sampling_params(llm["thinking"]),
            instruct_sampling=_sampling_params(llm["instruct"]),
        ),
        subtitle=SubtitleSettings(
            segmentation=segmentation,
            target_lang=target_lang,
            translation_position=translation_position,
            max_chars_per_cue=int(subtitle["max_chars_per_cue"]),
            max_seconds_per_cue=float(subtitle["max_seconds_per_cue"]),
            review_rounds=int(subtitle["review_rounds"]),
            review_pass_score=_review_pass_score(subtitle["review_pass_score"]),
        ),
    )


def _load_toml(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def _overlay(defaults: dict, overrides: dict) -> dict:
    """Recursively overlay a user configuration onto the packaged defaults."""
    merged = dict(defaults)
    for key, value in overrides.items():
        existing = merged.get(key)
        merged[key] = (
            _overlay(existing, value) if isinstance(value, dict) and isinstance(existing, dict) else value
        )
    return merged


def _require_tables(overrides: dict) -> None:
    """
    Reject a bare top-level key, which is almost always a forgotten section header.

    Raises
    ------
    ValueError
        If a top-level entry is not a table.
    """
    for key, value in overrides.items():
        if not isinstance(value, dict):
            raise ValueError(f"config entry '{key}' must sit inside a table such as [llm]")


def _sampling_params(values: dict) -> SamplingParams:
    return SamplingParams(
        temperature=float(values["temperature"]),
        top_p=float(values["top_p"]),
        top_k=int(values["top_k"]),
        min_p=float(values["min_p"]),
        presence_penalty=float(values["presence_penalty"]),
        repetition_penalty=float(values["repetition_penalty"]),
    )


def _review_pass_score(value: object) -> int:
    score = int(value)
    if not MIN_REVIEW_SCORE < score <= MAX_REVIEW_SCORE:
        raise ValueError(
            f"subtitle.review_pass_score must be between {MIN_REVIEW_SCORE + 1} and {MAX_REVIEW_SCORE}"
        )
    return score
