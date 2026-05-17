"""Command line interface for caption generation."""

import argparse
import os
import sys
from pathlib import Path

from caption.asr_mlx import LocalMlxAsr
from caption.config import RuntimeConfig, load_runtime_config
from caption.types import CaptionConfig
from caption.pipeline import run_pipeline
from caption.translator import OpenAICompatibleTranslator, create_openai_client

CONFIG_PATH = Path("config.toml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument list.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate source, target, and bilingual SRT subtitles.")
    parser.add_argument("input_path", type=Path, help="Input media file or folder.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Output directory. Defaults to config.toml [output].dir."
    )
    parser.add_argument("--text", action="store_true", help="Write TXT sidecar files outside plain-text mode.")
    parser.add_argument("--source-lang", default="", help="Optional ASR source language. Empty means auto-detect.")
    parser.add_argument(
        "--target-lang", default="zh", help='Translation target language. Use --target-lang "" to disable translation.'
    )
    parser.add_argument(
        "--plain-text", action="store_true", help="Only write raw source-language SRT and TXT from ASR; skip LLM."
    )
    return parser.parse_args(argv)


def require_llm_settings(config: RuntimeConfig) -> tuple[str, str | None, str]:
    """
    Read LLM settings from module constants.

    Returns
    -------
    tuple[str, str | None, str]
        API key, base URL, and model name.

    Raises
    ------
    ValueError
        If API key or model is missing.
    """
    if not config.llm.api_key:
        raise ValueError("missing llm.api_key in config.toml")
    if not config.llm.model:
        raise ValueError("missing llm.model in config.toml")
    return config.llm.api_key, config.llm.base_url, config.llm.model


def needs_llm(args: argparse.Namespace, config: RuntimeConfig) -> bool:
    """
    Return whether this invocation needs an LLM client.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    config : RuntimeConfig
        Runtime configuration.

    Returns
    -------
    bool
        True when translation or subtitle optimization will run.
    """
    return not args.plain_text and (bool(args.target_lang) or config.optimize_subtitles)


def apply_model_cache_dir(config: RuntimeConfig) -> None:
    """
    Apply the requested model cache directory to HuggingFace tooling.

    """
    if config.model_cache_dir is not None:
        os.environ["HF_HOME"] = str(config.model_cache_dir)


def main(argv: list[str] | None = None) -> int:
    """
    Run the CLI.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument list. Uses process arguments when omitted.

    Returns
    -------
    int
        Process exit code.
    """
    try:
        args = parse_args(argv)
        runtime_config = load_runtime_config(CONFIG_PATH)
        apply_model_cache_dir(runtime_config)
        config = CaptionConfig(
            source_language=args.source_lang or None,
            target_language=args.target_lang or None,
            translation_position=runtime_config.translation_position,
            max_chars_per_cue=runtime_config.max_chars_per_cue,
            max_seconds_per_cue=runtime_config.max_seconds_per_cue,
            plain_text=args.plain_text,
            write_text=args.text,
        )
        asr = LocalMlxAsr(model=runtime_config.asr_model, aligner_model=runtime_config.aligner_model)
        translator = None
        if needs_llm(args, runtime_config):
            api_key, base_url, llm_model = require_llm_settings(runtime_config)
            client = create_openai_client(api_key=api_key, base_url=base_url)
            translator = OpenAICompatibleTranslator(
                client=client,
                model=llm_model,
                target_language=args.target_lang or "",
                concurrency=runtime_config.llm.concurrency,
                enable_thinking=runtime_config.llm.enable_thinking,
                reasoning_effort=runtime_config.llm.reasoning_effort,
                optimization_retries=runtime_config.llm.optimization_retries,
                optimization_window_seconds=runtime_config.optimization_window_seconds,
                max_segment_seconds=runtime_config.max_optimized_seconds,
                max_target_chars=runtime_config.max_optimized_target_chars,
                min_segment_seconds=runtime_config.min_optimized_seconds,
                pause_seconds=runtime_config.optimization_pause_seconds,
            )
        outputs = run_pipeline(
            input_path=args.input_path,
            output_dir=args.output or runtime_config.output_dir,
            config=config,
            asr=asr,
            translator=translator,
            optimizer=translator if translator is not None and runtime_config.optimize_subtitles else None,
            save_asr_json=runtime_config.save_asr_json,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"caption: {exc}", file=sys.stderr)
        return 1

    for output in outputs:
        for path in output.written_paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
