# AGENTS.md

## Project Purpose

This project is a lightweight Python CLI for local ASR subtitle generation and optional LLM subtitle translation/optimization.

Expected flow:

1. Discover media files.
2. Run local MLX Qwen3-ASR with Qwen3 ForcedAligner.
3. Persist ASR artifacts immediately under `outputs/asr/`.
4. Optionally translate with an OpenAI-compatible LLM.
5. Optionally optimize subtitle segmentation using timestamps.
6. Persist pre-optimization LLM outputs under `outputs/raw/`.
7. Write final SRT and TXT outputs under `outputs/final/`.

## Environment

Use Python 3.10+.

Install in editable mode:

```bash
python -m pip install -e ".[dev]"
```

`ffmpeg` must be available on PATH.

## Model Setup

Use MLX models:

- `mlx-community/Qwen3-ASR-1.7B-8bit`
- `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`

Download them with `hf download` or another user-approved model management workflow.

The local model paths are configured in `config.toml` under `[asr]`.

## Configuration

Runtime settings belong in local `config.toml`, not in source code. Create it from `config-temp.toml`.

`config.toml` is ignored by git because it may contain secrets. Do not commit it.

Keep these in `config.toml`:

- LLM API key, base URL, model, thinking settings, concurrency, retry count, request timeout. These are required only when translation or optimization uses the LLM.
- ASR model paths and model cache directory.
- Output directory and ASR JSON persistence.
- Subtitle formatting, optimization settings, and the default target language (`subtitle.target_lang`).

Keep these as CLI flags:

- Input path.
- `--source-lang`
- `--target-lang`
- `--plain-text`
- `--text`

`subtitle.target_lang` in `config.toml` sets the default target language (`zh` when unset). `--target-lang` overrides it per run. An empty string in either place disables translation.

Do not add low-frequency runtime knobs to the CLI unless explicitly requested.

## Code Structure

- `src/caption/cli.py`: argument parsing and object wiring only.
- `src/caption/config.py`: TOML config loading.
- `src/caption/media.py`: media discovery and output path derivation only.
- `src/caption/asr_mlx.py`: MLX Qwen3-ASR adapter only.
- `src/caption/pipeline.py`: stage orchestration and incremental file writes.
- `src/caption/translator.py`: LLM translation/optimization orchestration.
- `src/caption/prompts.py`: prompt builders only.
- `src/caption/llm_json.py`: JSON parsing and validation only.
- `src/caption/progress.py`: tqdm progress bars and concise stage logging only.
- `src/caption/segment.py`: local cue construction from word timestamps.
- `src/caption/srt.py`: SRT rendering only.
- `src/caption/types.py`: dataclasses and shared types.

Keep files focused. Do not create large multi-purpose modules.

## Coding Rules

- Use English for code, identifiers, docstrings, and comments.
- Keep functions small and explicit.
- Avoid nested wrapper layers and hidden control flow.
- Prefer standard library unless a dependency is already justified.
- Fail fast on invalid LLM JSON or invalid timestamps.
- Do not silently fall back from optimized subtitles to raw subtitles.
- Retry optimization according to `llm.optimization_retries`.
- If a stage succeeds, persist its output before starting the next stage.
- When `asr/*.asr.json` already exists for a media file, reuse it and skip ASR. Invalid cache files must raise an error instead of triggering re-transcription.
- Keep output layout centralized in `src/caption/media.py`: ASR artifacts under `asr/`, unoptimized LLM artifacts under `raw/`, final artifacts under `final/`.
- For folder inputs, preserve only the input root's internal relative layout. Never include parent directories before the user-provided input root.
- Write TXT sidecar files only when `--text` is set, except `--plain-text`, which always writes ASR SRT and TXT.
- Use tqdm for countable long-running stages; use concise logs only for stage boundaries.

## Testing

Run before reporting completion:

```bash
python -m pytest -q
python -m ruff check .
```

Tests should be scenario-level, not tiny one-assertion noise. Add focused unit tests when they protect a real invariant:

- ASR result persistence before LLM work, including chunk-based `asr/*.asr.txt` when TXT output is enabled.
- Raw outputs retained under `raw/` before optimization.
- Final outputs written under `final/`, with folder input relative layout preserved.
- `--text` controls TXT outputs outside plain-text mode.
- LLM JSON validation and retry behavior.
- CLI/config boundaries.
- Progress-sensitive long-running flow behavior.

## Demo Command

Use a short sample when validating behavior:

```bash
caption examples/test_5min.mp4
```

Expected outputs go to `outputs/asr/`, `outputs/raw/`, and `outputs/final/`.
