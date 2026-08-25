# AGENTS.md

## Project Purpose

This project is a lightweight Python CLI for local ASR subtitle generation with optional LLM sentence segmentation, translation, and translation review.

Expected flow:

1. Discover media files.
2. Run local MLX Qwen3-ASR with Qwen3 ForcedAligner.
3. Persist ASR artifacts immediately under `outputs/asr/`.
4. Ask an LLM where sentences end in the word stream, then lay each sentence out as display lines.
5. Optionally extract a transcript glossary and translate sentence by sentence.
6. Persist pre-review LLM outputs under `outputs/raw/`.
7. Optionally refine translations through bounded score-and-revise rounds.
8. Write final SRT and TXT outputs under `outputs/final/`.

## Core Design Constraint

Display units and translation units obey unrelated constraints: a display line is bounded by duration and reading width, a translation unit must be a complete semantic span. Never conflate them. The sentence is the translation unit; display lines are derived from it.

The division of labour between program and model is fixed:

- The LLM makes semantic judgements only: where sentences end, what a term should be called, what a line says.
- The program makes every structural decision: which words a sentence owns, how many display lines it needs, where those lines start, and what timestamp each cue carries.

Choose the encoding that carries the least the model can get wrong. Segmentation reports boundaries as word ids rather than reproducing the transcript: repeating four hundred words to convey twenty positions makes every word a chance to invalidate the answer, and speech transcripts invite exactly that, since their stutters and repetitions read as typos a helpful model will quietly clean up. Ids also degrade one at a time, so a bad one merges two sentences instead of failing a batch.

Never split a requirement so that one request must both preserve and improve the same text. Fidelity to the word stream and readable prose are opposite instructions, and a model given both will oscillate. Segmentation therefore touches no text at all, and every correction - spelled-out numbers, misheard names, disfluencies, punctuation - happens in the translation stage, after timestamps are already fixed by word spans.

Consequences to preserve:

- A model response is verified by structure the model did not choose (entry counts, character streams), never by arithmetic the model had to perform.
- A retry is always scoped to one sentence or one batch. Never retry a whole document or window.
- A retry must change the request. Sampling is nearly deterministic, so resending an identical prompt reproduces the identical rejected response; the rejection reason is appended to the prompt on every attempt.
- Never hand the model a task that is unreasonable on its face. A validator that rejects a sane response is a bug in whatever produced the request, not in the model. When a stage repeatedly refuses, inspect the input before strengthening the prompt.
- Every loop has an explicit budget.

The model may advise on structure, as long as the advice cannot break anything. Segmentation reports `line_breaks` alongside `sentence_ends`, and the layout stage decides how many to use and may use none. An id that cannot be honoured is discarded rather than raising, because it changes no timestamp, text, or coverage. Distinguish this from a structural decision: advice is discardable, decisions are not, and only the program makes decisions.

Line placement combines two independent signals. The silence before a word is language-independent evidence, since speakers pause at syntactic boundaries and run compound nouns together; a break hint is the model's grammatical judgement. They add rather than override.

Measure a cut in the unit the constraint is written in. Display limits are stated in characters, so candidates are placed and windowed by character count, never by elapsed time: speech rate varies inside a sentence, and an even split in time is a lopsided split in text. The window is half the headroom between the even split and the character budget, which is what guarantees no line overflows; outside it, closeness to the split outranks every other signal.

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

Defaults have exactly one definition: the packaged `src/caption/defaults.toml`. A user's `config.toml` is overlaid on it section by section, so any key may be omitted. Never write a default into `config.py` as a `.get` fallback, and never let another module hold a default for something the configuration also sets; that reintroduces the drift this file exists to prevent.

A key belongs in `defaults.toml` only when a sensible default exists. Credentials, the chat model name, and the base URL are deliberately absent: omitting them means "not provided" and must fail with a clear local error rather than resolve to a placeholder. `config-temp.toml` is a starting point for users and is never read at runtime.

`[llm]` describes how to reach the model: provider, API key, base URL, model, reasoning settings, concurrency, retry count, request timeout, and the sampling parameters. Nothing about subtitle behaviour belongs there.

Sampling is split into `[llm.thinking]` and `[llm.instruct]` because models publish different recommendations for the two generation modes. The set in use follows the same per-task reasoning switch the stages already choose, so there is one decision, not two. Configuration merging is recursive, so a user may override a single sampling key without restating the rest.

`[subtitle]` describes how subtitles are produced: segmentation mode, target language, display limits, and the review loop. Nothing else. Batch sizes and translation context width are tuning constants in the module that owns them, not configuration; a value a user cannot reason about is not a setting.

`[asr]` holds model paths and the model cache directory. `[output]` holds the output directory and ASR JSON persistence.

Keep these as CLI flags:

- Input path.
- `--source-lang`
- `--target-lang`
- `--plain-text`
- `--text`

Capability switches are layered and each is independently expressible:

- `subtitle.segmentation = "asr"` cuts the word stream by display limits alone and never calls the LLM. It cannot be combined with a translation target; loading such a config must fail.
- `subtitle.segmentation = "llm"` restores sentences with the LLM. This is the prerequisite for translation.
- `subtitle.target_lang` sets the default target language (`zh` when unset); `--target-lang` overrides it per run, and an empty string in either place disables translation.
- `subtitle.review_rounds = 0` disables the review loop. There is no separate boolean switch.

Do not add low-frequency runtime knobs to the CLI unless explicitly requested.

## Code Structure

- `src/caption/cli.py`: argument parsing and object wiring only.
- `src/caption/config.py`: TOML config loading and validation.
- `src/caption/media.py`: media discovery and output path derivation only.
- `src/caption/asr_mlx.py`: MLX Qwen3-ASR adapter only.
- `src/caption/pipeline.py`: stage orchestration and incremental file writes. ASR (producer thread) overlaps the LLM stage (consumer) across files whenever an LLM stage is active.
- `src/caption/llm_client.py`: provider adapters, the retrying JSON task helper, and preflight. No subtitle logic.
- `src/caption/sentence.py`: deterministic word-stream handling: batching, turning reported boundaries into sentences, rendering words as text.
- `src/caption/transcript.py`: the sentence-segmentation and glossary stages.
- `src/caption/translator.py`: sentence translation and the review/revise loop.
- `src/caption/prompts.py`: prompt builders only.
- `src/caption/llm_json.py`: JSON parsing and validation only.
- `src/caption/progress.py`: tqdm progress bars and concise stage logging only.
- `src/caption/segment.py`: deterministic cue construction, both the ASR-only path and sentence layout.
- `src/caption/srt.py`: SRT rendering only.
- `src/caption/types.py`: dataclasses and shared types.

Keep files focused. Do not create large multi-purpose modules.

## Coding Rules

- Use English for code, identifiers, docstrings, and comments.
- Keep functions small and explicit.
- Avoid nested wrapper layers and hidden control flow.
- Prefer standard library unless a dependency is already justified.
- Fail fast on invalid LLM JSON or invalid timestamps.
- Never ask the model to satisfy a numeric or combinatorial constraint that the program can enforce itself.
- Never silently fall back from reviewed subtitles to unreviewed ones.
- Retry a failed LLM task according to `llm.retries`, always at the granularity of a single sentence or batch.
- Request reasoning tokens per task rather than globally: sentence splitting, translation, and scoring run without reasoning; glossary extraction and revision run with it.
- If a stage succeeds, persist its output before starting the next stage.
- When the LLM stage is active, ASR for the next file runs concurrently with the current file's LLM work; ASR artifacts must be persisted before a job enters the LLM stage. Without an LLM stage, files are processed sequentially.
- When `asr/*.asr.json` already exists for a media file, reuse it and skip ASR. Invalid cache files must raise an error instead of triggering re-transcription.
- Keep output layout centralized in `src/caption/media.py`: ASR artifacts under `asr/`, pre-review LLM artifacts under `raw/`, final artifacts under `final/`.
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

- Reported boundaries cover the batch exactly once, and unusable ids degrade instead of failing.
- Line layout honours whichever display limit binds, never asks for more lines than there is text to fill, and reassembles into the original sentence.
- ASR result persistence before LLM work, including chunk-based `asr/*.asr.txt` when TXT output is enabled.
- Raw outputs retained under `raw/` before review, and review output winning in `final/`.
- Final outputs written under `final/`, with folder input relative layout preserved.
- `--text` controls TXT outputs outside plain-text mode.
- LLM JSON validation and retry behavior, including the bounded review loop.
- Per-stage reasoning selection.
- CLI/config boundaries, including the capability layering.
- Progress-sensitive long-running flow behavior.

## Demo Command

Use a short sample when validating behavior:

```bash
caption examples/test_5min.mp4
```

Expected outputs go to `outputs/asr/`, `outputs/raw/`, and `outputs/final/`.
