# Caption

轻量级本地 ASR 字幕生成工具。输入视频/音频文件或文件夹，程序会用本地 Qwen3-ASR 识别语音并生成字幕；需要时再调用 LLM 做翻译和字幕分句优化。

## 功能

- 支持单个媒体文件或递归处理文件夹。
- 使用 MLX Qwen3-ASR 和 Qwen3 ForcedAligner 生成词级时间戳。
- 支持源语言自动识别，也可以手动指定源语言。
- 支持纯文本模式，只输出 ASR 原始语言 SRT/TXT，不需要 LLM 配置。
- 支持翻译为目标语言，并保留 ASR、raw、final 三类输出。
- ASR 完成后立即写出中间结果，后续 LLM 失败时不会丢掉识别结果。
- 处理文件夹时保留输入文件夹内部的相对层级。

## 安装

建议使用 Python 3.10+ 的独立环境。

```bash
python -m pip install -e ".[dev]"
```

需要本机可用 `ffmpeg`：

```bash
# macOS
brew install ffmpeg
```

## Qwen MLX 模型

本项目使用 MLX 版 Qwen3-ASR 和 Qwen3 ForcedAligner。推荐下载到固定模型目录，例如 `models/asr`，并在 `config.toml` 的 `[asr]` 中填写本地路径。下面命令需要 Hugging Face CLI。

```bash
mkdir -p models/asr
hf download mlx-community/Qwen3-ASR-1.7B-8bit --local-dir models/asr/Qwen3-ASR-1.7B-8bit
hf download mlx-community/Qwen3-ForcedAligner-0.6B-8bit --local-dir models/asr/Qwen3-ForcedAligner-0.6B-8bit
```

## 配置

从模板创建本地配置文件：

```bash
cp config-temp.toml config.toml
```

然后编辑 `config.toml`。`config.toml` 包含 API key 等敏感信息，已被 `.gitignore` 忽略；不要把真实配置提交到公开仓库。

如果只使用 `--plain-text`，可以不填写 `[llm]` 的 API key；翻译或字幕优化需要可用的 LLM 配置。

```toml
[llm]
provider = "openai"  # openai or anthropic
api_key = "..."
base_url = "https://api.deepseek.com"
model = "deepseek-v4-flash"
enable_thinking = true
reasoning_effort = "high"
concurrency = 4
optimization_retries = 3

[asr]
model = "models/asr/Qwen3-ASR-1.7B-8bit"
aligner_model = "models/asr/Qwen3-ForcedAligner-0.6B-8bit"
model_cache_dir = "models/asr"

[output]
dir = "outputs"
save_asr_json = true

[subtitle]
translation_position = "bottom"
max_chars_per_cue = 60
max_seconds_per_cue = 6.0
optimization_window_seconds = 30.0
max_optimized_seconds = 6.0
max_optimized_target_chars = 25
min_optimized_seconds = 2.5
optimization_pause_seconds = 1.0
optimize = true
```

## 使用

生成中文字幕，并让 LLM 优化分句：

```bash
caption input.mp4
```

默认目标语言是中文，由 `config.toml` 中 `subtitle.target_lang` 控制，改成空字符串即可默认关闭翻译。也可以按次用 CLI 覆盖：

```bash
caption input.mp4 --target-lang ""
```

不调用 LLM，只保留 ASR 结果：

```bash
caption input.mp4 --plain-text
```

指定源语言，或改成其他目标语言：

```bash
caption input.mp4 --source-lang en --target-lang ja
```

同时写出 TXT 文件：

```bash
caption input.mp4 --text
```

处理文件夹：

```bash
caption /path/to/media_folder
```

## 输出

输出目录默认为 `outputs/`，按阶段分为 `asr/`、`raw/`、`final/`。默认只写 JSON 和 SRT；传入 `--text` 时才写 TXT。`--plain-text` 模式始终写 ASR 的 SRT 和 TXT。

- `asr/<relative>/<name>.asr.json`：ASR 原始结构化结果。
- `asr/<relative>/<name>.asr.srt`：ASR 阶段直接生成的源语言字幕。
- `asr/<relative>/<name>.asr.txt`：ASR 阶段的源语言纯文本，仅 `--text` 或 `--plain-text` 时生成。
- `raw/<relative>/<name>.raw.source.srt`：LLM 优化前的源语言字幕。
- `raw/<relative>/<name>.raw.target.srt`、`raw/<relative>/<name>.raw.bilingual.srt`：LLM 优化前的翻译字幕。
- `final/<relative>/<name>.source.srt`：最终源语言字幕。
- `final/<relative>/<name>.target.srt`：最终目标语言字幕。默认目标语言是 `zh`；目标语言为空（config 或 `--target-lang ""`）时不生成。
- `final/<relative>/<name>.bilingual.srt`：最终双语字幕。
- `raw/<relative>/*.txt`、`final/<relative>/*.txt`：对应阶段的纯文本文件，仅 `--text` 时生成。

如果输入是单个文件，`<relative>` 为空；如果输入是文件夹，`<relative>` 是该输入文件夹内部的相对路径，不包含输入文件夹自身或它之前的父路径。例如输入 `/data/media`，其中有 `course/week1/clip.mp4`，最终双语字幕会写到 `outputs/final/course/week1/clip.bilingual.srt`。

## 工作流

1. 发现输入媒体文件。
2. 如果 `asr/<relative>/<name>.asr.json` 已存在，直接复用该缓存并跳过语音识别；否则本地 Qwen3-ASR 识别音频。
3. ASR 完成后立即写出 `asr/*.asr.json` 和 `asr/*.asr.srt`，传入 `--text` 或 `--plain-text` 时同时写出 `asr/*.asr.txt`。
4. 如果传入 `--plain-text`，流程到此结束。
5. 默认翻译到中文并保存 `raw/` 字幕；目标语言为空（config 或 `--target-lang ""`）时跳过翻译。
6. 如果 `config.toml` 中 `subtitle.optimize = true`，LLM 会按配置窗口结合时间戳重新做语义分句和字幕优化。
7. 写出 `final/` SRT；传入 `--text` 时同时写出 TXT。

运行时会用 `tqdm` 显示 ASR chunk、翻译 batch、优化窗口等可计数进度；阶段性事件会用简短日志打印。

## 验证

```bash
python -m pytest -q
python -m ruff check .
```
