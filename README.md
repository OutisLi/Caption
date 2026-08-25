# Caption

轻量级本地 ASR 字幕生成工具。输入视频/音频文件或文件夹，程序会用本地 Qwen3-ASR 识别语音并生成字幕；需要时再调用 LLM 做断句、翻译和译文精修。

## 功能

- 支持单个媒体文件或递归处理文件夹。
- 使用 MLX Qwen3-ASR 和 Qwen3 ForcedAligner 生成词级时间戳。
- 由 LLM 定位词流中的句子边界，字幕按语义断句而不是按固定长度切割。
- 支持源语言自动识别，也可以手动指定源语言。
- 支持纯文本模式，只输出 ASR 原始语言 SRT/TXT，不需要 LLM 配置。
- 支持翻译为目标语言，并保留 ASR、raw、final 三类输出。
- 译文可经过打分与定向重译的精修循环，轮数有上限。
- ASR 完成后立即写出中间结果，后续 LLM 失败时不会丢掉识别结果。
- 处理文件夹时保留输入文件夹内部的相对层级。
- 默认把完成的字幕嵌入源媒体，写成带语言标签的 MKV；翻译时挂原文、译文、双语三轨，双语为默认轨。

## 设计

字幕有两类互不相干的约束：显示单元受物理约束（时长、每行字数），翻译单元受语义约束（必须是完整意群）。把两者绑在一起，就会出现「翻译一个从句子中间切出来的片段」这种无解任务。

另一条同样重要的原则：**不要在同一个请求里既要求忠实于原词、又要求文本更通顺**。这两个指令方向相反，模型必然摇摆。所以断句阶段完全不碰文本，一切修正（数字、专名、口水词、标点）都留到翻译阶段——那时时间戳已由词区间锁定，改写无从破坏对齐。

因此流程把它们拆开，并让 LLM 只做语义判断、程序只做结构决策：

1. **断句**（LLM）：给模型编号的词流，让它报告每句最后一个词的编号，以及句内可以断行的词编号。它不复述任何文本，因此无从增删词；无法采纳的编号被直接丢弃，只是两句合并，不影响任何时间戳或覆盖。
2. **排版**（程序）：按时长和字数上限把每个句子切成显示行，时间戳直接取自词级时间戳。断点位置综合词间停顿和模型给的断行编号，并限制在均分点附近的窗口内，保证两行不会严重失衡。
3. **术语表**（LLM）：抽取全文主题和专名/术语的统一译法，作为并行翻译之间唯一的共享状态。
4. **翻译**（LLM）：逐句翻译，输入包含主题、术语表、前后邻句，以及程序已经切好的显示行。行数固定，模型只填文本。
5. **精修**（LLM，可选）：批量打分，只对被判定不合格的句子带着评分和原因重译，轮数有上限。

结构决策全部由程序完成，模型输出的结构错误无从产生，因此不存在「校验失败 → 整体重试」的死循环。模型可以对结构提**建议**（断行编号），但建议是可丢弃的：无法采纳的编号会被忽略而不是导致失败，因为它不改变任何时间戳、文本或覆盖关系。

词间停顿是排版的关键信号且免费：说话人在句法边界会停顿，而复合名词（如 `scaling laws`）是连读的，停顿恰好为零。ASR 错误的修正（例如把念出来的数字写成数字）放在第 4 步，此时时间戳已经锁定，改写不影响对齐。

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

`config.toml` 里省略的键会回退到随包分发的 `src/caption/defaults.toml`，可以只写要改的项，也可以整段省略。想改默认值就改那个文件，它是默认值的唯一定义处，`config-temp.toml` 只是模板，运行时不读取。

`api_key`、`model`、`base_url` 和 `[asr]` 的模型路径**没有**默认值，它们不在 `defaults.toml` 里：省略等于「未提供」，需要时会在本地直接报错，而不是拿一个占位符去请求。

`[llm]` 只描述如何连接模型，`[subtitle]` 描述字幕如何产出。使用 `--plain-text` 或 `segmentation = "asr"` 时可以不填写 API key，其余情况都需要可用的 LLM 配置。下面是完整形态，其中除 `api_key`、`model`、`base_url` 和 `[asr]` 路径外，每一项的值都等于内置默认值，可以删掉：

```toml
[llm]
provider = "openai"  # openai or anthropic
api_key = "..."
base_url = "https://api.deepseek.com"
model = "deepseek-v4-flash"
concurrency = 4
retries = 3
request_timeout = 30
enable_thinking = true
reasoning_effort = "high"

# 采样参数按生成模式分两组，用哪一组由每个阶段是否使用推理决定。
[llm.thinking]
temperature = 1.0
top_p = 0.95
top_k = 20
min_p = 0.0
presence_penalty = 0.0
repetition_penalty = 1.0

[llm.instruct]
temperature = 0.7
top_p = 0.80
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0

[asr]
model = "models/asr/Qwen3-ASR-1.7B-8bit"
aligner_model = "models/asr/Qwen3-ForcedAligner-0.6B-8bit"
model_cache_dir = "models/asr"

[output]
dir = "outputs"
save_asr_json = true
embed = true

[subtitle]
segmentation = "llm"  # "llm" 由模型断句，"asr" 只按显示上限机械切分
target_lang = "zh"
translation_position = "bottom"
max_seconds_per_cue = 6.0
max_chars_per_cue = 60
review_rounds = 2
review_pass_score = 4
```

各能力独立开关，逐层依赖：

| 配置 | 产物 |
| --- | --- |
| `segmentation = "asr"` | 机械切分的原文字幕，完全不调用 LLM |
| `segmentation = "llm"`，`target_lang = ""` | 按句子断开、带标点的原文字幕 |
| 再设 `target_lang = "zh"` | 双语字幕 |
| 再设 `review_rounds > 0` | 经过打分与重译精修的译文 |
| `output.embed = true`（默认） | `mkv/` 下的 MKV，翻译时挂原文、译文、双语三轨，双语为默认轨 |

`max_chars_per_cue` 按源语言计算，和 `max_seconds_per_cue` 一起决定每句切成几行；译文按各行的信息量对齐，不单独设字数。批次大小、翻译上下文宽度等实现细节不开放为配置项，它们是源码中的常量。

`enable_thinking` 只是允许推理，具体是否使用由阶段决定：断句、翻译和打分不使用推理，术语抽取和重译使用。采样参数跟着这个开关走——`[llm.thinking]` 用于后者，`[llm.instruct]` 用于前者，默认值取自 Qwen3 的官方推荐。`top_k`、`min_p`、`repetition_penalty` 不是 OpenAI 标准参数，通过 `extra_body` 发送，不支持的服务会忽略它们。

`[llm.instruct]` 的 `presence_penalty = 1.5` 值得留意：这是厂商针对散文生成的推荐值，而本项目每个请求都要求 JSON 输出，键名必然重复，高 presence penalty 恰好抑制这种重复。如果出现 JSON 解析失败，优先调低它。

`segmentation = "asr"` 与非空 `target_lang` 不能并存，因为按固定长度切出的片段无法翻译。

## 使用

生成中文双语字幕，并让 LLM 精修译文：

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

输出目录默认为 `outputs/`，按阶段分为 `asr/`、`srt/raw/`、`srt/final/`、`mkv/`。默认只写 JSON 和 SRT；传入 `--text` 时才写 TXT。`--plain-text` 模式始终写 ASR 的 SRT 和 TXT。

- `asr/<relative>/<name>.asr.json`：ASR 原始结构化结果。
- `asr/<relative>/<name>.asr.srt`：ASR 阶段直接生成的源语言字幕。
- `asr/<relative>/<name>.asr.txt`：ASR 阶段的源语言纯文本，仅 `--text` 或 `--plain-text` 时生成。
- `srt/raw/<relative>/<name>.raw.source.srt`：精修前的源语言字幕，仅 `review_rounds > 0` 时生成。
- `srt/raw/<relative>/<name>.raw.target.srt`、`srt/raw/<relative>/<name>.raw.bilingual.srt`：精修前的翻译字幕。
- `srt/final/<relative>/<name>_<lang>.srt`：单语字幕，后缀为源语言或目标语言代码，例如 `_en`、`_zh`。
- `srt/final/<relative>/<name>.srt`：双语字幕，无语言后缀。目标语言为空时不生成译文和双语文件。
- `mkv/<relative>/<name>.mkv`：源媒体加上嵌入字幕。翻译时含原文、译文、双语三轨，双语为默认轨；语言标签按源语言和 `target_lang` 写入。`output.embed = false` 时不生成。
- `srt/raw/<relative>/*.txt`、`srt/final/<relative>/*.txt`：对应阶段的纯文本文件，仅 `--text` 时生成。

如果输入是单个文件，`<relative>` 为空；如果输入是文件夹，`<relative>` 是该输入文件夹内部的相对路径，不包含输入文件夹自身或它之前的父路径。例如输入 `/data/media`，其中有 `course/week1/clip.mp4`，最终双语字幕会写到 `outputs/srt/final/course/week1/clip.srt`。

## 工作流

1. 发现输入媒体文件。
2. 如果 `asr/<relative>/<name>.asr.json` 已存在，直接复用该缓存并跳过语音识别；否则本地 Qwen3-ASR 识别音频。
3. ASR 完成后立即写出 `asr/*.asr.json` 和 `asr/*.asr.srt`，传入 `--text` 或 `--plain-text` 时同时写出 `asr/*.asr.txt`。
4. 如果传入 `--plain-text`，流程到此结束。
5. 如果 `srt/final/` 里已有该次运行需要的成品字幕，直接复用并跳过 LLM；缺一份或文件为空则报错，不会重跑翻译。
6. 否则由 LLM 报告词流中的句子边界，程序按显示上限把每个句子排版成显示行。`segmentation = "asr"` 时跳过这一步，直接机械切分。
7. 目标语言非空时，抽取术语表并逐句翻译，然后保存 `srt/raw/` 字幕。
8. `review_rounds > 0` 时对译文批量打分，只重译被判定不合格的句子，达到轮数上限或全部通过即停止。
9. 写出 `srt/final/` SRT；传入 `--text` 时同时写出 TXT。
10. `output.embed` 为真时，把完成的字幕轨 mux 进源媒体，写成 `mkv/<relative>/<name>.mkv`。

运行时会用 `tqdm` 显示 ASR chunk、断句 batch、逐句翻译、打分 batch、重译等可计数进度；阶段性事件会用简短日志打印。处理多个文件且启用 LLM 阶段时，下一个文件的语音识别会与当前文件的 LLM 阶段并行执行，ASR 结果落盘后才进入 LLM 阶段。

## 验证

```bash
python -m pytest -q
python -m ruff check .
```
