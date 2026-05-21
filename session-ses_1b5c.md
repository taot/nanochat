# 列出 emotion 实验相关文件

**Session ID:** ses_1b5c9b44bffe1QTOq2KSbuGH9v
**Created:** 5/21/2026, 7:05:32 AM
**Updated:** 5/21/2026, 7:24:13 AM

---

## User

@interp/emotion 是我添加的实现一点点简化版的 Anthropic 的 emotion paper 的实验。对比这个 branch 和 master branch 你应该能找到更多相关的脚本和 README 等文件。先帮我列出来相关文件

---

## Assistant

我会只做只读检查：先看当前分支相对 `master` 的文件差异，再结合 `interp/emotion`、README 和脚本内容筛出和 emotion 实验相关的文件。

**Tool: bash**

**Tool: bash**

**Tool: glob**

---

## Assistant

当前分支是 `emotion`，相对 `master` 新增的核心文件集中在 `interp/emotion` 和 `myscripts/interp/emotion`，另有依赖/忽略规则更新。我再搜索仓库里引用 `emotion` 或这些脚本的地方，避免漏掉相关文件。

**Tool: grep**

**Tool: bash**

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: bash**

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

当前分支：`emotion`  
相对 `master` 发现的相关文件如下：

**核心实现**
- `interp/emotion/emotion_probes.py`
  - emotion vector/probe 实验主脚本
  - 包含 `extract`、`eval-probe`、`logit-lens`、`steer`
  - 使用 `ryancodrai/emotion-probes` 数据集
  - 输出默认在 `out/emotion_probes`

**文档**
- `interp/emotion/README.md`
  - 实验说明、setup、smoke test、full run、Chat UI 运行方式
  - 解释 extract/eval/logit-lens/steer 做什么

**Chat UI**
- `interp/emotion/chatui/server.py`
  - FastAPI 服务
  - 支持 `llm` backend 和 `nanochat` backend
  - `nanochat` backend 使用 probes 做情绪检测和 activation steering
  - `llm` backend 使用 OpenRouter/Claude
- `interp/emotion/chatui/index.html`
  - 浏览器端 React UI
  - 调 `/api/config`、`/api/detect`、`/api/chat`
  - 支持选择回复情绪、strength、assistant prefix

**批量实验脚本**
- `myscripts/interp/emotion/emotion_sweep_params.sh`
  - 批量 sweep `layer`、`skip_tokens`、`max_len`
  - 支持 `extract` 和 `analyze`
  - analyze 会跑 `eval-probe`、`steer`、`logit-lens`
  - 输出日志到 `out/results*_eval.log`、`out/results*_steer.log`、`out/results*_logit_lens.log`

**项目配置/依赖**
- `pyproject.toml`
  - 新增依赖：`openai>=1.0.0`
  - 主要用于 Chat UI 的 OpenRouter LLM backend
- `uv.lock`
  - 依赖锁文件更新
- `.gitignore`
  - 新增忽略 `/out/`
  - 对应 probes、logs、实验输出目录

**本地生成文件，不属于 git diff**
- `interp/emotion/__pycache__/emotion_probes.cpython-310.pyc`
- `interp/emotion/__pycache__/emotion_probes.cpython-314.pyc`
- `interp/emotion/chatui/__pycache__/server.cpython-310.pyc`

这些 `__pycache__` 是运行 Python 后生成的缓存文件，不是这次相对 `master` 的代码改动核心。

---

## User

帮我看看 @interp/emotion/chatui/server.py 里有所有可能作为配置的选项（包括现在从命令行传入的参数，界面设置的参数，nanochat 本身支持的参数），给我列出来

---

## Assistant

我会继续只读检查 `index.html` 里的界面状态/API 参数，以及 `nanochat.engine`、checkpoint 加载相关函数的可配置项，最后按来源分类列出来。

**Tool: grep**

**Tool: grep**

**Tool: grep**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: glob**

**Tool: grep**

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: grep**

---

## Assistant

**Tool: read**

**Tool: grep**

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

下面按“已经可配置 / 代码里硬编码但明显可以配置 / nanochat 底层支持”分组列。

**命令行参数**
来自 `interp/emotion/chatui/server.py:350-366`：

- `--backend`
  - 可选：`llm`, `nanochat`
  - 默认：`llm`
  - 决定使用 OpenRouter LLM 还是本地 nanochat probes。
- `--host`
  - 默认：`127.0.0.1`
  - 传给 `uvicorn.run`。
- `--port`
  - 默认：`8001`
  - 传给 `uvicorn.run`。

nanochat-only：

- `--vectors`
  - 默认：`out/emotion_probes_layer_12_skiptokens_20_maxlen_256/vectors.pt`
  - probe 文件路径，里面还带有 `layer`、`max_len`、`skip_tokens`、`global_mean`、`emotions`。
- `--source`
  - 可选：`base`, `sft`, `rl`
  - 默认：`sft`
  - 传给 `load_model`，决定 checkpoint 目录。
- `--model-tag`
  - 默认：`None`
  - 如果不传，nanochat 会自动猜最大模型 tag。
- `--step`
  - 默认：`None`
  - 如果不传，nanochat 会自动找最后一个 checkpoint step。
- `--strength`
  - 默认：`2.0`
  - activation steering 强度。
  - 也会通过 `/api/config` 发给前端作为 UI 初始值。
- `--device-type`
  - 可选：`cuda`, `cpu`, `mps`
  - 默认：`None`
  - 如果不传，调用 `autodetect_device_type()`。
- `--gen-mode`
  - 可选：`engine`, `model`
  - 默认：`engine`
  - 目前两个分支都实际调用 `Engine.generate_batch()`；`model` 分支里真正的 `generate_steered/GPT.generate` 调用是注释掉的。
- `--gen-steps`
  - 默认：`512`
  - 当前只保存到 `self.gen_steps` 并打印；实际 active generation 里没有用它，两个分支都硬编码 `max_tokens=512`。

**环境变量**
- `OPENROUTER_API_KEY`
  - `llm` backend 必需。
- `NANOCHAT_BASE_DIR`
  - nanochat checkpoint/cache 根目录。
  - 不设时默认 `~/.cache/nanochat`。
- `NANOCHAT_DTYPE`
  - nanochat compute dtype。
  - 可选：`bfloat16`, `float16`, `float32`。
- `RANK`, `LOCAL_RANK`, `WORLD_SIZE`
  - nanochat DDP 检测用。
  - 这个 server 场景一般不会用，但底层 `compute_init()` 支持。

**前端 UI 可调参数**
来自 `interp/emotion/chatui/index.html`：

- `replyEmotion`
  - UI 下拉框。
  - 可选：`None` 加上 server 返回的 `EMOTIONS`。
  - 发送到 `/api/chat` 的 `replyEmotion`。
- `strength`
  - UI 数字输入框。
  - 只在 `backend !== "llm"` 时显示。
  - `step=0.1`, `min=0`。
  - 发送到 `/api/chat` 的 `strength`，覆盖 server 启动时的默认 `--strength`。
- `assistantPrefix`
  - UI 文本框。
  - 发送到 `/api/chat` 的 `assistantPrefix`。
  - 会拼进 assistant 开头。
- `input`
  - 当前用户输入。
  - `/api/detect` 用它做情绪检测，`/api/chat` 用它构造 messages。
- `messages`
  - 前端维护的完整聊天历史。
  - 发给 `/api/chat`。
- `Detect` / `Send`
  - 不是参数，但决定调用 `/api/detect` 或 `/api/chat`。

**API 请求参数**
`/api/detect`：

- `text`
  - 待检测文本。

`/api/chat`：

- `messages`
  - `list[{role, content}]`
  - `role` 目前前端用 `user` / `assistant`。
- `replyEmotion`
  - UI 选择的回复情绪。
- `strength`
  - 可空。
  - nanochat backend 中空值会 fallback 到 server 的 `self.strength`。
- `assistantPrefix`
  - 可空。

`/api/config` 返回给前端：

- `emotions`
  - 来自 server 里的 `EMOTIONS`。
- `backend`
  - `llm` 或 `nanochat`。
- `strength`
  - nanochat backend 返回当前 `self.strength`，llm backend 固定返回 `2.0`。

**server.py 里硬编码但明显可以作为配置的项**
- `EMOTIONS`
  - 当前：`["Happy", "Sad", "Angry", "Calm"]`
  - 影响 UI 选项、LLM 分类标签、nanochat detect 输出。
- `DETECT_MODEL`
  - 当前：`anthropic/claude-haiku-4-5`
  - OpenRouter 情绪检测模型。
- `REPLY_MODEL`
  - 当前：`anthropic/claude-sonnet-4-5`
  - OpenRouter 回复模型。
- OpenRouter `base_url`
  - 当前：`https://openrouter.ai/api/v1`
- LLM detect `max_tokens`
  - 当前：`80`
- LLM reply `max_tokens`
  - 当前：`512`
- LLM reply style prompt
  - 当前包括：
  - `Reply in a {emotion} tone.`
  - `Keep replies to 1–3 sentences unless the user asks for more.`
- nanochat detect softmax temperature
  - 当前：`0.1`
  - 在 `F.softmax(logits / 0.1)`。
- steering positions
  - 当前硬编码：`positions="all"`
  - 底层 `steer_layer` 其实支持 `all` / `last`。
- nanochat generation `max_tokens`
  - 当前两个 active 分支都硬编码：`512`
- nanochat generation `temperature`
  - `gen_mode == "model"` 分支当前 active 代码：`2.0`
  - `gen_mode == "engine"` 分支：`0.8`
- nanochat generation `top_k`
  - 当前两个 active 分支都硬编码：`50`
- nanochat generation `num_samples`
  - 当前硬编码：`1`
- nanochat generation `seed`
  - 没传，底层默认 `42`。
- nanochat steering vector 选择
  - 当前代码没有真正用 `replyEmotion`。
  - `vector = self.vectors.get("happy")`
  - `vector2 = self.vectors.get("sad")`
  - `gen_mode == "model"` 分支 steer 的是 `happy - sad`
  - `engine` 分支 steer 的是 `happy`
- prompt rendering special tokens
  - 当前硬编码使用 `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`。
- lazy backend behavior
  - 用 `uvicorn interp.emotion.chatui.server:app` 启动时，没有 CLI 参数，会 lazy 初始化 `LLMBackend()`。

**probe 文件内隐含配置**
这些不是命令行传给 chatui 的，但 `vectors.pt` 里会读出来并控制 nanochat backend 行为：

- `layer`
  - steering/detection hook 的 transformer layer。
- `max_len`
  - detect 时 `story_activation` 截断长度。
- `skip_tokens`
  - detect 时 activation averaging 跳过前多少 tokens。
- `global_mean`
  - detect 时中心化用。
- `vectors`
  - 每个 emotion 的 steering/probe vector。
- `emotions`
  - probe 文件里的 lowercase emotion keys。

**nanochat 底层 generation 支持的参数**
`Engine.generate()` / `Engine.generate_batch()` 支持：

- `num_samples`
  - 默认：`1`
- `max_tokens`
  - 默认：`None`
  - chatui 当前硬编码传 `512`。
- `temperature`
  - 默认：`1.0`
  - `0.0` 表示 greedy。
- `top_k`
  - 默认：`None`
- `seed`
  - 默认：`42`

`GPT.generate()` 支持：

- `max_tokens`
- `temperature`
- `top_k`
- `seed`

但当前 `server.py` active 路径没有直接用 `GPT.generate()`，只是在注释里的 `generate_steered()` 曾经会用它。

---

## User

帮我看看这些参数哪些合适放在 UI 界面，哪些适合放在命令行参数

---

## Assistant

建议按“会不会影响服务启动/模型加载/资源占用”和“是否需要实验时频繁调”来分。

**适合放 UI**
这些是用户在同一个 server 会话里经常试的，不应该重启服务：

- `replyEmotion`
  - 已在 UI。
  - 应该真正接到 nanochat vector 选择，而不是现在硬编码 `happy`。
- `strength`
  - 已在 UI。
  - 很适合 UI，因为需要交互式调 steering 强度。
- `assistantPrefix`
  - 已在 UI。
  - 适合快速测试不同生成开头。
- generation `temperature`
  - 建议放 UI。
  - 对回复质量影响很直观，用户会频繁调。
- generation `top_k`
  - 建议放 UI，但可以折叠到 Advanced。
- generation `max_tokens`
  - 建议放 UI，但可以叫 “Max reply tokens”。
  - 默认从 CLI/server config 来，UI 可覆盖。
- `positions`
  - 建议放 UI Advanced。
  - 可选 `all` / `last`，对 steering 机制影响大，适合实验比较。
- emotion detection softmax temperature
  - 可放 UI Advanced，或者先不暴露。
  - 它影响 detect 分布锐度，不影响模型加载。
- `show confidence` / `show distribution`
  - 纯 UI 选项，适合 UI。
- current `EMOTIONS` 的“reply choice”
  - UI 应展示，但 emotion 列表本身不建议在 UI 编辑。

**适合放命令行参数**
这些影响服务级配置、模型加载、外部依赖、checkpoint 或资源占用，适合启动时固定：

- `--backend`
  - `llm` / `nanochat`
  - 切 backend 通常需要不同初始化。
- `--host`
- `--port`
- `--vectors`
  - probe 文件路径，影响 layer/max_len/skip_tokens/vectors/global_mean。
- `--source`
  - `base` / `sft` / `rl`
- `--model-tag`
- `--step`
- `--device-type`
  - `cuda` / `cpu` / `mps`
- `EMOTIONS`
  - 建议变成 CLI 参数，比如 `--emotions Happy Sad Angry Calm`。
  - 原因：LLM backend 和 nanochat backend 都依赖这个列表，nanochat 还必须和 vectors keys 对齐。
- `DETECT_MODEL`
  - 建议 CLI：`--detect-model`
- `REPLY_MODEL`
  - 建议 CLI：`--reply-model`
- OpenRouter `base_url`
  - 建议 CLI 或环境变量。
- LLM detect `max_tokens`
  - 命令行即可，一般不用频繁调。
- LLM reply `max_tokens`
  - 如果只针对 LLM backend，可以放 CLI；如果希望前端控制回复长度，也可和 nanochat `max_tokens` 一起放 UI。
- default generation 参数
  - `--temperature`
  - `--top-k`
  - `--max-tokens`
  - `--seed`
  - 作为 server 默认值，UI 可以覆盖其中一部分。
- `--gen-mode`
  - 适合 CLI。
  - 这是实现路径选择，不是普通用户交互选项。
- `--num-samples`
  - 如果 UI 只显示一个回复，保持 CLI 或不暴露。
- debug logging 开关
  - 建议 CLI：`--debug`，替代现在大量 `print`。

**不建议直接暴露到 UI**
这些容易让界面复杂，或修改后需要重新 extract/reload：

- `layer`
  - 当前来自 `vectors.pt`。
  - 如果要改 layer，应重新选择另一个 vectors 文件，而不是 UI 里单独改。
- `max_len`
  - 当前来自 `vectors.pt`，detect activation 用。
- `skip_tokens`
  - 当前来自 `vectors.pt`。
- `global_mean`
- raw `vectors`
- `source/model-tag/step/device-type`
  - 都需要重载模型，不适合 UI。
- `NANOCHAT_DTYPE`
- `NANOCHAT_BASE_DIR`
- DDP 环境变量
- tokenizer special tokens
  - 不建议配置，除非在做 tokenizer/chat-template 实验。

**我建议的最终分层**
CLI/server config：

```text
backend, host, port
vectors
source, model-tag, step, device-type
emotions
detect-model, reply-model, openrouter-base-url
default-strength
default-temperature, default-top-k, default-max-tokens, seed
gen-mode
positions default
debug
```

UI/runtime config：

```text
reply emotion
steering strength
assistant prefix
temperature
top_k
max reply tokens
positions: all / last
optional: detection sharpness
```

**优先级建议**
第一批最值得改：

- 把 `replyEmotion` 真正接到 nanochat vector。
- 把 `max_tokens`、`temperature`、`top_k` 从硬编码变成配置。
- `strength` 保持 UI 覆盖 CLI 默认。
- `EMOTIONS`、`DETECT_MODEL`、`REPLY_MODEL` 变 CLI。
- `positions` 加到 UI Advanced 或 CLI 默认 + UI 覆盖。

第二批再考虑：

- detection softmax temperature。
- seed。
- OpenRouter base URL。
- debug 开关和日志清理。

---

