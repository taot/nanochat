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

