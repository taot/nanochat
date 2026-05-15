# Emotion Probes

Small interpretability experiments for reproducing a subset of Anthropic's emotion-vector results on nanochat.

The script uses `ryancodrai/emotion-probes` from Hugging Face, especially `expression/stories.parquet`, to extract residual-stream directions for a small set of emotions and test whether those directions activate on held-out stories or causally steer next-token probabilities.

## Setup

Use the repo environment:

```bash
uv run python -m interp.emotion.emotion_probes --help
```

The default model source is `sft`. If your checkpoint is the local d24 SFT checkpoint, pass:

```bash
--source sft --model-tag d24 --step 483
```

If your nanochat cache lives somewhere non-default, set:

```bash
export NANOCHAT_BASE_DIR=/path/to/nanochat-cache
```

## Quick Smoke Test

Use CPU if the available GPU cannot fit the d24 checkpoint:

```bash
uv run python -m interp.emotion.emotion_probes extract \
  --source sft \
  --model-tag d24 \
  --step 483 \
  --device-type cpu \
  --emotions happy sad \
  --seed 42 \
  --train-per-emotion 1 \
  --test-per-emotion 1 \
  --out-dir out/emotion_probes_smoke
```

Then run:

```bash
uv run python -m interp.emotion.emotion_probes eval-probe \
  --vectors out/emotion_probes_smoke/vectors.pt \
  --out-dir out/emotion_probes_smoke

uv run python -m interp.emotion.emotion_probes logit-lens \
  --source sft \
  --model-tag d24 \
  --step 483 \
  --device-type cpu \
  --vectors out/emotion_probes_smoke/vectors.pt \
  --top-k 5

uv run python -m interp.emotion.emotion_probes steer \
  --source sft \
  --model-tag d24 \
  --step 483 \
  --device-type cpu \
  --vectors out/emotion_probes_smoke/vectors.pt \
  --emotion happy \
  --targets happy sad \
  --strength 0.1
```

## Full First Pass

Recommended first real run, preferably on a GPU with enough memory:

```bash
uv run python -m interp.emotion.emotion_probes extract \
  --source sft \
  --model-tag d24 \
  --step 483 \
  --emotions happy sad angry calm afraid desperate proud loving \
  --seed 42 \
  --train-per-emotion 40 \
  --test-per-emotion 20
```

This writes:

```text
out/emotion_probes/vectors.pt
```

Evaluate held-out story projections:

```bash
uv run python -m interp.emotion.emotion_probes eval-probe
```

To center held-out activations before scoring (subtracts `global_mean`, consistent with how emotion vectors are constructed):

```bash
uv run python -m interp.emotion.emotion_probes eval-probe --center-act
```

Inspect unembed/logit-lens tokens for each direction:

```bash
uv run python -m interp.emotion.emotion_probes logit-lens --top-k 10
```

Measure causal steering on the prompt `How does he feel?` with assistant prefix `He feels`:

```bash
uv run python -m interp.emotion.emotion_probes steer --emotion happy --strength 2.0
uv run python -m interp.emotion.emotion_probes steer --emotion sad --strength 2.0
uv run python -m interp.emotion.emotion_probes steer --emotion angry --strength 2.0
```

## Chat UI

A small FastAPI app under `interp/emotion/chatui/` serves a browser-based chat that does live emotion detection on user messages and lets you pick the emotional tone of each reply. Two backends are available behind a single `--backend` flag:

- `llm` — uses OpenRouter-hosted Claude models (Haiku for classification, Sonnet for replies). Requires `OPENROUTER_API_KEY`. This is the default.
- `nanochat` — uses the local nanochat checkpoint plus the trained probes from `extract` above: cosine similarity against emotion vectors for `/api/detect`, and activation steering at the probe's layer for `/api/chat`.

### Run the LLM backend

```bash
export OPENROUTER_API_KEY=sk-...
uv run python -m interp.emotion.chatui.server --backend llm --port 8001
```

The old bare-uvicorn launch also still works and defaults to the LLM backend:

```bash
uv run uvicorn interp.emotion.chatui.server:app --port 8001
```

### Run the nanochat backend

First make sure you have a trained probe file (see [Full First Pass](#full-first-pass)). Then:

```bash
uv run python -m interp.emotion.chatui.server \
  --backend nanochat \
  --vectors out/emotion_probes_layer_12_skiptokens_0_maxlen_128/vectors.pt \
  --source sft --model-tag d24 --step 483 \
  --strength 2.0 \
  --port 8001
```

Then open `http://127.0.0.1:8001` in a browser. `OPENROUTER_API_KEY` is not required for the nanochat backend.

Flags:

- `--backend {llm, nanochat}`: which backend to serve from. Default `llm`.
- `--vectors PATH`: probe `vectors.pt` to load (nanochat backend only).
- `--source {base, sft, rl}`, `--model-tag`, `--step`: which nanochat checkpoint to load.
- `--strength FLOAT`: activation steering strength for the reply backend. Larger values push replies harder toward the chosen emotion at the cost of fluency.
- `--host`, `--port`: bind address.

The set of emotions shown in the UI is controlled by the `EMOTIONS` list at the top of `interp/emotion/chatui/server.py`. For the nanochat backend the chosen emotions must also exist as keys in the loaded `vectors.pt`.

## What The Script Does

`extract`:

- Loads `expression/stories.parquet` from `ryancodrai/emotion-probes`.
- Samples train/test stories for selected emotions.
- Hooks `model.transformer.h[layer]`, defaulting to `int(n_layer * 2 / 3)`.
- Averages residual activations after `--skip-tokens` tokens.
- Computes `v_emotion = mean(emotion) - mean(all selected emotions)`.
- Saves vectors and held-out activations to `vectors.pt`.

`eval-probe`:

- Computes cosine projection of held-out story activations onto each emotion vector.
- Prints a `true emotion x probe emotion` matrix and top-1 accuracy.
- `--center-act`: subtract `global_mean` from each held-out activation before scoring, matching the centered space in which vectors were computed.

`logit-lens`:

- Applies `model.lm_head` to each emotion vector.
- Prints top and bottom tokens.

`steer`:

- Builds a chat prompt.
- Adds `strength * v_emotion` at the extraction layer through a forward hook.
- Compares baseline vs steered next-token logprobs for target emotion words.

## Notes

- This is intentionally a minimal first pass. It does not yet implement neutral-story PCA confound removal.
- Very small smoke-test sample sizes are only for checking the code path. Use at least tens of stories per emotion for meaningful results.
- On this machine, CUDA auto-detection can fail for d24 because the available GPU is too small. Use `--device-type cpu` for correctness checks, or run on a larger GPU for real experiments.
