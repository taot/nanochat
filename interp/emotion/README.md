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
