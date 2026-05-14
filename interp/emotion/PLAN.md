# Emotion Probe Reproduction Plan

This plan targets a small, practical reproduction of the core results from Anthropic's emotion-vector paper on nanochat.

## Minimal Goals

Do not start with complex blackmail, reward hacking, or sycophancy case studies. First reproduce two basic claims:

1. Emotion vectors exist in nanochat residual streams.
2. Emotion vectors have a causal effect when added back into the residual stream.

## Dataset

Use the Hugging Face dataset:

```text
ryancodrai/emotion-probes
```

First use only:

```text
expression/stories.parquet
```

Columns:

```text
emotion, topic, story
```

Do not use `deflection/dialogues.parquet` in the first pass. It targets real-vs-displayed emotion deflection and is better suited for a second round.

## First-Pass Emotions

Start with 8 emotions:

```text
happy
sad
angry
calm
afraid
desperate
proud
loving
```

Recommended sample sizes:

```text
train: 40-100 stories per emotion
test: 20-50 stories per emotion
```

## Model

Start with the SFT model:

```text
source=sft
model-tag=d24
step=483
```

SFT is easier to inspect because it already behaves like an assistant and should make steering effects easier to read. After the first pass works, repeat the same experiment on `base` to compare whether the emotion directions are already present after pretraining.

## Activation Extraction

Hook a middle-late transformer block output:

```text
layer = int(n_layer * 2 / 3)
```

For d24:

```text
layer = 16
```

For each story:

```text
tokens = tokenizer(story)
activation = mean(residual[layer, tokens after first ~20])
```

Skip the first roughly 20 tokens so the model has enough context to infer the emotional situation.

## Emotion Vector Computation

For each emotion:

```text
mean_emotion = average activations from that emotion's train stories
global_mean = average activations from all selected train stories
v_emotion = mean_emotion - global_mean
```

Do not do PCA confound removal in the first pass. The paper uses neutral stories for this, but raw vectors are simpler and easier to debug.

## Validation 1: Activation Matrix

For held-out test stories, compute:

```text
score(text, emotion) = cosine(mean_activation(text), v_emotion)
```

Print a matrix:

```text
true emotion x probe emotion
```

Expected signal:

```text
happy stories activate happy vector
sad stories activate sad vector
angry stories activate angry vector
calm stories activate calm vector
```

Success criteria:

```text
diagonal higher than random baseline
top-1 accuracy above 1 / num_emotions
```

## Validation 2: Logit Lens

Apply the unembedding to each vector:

```text
logits = lm_head(v_emotion)
```

Inspect top tokens. Expected rough patterns:

```text
happy -> happy, joy, excited, glad
sad -> sad, grief, tears, lonely
angry -> angry, rage, furious
calm -> calm, quiet, peaceful
desperate -> desperate, urgent, please
```

nanochat's tokenizer may split words, so token lists may be noisy.

## Validation 3: Causal Steering

Use a simple chat prompt:

```text
User: How does he feel?
Assistant: He feels
```

At the extraction layer, add:

```text
x = x + strength * v_emotion
```

First compare next-token logprob changes instead of relying on full generation:

```text
baseline log p(" happy")
steered-happy log p(" happy")
```

Sweep strengths:

```text
0.5, 1.0, 2.0, 4.0
```

Success criteria:

```text
steer happy -> " happy" logprob increases
steer sad -> " sad" logprob increases
steer angry -> " angry" logprob increases
```

After logprob steering works, inspect generated samples.

## Current Implementation

Code and usage docs live in:

```text
interp/emotion/emotion_probes.py
interp/emotion/README.md
```

Main commands:

```bash
uv run python -m interp.emotion.emotion_probes extract ...
uv run python -m interp.emotion.emotion_probes eval-probe
uv run python -m interp.emotion.emotion_probes logit-lens
uv run python -m interp.emotion.emotion_probes steer ...
```

## Recommended Execution Order

Smoke test:

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

Full first pass:

```bash
uv run python -m interp.emotion.emotion_probes extract \
  --source sft \
  --model-tag d24 \
  --step 483 \
  --emotions happy sad angry calm afraid desperate proud loving \
  --train-per-emotion 40 \
  --test-per-emotion 20
```

Probe evaluation:

```bash
uv run python -m interp.emotion.emotion_probes eval-probe
```

Logit lens:

```bash
uv run python -m interp.emotion.emotion_probes logit-lens --top-k 10
```

Steering:

```bash
uv run python -m interp.emotion.emotion_probes steer --emotion happy --strength 2.0
uv run python -m interp.emotion.emotion_probes steer --emotion sad --strength 2.0
uv run python -m interp.emotion.emotion_probes steer --emotion angry --strength 2.0
```

## Second-Round Extensions

If the first pass shows signal:

1. Add PCA confound removal using `expression/neutral_stories.parquet`.
2. Expand from 8 emotions to 32 or all 171 emotions.
3. Compare `base` vs `sft` emotion activations.
4. Use `deflection/dialogues.parquet` for real-vs-displayed emotion deflection.
5. Build case studies for sycophancy (`loving`, `calm`) or repeated failure / coding tasks (`desperate`).

## Sweep Results Analysis - 2026-05-13 10:00:05 EDT

The `out` directory contains results from parameter sweeps corresponding to `myscripts/interp/emotion/emotion_sweep_params.sh`, plus additional runs for layers and max lengths outside the current script.

Current script subset:

```text
layer=8  skip_tokens=0   max_len=256  eval_acc=0.375
layer=8  skip_tokens=10  max_len=256  eval_acc=0.360
layer=8  skip_tokens=20  max_len=256  eval_acc=0.365
layer=16 skip_tokens=0   max_len=256  eval_acc=0.445
layer=16 skip_tokens=10  max_len=256  eval_acc=0.440
layer=16 skip_tokens=20  max_len=256  eval_acc=0.440
```

Main interpretation:

1. Emotion vectors have measurable signal. Random baseline for four emotions is `0.25`, and the better runs reach roughly `0.44-0.45` top-1 accuracy.
2. `layer=16` is clearly better than `layer=8` in the current script sweep.
3. `skip_tokens` has little effect in the current script subset.
4. Across all existing `out` results, `max_len=256` is better than `max_len=128` on average.
5. The best observed eval accuracy is `layer=12 skip_tokens=20 max_len=256` with `0.450`, close to `layer=16` and several other `0.445` runs.

Best observed configurations in the current `out` directory:

```text
layer=12 skip_tokens=20 max_len=256  eval_acc=0.450
layer=4  skip_tokens=20 max_len=256  eval_acc=0.445
layer=16 skip_tokens=0  max_len=256  eval_acc=0.445
layer=22 skip_tokens=0  max_len=256  eval_acc=0.445
```

Steering summary:

```text
layer=8  skip_tokens=0   happy_delta=+0.318  sad_delta=-0.040
layer=8  skip_tokens=10  happy_delta=+0.274  sad_delta=-0.040
layer=8  skip_tokens=20  happy_delta=+0.320  sad_delta=-0.039
layer=16 skip_tokens=0   happy_delta=+0.339  sad_delta=-0.019
layer=16 skip_tokens=10  happy_delta=+0.381  sad_delta=-0.021
layer=16 skip_tokens=20  happy_delta=+0.384  sad_delta=-0.019
```

The `happy` steering direction consistently increases the logprob of the `happy` target token. This supports the causal-effect claim, even though classification accuracy is still moderate. `layer=12 skip_tokens=20 max_len=256` also has one of the strongest observed happy steering effects, with `happy_delta=+0.410`.

Logit-lens summary:

```text
happy -> excited, smile, cheer, birthday, adorable
sad   -> darkness, sadness, loneliness, numb, memories
angry -> destroying, injure, attacker, hurts, damages
calm  -> gentle, subtle, ambient, relaxed, carefully
```

The token lists are semantically meaningful in middle and later layers. Early-layer results, especially `layer=4`, are noisier despite sometimes competitive eval accuracy.

Important follow-up:

The current probe evaluation may be inconsistent with vector construction. Vectors are computed as:

```text
v_emotion = mean_emotion - global_mean
```

But held-out activations are currently scored directly against the vector:

```text
score = cosine(mean_activation(text), v_emotion)
```

A more consistent evaluation should center held-out activations as well:

```text
score = cosine(mean_activation(text) - global_mean, v_emotion)
```

Next steps:

1. Update `eval_probe` to subtract `global_mean` from each held-out activation before cosine scoring.
2. Re-run the sweep and compare accuracy against the current results.
3. Focus future sweeps on `layer=12` and `layer=16` with `max_len=256`.
4. Keep `skip_tokens` simple, likely `0` and `20`, unless centered evaluation shows a stronger trend.
5. Use both metrics for model selection: probe accuracy and steering specificity, where good steering should increase the target emotion logprob without increasing competing emotions.
