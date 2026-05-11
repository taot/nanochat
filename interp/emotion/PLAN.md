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
