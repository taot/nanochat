# Emotion Chat UI 2 Server Design

## Goals

`chatui2` server provides a pure FastAPI API backend for emotion detection and steered reply generation.

The backend has two responsibilities:

1. Detect the emotion distribution of user text.
2. Generate assistant replies using request-scoped steering and generation parameters.

The backend does not serve client assets. API clients are responsible for their own hosting, state management, and connection configuration.

Clients own state for steering controls and generation parameters. The backend does not persist or manage those settings; it only receives them in `/api/chat` and applies them for that request.

## Backend Modes

The server supports two backend implementations behind the same HTTP API.

| Backend | Purpose |
| --- | --- |
| `llm` | Uses OpenRouter-hosted chat models for emotion detection and reply generation. |
| `nanochat` | Uses the local nanochat model, emotion probe vectors, and activation steering. |

The active backend is selected at server startup with `--backend`.

## API Boundary

The server only owns `/api/*` routes. It should not define `/`, asset routes, or fallback routes for client applications.

The API is designed around three client-facing operations:

- read backend configuration.
- detect emotion for a text input.
- generate a chat reply from messages and per-request generation controls.

If browser clients call the API from a different origin, the server may need configurable CORS. CORS should be opt-in through CLI or environment configuration rather than hard-coded to a specific client origin.

## API

### `GET /api/config`

Returns static backend configuration needed by API clients.

Response:

```json
{
    "emotions": ["happy", "sad", "angry", "calm"],
    "backend": "nanochat"
}
```

Fields:

| Field | Description |
| --- | --- |
| `emotions` | Canonical lowercase emotion labels and valid steering emotion names. |
| `backend` | Active backend name: `llm` or `nanochat`. |

### `POST /api/detect`

Detects the emotional tone of one text input.

Request:

```json
{
    "text": "I'm frustrated about a deadline."
}
```

Response:

```json
{
    "emotion": "angry",
    "confidence": 0.72,
    "scores": {
        "happy": 0.04,
        "sad": 0.13,
        "angry": 0.72,
        "calm": 0.11
    }
}
```

Validation:

- `text` must be non-empty after trimming.
- `scores` must contain all configured emotions.
- `confidence` is the score for the winning emotion.

### `POST /api/chat`

Generates one assistant reply from chat history and request-scoped generation controls.

Request:

```json
{
    "messages": [
        { "role": "user", "content": "I'm frustrated about a deadline." }
    ],
    "steering": {
        "emotions": [
            { "emotion": "calm", "strength": 1.5 },
            { "emotion": "angry", "strength": -0.5 }
        ],
        "position": "all"
    },
    "temperature": 0.7,
    "top_k": 50,
    "max_tokens": 256,
    "assistant_prefix": "Let's take this step by step:"
}
```

Response:

```json
{
    "reply": "Let's take this step by step: ..."
}
```

Fields:

| Field | Description |
| --- | --- |
| `messages` | Chat history, ordered oldest to newest. |
| `steering` | Per-request activation steering controls. |
| `steering.emotions` | Weighted emotion directions to combine for steering. |
| `steering.position` | Activation steering position: `all` or `last`. |
| `temperature` | Sampling temperature for generation. |
| `top_k` | Top-k sampling limit. `null` means disabled. |
| `max_tokens` | Maximum number of new tokens to generate. |
| `assistant_prefix` | Optional text that the assistant response must start with. |

Validation:

- `messages` must be non-empty.
- each message role must be `user` or `assistant`.
- `temperature` must be non-negative.
- `top_k` must be a positive integer when provided; `null` disables top-k sampling.
- `max_tokens` must be at least `1`.
- `steering.position` must be `all` or `last`.
- every steering emotion must exist in the configured emotion list; unknown emotions fail the request.

## Schemas

The server should model `/api/chat` around the client payload directly.

```python
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class EmotionSteer(BaseModel):
    emotion: str
    strength: float


class SteeringConfig(BaseModel):
    emotions: list[EmotionSteer] = Field(default_factory=list)
    position: Literal["all", "last"] = "all"


class ChatRequest(BaseModel):
    messages: list[Message]
    steering: SteeringConfig = Field(default_factory=SteeringConfig)
    temperature: float = 0.8
    top_k: int | None = 50
    max_tokens: int = 256
    assistant_prefix: str | None = None
```

Emotion names in requests and responses use canonical lowercase labels, for example `happy`, `sad`, `angry`, and `calm`.

The backend interface should accept the request as one object so future request controls can be added without widening every method signature.

```python
class Backend(ABC):
    @abstractmethod
    def detect(self, text: str) -> dict[str, float]:
        ...

    @abstractmethod
    def chat(self, request: ChatRequest) -> str:
        ...
```

## LLM Backend

### Detection

The LLM backend uses an OpenRouter chat completion model to classify text into the configured emotion labels.

Detection behavior:

- prompt the model to return only JSON.
- parse the first JSON object from the response.
- map emotion keys case-insensitively.
- clamp values into `[0, 1]`.
- fill missing emotions with `0`.
- normalize the scores if their sum is positive.
- return a safe default distribution if parsing fails.

### Chat

The LLM backend uses `messages`, `temperature`, `max_tokens`, and `assistant_prefix` directly.

`steering.emotions` are converted into natural-language style guidance because the LLM API does not support activation steering. Positive strength means lean into that emotion; negative strength means avoid it.

Example style guidance:

```text
Steer the assistant's tone according to these weighted emotion controls:
- calm: 1.5
- angry: -0.5

Positive strength means lean into that emotion.
Negative strength means avoid that emotion.
Keep replies concise unless the user asks for detail.
```

`top_k` and `steering.position` are accepted as part of the shared request shape but are not applied by the LLM backend.

## Nanochat Backend

### Initialization

The nanochat backend loads:

- nanochat model and tokenizer.
- `Engine` for generation.
- emotion probe data from `vectors.pt`.

Probe data fields used:

| Field | Purpose |
| --- | --- |
| `vectors` | Lowercase emotion name to activation direction. |
| `global_mean` | Activation mean used for centered detection. |
| `emotions` | Emotion names available in the probe file. |
| `layer` | Transformer layer used for detection and steering. |
| `max_len` | Detection token limit. |
| `skip_tokens` | Detection prefix tokens to skip before averaging activations. |

### Detection

Detection steps:

1. Compute the story activation for the input text.
2. Center it with `global_mean`.
3. Compute cosine similarity against each emotion vector.
4. Build logits in configured lowercase emotion order.
5. Apply low-temperature softmax to produce a probability distribution.
6. Return scores keyed by configured lowercase emotion labels.

### Chat Prompt Rendering

Nanochat chat history is rendered using the tokenizer's special chat tokens.

Rendering rules:

- start with BOS.
- user messages use `<|user_start|>` and `<|user_end|>`.
- assistant messages use `<|assistant_start|>` and `<|assistant_end|>`.
- if the last message is a user message, append `<|assistant_start|>` before generation.
- if `assistant_prefix` is present, encode it immediately after `<|assistant_start|>`.

### Steering Vector Construction

The nanochat backend constructs one steering vector from `request.steering.emotions`.

Algorithm:

```python
def build_steering_vector(items):
    vector = None
    for item in items:
        key = item.emotion.lower()
        emotion_vector = self.vectors.get(key)
        if emotion_vector is None:
            continue
        weighted = emotion_vector.float() * item.strength
        vector = weighted if vector is None else vector + weighted
    return vector
```

If no valid steering item is provided, generation runs without `steer_layer`.

If a vector is produced, it is applied with:

```python
steer_layer(
    self.model,
    self.layer,
    steering_vector,
    strength=1.0,
    positions=request.steering.position,
)
```

`strength=1.0` is used because individual steering strengths are already multiplied into the combined vector.

### Generation

Generation uses the request-provided parameters:

```python
result_tokens = self.engine.generate_batch(
    prompt_ids,
    num_samples=1,
    max_tokens=request.max_tokens,
    temperature=request.temperature,
    top_k=request.top_k,
)[0]
```

The reply is decoded from only the newly generated tokens. If `assistant_prefix` was provided, the returned reply includes the prefix followed by the decoded continuation.

## CLI

Expected commands:

```bash
python -m interp.emotion.chatui2.server --backend llm --port 8001
```

```bash
python -m interp.emotion.chatui2.server \
    --backend nanochat \
    --vectors out/emotion_probes_layer_12_skiptokens_20_maxlen_256/vectors.pt \
    --port 8001
```

Nanochat-specific CLI options:

| Option | Description |
| --- | --- |
| `--vectors` | Path to emotion probe vectors. |
| `--source` | Model source: `base`, `sft`, or `rl`. |
| `--model-tag` | Optional model tag. |
| `--step` | Optional checkpoint step. |
| `--device-type` | Device type: `cuda`, `cpu`, or `mps`; autodetect by default. |

## Verification

Backend implementation should be checked with:

```bash
python -m py_compile interp/emotion/chatui2/server.py
```

Manual smoke test:

1. start the API backend.
2. call `GET /api/config` and verify the configured emotions and backend name.
3. call `POST /api/detect` with non-empty text and verify the returned distribution.
4. call `POST /api/chat` with `messages`, `steering`, and generation parameters.
5. verify `/api/chat` applies the request parameters and returns a reply.
