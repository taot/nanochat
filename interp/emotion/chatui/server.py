#!/usr/bin/env python3
"""
Emotion Chat UI server.

Serves the HTML UI and provides AI-backed endpoints for emotion detection
and styled reply generation. Two backends are available:

  - llm      : OpenRouter-hosted Claude models (default)
  - nanochat : local nanochat model with probe-based detection and
               activation-steered generation

Run (default OpenRouter LLM backend, requires OPENROUTER_API_KEY):
    uvicorn interp.emotion.chatui.server:app --port 8001

Run with explicit CLI:
    python -m interp.emotion.chatui.server --backend llm --port 8001
    python -m interp.emotion.chatui.server --backend nanochat --port 8001
"""

import argparse
import json
import os
import re
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path

import openai
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from interp.emotion.emotion_probes import load_vectors, steer_layer, story_activation
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine

# ── Configurable emotion list ─────────────────────────────────────────────────
# Edit this list to change which emotions are available in the UI.
# Restart the server after editing.
EMOTIONS = ["Happy", "Sad", "Angry", "Calm"]

# ── Models (any model available on openrouter.ai) ─────────────────────────────
DETECT_MODEL = "anthropic/claude-haiku-4-5"   # fast + cheap for classification
REPLY_MODEL  = "anthropic/claude-sonnet-4-5"  # better quality for replies

# ── Server setup ──────────────────────────────────────────────────────────────
app = FastAPI()
HERE = Path(__file__).parent

# Active backend; set in __main__ (or lazily on first request when launched via `uvicorn …:app`).
backend: "Backend | None" = None

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    replyEmotion: str

# ── Backend abstraction ───────────────────────────────────────────────────────
class Backend(ABC):
    @abstractmethod
    def detect(self, text: str) -> dict[str, float]:
        """Return a Title-case emotion → probability dict summing to ~1."""

    @abstractmethod
    def chat(self, messages: list[Message], reply_emotion: str) -> str:
        """Return the assistant's reply text."""


class LLMBackend(Backend):
    def __init__(self, detect_model: str = DETECT_MODEL, reply_model: str = REPLY_MODEL):
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.detect_model = detect_model
        self.reply_model = reply_model

    def detect(self, text: str) -> dict[str, float]:
        emotion_labels = ", ".join(EMOTIONS)
        example_scores = ", ".join('"' + e + '":0.25' for e in EMOTIONS)
        prompt = (
            f"Classify the emotional tone of the following message. "
            f"For each label — {emotion_labels} — give a probability (0–1) that sums to 1. "
            "Reply with ONLY a JSON object, no prose, no markdown, "
            f'in this exact shape: {{"scores":{{{example_scores}}}}}.\n\n'
            f'Message:\n"""\n{text}\n"""'
        )
        resp = self.client.chat.completions.create(
            model=self.detect_model,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or ""
        try:
            m = re.search(r"\{[\s\S]*\}", raw)
            parsed = json.loads(m.group())
            raw_scores = {k.lower(): v for k, v in parsed.get("scores", {}).items()}
            return {
                e: max(0.0, min(1.0, float(raw_scores.get(e.lower(), 0))))
                for e in EMOTIONS
            }
        except Exception:
            return {e: (1.0 if e == EMOTIONS[-1] else 0.0) for e in EMOTIONS}

    def chat(self, messages: list[Message], reply_emotion: str) -> str:
        if reply_emotion.lower() == "none":
            style_note = "Keep replies to 1–3 sentences unless the user asks for more."
        else:
            style_note = (
                f"Reply in a {reply_emotion.lower()} tone. "
                "Keep replies to 1–3 sentences unless the user asks for more."
            )
        payload = [
            {"role": "user",      "content": f"[Style for all your replies: {style_note}] Acknowledge briefly."},
            {"role": "assistant", "content": "Understood."},
            *[{"role": m.role, "content": m.content} for m in messages],
        ]
        resp = self.client.chat.completions.create(
            model=self.reply_model,
            max_tokens=512,
            messages=payload,
        )
        return resp.choices[0].message.content


class NanochatBackend(Backend):
    def __init__(
        self,
        vectors_path: str,
        source: str = "sft",
        model_tag: str | None = None,
        step: int | None = None,
        strength: float = 2.0,
        device_type: str | None = None,
    ):
        if device_type is None:
            device_type = autodetect_device_type()
        _, _, _, _, device = compute_init(device_type)
        self.model, self.tokenizer, _ = load_model(
            source, device, phase="eval", model_tag=model_tag, step=step,
        )
        self.engine = Engine(self.model, self.tokenizer)

        self.data = load_vectors(vectors_path)
        print(self.data["vectors"])
        for emotion, vec in self.data["vectors"].items():
            print(emotion, vec, vec.shape)

        self.layer = self.data["layer"]
        self.max_len = self.data["max_len"]
        self.skip_tokens = self.data["skip_tokens"]
        self.global_mean = self.data["global_mean"]
        self.vectors = self.data["vectors"]            # lowercase keys
        self.probe_emotions = self.data["emotions"]
        self.strength = strength

        self.bos = self.tokenizer.get_bos_token_id()
        self.user_start = self.tokenizer.encode_special("<|user_start|>")
        self.user_end = self.tokenizer.encode_special("<|user_end|>")
        self.assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        self.assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        print(
            f"[nanochat backend] loaded probes from {vectors_path}: "
            f"layer={self.layer}, emotions={self.probe_emotions}, strength={self.strength}"
        )

    def detect(self, text: str) -> dict[str, float]:
        act = story_activation(
            self.model, self.tokenizer, text, self.layer, self.max_len, self.skip_tokens,
        )
        centered = act - self.global_mean
        raw = {
            e: F.cosine_similarity(centered, self.vectors[e].float(), dim=0).item()
            for e in self.probe_emotions
        }
        # Cosine similarities are small (typically 0.05–0.3); a low-temperature softmax
        # turns them into a usable 0–1 distribution while preserving relative order.
        logits = torch.tensor([raw.get(e.lower(), 0.0) for e in EMOTIONS])
        probs = F.softmax(logits / 0.1, dim=0).tolist()
        return {e: probs[i] for i, e in enumerate(EMOTIONS)}

    def chat(self, messages: list[Message], reply_emotion: str) -> str:
        print("messages:", messages)
        prompt_ids = self._render_multi_turn(messages)
        print("prompt_ids:", prompt_ids)
        vector = self.vectors.get(reply_emotion.lower())
        print("vector:", vector)
        ctx = (
            steer_layer(self.model, self.layer, vector, self.strength, positions="all")
            if vector is not None
            else nullcontext()
        )
        with ctx:
            result_tokens = self.engine.generate_batch(
                prompt_ids, num_samples=1, max_tokens=512, temperature=0.8, top_k=50,
            )[0]

        print("result_tokens:", result_tokens)
        new_tokens = result_tokens[0][len(prompt_ids):]

        print("New tokens:", new_tokens)
        print("New text:", self.tokenizer.decode(new_tokens))

        return self.tokenizer.decode(new_tokens)

    def _render_multi_turn(self, messages: list[Message]) -> list[int]:
        ids: list[int] = [self.bos]
        for i, m in enumerate(messages):
            is_last = i == len(messages) - 1
            start = self.user_start if m.role == "user" else self.assistant_start
            ids.append(start)
            ids.extend(self.tokenizer.encode(m.content))
            if is_last and m.role == "user":
                ids.append(self.user_end)
                ids.append(self.assistant_start)
            else:
                ids.append(self.user_end if m.role == "user" else self.assistant_end)
        return ids


# ── Routes ────────────────────────────────────────────────────────────────────
def _ensure_backend() -> "Backend":
    """Lazy default for the old `uvicorn …:app` launch path (no __main__ run)."""
    global backend
    if backend is None:
        backend = LLMBackend()
    return backend


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text()

@app.get("/api/config")
def config():
    return {"emotions": EMOTIONS}

@app.post("/api/detect")
def detect(req: DetectRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")
    scores = _ensure_backend().detect(text)
    emotion = max(scores, key=scores.__getitem__)
    return {"emotion": emotion, "confidence": scores[emotion], "scores": scores}

@app.post("/api/chat")
def chat(req: ChatRequest):
    return {"reply": _ensure_backend().chat(req.messages, req.replyEmotion)}


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "llm":
        return LLMBackend()
    return NanochatBackend(
        vectors_path=args.vectors,
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
        strength=args.strength,
        device_type=args.device,
    )


def main():
    p = argparse.ArgumentParser(description="Emotion chat UI server.")
    p.add_argument("--backend", choices=["llm", "nanochat"], default="llm")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="127.0.0.1")
    # nanochat-only knobs
    p.add_argument("--vectors", default="out/emotion_probes_layer_12_skiptokens_0_maxlen_128/vectors.pt")
    p.add_argument("--source", choices=["base", "sft", "rl"], default="sft")
    p.add_argument("--model-tag", default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--strength", type=float, default=2.0)
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], default=None,
                   help="Device type for nanochat: cuda|cpu|mps. Default: autodetect")
    args = p.parse_args()

    globals()["backend"] = _build_backend(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
