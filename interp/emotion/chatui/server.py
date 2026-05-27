#!/usr/bin/env python3
"""Pure FastAPI API server for emotion detection and steered replies."""

import argparse
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

EMOTIONS = ["happy", "sad", "angry", "calm"]
DETECT_MODEL = "anthropic/claude-haiku-4-5"
REPLY_MODEL = "anthropic/claude-sonnet-4-5"
DEFAULT_VECTORS = "out/emotion_probes_layer_12_skiptokens_20_maxlen_256/vectors.pt"

logging.basicConfig(
    level=os.environ.get("CHATUI_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("chatui.server")

class _ApiLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next) -> StarletteResponse:
        body = await request.body()
        if body:
            logger.info("[%s] request: %s", request.url.path, body.decode())

        response = await call_next(request)

        chunks = [chunk async for chunk in response.body_iterator]
        body = b"".join(chunks)
        if body:
            logger.info("[%s] response: %s", request.url.path, body.decode())
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        return StarletteResponse(body, status_code=response.status_code, headers=headers, media_type=response.media_type)


app = FastAPI()
app.add_middleware(_ApiLoggingMiddleware)
backend: "Backend | None" = None
backend_name = "llm"


class DetectRequest(BaseModel):
    text: str


class ConfigResponse(BaseModel):
    emotions: list[str]
    backend: str


class HealthResponse(BaseModel):
    ok: bool
    backend: str


class DetectResponse(BaseModel):
    emotion: str
    confidence: float
    scores: dict[str, float]


class ChatResponse(BaseModel):
    reply: str


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
    temperature: float = Field(default=0.8, ge=0.0)
    top_k: int | None = Field(default=50, ge=1)
    max_tokens: int = Field(default=256, ge=1)
    assistant_prefix: str | None = None


class Backend(ABC):
    @abstractmethod
    def detect(self, text: str) -> DetectResponse:
        """Return the detected emotion and lowercase emotion probabilities."""

    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        """Return the assistant reply for a full request object."""


class LLMBackend(Backend):
    def __init__(self, detect_model: str = DETECT_MODEL, reply_model: str = REPLY_MODEL) -> None:
        import openai

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.detect_model = detect_model
        self.reply_model = reply_model
        logger.info("LLMBackend initialized: detect_model=%s reply_model=%s", detect_model, reply_model)

    def detect(self, text: str) -> DetectResponse:
        labels = ", ".join(EMOTIONS)
        example = ", ".join(f'"{emotion}": 0.25' for emotion in EMOTIONS)
        prompt = (
            "Classify the emotional tone of the message. "
            f"Use exactly these lowercase labels: {labels}. "
            "Return only JSON, no prose or markdown, in this shape: "
            f'{{"scores": {{{example}}}}}.\n\nMessage:\n"""\n{text}\n"""'
        )
        try:
            response = self.client.chat.completions.create(
                model=self.detect_model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
            match = re.search(r"\{[\s\S]*\}", raw)
            parsed = json.loads(match.group() if match else raw)
            raw_scores = {str(k).lower(): v for k, v in parsed.get("scores", parsed).items()}
            scores = {
                emotion: max(0.0, min(1.0, float(raw_scores.get(emotion, 0.0))))
                for emotion in EMOTIONS
            }
            return _detect_response(_normalize_scores(scores))
        except Exception:
            logger.exception("detect failed for text=%r, returning default scores", text)
            return _detect_response(_default_scores())

    def chat(self, request: ChatRequest) -> ChatResponse:
        payload: list[dict[str, str]] = []
        style_note = _style_guidance(request.steering.emotions)
        if style_note:
            payload.append({"role": "system", "content": style_note})
        if request.assistant_prefix:
            payload.append({
                "role": "system",
                "content": f'Return a reply that starts exactly with: "{request.assistant_prefix}"',
            })
        payload.extend({"role": m.role, "content": m.content} for m in request.messages)

        response = self.client.chat.completions.create(
            model=self.reply_model,
            messages=payload,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        reply = response.choices[0].message.content or ""
        if request.assistant_prefix and not reply.startswith(request.assistant_prefix):
            reply = request.assistant_prefix + reply
        return ChatResponse(reply=reply)


class NanochatBackend(Backend):
    def __init__(
        self,
        vectors_path: str,
        source: str = "sft",
        model_tag: str | None = None,
        step: int | None = None,
        device_type: str | None = None,
    ) -> None:
        from interp.emotion.emotion_probes import load_vectors, steer_layer, story_activation
        from nanochat.checkpoint_manager import load_model
        from nanochat.common import autodetect_device_type, compute_init
        from nanochat.engine import Engine

        self.steer_layer = steer_layer
        self.story_activation = story_activation
        device_type = autodetect_device_type() if device_type is None else device_type
        _, _, _, _, device = compute_init(device_type)
        self.model, self.tokenizer, _ = load_model(
            source,
            device,
            phase="eval",
            model_tag=model_tag,
            step=step,
        )
        self.engine = Engine(self.model, self.tokenizer)

        data = load_vectors(vectors_path)
        self.layer = data["layer"]
        self.max_len = data["max_len"]
        self.skip_tokens = data["skip_tokens"]
        self.global_mean = data["global_mean"]
        self.vectors = {str(k).lower(): v for k, v in data["vectors"].items()}
        self.probe_emotions = [str(e).lower() for e in data["emotions"]]

        self.bos = self.tokenizer.get_bos_token_id()
        self.user_start = self.tokenizer.encode_special("<|user_start|>")
        self.user_end = self.tokenizer.encode_special("<|user_end|>")
        self.assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        self.assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        missing = [emotion for emotion in EMOTIONS if emotion not in self.vectors]
        if missing:
            raise ValueError(f"Probe file is missing configured emotions: {', '.join(missing)}")
        logger.info(
            "NanochatBackend loaded %s: layer=%s emotions=%s",
            vectors_path, self.layer, self.probe_emotions,
        )

    def detect(self, text: str) -> DetectResponse:
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            activation = self.story_activation(
                self.model,
                self.tokenizer,
                text,
                self.layer,
                self.max_len,
                self.skip_tokens,
            )
            centered = activation - self.global_mean
            raw = {
                emotion: F.cosine_similarity(centered, self.vectors[emotion].float(), dim=0).item()
                for emotion in EMOTIONS
            }
            logits = torch.tensor([raw[emotion] for emotion in EMOTIONS])
            probabilities = F.softmax(logits / 0.1, dim=0).tolist()
            scores = {emotion: probabilities[i] for i, emotion in enumerate(EMOTIONS)}
            return _detect_response(scores)

    def chat(self, request: ChatRequest) -> ChatResponse:
        import torch

        with torch.inference_mode():
            prompt_ids = self._render_chat_prompt(request.messages, request.assistant_prefix)
            steering_vector = self._build_steering_vector(request.steering.emotions)
            logger.debug(
                "chat: steering=%s temperature=%s top_k=%s max_tokens=%s",
                request.steering.model_dump(), request.temperature, request.top_k, request.max_tokens,
            )
            ctx = (
                self.steer_layer(
                    self.model,
                    self.layer,
                    steering_vector,
                    strength=1.0,
                    positions=request.steering.position,
                )
                if steering_vector is not None
                else nullcontext()
            )
            with ctx:
                results, _ = self.engine.generate_batch(
                    prompt_ids,
                    num_samples=1,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                )
            new_tokens = results[0][len(prompt_ids):]
            continuation = self.tokenizer.decode(new_tokens)
            if request.assistant_prefix:
                return ChatResponse(reply=request.assistant_prefix + continuation)
            return ChatResponse(reply=continuation)

    def _render_chat_prompt(self, messages: list[Message], assistant_prefix: str | None) -> list[int]:
        ids = [self.bos]
        for index, message in enumerate(messages):
            is_last = index == len(messages) - 1
            if message.role == "user":
                ids.append(self.user_start)
                ids.extend(self.tokenizer.encode(message.content))
                ids.append(self.user_end)
                if is_last:
                    ids.append(self.assistant_start)
                    if assistant_prefix:
                        ids.extend(self.tokenizer.encode(assistant_prefix))
            else:
                ids.append(self.assistant_start)
                ids.extend(self.tokenizer.encode(message.content))
                ids.append(self.assistant_end)
        return ids

    def _build_steering_vector(self, items: list[EmotionSteer]) -> Any | None:
        vector: Any | None = None
        for item in items:
            emotion_vector = self.vectors.get(item.emotion.lower())
            if emotion_vector is None:
                continue
            weighted = emotion_vector.float() * item.strength
            vector = weighted if vector is None else vector + weighted
        return vector


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return _default_scores()
    return {emotion: scores[emotion] / total for emotion in EMOTIONS}


def _default_scores() -> dict[str, float]:
    return {emotion: (1.0 if emotion == "calm" else 0.0) for emotion in EMOTIONS}


def _detect_response(scores: dict[str, float]) -> DetectResponse:
    scores = _normalize_scores({emotion: float(scores.get(emotion, 0.0)) for emotion in EMOTIONS})
    emotion = max(scores, key=scores.__getitem__)
    return DetectResponse(emotion=emotion, confidence=scores[emotion], scores=scores)


def _style_guidance(items: list[EmotionSteer]) -> str:
    if not items:
        return "Keep replies concise unless the user asks for detail."
    lines = [
        "Steer the assistant's tone according to these weighted emotion controls:",
        *[f"- {item.emotion.lower()}: {item.strength}" for item in items],
        "",
        "Positive strength means lean into that emotion.",
        "Negative strength means avoid that emotion.",
        "Keep replies concise unless the user asks for detail.",
    ]
    return "\n".join(lines)


def _ensure_backend() -> Backend:
    global backend
    if backend is None:
        backend = LLMBackend()
    return backend


def _validate_steering(request: ChatRequest) -> None:
    unknown = sorted({item.emotion.lower() for item in request.steering.emotions} - set(EMOTIONS))
    if unknown:
        raise HTTPException(400, f"unknown steering emotion(s): {', '.join(unknown)}")


@app.get("/api/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    return ConfigResponse(emotions=EMOTIONS, backend=backend_name)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if backend is None:
        raise HTTPException(503, "backend is not ready")
    return HealthResponse(ok=backend is not None, backend=backend_name)


@app.post("/api/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(400, "text must be non-empty")
    return _ensure_backend().detect(text)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        raise HTTPException(400, "messages must be non-empty")
    _validate_steering(request)
    return _ensure_backend().chat(request)


def _configure_cors(origins: list[str]) -> None:
    if not origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _build_backend(args: argparse.Namespace) -> Backend:
    global backend_name
    backend_name = args.backend
    if args.backend == "llm":
        return LLMBackend()
    return NanochatBackend(
        vectors_path=args.vectors,
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
        device_type=args.device_type,
    )


def _env_cors_origins() -> list[str]:
    raw = os.environ.get("CHATUI_CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Emotion chatui API server.")
    parser.add_argument("--backend", choices=["llm", "nanochat"], default="llm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--cors-origin", action="append", default=[], help="Allowed browser origin; repeatable")
    parser.add_argument("--vectors", default=DEFAULT_VECTORS)
    parser.add_argument("--source", choices=["base", "sft", "rl"], default="sft")
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default=None)
    args = parser.parse_args()

    logger.info("args: %s", vars(args))
    globals()["backend"] = _build_backend(args)
    _configure_cors([*args.cors_origin, *_env_cors_origins()])
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
