#!/usr/bin/env python3
"""
Emotion Chat UI server.

Serves the HTML UI and provides AI-backed endpoints for emotion detection
and styled reply generation via OpenRouter.

Run: uvicorn interp.emotion.chatui.server:app --port 8001
Requires: OPENROUTER_API_KEY env var
"""

import json
import re
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Configurable emotion list ─────────────────────────────────────────────────
# Edit this list to change which emotions are available in the UI.
# Restart the server after editing.
EMOTIONS = ["Happy", "Sad", "Angry", "Calm"]

# ── Models (any model available on openrouter.ai) ─────────────────────────────
DETECT_MODEL = "anthropic/claude-haiku-4-5"   # fast + cheap for classification
REPLY_MODEL  = "anthropic/claude-sonnet-4-5"  # better quality for replies

# ── Server setup ──────────────────────────────────────────────────────────────
app = FastAPI()
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=__import__("os").environ["OPENROUTER_API_KEY"],
)
HERE = Path(__file__).parent

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    replyEmotion: str

# ── Routes ────────────────────────────────────────────────────────────────────
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

    emotion_labels = ", ".join(EMOTIONS)
    example_scores = ", ".join('"' + e + '":0.25' for e in EMOTIONS)
    prompt = (
        f"Classify the emotional tone of the following message. "
        f"For each label — {emotion_labels} — give a probability (0–1) that sums to 1. "
        "Reply with ONLY a JSON object, no prose, no markdown, "
        f'in this exact shape: {{"scores":{{{example_scores}}}}}.\n\n'
        f'Message:\n"""\n{text}\n"""'
    )
    resp = client.chat.completions.create(
        model=DETECT_MODEL,
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content or ""
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(m.group())
        raw_scores = {k.lower(): v for k, v in parsed.get("scores", {}).items()}
        scores = {
            e: max(0.0, min(1.0, float(raw_scores.get(e.lower(), 0))))
            for e in EMOTIONS
        }
        emotion = max(scores, key=scores.__getitem__)
        confidence = scores[emotion]
    except Exception:
        scores = {e: (1.0 if e == EMOTIONS[-1] else 0.0) for e in EMOTIONS}
        emotion = EMOTIONS[-1]
        confidence = 1.0
    return {"emotion": emotion, "confidence": confidence, "scores": scores}

@app.post("/api/chat")
def chat(req: ChatRequest):
    style_note = (
        f"Reply in a {req.replyEmotion.lower()} tone. "
        "Keep replies to 1–3 sentences unless the user asks for more."
    )
    messages = [
        {"role": "user",      "content": f"[Style for all your replies: {style_note}] Acknowledge briefly."},
        {"role": "assistant", "content": "Understood."},
        *[{"role": m.role, "content": m.content} for m in req.messages],
    ]
    resp = client.chat.completions.create(
        model=REPLY_MODEL,
        max_tokens=512,
        messages=messages,
    )
    return {"reply": resp.choices[0].message.content}
