# chatui Usage Guide

`chatui` is a Web UI for emotion detection and emotion-guided responses. The backend uses FastAPI, and the frontend uses Vite + React.

## Setup

Install the Python dependencies from the repository root:

```bash
uv sync
source .venv/bin/activate
```

The frontend requires `npm` to be installed locally. The one-command startup script automatically runs `npm install` under `frontend/`.

## One-Command Startup

By default, the backend uses the OpenRouter LLM, so you need to set the API key first:

```bash
export OPENROUTER_API_KEY=your_key
python interp/emotion/chatui/start.py
```

After startup, open:

```text
http://127.0.0.1:5173
```

Default ports:

- Backend API: `http://127.0.0.1:8001`
- Frontend page: `http://127.0.0.1:5173`

## Using the nanochat Backend

To use a local nanochat model and emotion vectors, pass backend arguments after `start.py` with `--`:

```bash
python interp/emotion/chatui/start.py -- --backend nanochat \
    --vectors out/emotion_probes_layer_12_skiptokens_20_maxlen_256/vectors.pt \
    --source sft
```

Common arguments:

- `--backend llm|nanochat`: Backend type. Defaults to `llm`.
- `--vectors`: Path to the emotion vector file. Defaults to `out/emotion_probes_layer_12_skiptokens_20_maxlen_256/vectors.pt`.
- `--source base|sft|rl`: Model stage to load. Defaults to `sft`.
- `--model-tag`: Model tag to use.
- `--step`: Checkpoint step to use.
- `--device-type cuda|cpu|mps`: Runtime device to use.

## Manual Startup

You can also start the backend and frontend separately.

Terminal 1, start the backend:

```bash
python -m interp.emotion.chatui.server --host 127.0.0.1 --port 8001
```

Terminal 2, start the frontend:

```bash
cd interp/emotion/chatui/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

The frontend development server proxies `/api` requests to `http://127.0.0.1:8001`.

## Stopping Services

Press `Ctrl+C` in the terminal running `start.py`. The script stops both the backend and frontend processes.
