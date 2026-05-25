# chatui2 运行说明

`chatui2` 是一个情绪检测和情绪引导回复的 Web UI。后端是 FastAPI，前端是 Vite + React。

## 准备环境

在仓库根目录安装 Python 依赖：

```bash
uv sync
source .venv/bin/activate
```

前端需要本机已安装 `npm`。一键启动脚本会自动在 `frontend/` 下执行 `npm install`。

## 一键启动

默认后端使用 OpenRouter LLM，需要先设置 API key：

```bash
export OPENROUTER_API_KEY=你的_key
python interp/emotion/chatui2/start.py
```

启动后访问：

```text
http://127.0.0.1:5173
```

默认端口：

- 后端 API: `http://127.0.0.1:8001`
- 前端页面: `http://127.0.0.1:5173`

## 使用 nanochat 后端

如果要使用本地 nanochat 模型和情绪向量，在 `start.py` 后通过 `--` 传递后端参数：

```bash
python interp/emotion/chatui2/start.py -- --backend nanochat \
    --vectors out/emotion_probes_layer_12_skiptokens_20_maxlen_256/vectors.pt \
    --source sft
```

常用参数：

- `--backend llm|nanochat`: 后端类型，默认 `llm`
- `--vectors`: 情绪向量文件路径，默认 `out/emotion_probes_layer_12_skiptokens_20_maxlen_256/vectors.pt`
- `--source base|sft|rl`: 加载的模型阶段，默认 `sft`
- `--model-tag`: 指定模型 tag
- `--step`: 指定 checkpoint step
- `--device-type cuda|cpu|mps`: 指定运行设备

## 手动启动

也可以分别启动后端和前端。

终端 1，启动后端：

```bash
python -m interp.emotion.chatui2.server --host 127.0.0.1 --port 8001
```

终端 2，启动前端：

```bash
cd interp/emotion/chatui2/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

前端开发服务器会把 `/api` 请求代理到 `http://127.0.0.1:8001`。

## 停止服务

在运行 `start.py` 的终端按 `Ctrl+C`，脚本会同时停止后端和前端进程。
