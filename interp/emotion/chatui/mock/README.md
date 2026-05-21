# Emotion Chat UI Mock

This is a React mock version of `interp/emotion/chatui/index.html`.

It keeps the desktop chat UI and uses local mock functions instead of real backend APIs. The mock API lives in `src/api.js`.

## Run

```bash
cd interp/emotion/chatui/mock
npm install
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`.

## Build

```bash
npm run build
npm run preview
```

## Mock API

Current mock functions are in `src/api.js`:

- `getConfig()` returns emotion labels and mock backend config.
- `detectEmotion(text)` returns a mock emotion, confidence, and score distribution.
- `generateReply(...)` returns a mock assistant reply after a short delay.

To connect real APIs later, replace the implementations in `src/api.js` with `fetch` calls to your backend endpoints, for example `/api/config`, `/api/detect`, and `/api/chat`.
