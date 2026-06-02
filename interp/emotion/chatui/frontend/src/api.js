function titleCase(value) {
  return value ? `${value[0].toUpperCase()}${value.slice(1).toLowerCase()}` : value;
}

function normalizeDetection(result) {
  return {
    ...result,
    emotion: titleCase(result.emotion),
    scores: Object.fromEntries(Object.entries(result.scores || {}).map(([emotion, score]) => [titleCase(emotion), score])),
  };
}

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

export async function getConfig() {
  const config = await request("/api/config");
  return { ...config, emotions: config.emotions.map(titleCase), strength: 2 };
}

export async function detectEmotion(text) {
  const result = await request("/api/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return normalizeDetection(result);
}

export async function generateReply({
  messages,
  replyEmotion,
  strength,
  steeringItems,
  temperature,
  topK,
  maxTokens,
  position,
  assistantPrefix,
}) {
  const emotions = [...steeringItems];
  if (replyEmotion !== "None") {
    emotions.push({ emotion: replyEmotion, strength });
  }
  const result = await request("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      steering: {
        emotions: emotions.map((item) => ({
          emotion: item.emotion.toLowerCase(),
          strength: item.strength,
        })),
        position,
      },
      temperature,
      top_k: topK,
      max_tokens: maxTokens,
      assistant_prefix: assistantPrefix.trim() || null,
    }),
  });
  return result.reply;
}
