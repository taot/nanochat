const EMOTIONS = ["Happy", "Sad", "Angry", "Calm"];

const KEYWORDS = {
  Happy: ["happy", "hopeful", "great", "thanks", "glad", "excited", "love"],
  Sad: ["sad", "upset", "tired", "lonely", "sorry", "hurt", "down"],
  Angry: ["angry", "frustrated", "deadline", "annoyed", "mad", "unfair", "blocked"],
  Calm: ["breathe", "calm", "relax", "focus", "quiet", "steady", "peace"],
};

const REPLIES = {
  None: "I hear you. Let's make this concrete: name the main issue, then choose one small next step.",
  Happy: "That sounds encouraging. We can build on that momentum and turn it into something useful today.",
  Sad: "I'm sorry this feels heavy. We can slow it down, sort through it, and find one manageable step.",
  Angry: "That frustration makes sense. Let's separate what happened, what you need, and what response would help.",
  Calm: "Let's keep this steady. Take a breath, then focus on the next useful move rather than the whole problem.",
};

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export async function getConfig() {
  await delay(150);
  return { emotions: EMOTIONS, strength: 2, backend: "mock" };
}

export async function detectEmotion(text) {
  await delay(400);
  const lower = text.toLowerCase();
  const raw = Object.fromEntries(
    EMOTIONS.map((emotion) => [
      emotion,
      KEYWORDS[emotion].reduce((count, keyword) => count + (lower.includes(keyword) ? 1 : 0), 0),
    ]),
  );

  if (Object.values(raw).every((count) => count === 0)) raw.Calm = 1;

  const total = Object.values(raw).reduce((sum, count) => sum + count, 0);
  const scores = Object.fromEntries(
    EMOTIONS.map((emotion) => [emotion, Number(((raw[emotion] + 0.2) / (total + 0.8)).toFixed(2))]),
  );
  const emotion = EMOTIONS.reduce((best, current) => (scores[current] > scores[best] ? current : best), EMOTIONS[0]);

  return { emotion, confidence: scores[emotion], scores };
}

export async function generateReply({ replyEmotion, assistantPrefix }) {
  await delay(750);
  const mode = REPLIES[replyEmotion] ? replyEmotion : "None";
  const prefix = assistantPrefix.trim() ? `${assistantPrefix.trim()} ` : "";
  return `${prefix}${REPLIES[mode]}`;
}
