import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { detectEmotion, generateReply, getConfig } from "./api";

const PROMPTS = [
  "Help me draft a polite decline to a meeting.",
  "I'm frustrated about a deadline - can we talk through it?",
  "Tell me something hopeful about today.",
  "Walk me through a breathing exercise.",
];

function useChat() {
  const [emotions, setEmotions] = useState(["Happy", "Sad", "Angry", "Calm"]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [replyEmotion, setReplyEmotion] = useState("None");
  const [strength, setStrength] = useState(2);
  const [steeringItems, setSteeringItems] = useState([]);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(256);
  const [position, setPosition] = useState("all");
  const [assistantPrefix, setAssistantPrefix] = useState("");
  const [backend, setBackend] = useState("mock");
  const [detecting, setDetecting] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [lastDetection, setLastDetection] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getConfig().then((config) => {
      if (cancelled) return;
      if (Array.isArray(config.emotions) && config.emotions.length) setEmotions(config.emotions);
      if (typeof config.strength === "number") setStrength(config.strength);
      if (config.backend) setBackend(config.backend);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const detect = useCallback(async () => {
    const text = input.trim();
    if (!text || detecting) return;
    setDetecting(true);
    try {
      const result = await detectEmotion(text);
      setLastDetection({ ...result, text });
    } finally {
      setDetecting(false);
    }
  }, [detecting, input]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || thinking) return;

    const userMessage = { id: `${Date.now()}-u`, role: "user", content: text };
    const history = [...messages, userMessage].map((message) => ({ role: message.role, content: message.content }));

    setInput("");
    setMessages((previous) => [...previous, userMessage]);
    setThinking(true);

    const detection = detectEmotion(text).then((result) => {
      setLastDetection({ ...result, text });
      setMessages((previous) =>
        previous.map((message) => (message.id === userMessage.id ? { ...message, emotion: result } : message)),
      );
    });

    try {
      const reply = await generateReply({
        messages: history,
        replyEmotion,
        strength,
        steeringItems,
        temperature,
        topK,
        maxTokens,
        position,
        assistantPrefix,
      });
      await detection;
      setMessages((previous) => [
        ...previous,
        { id: `${Date.now()}-a`, role: "assistant", content: reply, replyEmotion },
      ]);
    } finally {
      setThinking(false);
    }
  }, [assistantPrefix, input, maxTokens, messages, position, replyEmotion, steeringItems, strength, temperature, thinking, topK]);

  const clear = useCallback(() => {
    setMessages([]);
    setLastDetection(null);
    setInput("");
  }, []);

  return {
    emotions,
    messages,
    input,
    setInput,
    replyEmotion,
    setReplyEmotion,
    strength,
    setStrength,
    steeringItems,
    setSteeringItems,
    temperature,
    setTemperature,
    topK,
    setTopK,
    maxTokens,
    setMaxTokens,
    position,
    setPosition,
    backend,
    assistantPrefix,
    setAssistantPrefix,
    detecting,
    thinking,
    lastDetection,
    detect,
    send,
    clear,
  };
}

function LeftRail({ chat }) {
  const userMessages = chat.messages.filter((message) => message.role === "user");
  return (
    <aside className="left-rail">
      <div className="rail-title-row">
        <div className="section-title">Conversations</div>
        <button className="new-chat" onClick={chat.clear} title="New chat">+</button>
      </div>
      <div className="conversation-card">
        <div className="conversation-name">{userMessages[0]?.content?.slice(0, 28) || "Untitled session"}</div>
        <div className="conversation-meta">{userMessages.length} message{userMessages.length === 1 ? "" : "s"}</div>
      </div>
    </aside>
  );
}

function Distribution({ detection, emotions }) {
  const scores = detection?.scores || {};
  return (
    <div>
      <div className="eyebrow likelihood-title">LIKELIHOOD</div>
      <div className="bars">
        {emotions.map((emotion) => {
          const percent = detection ? Math.round((scores[emotion] || 0) * 100) : 0;
          return (
            <div key={emotion}>
              <div className="bar-label-row">
                <span>{emotion}</span>
                <span className="mono-muted">{detection ? `${percent}%` : "-"}</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${percent}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RightRail({ chat }) {
  const detection = chat.lastDetection;
  const percent = detection ? Math.round(detection.confidence * 100) : 0;
  const addSteeringItem = () => {
    chat.setSteeringItems((items) => [
      ...items,
      { id: `${Date.now()}-${items.length}`, emotion: chat.emotions[0] || "", strength: 1 },
    ]);
  };
  const updateSteeringItem = (id, patch) => {
    chat.setSteeringItems((items) => items.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  };
  const deleteSteeringItem = (id) => {
    chat.setSteeringItems((items) => items.filter((item) => item.id !== id));
  };
  return (
    <aside className="right-rail">
      <div className="analysis-header">
        <div className="section-title">Prompt emotion</div>
        <div className="status-label">{chat.detecting ? "WORKING..." : detection ? "LATEST" : "IDLE"}</div>
      </div>
      <div className="analysis-card">
        <div className="emotion-row">
          <div className="emotion-name">{detection ? detection.emotion : "-"}</div>
          <div className="confidence">{detection ? `${percent}%` : "-"}</div>
        </div>
        <div className="confidence-track">
          <div className="confidence-fill" style={{ width: `${percent}%` }} />
        </div>
        {detection && (
          <div className="detected-text">
            "{detection.text.slice(0, 80)}{detection.text.length > 80 ? "..." : ""}"
          </div>
        )}
      </div>
      <Distribution detection={detection} emotions={chat.emotions} />
      <section className="steering-section">
        <div className="section-title">Steering</div>
        <div className="steering-subitem">
          <div className="steering-title-row">
            <span>Emotion</span>
            <button className="add-steering-button" onClick={addSteeringItem} type="button">Add</button>
          </div>
          <div className="steering-list">
            {chat.steeringItems.length === 0 ? (
              <div className="empty-steering">No steering emotions</div>
            ) : chat.steeringItems.map((item) => (
              <div className="steering-item" key={item.id}>
                <select
                  value={item.emotion}
                  onChange={(event) => updateSteeringItem(item.id, { emotion: event.target.value })}
                >
                  {chat.emotions.map((emotion) => <option key={emotion} value={emotion}>{emotion}</option>)}
                </select>
                <input
                  type="number"
                  value={item.strength}
                  onChange={(event) => updateSteeringItem(item.id, { strength: Number(event.target.value) })}
                  step={0.1}
                />
                <button className="delete-steering-button" onClick={() => deleteSteeringItem(item.id)} type="button">Remove</button>
              </div>
            ))}
          </div>
        </div>
        <div className="steering-field">
          <span>Position</span>
          <div className="radio-row">
            {[
              ["all", "All"],
              ["last", "Last"],
            ].map(([value, label]) => (
              <label key={value} className="radio-option">
                <input
                  type="radio"
                  name="steering-position"
                  value={value}
                  checked={chat.position === value}
                  onChange={() => chat.setPosition(value)}
                />
                {label}
              </label>
            ))}
          </div>
        </div>
      </section>
      <section className="parameters-section">
        <div className="section-title">Parameters</div>
        <label className="steering-field">
          <span>Temperature</span>
          <input
            type="number"
            value={chat.temperature}
            onChange={(event) => chat.setTemperature(Number(event.target.value))}
            step={0.1}
            min={0}
          />
        </label>
        <label className="steering-field">
          <span>Top k</span>
          <input
            type="number"
            value={chat.topK}
            onChange={(event) => chat.setTopK(Number.parseInt(event.target.value, 10) || 0)}
            step={1}
            min={0}
          />
        </label>
        <label className="steering-field">
          <span>Max tokens</span>
          <input
            type="number"
            value={chat.maxTokens}
            onChange={(event) => chat.setMaxTokens(Number.parseInt(event.target.value, 10) || 0)}
            step={1}
            min={1}
          />
        </label>
      </section>
      <div className="analysis-note">Analysis is per-message and may not reflect underlying intent.</div>
    </aside>
  );
}

function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="user-message-row">
        <div className="user-message-wrap">
          <div className="user-bubble">{message.content}</div>
          {message.emotion && (
            <div className="message-emotion">
              {message.emotion.emotion.toUpperCase()} - {Math.round(message.emotion.confidence * 100)}%
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="assistant-message-row">
      <div className="avatar">A</div>
      <div className="assistant-content">
        <div className="assistant-title-row">
          <div className="assistant-title">Assistant</div>
          <div className="assistant-mode">REPLY - {(message.replyEmotion || "").toUpperCase()}</div>
        </div>
        <div className="assistant-text">{message.content}</div>
      </div>
    </div>
  );
}

function Empty({ setInput }) {
  return (
    <div className="empty-state">
      <div className="empty-title">Start a conversation</div>
      <div className="empty-copy">
        Type a message below. Press <span className="keycap">Detect</span> to analyze the tone of your draft, or send it to get a reply in your chosen mode.
      </div>
      <div className="prompt-grid">
        {PROMPTS.map((prompt) => (
          <button className="prompt-card" key={prompt} onClick={() => setInput(prompt)}>{prompt}</button>
        ))}
      </div>
    </div>
  );
}

function Thinking() {
  return (
    <div className="assistant-message-row thinking-row">
      <div className="avatar muted-avatar">A</div>
      <div className="thinking-dots">
        {[0, 1, 2].map((index) => <span key={index} style={{ animationDelay: `${index * 0.15}s` }} />)}
      </div>
    </div>
  );
}

function Workspace() {
  const chat = useChat();
  const scrollerRef = useRef(null);
  const userMessages = useMemo(() => chat.messages.filter((message) => message.role === "user"), [chat.messages]);

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
  }, [chat.messages.length, chat.thinking]);

  const onKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      chat.send();
    }
  };

  return (
    <div className="workspace">
      <LeftRail chat={chat} />
      <main className="main-panel">
        <header className="chat-header">
          <div>
            <div className="session-title">{userMessages[0]?.content?.slice(0, 38) || "New session"}</div>
            <div className="session-meta">{chat.messages.length} turns - reply mode: {chat.replyEmotion.toLowerCase()}</div>
          </div>
          <div className="auto-detect">auto-detect ON</div>
        </header>

        <section className="messages" ref={scrollerRef}>
          <div className="message-column">
            {chat.messages.length === 0 ? <Empty setInput={chat.setInput} /> : chat.messages.map((message) => <Message key={message.id} message={message} />)}
            {chat.thinking && <Thinking />}
          </div>
        </section>

        <footer className="composer-area">
          <div className="composer">
            <textarea
              value={chat.input}
              onChange={(event) => chat.setInput(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Type a message..."
              rows={2}
            />
            <div className="prefix-row">
              <span>ASSISTANT PREFIX</span>
              <input
                type="text"
                value={chat.assistantPrefix}
                onChange={(event) => chat.setAssistantPrefix(event.target.value)}
                placeholder="Assistant starts with..."
              />
            </div>
            <div className="action-row">
              <button className="secondary-button" onClick={chat.detect} disabled={!chat.input.trim() || chat.detecting}>
                {chat.detecting ? "Analyzing..." : "Detect"}
              </button>
              <div className="spacer" />
              {chat.backend !== "mock" && (
                <input
                  className="strength-input"
                  type="number"
                  value={chat.strength}
                  onChange={(event) => chat.setStrength(Number(event.target.value))}
                  step={0.1}
                  min={0}
                  title="Steering strength"
                />
              )}
              <button className="send-button" onClick={chat.send} disabled={!chat.input.trim() || chat.thinking}>Send</button>
            </div>
          </div>
        </footer>
      </main>
      <RightRail chat={chat} />
    </div>
  );
}

export default function App() {
  return <Workspace />;
}
