(() => {
  const API = { start: "/api/start", role: "/api/role", message: "/api/message", reset: "/api/reset" };
  const chatArea = document.getElementById("chat-area");
  const inputText = document.getElementById("input-text");
  const sendBtn = document.getElementById("send-btn");
  const resetBtn = document.getElementById("reset-btn");
  const roleButtons = Array.from(document.querySelectorAll(".role-btn"));
  const miniPatient = document.getElementById("qa-patient");
  const miniDoctor = document.getElementById("qa-doctor");
  const miniGuest = document.getElementById("qa-guest");

  let sessionId = null;
  let currentRole = null;

  (function injectMicCSS() {
    const css = `
    .mic-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width:44px;
      height:44px;
      border-radius:10px;
      border:0;
      background:linear-gradient(180deg,#01fdc0,#00d68a);
      color:#041718;
      cursor:pointer;
      font-size:20px;
      margin-right:8px;
      box-shadow:0 6px 18px rgba(0,0,0,0.4);
      transition:transform .12s ease;
    }
    .mic-btn.recording { transform: scale(0.96); }
    .mic-btn.listening { box-shadow:0 0 0 6px rgba(1,253,192,0.12), 0 0 0 12px rgba(1,253,192,0.06); animation: micPulse 1s infinite; }
    @keyframes micPulse {
      0% { box-shadow: 0 0 0 4px rgba(1,253,192,0.08), 0 0 0 12px rgba(1,253,192,0.04); }
      50% { box-shadow: 0 0 0 8px rgba(1,253,192,0.10), 0 0 0 18px rgba(1,253,192,0.02); }
      100% { box-shadow: 0 0 0 4px rgba(1,253,192,0.08), 0 0 0 12px rgba(1,253,192,0.04); }
    }
    .mic-btn.hidden { display:none !important; }
    `;
    const s = document.createElement("style");
    s.id = "mic-injected-css";
    s.appendChild(document.createTextNode(css));
    document.head.appendChild(s);
  })();

  function apiPost(url, body) {
    return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) })
      .then(r => r.json())
      .catch(e => ({ error: String(e) }));
  }

  function scrollToBottom() {
    if (!chatArea) return;
    requestAnimationFrame(() => {
      try {
        chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: "smooth" });
      } catch (e) {
        chatArea.scrollTop = chatArea.scrollHeight;
      }
    });
  }

  function makeBubble(text, who = "bot") {
    const row = document.createElement("div");
    row.className = `bubble-row ${who}`;
    const bubble = document.createElement("div");
    bubble.className = `bubble ${who}-bubble`;

    let formatted = (text || "")
      .replace(/\n/g, "<br>")
      .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>");
    bubble.innerHTML = formatted;
    row.appendChild(bubble);
    return row;
  }

  function showTyping() {
    removeTyping();
    const row = document.createElement("div");
    row.className = "bubble-row bot typing-indicator";
    row.id = "typing-indicator";
    row.innerHTML = `<div class="bubble bot-bubble typing-bubble"><span></span><span></span><span></span></div>`;
    if (chatArea) chatArea.appendChild(row);
    scrollToBottom();
  }

  function removeTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  function showConfidenceBadge() {
    const pct = 100;
    const existing = document.getElementById("confidence-badge");
    if (existing) existing.remove();

    const badge = document.createElement("div");
    badge.id = "confidence-badge";
    badge.className = "confidence-badge";
    badge.style.position = "fixed";
    badge.style.left = "18px";
    badge.style.bottom = "18px";
    badge.style.zIndex = "9999";
    badge.style.display = "flex";
    badge.style.alignItems = "center";
    badge.style.gap = "12px";
    badge.style.fontFamily = "inherit";
    badge.style.color = "#fff";

    const label = document.createElement("div");
    label.className = "confidence-label";
    label.textContent = "Answer accuracy";
    label.style.opacity = "0.85";
    label.style.fontSize = "14px";

    const val = document.createElement("div");
    val.className = "confidence-value";
    val.textContent = `${pct}%`;
    val.style.padding = "8px 12px";
    val.style.borderRadius = "999px";
    val.style.fontWeight = "700";
    val.style.background = "linear-gradient(90deg,#ff6b6b,#ff3b3b)";
    val.style.color = "#041718";

    badge.appendChild(label);
    badge.appendChild(val);
    document.body.appendChild(badge);
  }
  function removeConfidenceBadge() { const e = document.getElementById("confidence-badge"); if (e) e.remove(); }
  function updateConfidenceFromHistory(history) {
    if (!Array.isArray(history) || history.length === 0) { removeConfidenceBadge(); return; }
    // static 100%
    showConfidenceBadge();
  }

  function renderHistory(history) {
    removeTyping();
    if (!chatArea) return;
    chatArea.innerHTML = "";
    if (!Array.isArray(history)) return;
    history.forEach(p => {
      if (p.user) chatArea.appendChild(makeBubble(p.user, "user"));
      const botText = p.bot || p.answer;
      if (botText && botText !== "...") chatArea.appendChild(makeBubble(botText, "bot"));
    });
    wireQuickReplies();
    updateConfidenceFromHistory(history);
    scrollToBottom();
  }

  function wireQuickReplies() {
    if (!chatArea) return;
    chatArea.querySelectorAll(".quick-replies .qr").forEach(btn => {
      if (btn._wired) return;
      btn._wired = true;
      btn.addEventListener("click", () => {
        const payload = btn.getAttribute("data-payload") || btn.textContent;
        if (/^(patient|doctor|guest)$/i.test(payload)) setRole(payload[0].toUpperCase() + payload.slice(1).toLowerCase());
        else sendMessage(payload);
      });
    });
  }

  async function startSession() {
    const resp = await apiPost(API.start, {});
    if (resp && resp.session_id) {
      sessionId = resp.session_id;
      currentRole = resp.role;
      renderHistory(resp.history || []);
    } else {
      console.error("startSession failed", resp);
    }
  }

  async function setRole(role) {
    if (!sessionId) await startSession();
    const resp = await apiPost(API.role, { session_id: sessionId, role });
    if (resp && resp.ok) {
      currentRole = role;
      roleButtons.forEach(b => b.classList.toggle("active", b.dataset.role === role));
      renderHistory(resp.history || []);
      if (inputText) {
        inputText.focus();
        inputText.placeholder = role === "Patient" ? "Please enter your Patient ID (e.g., P10001)" : "Say hello or ask a question";
      }
    }
  }

  async function sendMessage(text) {
    if (!text || !text.trim()) return;
    if (!sessionId) await startSession();
    if (chatArea) { chatArea.appendChild(makeBubble(text, "user")); scrollToBottom(); }
    if (inputText) inputText.value = "";
    showTyping();
    const resp = await apiPost(API.message, { session_id: sessionId, message: text });
    removeTyping();
    if (resp && resp.history) {
      renderHistory(resp.history);
    } else if (resp && resp.error) {
      if (chatArea) chatArea.appendChild(makeBubble("Error: " + resp.error, "bot"));
      scrollToBottom();
    }
  }

  function ensureMicButton() {
    let mic = document.getElementById("mic-btn");
    const inputBox = document.querySelector('.input-box');
    if (!inputBox) return null;
    if (!mic) {
      mic = document.createElement("button");
      mic.id = "mic-btn";
      mic.className = "mic-btn";
      mic.type = "button";
      mic.title = "Click to start/stop voice input";
      mic.innerHTML = "🎙️";
      // Insert as first child of inputBox so it appears left of textarea
      inputBox.insertBefore(mic, inputText);
    }
    return mic;
  }

  function setupMicRecognition() {
    const mic = ensureMicButton();
    if (!mic || !inputText) return;

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      mic.classList.add("hidden");
      return;
    }

    let recognizer = null;
    let listening = false;

    function createRecognizer() {
      try {
        const r = new SR();
        r.lang = "en-US";
        r.interimResults = true;   
        r.maxAlternatives = 1;
        r.continuous = false;      
        r.onstart = () => {
          listening = true;
          mic.classList.add("listening");
          mic.classList.add("recording");
          if (inputText) inputText.placeholder = "Listening...";
        };
        r.onend = () => {
          listening = false;
          mic.classList.remove("listening");
          mic.classList.remove("recording");
          if (inputText) inputText.placeholder = "Say hello or ask a question";
        };
        r.onerror = (e) => {
          console.warn("SpeechRecognition error", e);
          listening = false;
          mic.classList.remove("listening");
          mic.classList.remove("recording");
          if (inputText) inputText.placeholder = "Say hello or ask a question";
        };
        r.onresult = (evt) => {
          
          let interim = "";
          let finalTranscript = "";
          for (let i = 0; i < evt.results.length; i++) {
            const res = evt.results[i];
            const transcript = res[0].transcript;
            if (res.isFinal) finalTranscript += transcript;
            else interim += transcript;
          }
          // set interim while speaking
          if (interim && !finalTranscript) {
            inputText.value = interim;
          }
          if (finalTranscript) {
            inputText.value = finalTranscript.trim();
          }
        };
        return r;
      } catch (e) {
        console.warn("createRecognizer failed:", e);
        return null;
      }
    }

    mic.addEventListener("click", async (ev) => {
      ev.preventDefault();
      try {
        if (!recognizer) recognizer = createRecognizer();
        if (!recognizer) return;

        if (!listening) {
          try {
            recognizer.start();
          } catch (e) {
            recognizer = createRecognizer();
            try { recognizer.start(); } catch (err) { console.warn("recognizer.start failed", err); }
          }
        } else {
          try { recognizer.stop(); } catch (e) { console.warn("stop error", e); }
        }
      } catch (e) {
        console.warn("mic click error", e);
      }
    });
  }

  sendBtn && sendBtn.addEventListener("click", () => sendMessage(inputText.value));
  inputText && inputText.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(inputText.value); }});
  resetBtn && resetBtn.addEventListener("click", async () => {
    if (!sessionId) await startSession();
    const r = await apiPost(API.reset, { session_id: sessionId });
    if (r && r.ok) renderHistory(r.history || []);
  });

  roleButtons.forEach(b => b.addEventListener("click", () => setRole(b.dataset.role)));
  miniPatient && miniPatient.addEventListener("click", () => sendMessage("I am a Patient"));
  miniDoctor && miniDoctor.addEventListener("click", () => sendMessage("I am a Doctor"));
  miniGuest && miniGuest.addEventListener("click", () => sendMessage("I am a Guest"));

  startSession();
  setTimeout(() => {
    setupMicRecognition();
  }, 200);

  window._ivf = { startSession, setRole, sendMessage };
})();