const PREFERRED_MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
const ASSISTANT_SEGMENT_MS = 1200;

const COMMON_PROMPTS = [
  { group: "P1_vigil_only", text: "VIGIL", positive: true },
  { group: "P2_phrase_plus_vigil", text: "Hi VIGIL.", positive: true },
  { group: "P2_phrase_plus_vigil", text: "Hey VIGIL.", positive: true },
  { group: "P2_phrase_plus_vigil", text: "What's next, VIGIL?", positive: true },
  { group: "P3_vigil_plus_phrase", text: "VIGIL, next.", positive: true },
  { group: "P3_vigil_plus_phrase", text: "VIGIL, am I doing it right?", positive: true },
];

const state = {
  profileId: null,
  displayName: null,
  selectedPrompt: COMMON_PROMPTS[0],
  onboardingRecorder: null,
  onboardingStream: null,
  onboardingChunks: [],
  onboardingDraftBlob: null,
  onboardingDraftUrl: null,
  onboardingTimer: null,
  onboardingElapsedSec: 0,
  clips: [],
  calibration: null,
  assistantSessionId: null,
  assistantRecorder: null,
  assistantStream: null,
  assistantActive: false,
  assistantSegmentTimer: null,
  assistantUploadQueue: Promise.resolve(),
  transcriptEvents: [],
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeTranscript(value) {
  return String(value || "")
    .trim()
    .replace(/\s+/g, " ")
    .replace(/\bvigil\b/gi, "VIGIL");
}

function chooseMimeType() {
  if (typeof MediaRecorder === "undefined") return undefined;
  return PREFERRED_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
}

function promptLabel(prompt) {
  return prompt.text;
}

function showScreen(id) {
  document.querySelectorAll(".screen").forEach((el) => el.classList.toggle("active", el.id === id));
  document.querySelectorAll(".step").forEach((el) => el.classList.toggle("active", el.dataset.screen === id));
}

function setAssistantAvailable(available) {
  const ready = Boolean(available);
  $("openAssistantButton").disabled = !ready;
  $("startAssistant").disabled = !ready || state.assistantActive;
  $("calibrationActive").textContent = ready ? "active" : "inactive";
}

function invalidateCalibration() {
  state.calibration = null;
  setAssistantAvailable(false);
  $("calibrationResult").innerHTML = `<div><span>status</span><strong>needs calibration</strong></div>`;
}

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}

async function refreshHealth() {
  try {
    const health = await jsonFetch("/health");
    $("healthBadge").textContent = `${health.mode}: ${health.message}`;
  } catch {
    $("healthBadge").textContent = "offline";
  }
}

async function createProfile() {
  const name = $("nameInput").value.trim();
  if (!name) return;
  const profile = await jsonFetch("/api/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  state.profileId = profile.profile_id;
  state.displayName = profile.display_name;
  state.clips = [];
  state.calibration = null;
  $("profileStatus").textContent = `Ready, ${profile.display_name}`;
  setAssistantAvailable(false);
  renderOnboarding();
  showScreen("onboarding");
}

function setOnboardingStatus(text, error = "") {
  $("recordingStatus").textContent = text;
  $("recordingError").textContent = error;
}

function stopOnboardingTimer() {
  if (state.onboardingTimer) window.clearInterval(state.onboardingTimer);
  state.onboardingTimer = null;
}

function stopOnboardingStream() {
  state.onboardingStream?.getTracks().forEach((track) => track.stop());
  state.onboardingStream = null;
}

function clearOnboardingDraft(status = "No recording yet") {
  state.onboardingDraftBlob = null;
  if (state.onboardingDraftUrl) URL.revokeObjectURL(state.onboardingDraftUrl);
  state.onboardingDraftUrl = null;
  $("playback").removeAttribute("src");
  $("acceptOnboarding").disabled = true;
  $("deleteDraft").disabled = true;
  setOnboardingStatus(status);
}

function selectPrompt(index) {
  state.selectedPrompt = COMMON_PROMPTS[index] || COMMON_PROMPTS[0];
  clearOnboardingDraft("No recording yet");
  renderOnboarding();
}

async function recordOnboarding() {
  if (!state.profileId) return;
  clearOnboardingDraft("Starting microphone. Wait for Recording before speaking.");
  $("recordOnboarding").disabled = true;
  $("stopOnboarding").disabled = false;
  state.onboardingChunks = [];
  state.onboardingElapsedSec = 0;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = chooseMimeType();
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    state.onboardingStream = stream;
    state.onboardingRecorder = recorder;

    recorder.onstart = () => {
      setOnboardingStatus("Recording 0s");
      stopOnboardingTimer();
      state.onboardingTimer = window.setInterval(() => {
        state.onboardingElapsedSec += 1;
        setOnboardingStatus(`Recording ${state.onboardingElapsedSec}s`);
      }, 1000);
    };
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) state.onboardingChunks.push(event.data);
    };
    recorder.onstop = () => {
      stopOnboardingTimer();
      stopOnboardingStream();
      $("recordOnboarding").disabled = false;
      $("stopOnboarding").disabled = true;
      const blob = new Blob(state.onboardingChunks, { type: recorder.mimeType || "audio/webm" });
      if (blob.size === 0) {
        clearOnboardingDraft("No recording yet");
        setOnboardingStatus("No recording yet", "The recording is empty. Please redo it.");
        return;
      }
      state.onboardingDraftBlob = blob;
      state.onboardingDraftUrl = URL.createObjectURL(blob);
      $("playback").src = state.onboardingDraftUrl;
      $("acceptOnboarding").disabled = false;
      $("deleteDraft").disabled = false;
      setOnboardingStatus("Ready for playback");
    };
    recorder.onerror = () => {
      stopOnboardingTimer();
      stopOnboardingStream();
      $("recordOnboarding").disabled = false;
      $("stopOnboarding").disabled = true;
      setOnboardingStatus("Recording failed", "Recording failed. Please redo it.");
    };
    recorder.start();
  } catch {
    $("recordOnboarding").disabled = false;
    $("stopOnboarding").disabled = true;
    setOnboardingStatus("No recording yet", "Microphone permission was not granted.");
  }
}

function stopOnboarding() {
  if (state.onboardingRecorder?.state === "recording") {
    state.onboardingRecorder.stop();
  }
}

async function acceptOnboarding() {
  if (!state.profileId || !state.onboardingDraftBlob) return;
  const prompt = state.selectedPrompt;
  const form = new FormData();
  form.append("profile_id", state.profileId);
  form.append("prompt_group", prompt.group);
  form.append("transcript", normalizeTranscript(prompt.text));
  form.append("is_positive", prompt.positive ? "true" : "false");
  form.append("accepted", "true");
  form.append("file", state.onboardingDraftBlob, `${prompt.group}.webm`);
  const clip = await jsonFetch("/api/onboarding/clip", { method: "POST", body: form });
  state.clips.push(clip);
  invalidateCalibration();
  clearOnboardingDraft("Accepted");
  renderOnboarding();
}

function deleteDraft() {
  clearOnboardingDraft("Deleted");
}

async function deleteAcceptedClip(clipId) {
  if (!state.profileId || !clipId) return;
  await jsonFetch(`/api/onboarding/clip/${clipId}?profile_id=${encodeURIComponent(state.profileId)}`, { method: "DELETE" });
  state.clips = state.clips.filter((clip) => clip.clip_id !== clipId);
  invalidateCalibration();
  renderOnboarding();
}

function renderOnboarding() {
  const promptButtons = COMMON_PROMPTS.map((prompt, index) => {
    const active = prompt.text === state.selectedPrompt.text && prompt.group === state.selectedPrompt.group;
    const count = state.clips.filter((clip) => normalizeTranscript(clip.transcript) === normalizeTranscript(prompt.text)).length;
    return `<button class="prompt-chip ${active ? "active" : ""}" type="button" data-prompt-index="${index}">
      <span>${escapeHtml(promptLabel(prompt))}</span><strong>${count}</strong>
    </button>`;
  }).join("");
  $("promptChoices").innerHTML = promptButtons;
  $("selectedTranscript").textContent = normalizeTranscript(state.selectedPrompt.text);
  $("selectedPromptGroup").textContent = state.selectedPrompt.group.replaceAll("_", " ");

  const positives = state.clips.filter((clip) => clip.is_positive).length;
  $("totalClips").textContent = String(state.clips.length);
  $("positiveClips").textContent = String(positives);
  $("calibrateButton").disabled = positives < 3;

  $("clipList").innerHTML = state.clips.length
    ? state.clips
        .map(
          (clip) => `<li>
            <div><strong>${escapeHtml(clip.transcript || "audio")}</strong><span>${clip.prompt_group}</span></div>
            <audio controls src="${escapeHtml(clip.playback_url)}"></audio>
            <button type="button" data-delete-clip="${escapeHtml(clip.clip_id)}">Delete</button>
          </li>`,
        )
        .join("")
    : `<li class="empty-row">No accepted recordings yet.</li>`;
}

async function calibrate() {
  showScreen("calibration");
  setAssistantAvailable(false);
  $("calibrateButton").disabled = true;
  $("calibrateButton").textContent = "Calibrating...";
  $("calibrationResult").innerHTML = Object.entries({
    status: "extracting voice features",
    support: state.clips.filter((clip) => clip.is_positive).length,
    method: "few-shot prototype",
    active: "loading",
  })
    .map(([key, value]) => `<div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
  try {
    const result = await jsonFetch("/api/onboarding/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile_id: state.profileId }),
    });
    state.calibration = result;
    const cards = {
      status: result.calibration_status || "unknown",
      support: result.support_count,
      method: result.method || "none",
      prototype: result.prototype_embedding_dim ? `${result.prototype_embedding_dim}D` : "none",
      similarity:
        result.support_pairwise_mean_similarity == null ? "-" : Number(result.support_pairwise_mean_similarity).toFixed(3),
      threshold: result.prototype_threshold == null ? "-" : Number(result.prototype_threshold).toFixed(3),
      active: result.calibration_active ? "yes" : "no",
    };
    $("calibrationResult").innerHTML = Object.entries(cards)
      .map(([key, value]) => `<div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`)
      .join("");
    setAssistantAvailable(Boolean(result.calibration_active));
  } catch (error) {
    state.calibration = null;
    $("calibrationResult").innerHTML = Object.entries({
      status: "failed",
      error: error.message,
      active: "no",
    })
      .map(([key, value]) => `<div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`)
      .join("");
    setAssistantAvailable(false);
  } finally {
    const positives = state.clips.filter((clip) => clip.is_positive).length;
    $("calibrateButton").disabled = positives < 3;
    $("calibrateButton").textContent = "Calibrate my VIGIL voice";
  }
}

async function startAssistant() {
  if (!state.calibration?.calibration_active) {
    $("debugPanel").textContent = "Few-shot calibration is required before starting the assistant.";
    setAssistantAvailable(false);
    return;
  }
  const session = await jsonFetch("/api/assistant/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: state.profileId }),
  });
  state.assistantSessionId = session.assistant_session_id;
  state.assistantActive = true;
  state.transcriptEvents = [];
  $("assistantState").textContent = "LISTENING";
  $("startAssistant").disabled = true;
  $("stopAssistant").disabled = false;
  $("resetAssistant").disabled = false;
  $("debugPanel").textContent = "";
  renderRollingTranscript("");
  await startAssistantStream();
  startAssistantSegment();
}

async function startAssistantStream() {
  $("micState").textContent = "Starting microphone...";
  state.assistantStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  $("micState").textContent = "Recording - speak now";
}

function startAssistantSegment() {
  if (!state.assistantActive || !state.assistantStream) return;
  const chunks = [];
  const mimeType = chooseMimeType();
  const recorder = mimeType ? new MediaRecorder(state.assistantStream, { mimeType }) : new MediaRecorder(state.assistantStream);
  state.assistantRecorder = recorder;

  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  recorder.onstop = () => {
    if (chunks.length > 0) {
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      if (blob.size > 0) queueAssistantChunk(blob);
    }
    if (state.assistantActive) {
      window.setTimeout(startAssistantSegment, 50);
    }
  };
  recorder.onerror = () => {
    $("micState").textContent = "Recording segment failed";
    if (state.assistantActive) {
      window.setTimeout(startAssistantSegment, 250);
    }
  };
  recorder.start();
  state.assistantSegmentTimer = window.setTimeout(() => {
    if (recorder.state === "recording") recorder.stop();
  }, ASSISTANT_SEGMENT_MS);
}

function queueAssistantChunk(blob) {
  state.assistantUploadQueue = state.assistantUploadQueue
    .then(() => sendAssistantChunk(blob))
    .catch((error) => {
      $("debugPanel").textContent = `assistant chunk failed: ${error.message}`;
    });
}

function renderRollingTranscript(text) {
  const escaped = escapeHtml(text || "");
  const highlighted = escaped.replace(/\b(VIGIL|Vigil|vigil|Virgil|virgil)\b/g, '<span class="wake-word">$1</span>');
  const events = state.transcriptEvents
    .map((event) => `<div class="activation-line">${escapeHtml(event)}</div>`)
    .join("");
  $("transcriptPanel").innerHTML = [highlighted, events].filter(Boolean).join(events && highlighted ? "\n" : "");
}

async function sendAssistantChunk(blob) {
  if (!state.assistantSessionId) return;
  const form = new FormData();
  form.append("profile_id", state.profileId);
  form.append("assistant_session_id", state.assistantSessionId);
  form.append("file", blob, "assistant.webm");
  const result = await jsonFetch("/api/assistant/chunk", { method: "POST", body: form });
  if (result.trigger_detected) {
    state.transcriptEvents.push("VIGIL Assistant activated");
  }
  renderRollingTranscript(result.rolling_transcript || "");
  $("stage1Score").textContent = Number(result.stage1_score || 0).toFixed(4);
  $("stage2Score").textContent = result.stage2_score == null ? "-" : Number(result.stage2_score).toFixed(4);
  $("thresholds").textContent = `${Number(result.theta_1).toFixed(3)} / ${Number(result.theta_2).toFixed(3)}`;
  $("assistantState").textContent = result.assistant_state;
  $("calibrationActive").textContent = state.calibration?.calibration_active ? "active" : "inactive";
  $("extraEncoder").textContent = result.debug?.qwen_extra_encoder_forward ? "yes" : "no";
  $("qwenWeights").textContent = result.debug?.qwen_weight_instances === 1 ? "one frozen instance" : "unknown";
  $("debugPanel").textContent = JSON.stringify(result.debug, null, 2);
  $("detectedBanner").classList.toggle("hidden", result.assistant_state !== "ASSISTANT_STATE");
}

async function stopAssistant() {
  state.assistantActive = false;
  if (state.assistantSegmentTimer) window.clearTimeout(state.assistantSegmentTimer);
  if (state.assistantRecorder?.state === "recording") state.assistantRecorder.stop();
  state.assistantStream?.getTracks().forEach((track) => track.stop());
  state.assistantStream = null;
  await state.assistantUploadQueue;
  await jsonFetch("/api/assistant/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: state.profileId, assistant_session_id: state.assistantSessionId }),
  });
  $("micState").textContent = "IDLE";
  $("assistantState").textContent = "IDLE";
  $("startAssistant").disabled = !state.calibration?.calibration_active;
  $("stopAssistant").disabled = true;
}

async function resetAssistant() {
  await jsonFetch("/api/assistant/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: state.profileId, assistant_session_id: state.assistantSessionId }),
  });
  $("assistantState").textContent = "IDLE";
  state.transcriptEvents = [];
  renderRollingTranscript("");
  $("detectedBanner").classList.add("hidden");
}

document.querySelectorAll(".step").forEach((button) => button.addEventListener("click", () => showScreen(button.dataset.screen)));
document.querySelectorAll("[data-screen-target]").forEach((button) => button.addEventListener("click", () => showScreen(button.dataset.screenTarget)));
$("createProfile").addEventListener("click", createProfile);
$("promptChoices").addEventListener("click", (event) => {
  const button = event.target instanceof Element ? event.target.closest("button[data-prompt-index]") : null;
  if (button) selectPrompt(Number(button.dataset.promptIndex));
});
$("clipList").addEventListener("click", (event) => {
  const button = event.target instanceof Element ? event.target.closest("button[data-delete-clip]") : null;
  if (button) deleteAcceptedClip(button.dataset.deleteClip);
});
$("recordOnboarding").addEventListener("click", recordOnboarding);
$("stopOnboarding").addEventListener("click", stopOnboarding);
$("acceptOnboarding").addEventListener("click", acceptOnboarding);
$("deleteDraft").addEventListener("click", deleteDraft);
$("calibrateButton").addEventListener("click", calibrate);
$("startAssistant").addEventListener("click", startAssistant);
$("stopAssistant").addEventListener("click", stopAssistant);
$("resetAssistant").addEventListener("click", resetAssistant);
renderOnboarding();
refreshHealth();
