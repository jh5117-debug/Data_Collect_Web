const state = {
  profileId: null,
  displayName: null,
  selectedPrompt: { group: "P1_vigil_only", text: "VIGIL", positive: true },
  draftBlob: null,
  draftUrl: null,
  clips: [],
  calibration: null,
  assistantSessionId: null,
  assistantRecorder: null,
  assistantStream: null,
};

const $ = (id) => document.getElementById(id);

function showScreen(id) {
  document.querySelectorAll(".screen").forEach((el) => el.classList.toggle("active", el.id === id));
  document.querySelectorAll(".step").forEach((el) => el.classList.toggle("active", el.dataset.screen === id));
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
  } catch (error) {
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
  $("profileStatus").textContent = `Ready, ${profile.display_name}`;
  showScreen("onboarding");
}

function selectPrompt(button) {
  document.querySelectorAll(".prompt").forEach((el) => el.classList.remove("active"));
  button.classList.add("active");
  state.selectedPrompt = {
    group: button.dataset.group,
    text: button.dataset.text,
    positive: !button.classList.contains("negative"),
  };
}

async function startRecording(kind) {
  const status = kind === "assistant" ? $("micState") : $("recordingStatus");
  status.textContent = "Starting microphone...";
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  recorder.onstart = () => {
    status.textContent = "Recording — speak now";
  };
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    stream.getTracks().forEach((track) => track.stop());
    if (kind === "assistant") {
      sendAssistantChunk(blob);
      return;
    }
    state.draftBlob = blob;
    if (state.draftUrl) URL.revokeObjectURL(state.draftUrl);
    state.draftUrl = URL.createObjectURL(blob);
    $("playback").src = state.draftUrl;
    $("acceptOnboarding").disabled = false;
    $("deleteDraft").disabled = false;
    status.textContent = "Ready for playback";
  };
  recorder.start();
  return recorder;
}

async function recordOnboarding() {
  $("recordOnboarding").disabled = true;
  $("stopOnboarding").disabled = false;
  state.onboardingRecorder = await startRecording("onboarding");
}

function stopOnboarding() {
  if (state.onboardingRecorder?.state === "recording") state.onboardingRecorder.stop();
  $("recordOnboarding").disabled = false;
  $("stopOnboarding").disabled = true;
}

async function acceptOnboarding() {
  if (!state.profileId || !state.draftBlob) return;
  const form = new FormData();
  form.append("profile_id", state.profileId);
  form.append("prompt_group", state.selectedPrompt.group);
  form.append("transcript", state.selectedPrompt.text);
  form.append("is_positive", state.selectedPrompt.positive ? "true" : "false");
  form.append("accepted", "true");
  form.append("file", state.draftBlob, "onboarding.webm");
  const clip = await jsonFetch("/api/onboarding/clip", { method: "POST", body: form });
  state.clips.push(clip);
  state.draftBlob = null;
  $("acceptOnboarding").disabled = true;
  $("deleteDraft").disabled = true;
  $("recordingStatus").textContent = "Accepted";
  renderClips();
}

function deleteDraft() {
  state.draftBlob = null;
  if (state.draftUrl) URL.revokeObjectURL(state.draftUrl);
  $("playback").removeAttribute("src");
  $("acceptOnboarding").disabled = true;
  $("deleteDraft").disabled = true;
  $("recordingStatus").textContent = "Deleted";
}

function renderClips() {
  $("clipList").innerHTML = "";
  state.clips.forEach((clip) => {
    const li = document.createElement("li");
    li.textContent = `${clip.prompt_group} — ${clip.transcript || "audio"} — ${clip.is_positive ? "positive" : "negative"}`;
    $("clipList").appendChild(li);
  });
  const positives = state.clips.filter((clip) => clip.is_positive).length;
  $("calibrateButton").disabled = positives < 3;
}

async function calibrate() {
  const result = await jsonFetch("/api/onboarding/calibrate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: state.profileId }),
  });
  state.calibration = result;
  $("calibrationResult").innerHTML = Object.entries({
    support: result.support_count,
    method: result.method || "none",
    bias: Number(result.bias || 0).toFixed(4),
    active: result.calibration_active ? "yes" : "no",
  })
    .map(([key, value]) => `<div><span>${key}</span><strong>${value}</strong></div>`)
    .join("");
  showScreen("calibration");
}

async function startAssistant() {
  const session = await jsonFetch("/api/assistant/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: state.profileId }),
  });
  state.assistantSessionId = session.assistant_session_id;
  $("assistantState").textContent = "LISTENING";
  $("startAssistant").disabled = true;
  $("stopAssistant").disabled = false;
  $("resetAssistant").disabled = false;
  state.assistantRecorder = await startRecording("assistant");
  state.assistantInterval = setInterval(() => {
    if (state.assistantRecorder?.state === "recording") {
      state.assistantRecorder.stop();
      startRecording("assistant").then((rec) => {
        state.assistantRecorder = rec;
      });
    }
  }, 2500);
}

async function sendAssistantChunk(blob) {
  if (!state.assistantSessionId) return;
  const form = new FormData();
  form.append("profile_id", state.profileId);
  form.append("assistant_session_id", state.assistantSessionId);
  form.append("file", blob, "assistant.webm");
  const result = await jsonFetch("/api/assistant/chunk", { method: "POST", body: form });
  $("transcriptPanel").textContent = result.rolling_transcript || "";
  $("stage1Score").textContent = Number(result.stage1_score || 0).toFixed(4);
  $("stage2Score").textContent = result.stage2_score == null ? "-" : Number(result.stage2_score).toFixed(4);
  $("thresholds").textContent = `${Number(result.theta_1).toFixed(3)} / ${Number(result.theta_2).toFixed(3)}`;
  $("assistantState").textContent = result.assistant_state;
  $("calibrationActive").textContent = state.calibration?.calibration_active ? "active" : "inactive";
  $("extraEncoder").textContent = result.debug.qwen_extra_encoder_forward ? "yes" : "no";
  $("debugPanel").textContent = JSON.stringify(result.debug, null, 2);
  $("detectedBanner").classList.toggle("hidden", !result.trigger_detected);
}

async function stopAssistant() {
  clearInterval(state.assistantInterval);
  if (state.assistantRecorder?.state === "recording") state.assistantRecorder.stop();
  await jsonFetch("/api/assistant/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: state.profileId, assistant_session_id: state.assistantSessionId }),
  });
  $("micState").textContent = "IDLE";
  $("assistantState").textContent = "IDLE";
  $("startAssistant").disabled = false;
  $("stopAssistant").disabled = true;
}

async function resetAssistant() {
  await jsonFetch("/api/assistant/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: state.profileId, assistant_session_id: state.assistantSessionId }),
  });
  $("assistantState").textContent = "IDLE";
  $("detectedBanner").classList.add("hidden");
}

document.querySelectorAll(".step").forEach((button) => button.addEventListener("click", () => showScreen(button.dataset.screen)));
document.querySelectorAll("[data-screen-target]").forEach((button) => button.addEventListener("click", () => showScreen(button.dataset.screenTarget)));
document.querySelectorAll(".prompt").forEach((button) => button.addEventListener("click", () => selectPrompt(button)));
$("createProfile").addEventListener("click", createProfile);
$("recordOnboarding").addEventListener("click", recordOnboarding);
$("stopOnboarding").addEventListener("click", stopOnboarding);
$("acceptOnboarding").addEventListener("click", acceptOnboarding);
$("deleteDraft").addEventListener("click", deleteDraft);
$("calibrateButton").addEventListener("click", calibrate);
$("startAssistant").addEventListener("click", startAssistant);
$("stopAssistant").addEventListener("click", stopAssistant);
$("resetAssistant").addEventListener("click", resetAssistant);
refreshHealth();
