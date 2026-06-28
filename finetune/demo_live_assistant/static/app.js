const PREFERRED_MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];

const PROMPT_GROUPS = [
  {
    id: "P1_vigil_only",
    title: "Prompt 1 - VIGIL Only",
    instruction: 'Please say "VIGIL" once per recording.',
    fixedTranscript: "VIGIL",
    examples: [],
    containsVigil: true,
    isNegative: false,
  },
  {
    id: "P2_phrase_plus_vigil",
    title: "Prompt 2 - Phrase/Sentence + VIGIL",
    instruction: 'Please say a phrase or sentence ending with or followed by "VIGIL" in one recording.',
    inputLabel: "Exact phrase/sentence you will say",
    examples: ["Hi VIGIL.", "Hey VIGIL.", "Hello VIGIL.", "Next, VIGIL.", "What's next, VIGIL?", "Am I doing it right, VIGIL?"],
    containsVigil: true,
    isNegative: false,
  },
  {
    id: "P3_vigil_plus_phrase",
    title: "Prompt 3 - VIGIL + Phrase/Sentence",
    instruction: 'Please say "VIGIL" plus a phrase or sentence in one recording.',
    inputLabel: "Exact phrase/sentence you will say",
    examples: ["VIGIL, next.", "VIGIL, go back.", "VIGIL, what's next?", "VIGIL, am I doing it right?"],
    containsVigil: true,
    isNegative: false,
  },
  {
    id: "P4_negative",
    title: "Prompt 4 - Negative Examples",
    instruction: 'Please record confusing common words or sentences. Do not say the exact word "Vigil" in this section.',
    inputLabel: "Exact negative word or sentence you will say",
    examples: [
      "visual",
      "visuals",
      "visible",
      "digital",
      "individual",
      "residual",
      "video",
      "vital",
      "vigilant",
      "This is a visual input.",
      "The video is clear.",
      "The image is visible.",
      "This is a digital system.",
      "The individual is moving.",
      "The vital signs are normal.",
    ],
    containsVigil: false,
    isNegative: true,
  },
];

const state = {
  profileId: null,
  displayName: null,
  clips: [],
  cardState: Object.fromEntries(
    PROMPT_GROUPS.map((group) => [
      group.id,
      {
        transcript: group.fixedTranscript || "",
        draftBlob: null,
        draftUrl: null,
        stream: null,
        recorder: null,
        chunks: [],
        isStarting: false,
        isRecording: false,
        elapsedSec: 0,
        timer: null,
        status: "No recording yet",
        error: null,
      },
    ]),
  ),
  calibration: null,
  assistantSessionId: null,
  assistantRecorder: null,
  assistantStream: null,
  assistantUploadQueue: Promise.resolve(),
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
    .replace(/\bvigil\b/gi, "Vigil");
}

function containsExactVigil(value) {
  return /\bvigil\b/i.test(value);
}

function transcriptCountKey(value) {
  return normalizeTranscript(value).toLowerCase();
}

function groupById(groupId) {
  return PROMPT_GROUPS.find((group) => group.id === groupId);
}

function chooseMimeType() {
  if (typeof MediaRecorder === "undefined") return undefined;
  return PREFERRED_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
}

function validationMessage(group, transcriptValue) {
  const transcript = normalizeTranscript(group.fixedTranscript || transcriptValue);
  if (group.fixedTranscript) return null;
  if (!transcript) return "Choose an example or type the exact transcript before recording.";
  if ((group.id === "P2_phrase_plus_vigil" || group.id === "P3_vigil_plus_phrase") && !containsExactVigil(transcript)) {
    return 'This prompt should include the word "Vigil".';
  }
  if (group.id === "P4_negative" && containsExactVigil(transcript)) {
    return 'Negative examples should not contain the exact word "Vigil".';
  }
  return null;
}

function countRowsForTranscript(groupId, transcript) {
  const key = transcriptCountKey(transcript);
  return state.clips.filter((clip) => clip.prompt_group === groupId && transcriptCountKey(clip.transcript) === key).length;
}

function countToneClass(count) {
  if (count <= 0) return "empty";
  if (count < 3) return "some";
  return "enough";
}

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
  $("profileStatus").textContent = `Ready, ${profile.display_name}`;
  renderPromptCards();
  showScreen("onboarding");
}

function clearCardDraft(groupId) {
  const card = state.cardState[groupId];
  card.draftBlob = null;
  if (card.draftUrl) URL.revokeObjectURL(card.draftUrl);
  card.draftUrl = null;
}

function stopCardTimer(card) {
  if (card.timer) window.clearInterval(card.timer);
  card.timer = null;
}

function stopCardStream(card) {
  card.stream?.getTracks().forEach((track) => track.stop());
  card.stream = null;
}

async function requestMicrophone(card) {
  card.error = null;
  if (!navigator.mediaDevices?.getUserMedia) {
    card.error = "This browser does not support microphone recording.";
    return null;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    card.stream = stream;
    return stream;
  } catch {
    card.error = "Microphone permission was not granted.";
    return null;
  }
}

async function startCardRecording(groupId) {
  const group = groupById(groupId);
  const card = state.cardState[groupId];
  const validation = validationMessage(group, card.transcript);
  if (validation || card.isRecording || card.isStarting) {
    card.error = validation;
    renderPromptCards();
    return;
  }
  clearCardDraft(groupId);
  card.chunks = [];
  card.error = null;
  card.isStarting = true;
  card.status = "Starting microphone. Wait for Recording before speaking.";
  renderPromptCards();

  const stream = await requestMicrophone(card);
  if (!stream) {
    card.isStarting = false;
    card.status = "No recording yet";
    renderPromptCards();
    return;
  }

  const mimeType = chooseMimeType();
  let recorder;
  try {
    recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  } catch {
    stopCardStream(card);
    card.isStarting = false;
    card.status = "No recording yet";
    card.error = "Could not start recording. Please enable the microphone again.";
    renderPromptCards();
    return;
  }

  recorder.onstart = () => {
    card.elapsedSec = 0;
    card.isStarting = false;
    card.isRecording = true;
    card.status = "Recording 0s";
    stopCardTimer(card);
    card.timer = window.setInterval(() => {
      card.elapsedSec += 1;
      card.status = `Recording ${card.elapsedSec}s`;
      renderPromptCards();
    }, 1000);
    renderPromptCards();
  };

  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) card.chunks.push(event.data);
  };

  recorder.onstop = () => {
    stopCardTimer(card);
    card.isStarting = false;
    card.isRecording = false;
    stopCardStream(card);
    const blobType = mimeType || card.chunks[0]?.type || "audio/webm";
    const blob = new Blob(card.chunks, { type: blobType });
    if (blob.size === 0) {
      card.error = "The recording is empty. Please redo it.";
      card.status = "No recording yet";
      renderPromptCards();
      return;
    }
    card.draftBlob = blob;
    card.draftUrl = URL.createObjectURL(blob);
    card.status = "Ready for playback";
    renderPromptCards();
  };

  recorder.onerror = () => {
    stopCardTimer(card);
    stopCardStream(card);
    card.isStarting = false;
    card.isRecording = false;
    card.status = "Recording failed";
    card.error = "Recording failed. Please redo it.";
    renderPromptCards();
  };

  card.recorder = recorder;
  recorder.start();
}

function stopCardRecording(groupId) {
  const card = state.cardState[groupId];
  if (card.recorder?.state === "recording") {
    card.recorder.stop();
  }
  card.isStarting = false;
  renderPromptCards();
}

function redoCardDraft(groupId) {
  const card = state.cardState[groupId];
  if (card.recorder?.state === "recording") {
    card.recorder.stop();
  }
  clearCardDraft(groupId);
  stopCardTimer(card);
  stopCardStream(card);
  card.chunks = [];
  card.elapsedSec = 0;
  card.isStarting = false;
  card.isRecording = false;
  card.status = "No recording yet";
  card.error = null;
  renderPromptCards();
}

async function acceptCardRecording(groupId) {
  if (!state.profileId) return;
  const group = groupById(groupId);
  const card = state.cardState[groupId];
  if (!card.draftBlob) return;
  const validation = validationMessage(group, card.transcript);
  if (validation) {
    card.error = validation;
    renderPromptCards();
    return;
  }
  const transcript = normalizeTranscript(group.fixedTranscript || card.transcript);
  const form = new FormData();
  form.append("profile_id", state.profileId);
  form.append("prompt_group", group.id);
  form.append("transcript", transcript);
  form.append("is_positive", group.isNegative ? "false" : "true");
  form.append("accepted", "true");
  form.append("file", card.draftBlob, `${group.id}.webm`);
  const clip = await jsonFetch("/api/onboarding/clip", { method: "POST", body: form });
  state.clips.push(clip);
  clearCardDraft(groupId);
  card.status = "Accepted";
  card.error = null;
  renderPromptCards();
}

async function deleteAcceptedClip(groupId, clipId) {
  if (!state.profileId || !clipId) return;
  await jsonFetch(`/api/onboarding/clip/${clipId}?profile_id=${encodeURIComponent(state.profileId)}`, { method: "DELETE" });
  state.clips = state.clips.filter((clip) => clip.clip_id !== clipId);
  const card = state.cardState[groupId];
  card.status = "Deleted accepted recording";
  renderPromptCards();
}

function setGroupTranscript(groupId, value) {
  const card = state.cardState[groupId];
  card.transcript = value;
  card.error = null;
  clearCardDraft(groupId);
  card.status = "No recording yet";
  renderPromptCards();
}

function renderPromptCards() {
  const container = $("promptCards");
  if (!container) return;
  container.innerHTML = PROMPT_GROUPS.map((group) => renderPromptCard(group)).join("");
  renderOnboardingSummary();
}

function renderPromptCard(group) {
  const card = state.cardState[group.id];
  const rows = state.clips.filter((clip) => clip.prompt_group === group.id);
  const validation = validationMessage(group, card.transcript);
  const canRecord = Boolean(state.profileId) && !validation;
  const fixedCount = group.fixedTranscript ? countRowsForTranscript(group.id, group.fixedTranscript) : 0;
  const draftAudio = card.draftUrl ? `<audio class="playback" controls src="${escapeHtml(card.draftUrl)}"></audio>` : "";
  const examples = group.fixedTranscript
    ? `<div class="fixed-transcript ${countToneClass(fixedCount)}"><div><span>Transcript</span><strong>${fixedCount}</strong></div><p>${escapeHtml(group.fixedTranscript)}</p></div>`
    : `
      <div class="example-chips" aria-label="${escapeHtml(group.title)} examples">
        ${group.examples
          .map((example) => {
            const count = countRowsForTranscript(group.id, example);
            const active = transcriptCountKey(card.transcript) === transcriptCountKey(example);
            return `<button class="chip ${countToneClass(count)} ${active ? "active" : ""}" type="button" data-action="choose-example" data-group="${group.id}" data-value="${escapeHtml(example)}"><span>${escapeHtml(example)}</span><strong>${count}</strong></button>`;
          })
          .join("")}
      </div>
      <label class="compact-field">
        <span>${escapeHtml(group.inputLabel)}</span>
        <input data-action="transcript-input" data-group="${group.id}" value="${escapeHtml(card.transcript)}" />
      </label>
      ${validation && card.transcript.trim() ? `<p class="error-text">${escapeHtml(validation)}</p>` : ""}
      ${validation && !card.transcript.trim() ? `<p class="helper-text">${escapeHtml(validation)}</p>` : ""}
    `;
  const rowsHtml = rows.length
    ? `<ul class="clip-list">
        ${rows
          .map(
            (clip) => `<li>
              <div>
                <strong>${escapeHtml(clip.transcript || "audio")}</strong>
                <span>${clip.is_positive ? "positive" : "negative"}</span>
              </div>
              <audio controls src="${escapeHtml(clip.playback_url)}"></audio>
              <button type="button" data-action="delete-clip" data-group="${group.id}" data-clip-id="${escapeHtml(clip.clip_id)}">Delete</button>
            </li>`,
          )
          .join("")}
      </ul>`
    : `<p class="helper-text">No accepted recordings yet.</p>`;
  return `
    <section class="prompt-group-card ${group.isNegative ? "negative-card" : "positive-card"}">
      <div class="prompt-card-head">
        <div>
          <h3>${escapeHtml(group.title)}</h3>
          <p>${escapeHtml(group.instruction)}</p>
        </div>
        <span class="count-badge ${countToneClass(rows.length)}">${rows.length}</span>
      </div>
      ${examples}
      <div class="record-row">
        <button type="button" data-action="record" data-group="${group.id}" ${!canRecord || card.isRecording || card.isStarting ? "disabled" : ""}>Record</button>
        <button type="button" data-action="stop" data-group="${group.id}" ${!card.isRecording ? "disabled" : ""}>Stop</button>
        <button type="button" data-action="accept" data-group="${group.id}" ${!card.draftBlob || !canRecord ? "disabled" : ""}>Accept</button>
        <button type="button" data-action="redo" data-group="${group.id}" ${!card.draftBlob && !card.isRecording && !card.isStarting ? "disabled" : ""}>Delete</button>
      </div>
      ${draftAudio}
      <p class="status">${escapeHtml(card.status)}</p>
      ${card.error ? `<p class="error-text">${escapeHtml(card.error)}</p>` : ""}
      ${rowsHtml}
    </section>
  `;
}

function renderOnboardingSummary() {
  const positives = state.clips.filter((clip) => clip.is_positive).length;
  const negatives = state.clips.filter((clip) => !clip.is_positive).length;
  $("totalClips").textContent = String(state.clips.length);
  $("positiveClips").textContent = String(positives);
  $("negativeClips").textContent = String(negatives);
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
    .map(([key, value]) => `<div><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`)
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
  await startAssistantRecorder();
}

async function startAssistantRecorder() {
  $("micState").textContent = "Starting microphone...";
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mimeType = chooseMimeType();
  const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  state.assistantStream = stream;
  state.assistantRecorder = recorder;

  recorder.onstart = () => {
    $("micState").textContent = "Recording - speak now";
  };
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) queueAssistantChunk(event.data);
  };
  recorder.onerror = () => {
    $("micState").textContent = "Recording failed";
  };
  recorder.onstop = () => {
    stream.getTracks().forEach((track) => track.stop());
    state.assistantStream = null;
  };
  recorder.start(2500);
}

function queueAssistantChunk(blob) {
  state.assistantUploadQueue = state.assistantUploadQueue
    .then(() => sendAssistantChunk(blob))
    .catch((error) => {
      $("debugPanel").textContent = `assistant chunk failed: ${error.message}`;
    });
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
  $("extraEncoder").textContent = result.debug?.qwen_extra_encoder_forward ? "yes" : "no";
  $("qwenWeights").textContent = result.debug?.qwen_weight_instances === 1 ? "one frozen instance" : "unknown";
  $("debugPanel").textContent = JSON.stringify(result.debug, null, 2);
  $("detectedBanner").classList.toggle("hidden", !result.trigger_detected);
}

async function stopAssistant() {
  if (state.assistantRecorder?.state === "recording") state.assistantRecorder.stop();
  state.assistantStream?.getTracks().forEach((track) => track.stop());
  await state.assistantUploadQueue;
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
  $("transcriptPanel").textContent = "";
  $("detectedBanner").classList.add("hidden");
}

document.querySelectorAll(".step").forEach((button) => button.addEventListener("click", () => showScreen(button.dataset.screen)));
document.querySelectorAll("[data-screen-target]").forEach((button) => button.addEventListener("click", () => showScreen(button.dataset.screenTarget)));
$("createProfile").addEventListener("click", createProfile);
$("promptCards").addEventListener("click", (event) => {
  const button = event.target instanceof Element ? event.target.closest("button[data-action]") : null;
  if (!button) return;
  const groupId = button.dataset.group;
  if (button.dataset.action === "choose-example") setGroupTranscript(groupId, button.dataset.value || "");
  if (button.dataset.action === "record") startCardRecording(groupId);
  if (button.dataset.action === "stop") stopCardRecording(groupId);
  if (button.dataset.action === "accept") acceptCardRecording(groupId);
  if (button.dataset.action === "redo") redoCardDraft(groupId);
  if (button.dataset.action === "delete-clip") deleteAcceptedClip(groupId, button.dataset.clipId);
});
$("promptCards").addEventListener("input", (event) => {
  const input = event.target instanceof HTMLInputElement ? event.target : null;
  if (input?.dataset?.action === "transcript-input") {
    const groupId = input.dataset.group;
    state.cardState[groupId].transcript = input.value;
    state.cardState[groupId].error = null;
  }
});
$("calibrateButton").addEventListener("click", calibrate);
$("startAssistant").addEventListener("click", startAssistant);
$("stopAssistant").addEventListener("click", stopAssistant);
$("resetAssistant").addEventListener("click", resetAssistant);
renderPromptCards();
refreshHealth();
