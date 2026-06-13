import type {
  AdminSummary,
  AdminClient,
  AdminClip,
  AccountSession,
  AccountClip,
  EnglishNativeSpeaker,
  ExportResponse,
  FlaggedClip,
  Prompt,
  RecordingDeviceType,
  UploadResponse
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail ?? message;
    } catch {
      // Keep the HTTP status as the error message.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function createParticipant(payload: {
  user_email?: string;
  english_native_speaker: EnglishNativeSpeaker;
  recording_device_type: RecordingDeviceType;
}): Promise<{ participant_id: string }> {
  return request("/api/participants", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function createSession(payload: {
  participant_id: string;
  batch_id: string;
}): Promise<{ session_id: string }> {
  return request("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function getPrompts(): Promise<Prompt[]> {
  return request("/api/prompts");
}

function filenameForBlob(blob: Blob): string {
  if (blob.type.includes("mp4")) return "recording.m4a";
  if (blob.type.includes("ogg")) return "recording.ogg";
  if (blob.type.includes("wav")) return "recording.wav";
  return "recording.webm";
}

export async function uploadClip(payload: {
  blob: Blob;
  participant_id: string;
  session_id: string;
  prompt_id?: string;
  prompt_group?: string;
  transcript?: string;
  clip_type: "normal" | "calibration";
}): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("audio", payload.blob, filenameForBlob(payload.blob));
  formData.append("participant_id", payload.participant_id);
  formData.append("session_id", payload.session_id);
  if (payload.prompt_id) formData.append("prompt_id", payload.prompt_id);
  if (payload.prompt_group) formData.append("prompt_group", payload.prompt_group);
  if (payload.transcript) formData.append("transcript", payload.transcript);
  formData.append("clip_type", payload.clip_type);

  return request("/api/clips", {
    method: "POST",
    body: formData
  });
}

export async function submitSession(sessionId: string): Promise<{ status: string }> {
  return request(`/api/sessions/${sessionId}/submit`, { method: "POST" });
}

export async function getAdminSummary(): Promise<AdminSummary> {
  return request("/api/admin/summary");
}

export async function getFlaggedClips(): Promise<FlaggedClip[]> {
  return request("/api/admin/flagged");
}

export async function getAdminClients(): Promise<AdminClient[]> {
  return request("/api/admin/clients");
}

export async function getAdminClips(): Promise<AdminClip[]> {
  return request("/api/admin/clips");
}

export async function deleteAdminClip(clipId: string): Promise<{ status: string; clip_id: string; deleted_files: string[] }> {
  return request(`/api/admin/clips/${encodeURIComponent(clipId)}`, { method: "DELETE" });
}

export async function createExport(): Promise<ExportResponse> {
  return request("/api/admin/export", { method: "POST" });
}

export async function requestLoginCode(email: string): Promise<{ status: string; dev_code?: string | null }> {
  return request("/api/auth/request-code", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email })
  });
}

export async function verifyLoginCode(email: string, code: string): Promise<{ status: string; email: string; auth_token: string; expires_at_utc: string }> {
  return request("/api/auth/verify-code", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, code })
  });
}

function authHeaders(authToken: string): HeadersInit {
  return { "X-Auth-Token": authToken };
}

export async function getAccountSessions(email: string, authToken: string): Promise<AccountSession[]> {
  return request(`/api/auth/accounts/${encodeURIComponent(email)}/sessions`, {
    headers: authHeaders(authToken)
  });
}

export async function getAccountSessionClips(email: string, sessionId: string, authToken: string): Promise<AccountClip[]> {
  return request(`/api/auth/accounts/${encodeURIComponent(email)}/sessions/${encodeURIComponent(sessionId)}/clips`, {
    headers: authHeaders(authToken)
  });
}

export async function deleteAccountClip(email: string, clipId: string, authToken: string): Promise<{ status: string; clip_id: string; deleted_files: string[] }> {
  return request(`/api/auth/accounts/${encodeURIComponent(email)}/clips/${encodeURIComponent(clipId)}`, {
    method: "DELETE",
    headers: authHeaders(authToken)
  });
}

export function getAccountClipAudioUrl(email: string, clipId: string, authToken: string): string {
  const params = new URLSearchParams({ token: authToken });
  return `${API_BASE_URL}/api/auth/accounts/${encodeURIComponent(email)}/clips/${encodeURIComponent(clipId)}/audio?${params.toString()}`;
}

export function getAdminClipAudioUrl(clipId: string): string {
  return `${API_BASE_URL}/api/admin/clips/${encodeURIComponent(clipId)}/audio`;
}

export async function getAdminClientSessions(email: string): Promise<AccountSession[]> {
  return request(`/api/admin/clients/${encodeURIComponent(email)}/sessions`);
}

export async function getAdminClientClips(email: string): Promise<AdminClip[]> {
  return request(`/api/admin/clients/${encodeURIComponent(email)}/clips`);
}

export async function deleteAdminClient(email: string): Promise<{ status: string; email: string; deleted_files: string[] }> {
  return request(`/api/admin/clients/${encodeURIComponent(email)}`, { method: "DELETE" });
}

export async function getAdminSessionClips(sessionId: string): Promise<AdminClip[]> {
  return request(`/api/admin/sessions/${encodeURIComponent(sessionId)}/clips`);
}

export async function deleteAdminSession(sessionId: string): Promise<{ status: string; session_id: string; deleted_files: string[] }> {
  return request(`/api/admin/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}

export async function deleteAdminClientSessions(email: string): Promise<{ status: string; email: string; deleted_sessions: number; deleted_files: string[] }> {
  return request(`/api/admin/clients/${encodeURIComponent(email)}/sessions`, { method: "DELETE" });
}
