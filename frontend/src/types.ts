export type EnglishNativeSpeaker =
  | "native_english_speaker"
  | "non_native_english_speaker"
  | "prefer_not_to_say";

export type RecordingDeviceType =
  | "smartphone"
  | "laptop_builtin_microphone"
  | "webcam_microphone"
  | "headset_or_airpods"
  | "attached_external_microphone"
  | "other"
  | "not_sure";

export interface ParticipantMetadata {
  english_native_speaker: EnglishNativeSpeaker;
  recording_device_type: RecordingDeviceType;
}

export interface Prompt {
  prompt_id: string;
  instruction_text: string;
  target_phrase: string;
  display_text: string;
  label_type: string;
  recording_mode: "single" | "repeat";
  target_repetition_count: number;
  contains_vigil: boolean;
  wake_intent: boolean;
  segmentation_required: boolean;
  expected_transcript: string;
  prompt_version: string;
}

export interface UploadResponse {
  status: string;
  clip_id: string;
  auto_qc_status: "auto_accepted" | "auto_rejected" | "flagged_for_review";
  auto_qc_flags: string[];
  segmentation_status: string;
  detected_segment_count: number;
}

export interface RecordingStats {
  completedPrompts: number;
  uploadedClips: number;
  failedUploads: number;
  qcWarnings: number;
  generatedSegments: number;
}

export interface UploadedRecordingRow {
  row_id: string;
  prompt_id: string;
  target_phrase: string;
  take_number: number;
  blob: Blob;
  playback_url: string;
  clip_id?: string;
  upload_status: "local_only" | "uploading" | "uploaded" | "failed";
  auto_qc_status?: UploadResponse["auto_qc_status"];
  auto_qc_flags: string[];
  segmentation_status?: string;
  detected_segment_count: number;
}

export interface RecordingProgressState {
  currentPromptIndex: number;
  uploadedRows: UploadedRecordingRow[];
  failedUploads: number;
}

export interface AdminSummary {
  batch_id: string;
  participants: number;
  sessions: number;
  submitted_sessions: number;
  total_clips: number;
  total_segments: number;
  auto_accepted: number;
  flagged: number;
  rejected: number;
}

export interface FlaggedClip {
  clip_id: string;
  participant_id: string;
  session_id: string;
  prompt_id: string;
  clip_type: string;
  duration_sec: number | null;
  auto_qc_status: string;
  auto_qc_flags: string;
  segmentation_status: string;
  detected_segment_count: number;
  expected_segment_count: number;
  created_at_utc: string;
}

export interface AdminClient {
  email: string | null;
  verified: boolean;
  account_created_at_utc: string | null;
  last_login_at_utc: string | null;
  participant_count: number;
  session_count: number;
  submitted_session_count: number;
  clip_count: number;
  segment_count: number;
}

export interface AdminClip {
  clip_id: string;
  participant_id: string;
  user_email: string | null;
  session_id: string;
  prompt_id: string;
  clip_type: string;
  raw_audio_path: string;
  processed_wav_path: string | null;
  duration_sec: number | null;
  file_size_bytes: number;
  auto_qc_status: string;
  auto_qc_flags: string;
  segmentation_status: string;
  detected_segment_count: number;
  expected_segment_count: number;
  created_at_utc: string;
}

export interface ExportResponse {
  status: string;
  file_name: string;
  download_path: string;
}

export interface AccountSession {
  session_id: string;
  batch_id: string;
  status: string;
  created_at_utc: string;
  submitted_at_utc: string | null;
  clip_count: number;
}

export interface AccountClip {
  clip_id: string;
  session_id: string;
  prompt_id: string;
  clip_type: string;
  duration_sec: number | null;
  file_size_bytes: number;
  auto_qc_status: string;
  auto_qc_flags: string;
  segmentation_status: string;
  detected_segment_count: number;
  expected_segment_count: number;
  created_at_utc: string;
}

export interface AuthSession {
  email: string;
  auth_token: string;
  expires_at_utc: string;
}
