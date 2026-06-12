import { ArrowRight, Loader2 } from "lucide-react";
import { FormEvent, useState } from "react";
import { getPrompts } from "../api";
import { BackButton } from "../components/BackButton";
import type { EnglishNativeSpeaker, ParticipantMetadata, Prompt, RecordingDeviceType } from "../types";

interface ParticipantPageProps {
  onBack: () => void;
  userEmail: string;
  onReady: (metadata: ParticipantMetadata, prompts: Prompt[]) => void;
}

const ENGLISH_OPTIONS: Array<{ value: EnglishNativeSpeaker; label: string }> = [
  { value: "native_english_speaker", label: "Native English speaker" },
  { value: "non_native_english_speaker", label: "Non-native English speaker" },
  { value: "prefer_not_to_say", label: "Prefer not to say" }
];

const DEVICE_OPTIONS: Array<{ value: RecordingDeviceType; label: string }> = [
  { value: "smartphone", label: "Smartphone" },
  { value: "laptop_builtin_microphone", label: "Laptop built-in microphone" },
  { value: "webcam_microphone", label: "Webcam microphone" },
  { value: "headset_or_airpods", label: "Headset or AirPods" },
  { value: "attached_external_microphone", label: "Attached external microphone" },
  { value: "other", label: "Other" },
  { value: "not_sure", label: "Not sure" }
];

export function ParticipantPage({ onBack, userEmail, onReady }: ParticipantPageProps) {
  const [englishNativeSpeaker, setEnglishNativeSpeaker] =
    useState<EnglishNativeSpeaker>("native_english_speaker");
  const [recordingDeviceType, setRecordingDeviceType] =
    useState<RecordingDeviceType>("laptop_builtin_microphone");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const prompts = await getPrompts();
      onReady(
        {
          english_native_speaker: englishNativeSpeaker,
          recording_device_type: recordingDeviceType
        },
        prompts
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load prompts.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell narrow">
      <form className="form-panel" onSubmit={handleSubmit}>
        <BackButton onClick={onBack} disabled={loading} />
        <h1>Participant Details</h1>
        <label className="field">
          <span>Are you a native English speaker?</span>
          <select value={englishNativeSpeaker} onChange={(event) => setEnglishNativeSpeaker(event.target.value as EnglishNativeSpeaker)}>
            {ENGLISH_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Which device are you using to record?</span>
          <select value={recordingDeviceType} onChange={(event) => setRecordingDeviceType(event.target.value as RecordingDeviceType)}>
            {DEVICE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        {error && <p className="error-text">{error}</p>}
        <button className="button primary wide" type="submit" disabled={loading}>
          {loading ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <ArrowRight size={18} aria-hidden="true" />}
          Continue
        </button>
      </form>
    </main>
  );
}
