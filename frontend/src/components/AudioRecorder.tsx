import { Mic, Play, RotateCcw, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";

const PREFERRED_MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];

function chooseMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") {
    return undefined;
  }
  return PREFERRED_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
}

interface AudioRecorderProps {
  onBlobChange: (blob: Blob | null) => void;
  disabled?: boolean;
}

export function AudioRecorder({ onBlobChange, disabled = false }: AudioRecorderProps) {
  const [hasPermission, setHasPermission] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [playbackUrl, setPlaybackUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const playbackUrlRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      if (playbackUrlRef.current) URL.revokeObjectURL(playbackUrlRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  function setNextPlaybackUrl(url: string | null) {
    if (playbackUrlRef.current) URL.revokeObjectURL(playbackUrlRef.current);
    playbackUrlRef.current = url;
    setPlaybackUrl(url);
  }

  function activeStream() {
    const stream = streamRef.current;
    if (!stream) return null;
    return stream.getTracks().some((track) => track.readyState === "live") ? stream : null;
  }

  async function requestMicrophone() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not support microphone recording.");
      return null;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setHasPermission(true);
      return stream;
    } catch {
      setError("Microphone permission was not granted.");
      return null;
    }
  }

  async function startRecording() {
    setError(null);
    onBlobChange(null);
    setNextPlaybackUrl(null);

    const stream = activeStream() ?? (await requestMicrophone());
    if (!stream) return;

    chunksRef.current = [];
    const mimeType = chooseMimeType();
    let recorder: MediaRecorder;
    try {
      recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    } catch {
      streamRef.current = null;
      setHasPermission(false);
      setError("Could not start recording. Please enable the microphone again.");
      return;
    }

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };

    recorder.onstop = () => {
      const blobType = mimeType ?? chunksRef.current[0]?.type ?? "audio/webm";
      const blob = new Blob(chunksRef.current, { type: blobType });
      if (blob.size === 0) {
        setError("The recording is empty. Please redo it.");
        onBlobChange(null);
        return;
      }
      const nextUrl = URL.createObjectURL(blob);
      setNextPlaybackUrl(nextUrl);
      onBlobChange(blob);
    };

    recorderRef.current = recorder;
    recorder.start();
    setElapsedSec(0);
    setIsRecording(true);
    timerRef.current = window.setInterval(() => {
      setElapsedSec((value) => value + 1);
    }, 1000);
  }

  function stopRecording() {
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
  }

  function redo() {
    if (isRecording) stopRecording();
    chunksRef.current = [];
    setNextPlaybackUrl(null);
    setElapsedSec(0);
    setError(null);
    onBlobChange(null);
  }

  return (
    <div className="recorder">
      <div className="recorder-actions">
        {!hasPermission && (
          <button className="button secondary" type="button" onClick={requestMicrophone} disabled={disabled}>
            <Mic size={18} aria-hidden="true" />
            Enable Microphone
          </button>
        )}
        <button
          className="button primary"
          type="button"
          onClick={startRecording}
          disabled={disabled || isRecording}
          title="Record"
        >
          <Mic size={18} aria-hidden="true" />
          Record
        </button>
        <button
          className="button danger"
          type="button"
          onClick={stopRecording}
          disabled={disabled || !isRecording}
          title="Stop"
        >
          <Square size={18} aria-hidden="true" />
          Stop
        </button>
        <button
          className="button secondary"
          type="button"
          onClick={redo}
          disabled={disabled || (!playbackUrl && !isRecording)}
          title="Redo"
        >
          <RotateCcw size={18} aria-hidden="true" />
          Redo
        </button>
      </div>

      <div className="recording-state" aria-live="polite">
        {isRecording ? <span className="status-dot" /> : <Play size={16} aria-hidden="true" />}
        <span>{isRecording ? `Recording ${elapsedSec}s` : playbackUrl ? "Ready for playback" : "No recording yet"}</span>
      </div>

      {playbackUrl && <audio className="playback" controls src={playbackUrl} />}
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}
