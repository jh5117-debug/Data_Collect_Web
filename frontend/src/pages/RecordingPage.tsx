import { ArrowRight, CheckCircle2 } from "lucide-react";
import { useMemo, useState } from "react";
import { AudioRecorder } from "../components/AudioRecorder";
import { BackButton } from "../components/BackButton";
import { ProgressBar } from "../components/ProgressBar";
import { PromptCard } from "../components/PromptCard";
import type { Prompt, RecordingProgressState, RecordingStats, UploadedRecordingRow } from "../types";

interface RecordingPageProps {
  prompts: Prompt[];
  progress: RecordingProgressState;
  onProgressChange: (progress: RecordingProgressState) => void;
  onBack: () => void;
  onFinished: (stats: RecordingStats) => void;
}

const MIN_RECORDINGS_PER_PROMPT = 2;

export function RecordingPage({
  prompts,
  progress,
  onProgressChange,
  onBack,
  onFinished
}: RecordingPageProps) {
  const [blob, setBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recorderKey, setRecorderKey] = useState(0);

  const currentIndex = Math.min(progress.currentPromptIndex, Math.max(prompts.length - 1, 0));
  const prompt = prompts[currentIndex];
  const rows = progress.uploadedRows;
  const rowsForCurrentPrompt = prompt
    ? rows.filter((row) => row.prompt_id === prompt.prompt_id)
    : [];
  const currentPromptReady = rowsForCurrentPrompt.length >= MIN_RECORDINGS_PER_PROMPT;
  const isLastPrompt = currentIndex >= prompts.length - 1;
  const stats = useMemo<RecordingStats>(() => {
    const completedPrompts = prompts.filter(
      (item) => rows.filter((row) => row.prompt_id === item.prompt_id).length >= MIN_RECORDINGS_PER_PROMPT
    ).length;
    const qcWarnings = rows.filter((row) => row.auto_qc_status && row.auto_qc_status !== "auto_accepted").length;
    const generatedSegments = rows.reduce((sum, row) => sum + row.detected_segment_count, 0);
    return {
      completedPrompts,
      uploadedClips: rows.length,
      failedUploads: progress.failedUploads,
      qcWarnings,
      generatedSegments
    };
  }, [prompts, progress.failedUploads, rows]);

  if (!prompt) {
    return (
      <main className="shell narrow">
        <section className="form-panel">
          <h1>No prompts loaded</h1>
          <p className="instruction">Please refresh and try again.</p>
        </section>
      </main>
    );
  }

  function acceptRecording() {
    if (!blob) return;
    setError(null);
    const takeNumber = rowsForCurrentPrompt.length + 1;
    const row: UploadedRecordingRow = {
      row_id: `${prompt.prompt_id}-${Date.now()}-${takeNumber}`,
      prompt_id: prompt.prompt_id,
      target_phrase: prompt.target_phrase,
      take_number: takeNumber,
      blob,
      playback_url: URL.createObjectURL(blob),
      upload_status: "local_only",
      auto_qc_flags: [],
      detected_segment_count: 0
    };
    onProgressChange({
      ...progress,
      uploadedRows: [...rows, row]
    });
    setBlob(null);
    setRecorderKey((value) => value + 1);
  }

  function handleBack() {
    setBlob(null);
    setError(null);
    setRecorderKey((value) => value + 1);
    if (currentIndex > 0) {
      onProgressChange({
        ...progress,
        currentPromptIndex: currentIndex - 1
      });
      return;
    }
    onBack();
  }

  function handleNext() {
    if (!currentPromptReady) return;
    setBlob(null);
    setError(null);
    setRecorderKey((value) => value + 1);
    if (isLastPrompt) {
      onFinished(stats);
      return;
    }
    onProgressChange({
      ...progress,
      currentPromptIndex: currentIndex + 1
    });
  }

  return (
    <main className="shell recorder-shell">
      <section className="recording-layout">
        <div className="recording-header">
          <div>
            <BackButton onClick={handleBack} />
            <p className="eyebrow">Prompt {currentIndex + 1} / {prompts.length}</p>
            <h1>Record Prompt</h1>
          </div>
          <ProgressBar current={currentIndex + 1} total={prompts.length} />
        </div>

        <PromptCard prompt={prompt} />

        <section className="form-panel recording-panel">
          <p className="recording-guidance">
            Please record this phrase at least {MIN_RECORDINGS_PER_PROMPT} times. You may add more recordings if you want.
          </p>
          <AudioRecorder key={`${prompt.prompt_id}-${recorderKey}`} onBlobChange={setBlob} />
          {error && <p className="error-text">{error}</p>}
          <div className="button-row">
            <button
              className="button primary"
              type="button"
              onClick={acceptRecording}
              disabled={!blob}
            >
              <CheckCircle2 size={18} aria-hidden="true" />
              Accept Recording
            </button>
            <button className="button secondary" type="button" onClick={handleNext} disabled={!currentPromptReady}>
              {isLastPrompt ? "Finish Recording" : "Next Prompt"}
              <ArrowRight size={18} aria-hidden="true" />
            </button>
          </div>
          <p className="recording-count">
            Current phrase recordings: {rowsForCurrentPrompt.length} / {MIN_RECORDINGS_PER_PROMPT} minimum
          </p>
          <RecordingRowsTable rows={rows} />
        </section>
      </section>
    </main>
  );
}

function RecordingRowsTable({ rows }: { rows: UploadedRecordingRow[] }) {
  return (
    <section className="embedded-table">
      <div className="section-title">
        <h2>Uploaded Recordings</h2>
        <span>{rows.length} clips</span>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Phrase</th>
              <th>Take</th>
              <th>Playback</th>
              <th>Clip</th>
              <th>QC</th>
              <th>Flags</th>
              <th>Segments</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={7}>No accepted recordings yet.</td>
              </tr>
            )}
            {rows.map((row) => (
              <tr key={row.row_id}>
                <td>{row.target_phrase}</td>
                <td>{row.take_number}</td>
                <td><audio className="table-audio" controls src={row.playback_url} /></td>
                <td>{row.clip_id ?? "Not uploaded yet"}</td>
                <td>{row.auto_qc_status ?? row.upload_status}</td>
                <td>{row.auto_qc_flags.length > 0 ? row.auto_qc_flags.join(", ") : "None"}</td>
                <td>{row.detected_segment_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
