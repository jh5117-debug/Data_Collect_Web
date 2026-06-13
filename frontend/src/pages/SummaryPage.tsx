import { CheckCircle2, Loader2 } from "lucide-react";
import { useMemo, useState } from "react";
import { createParticipant, createSession, submitSession, uploadClip } from "../api";
import { BackButton } from "../components/BackButton";
import type { ParticipantMetadata, RecordingProgressState, RecordingStats, UploadedRecordingRow } from "../types";

interface SummaryPageProps {
  userEmail: string;
  participantMetadata: ParticipantMetadata | null;
  progress: RecordingProgressState;
  stats: RecordingStats;
  onProgressChange: (progress: RecordingProgressState) => void;
  onBack: () => void;
  onSubmitted: () => void;
}

export function SummaryPage({
  userEmail,
  participantMetadata,
  progress,
  stats,
  onProgressChange,
  onBack,
  onSubmitted
}: SummaryPageProps) {
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const [participantId, setParticipantId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const liveStats = useMemo<RecordingStats>(() => {
    const qcWarnings = progress.uploadedRows.filter(
      (row) => row.auto_qc_status && row.auto_qc_status !== "auto_accepted"
    ).length;
    const positiveRecordings = progress.uploadedRows.filter((row) => !row.is_negative).length;
    const negativeRecordings = progress.uploadedRows.filter((row) => row.is_negative).length;
    return {
      ...stats,
      uploadedClips: progress.uploadedRows.length,
      failedUploads: progress.failedUploads,
      qcWarnings,
      positiveRecordings,
      negativeRecordings
    };
  }, [progress.failedUploads, progress.uploadedRows, stats]);

  async function handleSubmitSession() {
    setSubmitting(true);
    setError(null);
    setStatusText("Uploading recordings...");
    try {
      if (!participantMetadata) {
        throw new Error("Missing participant metadata.");
      }
      let activeParticipantId = participantId;
      let activeSessionId = sessionId;
      if (!activeParticipantId || !activeSessionId) {
        setStatusText("Creating participant session...");
        const participant = await createParticipant({
          user_email: userEmail,
          english_native_speaker: participantMetadata.english_native_speaker,
          recording_device_type: participantMetadata.recording_device_type
        });
        const session = await createSession({
          participant_id: participant.participant_id,
          batch_id: "vigil_batch_v0_1"
        });
        activeParticipantId = participant.participant_id;
        activeSessionId = session.session_id;
        setParticipantId(activeParticipantId);
        setSessionId(activeSessionId);
      }

      let nextRows = [...progress.uploadedRows];
      for (const row of progress.uploadedRows) {
        if (row.upload_status === "uploaded") continue;
        setStatusText(`Uploading ${row.transcript}, take ${row.take_number}...`);
        nextRows = nextRows.map((item) =>
          item.row_id === row.row_id ? { ...item, upload_status: "uploading" } : item
        );
        onProgressChange({ ...progress, uploadedRows: nextRows });

        const result = await uploadClip({
          blob: row.blob,
          participant_id: activeParticipantId,
          session_id: activeSessionId,
          prompt_id: row.prompt_id,
          prompt_group: row.prompt_group,
          transcript: row.transcript,
          clip_type: "normal"
        });

        nextRows = nextRows.map((item) =>
          item.row_id === row.row_id
            ? {
                ...item,
                clip_id: result.clip_id,
                prompt_group: result.prompt_group as typeof item.prompt_group,
                transcript: result.transcript,
                normalized_transcript: result.normalized_transcript,
                contains_vigil: result.contains_vigil,
                wake_intent: result.wake_intent,
                is_negative: result.is_negative,
                upload_status: "uploaded",
                auto_qc_status: result.auto_qc_status,
                auto_qc_flags: result.auto_qc_flags,
                segmentation_status: result.segmentation_status,
                detected_segment_count: result.detected_segment_count
              }
            : item
        );
        onProgressChange({ ...progress, uploadedRows: nextRows });
      }
      setStatusText("Finalizing session...");
      await submitSession(activeSessionId);
      setSubmitted(true);
      setStatusText("Session submitted.");
    } catch (err) {
      const nextRows = progress.uploadedRows.map((row) =>
        row.upload_status === "uploading" ? { ...row, upload_status: "failed" as const } : row
      );
      onProgressChange({
        ...progress,
        uploadedRows: nextRows,
        failedUploads: progress.failedUploads + 1
      });
      setError(err instanceof Error ? err.message : "Could not submit session.");
      setStatusText(null);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="shell narrow">
      <section className="form-panel">
        <BackButton onClick={onBack} disabled={submitting} />
        <p className="eyebrow">Session Summary</p>
        <h1>{submitted ? "Thank you. Your session has been submitted." : "Review Session"}</h1>
        <div className="summary-grid">
          <div><span>Total recordings</span><strong>{liveStats.uploadedClips}</strong></div>
          <div><span>Positive recordings</span><strong>{liveStats.positiveRecordings}</strong></div>
          <div><span>Negative recordings</span><strong>{liveStats.negativeRecordings}</strong></div>
          <div><span>Failed uploads</span><strong>{liveStats.failedUploads}</strong></div>
          <div><span>Auto QC warnings</span><strong>{liveStats.qcWarnings}</strong></div>
        </div>
        <RecordingRows rows={progress.uploadedRows} />
        {statusText && <p className="success-text">{statusText}</p>}
        {error && <p className="error-text">{error}</p>}
        {!submitted && (
          <button className="button primary wide" type="button" onClick={handleSubmitSession} disabled={submitting}>
            {submitting ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <CheckCircle2 size={18} aria-hidden="true" />}
            Upload All & Submit Session
          </button>
        )}
        {submitted && (
          <button className="button primary wide" type="button" onClick={onSubmitted}>
            Back to Workspace
          </button>
        )}
      </section>
    </main>
  );
}

function RecordingRows({ rows }: { rows: UploadedRecordingRow[] }) {
  return (
    <section className="embedded-table">
      <div className="section-title">
        <h2>Recordings</h2>
        <span>{rows.length} clips</span>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Transcript</th>
              <th>Prompt Group</th>
              <th>Take</th>
              <th>Playback</th>
              <th>Upload</th>
              <th>QC</th>
              <th>Flags</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.row_id}>
                <td>{row.transcript}</td>
                <td>{row.prompt_group}</td>
                <td>{row.take_number}</td>
                <td><audio className="table-audio" controls src={row.playback_url} /></td>
                <td>{row.clip_id ?? row.upload_status}</td>
                <td>{row.auto_qc_status ?? "Pending final upload"}</td>
                <td>{row.auto_qc_flags.length > 0 ? row.auto_qc_flags.join(", ") : "None"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
