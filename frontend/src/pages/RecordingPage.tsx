import { CheckCircle2, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { deleteAccountClip } from "../api";
import { AudioRecorder } from "../components/AudioRecorder";
import { BackButton } from "../components/BackButton";
import { containsExactVigil, normalizeTranscript, PROMPT_GROUPS, type PromptGroupConfig } from "../promptGroups";
import type { Prompt, PromptGroupId, RecordingProgressState, RecordingStats, UploadedRecordingRow } from "../types";

interface RecordingPageProps {
  prompts: Prompt[];
  progress: RecordingProgressState;
  userEmail: string;
  authToken: string;
  onProgressChange: (progress: RecordingProgressState) => void;
  onBack: () => void;
  onFinished: (stats: RecordingStats) => void;
}

type BlobMap = Partial<Record<PromptGroupId, Blob | null>>;
type TextMap = Record<PromptGroupId, string>;
type ErrorMap = Partial<Record<PromptGroupId, string | null>>;
type KeyMap = Record<PromptGroupId, number>;

const INITIAL_TEXT: TextMap = {
  P1_vigil_only: "VIGIL",
  P2_phrase_plus_vigil: "",
  P3_vigil_plus_phrase: "",
  P4_negative: ""
};

const INITIAL_KEYS: KeyMap = {
  P1_vigil_only: 0,
  P2_phrase_plus_vigil: 0,
  P3_vigil_plus_phrase: 0,
  P4_negative: 0
};

export function RecordingPage({
  progress,
  userEmail,
  authToken,
  onProgressChange,
  onBack,
  onFinished
}: RecordingPageProps) {
  const [blobByGroup, setBlobByGroup] = useState<BlobMap>({});
  const [transcripts, setTranscripts] = useState<TextMap>(INITIAL_TEXT);
  const [errors, setErrors] = useState<ErrorMap>({});
  const [deletingRowId, setDeletingRowId] = useState<string | null>(null);
  const [recorderKeys, setRecorderKeys] = useState<KeyMap>(INITIAL_KEYS);

  const rows = progress.uploadedRows;
  const stats = useMemo<RecordingStats>(() => {
    const qcWarnings = rows.filter((row) => row.auto_qc_status && row.auto_qc_status !== "auto_accepted").length;
    const positiveRecordings = rows.filter((row) => !row.is_negative).length;
    const negativeRecordings = rows.filter((row) => row.is_negative).length;
    return {
      completedPrompts: PROMPT_GROUPS.filter((group) => rows.some((row) => row.prompt_group === group.id)).length,
      uploadedClips: rows.length,
      failedUploads: progress.failedUploads,
      qcWarnings,
      positiveRecordings,
      negativeRecordings
    };
  }, [progress.failedUploads, rows]);

  function validationMessage(group: PromptGroupConfig, transcriptValue: string): string | null {
    const transcript = normalizeTranscript(group.fixedTranscript ?? transcriptValue);
    if (group.fixedTranscript) return null;
    if (!transcript) return "Enter or choose the exact transcript before recording.";
    if ((group.id === "P2_phrase_plus_vigil" || group.id === "P3_vigil_plus_phrase") && !containsExactVigil(transcript)) {
      return "This prompt should include the word 'Vigil'.";
    }
    if (group.id === "P4_negative" && containsExactVigil(transcript)) {
      return "Negative examples should not contain the exact word 'Vigil'. Please use Prompt 2 or Prompt 3 instead.";
    }
    return null;
  }

  function setGroupTranscript(groupId: PromptGroupId, value: string) {
    setTranscripts((current) => ({ ...current, [groupId]: value }));
    setErrors((current) => ({ ...current, [groupId]: null }));
    setBlobByGroup((current) => ({ ...current, [groupId]: null }));
    setRecorderKeys((current) => ({ ...current, [groupId]: current[groupId] + 1 }));
  }

  function acceptRecording(group: PromptGroupConfig) {
    const blob = blobByGroup[group.id];
    if (!blob) return;

    const message = validationMessage(group, transcripts[group.id]);
    if (message) {
      setErrors((current) => ({ ...current, [group.id]: message }));
      return;
    }

    const normalized = normalizeTranscript(group.fixedTranscript ?? transcripts[group.id]);
    const takeNumber =
      rows.filter(
        (row) => row.prompt_group === group.id && transcriptCountKey(row.normalized_transcript || row.transcript) === transcriptCountKey(normalized)
      ).length + 1;
    const row: UploadedRecordingRow = {
      row_id: `${group.id}-${Date.now()}-${takeNumber}`,
      prompt_id: group.id,
      prompt_group: group.id,
      prompt_title: group.title.replace(/^Prompt \d+ — /, ""),
      transcript: normalized,
      normalized_transcript: normalized,
      contains_vigil: group.contains_vigil,
      wake_intent: group.wake_intent,
      is_negative: group.is_negative,
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
    setBlobByGroup((current) => ({ ...current, [group.id]: null }));
    setErrors((current) => ({ ...current, [group.id]: null }));
    setRecorderKeys((current) => ({ ...current, [group.id]: current[group.id] + 1 }));
  }

  async function deleteRecording(row: UploadedRecordingRow) {
    const confirmed = window.confirm(
      row.clip_id
        ? `Delete clip ${row.clip_id} from this draft and from the server? This cannot be undone.`
        : "Delete this accepted recording from the current draft?"
    );
    if (!confirmed) return;

    setDeletingRowId(row.row_id);
    try {
      if (row.clip_id) {
        await deleteAccountClip(userEmail, row.clip_id, authToken);
      }
      URL.revokeObjectURL(row.playback_url);
      onProgressChange({
        ...progress,
        uploadedRows: renumberTakeNumbers(rows.filter((item) => item.row_id !== row.row_id))
      });
    } catch (err) {
      setErrors((current) => ({
        ...current,
        [row.prompt_group]: err instanceof Error ? err.message : "Could not delete recording."
      }));
    } finally {
      setDeletingRowId(null);
    }
  }

  return (
    <main className="shell recorder-shell">
      <section className="recording-layout">
        <div className="recording-header">
          <div>
            <BackButton onClick={onBack} />
            <p className="eyebrow">Session Draft</p>
            <h1>Recording Workspace</h1>
            <p className="instruction">
              Please record clean voice samples. You can record as many examples as you like in each section.
            </p>
          </div>
          <div className="recording-total">
            <span>Total recordings</span>
            <strong>{rows.length}</strong>
          </div>
        </div>

        <section className="prompt-group-grid">
          {PROMPT_GROUPS.map((group) => {
            const groupRows = renumberTakeNumbers(rows.filter((row) => row.prompt_group === group.id));
            const validation = validationMessage(group, transcripts[group.id]);
            return (
              <PromptGroupCard
                key={group.id}
                group={group}
                rows={groupRows}
                transcript={transcripts[group.id]}
                validation={validation}
                blob={blobByGroup[group.id] ?? null}
                recorderKey={recorderKeys[group.id]}
                error={errors[group.id] ?? null}
                deletingRowId={deletingRowId}
                onTranscriptChange={(value) => setGroupTranscript(group.id, value)}
                onBlobChange={(blob) => setBlobByGroup((current) => ({ ...current, [group.id]: blob }))}
                onAccept={() => acceptRecording(group)}
                onDelete={deleteRecording}
              />
            );
          })}
        </section>

        <section className="form-panel recording-panel">
          <div className="summary-grid compact-summary">
            <div><span>Total recordings</span><strong>{stats.uploadedClips}</strong></div>
            <div><span>Positive recordings</span><strong>{stats.positiveRecordings}</strong></div>
            <div><span>Negative recordings</span><strong>{stats.negativeRecordings}</strong></div>
            <div><span>Failed uploads</span><strong>{stats.failedUploads}</strong></div>
          </div>
          <button className="button primary wide" type="button" onClick={() => onFinished(stats)}>
            <CheckCircle2 size={18} aria-hidden="true" />
            Finish Recording
          </button>
        </section>
      </section>
    </main>
  );
}

function PromptGroupCard({
  group,
  rows,
  transcript,
  validation,
  blob,
  recorderKey,
  error,
  deletingRowId,
  onTranscriptChange,
  onBlobChange,
  onAccept,
  onDelete
}: {
  group: PromptGroupConfig;
  rows: UploadedRecordingRow[];
  transcript: string;
  validation: string | null;
  blob: Blob | null;
  recorderKey: number;
  error: string | null;
  deletingRowId: string | null;
  onTranscriptChange: (value: string) => void;
  onBlobChange: (blob: Blob | null) => void;
  onAccept: () => void;
  onDelete: (row: UploadedRecordingRow) => void;
}) {
  const canRecord = !validation;
  const fixedTranscriptCount = group.fixedTranscript ? countRowsForTranscript(rows, group.fixedTranscript) : 0;
  return (
    <section className={`prompt-group-card ${group.is_negative ? "negative-card" : "positive-card"}`}>
      <div className="prompt-group-card-head">
        <div>
          <h2>{group.title}</h2>
          <p>{group.instruction}</p>
        </div>
        <span className={`count-badge ${countToneClass(rows.length)}`}>{rows.length}</span>
      </div>

      {group.fixedTranscript ? (
        <div className={`fixed-transcript ${countToneClass(fixedTranscriptCount)}`}>
          <div className="fixed-transcript-head">
            <span>Transcript</span>
            <span className={`transcript-count-pill ${countToneClass(fixedTranscriptCount)}`}>{fixedTranscriptCount}</span>
          </div>
          <strong>{group.fixedTranscript}</strong>
        </div>
      ) : (
        <>
          <div className="example-chips" aria-label={`${group.title} examples`}>
            {group.examples.map((example) => {
              const count = countRowsForTranscript(rows, example);
              const active = transcriptCountKey(transcript) === transcriptCountKey(example);
              return (
                <button
                  className={`chip count-chip ${countToneClass(count)} ${active ? "active-chip" : ""}`}
                  type="button"
                  key={example}
                  onClick={() => onTranscriptChange(example)}
                >
                  <span className="chip-label">{example}</span>
                  <span className="chip-count">{count}</span>
                </button>
              );
            })}
          </div>
          <label className="field compact-field">
            <span>{group.inputLabel}</span>
            <input value={transcript} onChange={(event) => onTranscriptChange(event.target.value)} />
          </label>
          {validation && transcript.trim() && <p className="error-text">{validation}</p>}
          {validation && !transcript.trim() && (
            <p className="helper-text">Choose an example or type the exact transcript before recording.</p>
          )}
        </>
      )}

      <AudioRecorder key={`${group.id}-${recorderKey}`} onBlobChange={onBlobChange} disabled={!canRecord} />
      {error && <p className="error-text">{error}</p>}
      <button className="button primary" type="button" onClick={onAccept} disabled={!blob || !canRecord}>
        <CheckCircle2 size={18} aria-hidden="true" />
        Accept Recording
      </button>

      <RecordingRowsTable rows={rows} deletingRowId={deletingRowId} onDelete={onDelete} />
    </section>
  );
}

function renumberTakeNumbers(rows: UploadedRecordingRow[]): UploadedRecordingRow[] {
  const counts = new Map<string, number>();
  return rows.map((row) => {
    const key = `${row.prompt_group}:${transcriptCountKey(row.normalized_transcript || row.transcript)}`;
    const nextTakeNumber = (counts.get(key) ?? 0) + 1;
    counts.set(key, nextTakeNumber);
    return { ...row, take_number: nextTakeNumber };
  });
}

function transcriptCountKey(value: string): string {
  return normalizeTranscript(value).toLowerCase();
}

function countRowsForTranscript(rows: UploadedRecordingRow[], transcript: string): number {
  const key = transcriptCountKey(transcript);
  return rows.filter((row) => transcriptCountKey(row.normalized_transcript || row.transcript) === key).length;
}

function countToneClass(count: number): string {
  if (count === 0) return "count-zero";
  if (count === 1) return "count-one";
  return "count-many";
}

function uploadStatusLabel(row: UploadedRecordingRow): string {
  if (row.upload_status === "local_only") return "Draft";
  if (row.upload_status === "uploaded") return "Uploaded";
  if (row.upload_status === "uploading") return "Uploading";
  return "Failed";
}

function RecordingRowsTable({
  rows,
  deletingRowId,
  onDelete
}: {
  rows: UploadedRecordingRow[];
  deletingRowId: string | null;
  onDelete: (row: UploadedRecordingRow) => void;
}) {
  const displayRows = renumberTakeNumbers(rows);
  return (
    <section className="embedded-table compact-recordings">
      <div className="section-title">
        <h3>Accepted Recordings</h3>
        <span>{rows.length} clips</span>
      </div>
      <div className="table-scroll">
        <table className="compact-table">
          <thead>
            <tr>
              <th>Transcript</th>
              <th>Playback</th>
              <th>Status</th>
              <th>Delete</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={4}>No accepted recordings yet.</td>
              </tr>
            )}
            {displayRows.map((row) => (
              <tr key={row.row_id}>
                <td>
                  <div className="transcript-cell">
                    <span>{row.transcript}</span>
                    <span className="take-pill">Take {row.take_number}</span>
                  </div>
                </td>
                <td><audio className="table-audio" controls src={row.playback_url} /></td>
                <td>{row.auto_qc_status ?? uploadStatusLabel(row)}</td>
                <td>
                  <button
                    className="button danger icon-button"
                    type="button"
                    onClick={() => onDelete(row)}
                    disabled={deletingRowId === row.row_id || row.upload_status === "uploading"}
                  >
                    <Trash2 size={16} aria-hidden="true" />
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
