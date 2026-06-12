import { ArrowLeft, ArrowRight, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { deleteAccountClip, getAccountClipAudioUrl, getAccountSessionClips, getAccountSessions } from "../api";
import { BackButton } from "../components/BackButton";
import type { AccountClip, AccountSession } from "../types";

interface AccountPageProps {
  email: string;
  authToken: string;
  onBack: () => void;
  onStart: () => void;
}

export function AccountPage({ email, authToken, onBack, onStart }: AccountPageProps) {
  const [sessions, setSessions] = useState<AccountSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<AccountSession | null>(null);
  const [clips, setClips] = useState<AccountClip[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadSessions() {
    setLoading(true);
    setError(null);
    try {
      setSessions(await getAccountSessions(email, authToken));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load account sessions.");
    } finally {
      setLoading(false);
    }
  }

  async function openSession(session: AccountSession) {
    setSelectedSession(session);
    setLoading(true);
    setError(null);
    try {
      setClips(await getAccountSessionClips(email, session.session_id, authToken));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load session clips.");
    } finally {
      setLoading(false);
    }
  }

  async function handleDeleteClip(clipId: string) {
    const confirmed = window.confirm(`Delete clip ${clipId}? This removes it from your submitted data.`);
    if (!confirmed || !selectedSession) return;
    setError(null);
    try {
      await deleteAccountClip(email, clipId, authToken);
      setClips(await getAccountSessionClips(email, selectedSession.session_id, authToken));
      setSessions(await getAccountSessions(email, authToken));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete clip.");
    }
  }

  useEffect(() => {
    void loadSessions();
  }, [email, authToken]);

  if (selectedSession) {
    return (
      <main className="shell">
        <section className="form-panel">
          <button className="button secondary back-button" type="button" onClick={() => setSelectedSession(null)}>
            <ArrowLeft size={18} aria-hidden="true" />
            Back to Sessions
          </button>
          <p className="eyebrow">Participant Workspace</p>
          <h1>Session {selectedSession.session_id}</h1>
          <p className="instruction">{email}</p>
          {error && <p className="error-text">{error}</p>}
          <div className="embedded-table">
            <div className="section-title">
              <h2>Your Clips</h2>
              <span>{clips.length} clips</span>
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Clip</th>
                    <th>Prompt</th>
                    <th>Playback</th>
                    <th>Status</th>
                    <th>Flags</th>
                    <th>Duration</th>
                    <th>Size</th>
                    <th>Created</th>
                    <th>Delete</th>
                  </tr>
                </thead>
                <tbody>
                  {clips.length === 0 && (
                    <tr>
                      <td colSpan={9}>No clips in this session.</td>
                    </tr>
                  )}
                  {clips.map((clip) => (
                    <tr key={clip.clip_id}>
                      <td>{clip.clip_id}</td>
                      <td>{clip.prompt_id}</td>
                      <td><audio className="table-audio" controls src={getAccountClipAudioUrl(email, clip.clip_id, authToken)} /></td>
                      <td>{clip.auto_qc_status}</td>
                      <td>{clip.auto_qc_flags}</td>
                      <td>{clip.duration_sec?.toFixed(2) ?? "n/a"}s</td>
                      <td>{Math.round(clip.file_size_bytes / 1024)} KB</td>
                      <td>{new Date(clip.created_at_utc).toLocaleString()}</td>
                      <td>
                        <button className="button danger icon-button" type="button" onClick={() => handleDeleteClip(clip.clip_id)}>
                          <Trash2 size={16} aria-hidden="true" />
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="shell narrow">
      <section className="form-panel">
        <BackButton onClick={onBack} disabled={loading} />
        <p className="eyebrow">Participant Workspace</p>
        <h1>Your Sessions</h1>
        <p className="instruction">{email}</p>
        <div className="button-row">
          <button className="button primary" type="button" onClick={onStart}>
            Start New Session
            <ArrowRight size={18} aria-hidden="true" />
          </button>
          <button className="button secondary" type="button" onClick={loadSessions} disabled={loading}>
            <RefreshCw size={18} aria-hidden="true" />
            Refresh
          </button>
        </div>
        {error && <p className="error-text">{error}</p>}
        <div className="embedded-table">
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Batch</th>
                  <th>Session</th>
                  <th>Status</th>
                  <th>Clips</th>
                  <th>Submitted</th>
                </tr>
              </thead>
              <tbody>
                {sessions.length === 0 && (
                  <tr>
                    <td colSpan={5}>No submitted sessions yet.</td>
                  </tr>
                )}
                {sessions.map((session) => (
                  <tr key={session.session_id} className="clickable-row" onClick={() => openSession(session)}>
                    <td>{session.batch_id}</td>
                    <td>{session.session_id}</td>
                    <td>{session.status}</td>
                    <td>{session.clip_count}</td>
                    <td>{session.submitted_at_utc ? new Date(session.submitted_at_utc).toLocaleString() : "Not submitted"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </main>
  );
}
