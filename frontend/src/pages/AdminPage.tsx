import { ArrowLeft, Download, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import {
  createExport,
  deleteAdminClient,
  deleteAdminClientSessions,
  deleteAdminClip,
  deleteAdminSession,
  getAdminClientSessions,
  getAdminClipAudioUrl,
  getAdminClients,
  getAdminSessionClips,
  getAdminSummary,
  getFlaggedClips
} from "../api";
import { BackButton } from "../components/BackButton";
import type { AccountSession, AdminClient, AdminClip, AdminSummary, ExportResponse, FlaggedClip } from "../types";

export function AdminPage() {
  const [summary, setSummary] = useState<AdminSummary | null>(null);
  const [flagged, setFlagged] = useState<FlaggedClip[]>([]);
  const [clients, setClients] = useState<AdminClient[]>([]);
  const [selectedClient, setSelectedClient] = useState<AdminClient | null>(null);
  const [clientSessions, setClientSessions] = useState<AccountSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<AccountSession | null>(null);
  const [sessionClips, setSessionClips] = useState<AdminClip[]>([]);
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function refreshAdminData(activeEmail = selectedClient?.email ?? null, activeSessionId = selectedSession?.session_id ?? null) {
    setLoading(true);
    setError(null);
    try {
      const [nextSummary, nextFlagged, nextClients] = await Promise.all([
        getAdminSummary(),
        getFlaggedClips(),
        getAdminClients()
      ]);
      setSummary(nextSummary);
      setFlagged(nextFlagged);
      setClients(nextClients);

      if (!activeEmail) {
        setSelectedClient(null);
        setClientSessions([]);
        setSelectedSession(null);
        setSessionClips([]);
        return;
      }

      const nextClient = nextClients.find((client) => client.email === activeEmail) ?? null;
      if (!nextClient) {
        setSelectedClient(null);
        setClientSessions([]);
        setSelectedSession(null);
        setSessionClips([]);
        return;
      }

      const nextSessions = await getAdminClientSessions(activeEmail);
      setSelectedClient(nextClient);
      setClientSessions(nextSessions);

      if (!activeSessionId) {
        setSelectedSession(null);
        setSessionClips([]);
        return;
      }

      const nextSession = nextSessions.find((session) => session.session_id === activeSessionId) ?? null;
      setSelectedSession(nextSession);
      setSessionClips(nextSession ? await getAdminSessionClips(activeSessionId) : []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load admin data.");
    } finally {
      setLoading(false);
    }
  }

  async function openClient(client: AdminClient) {
    if (!client.email) return;
    setSelectedClient(client);
    setSelectedSession(null);
    setSessionClips([]);
    await refreshAdminData(client.email, null);
  }

  async function openSession(session: AccountSession) {
    setSelectedSession(session);
    setLoading(true);
    setError(null);
    try {
      setSessionClips(await getAdminSessionClips(session.session_id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load session clips.");
    } finally {
      setLoading(false);
    }
  }

  async function handleDeleteClip(clipId: string) {
    if (!selectedClient?.email || !selectedSession) return;
    const confirmed = window.confirm(`Delete clip ${clipId} and its audio files? This cannot be undone.`);
    if (!confirmed) return;

    setError(null);
    try {
      await deleteAdminClip(clipId);
      await refreshAdminData(selectedClient.email, selectedSession.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete clip.");
    }
  }

  async function handleDeleteSession(session: AccountSession) {
    if (!selectedClient?.email) return;
    const confirmed = window.confirm(
      `Delete session ${session.session_id}, all clips, and all audio files? This cannot be undone.`
    );
    if (!confirmed) return;

    setError(null);
    try {
      await deleteAdminSession(session.session_id);
      setSelectedSession(null);
      setSessionClips([]);
      await refreshAdminData(selectedClient.email, null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete session.");
    }
  }

  async function handleDeleteAllSessions() {
    if (!selectedClient?.email || clientSessions.length === 0) return;
    const confirmed = window.confirm(
      `Delete all ${clientSessions.length} sessions for ${selectedClient.email}, including every clip and audio file? This cannot be undone.`
    );
    if (!confirmed) return;

    setError(null);
    try {
      await deleteAdminClientSessions(selectedClient.email);
      setSelectedSession(null);
      setSessionClips([]);
      await refreshAdminData(selectedClient.email, null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete all sessions.");
    }
  }

  async function handleDeleteAccount() {
    if (!selectedClient?.email) return;
    const confirmed = window.confirm(
      `Delete account ${selectedClient.email}, all sessions, all clips, and all audio files? This cannot be undone.`
    );
    if (!confirmed) return;

    setError(null);
    try {
      await deleteAdminClient(selectedClient.email);
      setSelectedClient(null);
      setClientSessions([]);
      setSelectedSession(null);
      setSessionClips([]);
      await refreshAdminData(null, null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete account.");
    }
  }

  async function handleExport() {
    setError(null);
    try {
      const result = await createExport();
      setExportResult(result);
      window.open(result.download_path, "_blank");
      await refreshAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed.");
    }
  }

  useEffect(() => {
    void refreshAdminData(null, null);
  }, []);

  if (selectedClient && selectedSession) {
    return (
      <main className="shell admin-shell">
        <section className="admin-header">
          <div>
            <button className="button secondary back-button" type="button" onClick={() => setSelectedSession(null)}>
              <ArrowLeft size={18} aria-hidden="true" />
              Back to Account
            </button>
            <p className="eyebrow">Session Detail</p>
            <h1>Session {selectedSession.session_id}</h1>
            <p className="instruction">{selectedClient.email}</p>
          </div>
          <div className="button-row">
            <button
              className="button secondary"
              type="button"
              onClick={() => selectedClient.email && refreshAdminData(selectedClient.email, selectedSession.session_id)}
              disabled={loading}
            >
              <RefreshCw size={18} aria-hidden="true" />
              Refresh
            </button>
            <button className="button danger" type="button" onClick={() => handleDeleteSession(selectedSession)}>
              <Trash2 size={18} aria-hidden="true" />
              Delete Session
            </button>
          </div>
        </section>

        {error && <p className="error-text">{error}</p>}

        <section className="metric-grid">
          <div><span>Status</span><strong>{selectedSession.status}</strong></div>
          <div><span>Clips</span><strong>{sessionClips.length}</strong></div>
          <div><span>Submitted</span><strong>{selectedSession.submitted_at_utc ? "Yes" : "No"}</strong></div>
          <div><span>Batch</span><strong>{selectedSession.batch_id}</strong></div>
        </section>

        <ClipsTable clips={sessionClips} onDeleteClip={handleDeleteClip} />
      </main>
    );
  }

  if (selectedClient) {
    const submittedCount = clientSessions.filter((session) => session.status === "submitted").length;
    const clipCount = clientSessions.reduce((sum, session) => sum + session.clip_count, 0);

    return (
      <main className="shell admin-shell">
        <section className="admin-header">
          <div>
            <button className="button secondary back-button" type="button" onClick={() => setSelectedClient(null)}>
              <ArrowLeft size={18} aria-hidden="true" />
              Back to Admin
            </button>
            <p className="eyebrow">Account Detail</p>
            <h1>{selectedClient.email ?? "Unknown Account"}</h1>
          </div>
          <div className="button-row">
            <button
              className="button secondary"
              type="button"
              onClick={() => selectedClient.email && refreshAdminData(selectedClient.email, null)}
              disabled={loading}
            >
              <RefreshCw size={18} aria-hidden="true" />
              Refresh
            </button>
            <button className="button danger" type="button" onClick={handleDeleteAllSessions} disabled={clientSessions.length === 0}>
              <Trash2 size={18} aria-hidden="true" />
              Delete All Sessions
            </button>
            <button className="button danger" type="button" onClick={handleDeleteAccount}>
              <Trash2 size={18} aria-hidden="true" />
              Delete Account
            </button>
          </div>
        </section>

        {error && <p className="error-text">{error}</p>}

        <section className="metric-grid">
          <div><span>Verified</span><strong>{selectedClient.verified ? "Yes" : "No"}</strong></div>
          <div><span>Sessions</span><strong>{clientSessions.length}</strong></div>
          <div><span>Submitted</span><strong>{submittedCount}</strong></div>
          <div><span>Clips</span><strong>{clipCount}</strong></div>
          <div><span>Segments</span><strong>{selectedClient.segment_count}</strong></div>
          <div><span>Last login</span><strong>{selectedClient.last_login_at_utc ? new Date(selectedClient.last_login_at_utc).toLocaleString() : "Never"}</strong></div>
        </section>

        <section className="table-panel">
          <div className="section-title">
            <h2>Sessions</h2>
            <span>{clientSessions.length} sessions</span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Batch</th>
                  <th>Session</th>
                  <th>Status</th>
                  <th>Clips</th>
                  <th>Created</th>
                  <th>Submitted</th>
                  <th>Delete</th>
                </tr>
              </thead>
              <tbody>
                {clientSessions.length === 0 && (
                  <tr><td colSpan={7}>No sessions for this account.</td></tr>
                )}
                {clientSessions.map((session) => (
                  <tr key={session.session_id} className="clickable-row" onClick={() => openSession(session)}>
                    <td>{session.batch_id}</td>
                    <td>{session.session_id}</td>
                    <td>{session.status}</td>
                    <td>{session.clip_count}</td>
                    <td>{new Date(session.created_at_utc).toLocaleString()}</td>
                    <td>{session.submitted_at_utc ? new Date(session.submitted_at_utc).toLocaleString() : "Not submitted"}</td>
                    <td>
                      <button
                        className="button danger icon-button"
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          void handleDeleteSession(session);
                        }}
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
      </main>
    );
  }

  return (
    <main className="shell admin-shell">
      <section className="admin-header">
        <div>
          <BackButton
            onClick={() => {
              if (window.history.length > 1) {
                window.history.back();
                return;
              }
              window.location.href = "/";
            }}
          />
          <p className="eyebrow">Coordinator</p>
          <h1>Vigil Recorder Admin</h1>
        </div>
        <div className="button-row">
          <button className="button secondary" type="button" onClick={() => refreshAdminData(null, null)} disabled={loading}>
            <RefreshCw size={18} aria-hidden="true" />
            Refresh
          </button>
          <button className="button primary" type="button" onClick={handleExport}>
            <Download size={18} aria-hidden="true" />
            Export
          </button>
        </div>
      </section>

      {error && <p className="error-text">{error}</p>}

      {summary && (
        <section className="metric-grid">
          <div><span>Batch</span><strong>{summary.batch_id}</strong></div>
          <div><span>Accounts</span><strong>{summary.participants}</strong></div>
          <div><span>Sessions</span><strong>{summary.sessions}</strong></div>
          <div><span>Submitted</span><strong>{summary.submitted_sessions}</strong></div>
          <div><span>Raw clips</span><strong>{summary.total_clips}</strong></div>
          <div><span>Segments</span><strong>{summary.total_segments}</strong></div>
          <div><span>Auto accepted</span><strong>{summary.auto_accepted}</strong></div>
          <div><span>Flagged</span><strong>{summary.flagged}</strong></div>
          <div><span>Rejected</span><strong>{summary.rejected}</strong></div>
        </section>
      )}

      {exportResult && (
        <p className="success-text">
          Export created: {exportResult.file_name}
        </p>
      )}

      <section className="table-panel">
        <div className="section-title">
          <h2>Accounts</h2>
          <span>{clients.length} accounts</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Email</th>
                <th>Verified</th>
                <th>Sessions</th>
                <th>Submitted</th>
                <th>Clips</th>
                <th>Segments</th>
                <th>Last login</th>
              </tr>
            </thead>
            <tbody>
              {clients.length === 0 && (
                <tr>
                  <td colSpan={7}>No accounts yet.</td>
                </tr>
              )}
              {clients.map((client) => (
                <tr key={client.email ?? "unknown"} className="clickable-row" onClick={() => openClient(client)}>
                  <td>{client.email ?? "Unknown"}</td>
                  <td>{client.verified ? "Yes" : "No"}</td>
                  <td>{client.session_count}</td>
                  <td>{client.submitted_session_count}</td>
                  <td>{client.clip_count}</td>
                  <td>{client.segment_count}</td>
                  <td>{client.last_login_at_utc ? new Date(client.last_login_at_utc).toLocaleString() : "Never"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="table-panel">
        <div className="section-title">
          <h2>Needs Review</h2>
          <span>{flagged.length} clips</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Clip</th>
                <th>Prompt</th>
                <th>Status</th>
                <th>Flags</th>
                <th>Segments</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {flagged.length === 0 && (
                <tr>
                  <td colSpan={6}>No flagged clips.</td>
                </tr>
              )}
              {flagged.map((clip) => (
                <tr key={clip.clip_id}>
                  <td>{clip.clip_id}</td>
                  <td>{clip.prompt_id}</td>
                  <td>{clip.auto_qc_status}</td>
                  <td>{clip.auto_qc_flags}</td>
                  <td>{clip.detected_segment_count} / {clip.expected_segment_count}</td>
                  <td>{new Date(clip.created_at_utc).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function ClipsTable({
  clips,
  onDeleteClip
}: {
  clips: AdminClip[];
  onDeleteClip: (clipId: string) => void;
}) {
  return (
    <section className="table-panel">
      <div className="section-title">
        <h2>Clips</h2>
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
              <tr><td colSpan={9}>No clips in this session.</td></tr>
            )}
            {clips.map((clip) => (
              <tr key={clip.clip_id}>
                <td>{clip.clip_id}</td>
                <td>{clip.prompt_id}</td>
                <td><audio className="table-audio" controls src={getAdminClipAudioUrl(clip.clip_id)} /></td>
                <td>{clip.auto_qc_status}</td>
                <td>{clip.auto_qc_flags}</td>
                <td>{clip.duration_sec?.toFixed(2) ?? "n/a"}s</td>
                <td>{Math.round(clip.file_size_bytes / 1024)} KB</td>
                <td>{new Date(clip.created_at_utc).toLocaleString()}</td>
                <td>
                  <button className="button danger icon-button" type="button" onClick={() => onDeleteClip(clip.clip_id)}>
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
