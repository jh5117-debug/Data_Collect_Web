import { ArrowLeft, Download, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import {
  createExport,
  deleteAdminClient,
  deleteAdminClip,
  getAdminClientClips,
  getAdminClientSessions,
  getAdminClipAudioUrl,
  getAdminClients,
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
  const [clientClips, setClientClips] = useState<AdminClip[]>([]);
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadAdminData() {
    setLoading(true);
    setError(null);
    try {
      const [nextSummary, nextFlagged, nextClients, nextClips] = await Promise.all([
        getAdminSummary(),
        getFlaggedClips(),
        getAdminClients(),
        selectedClient?.email ? getAdminClientClips(selectedClient.email) : Promise.resolve([])
      ]);
      setSummary(nextSummary);
      setFlagged(nextFlagged);
      setClients(nextClients);
      if (selectedClient?.email) setClientClips(nextClips);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load admin data.");
    } finally {
      setLoading(false);
    }
  }

  async function handleDeleteClip(clipId: string) {
    const confirmed = window.confirm(`Delete clip ${clipId} and its audio files? This cannot be undone.`);
    if (!confirmed) return;

    setError(null);
    try {
      await deleteAdminClip(clipId);
      if (selectedClient?.email) {
        setClientClips(await getAdminClientClips(selectedClient.email));
        setClientSessions(await getAdminClientSessions(selectedClient.email));
      }
      await loadAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete clip.");
    }
  }

  async function openClient(client: AdminClient) {
    if (!client.email) return;
    setSelectedClient(client);
    setLoading(true);
    setError(null);
    try {
      const [sessions, clips] = await Promise.all([
        getAdminClientSessions(client.email),
        getAdminClientClips(client.email)
      ]);
      setClientSessions(sessions);
      setClientClips(clips);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load client detail.");
    } finally {
      setLoading(false);
    }
  }

  async function handleDeleteClient() {
    if (!selectedClient?.email) return;
    const confirmed = window.confirm(`Delete client ${selectedClient.email}, all sessions, all clips, and all audio files? This cannot be undone.`);
    if (!confirmed) return;
    setError(null);
    try {
      await deleteAdminClient(selectedClient.email);
      setSelectedClient(null);
      setClientSessions([]);
      setClientClips([]);
      await loadAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete client.");
    }
  }

  async function handleExport() {
    setError(null);
    try {
      const result = await createExport();
      setExportResult(result);
      window.open(result.download_path, "_blank");
      await loadAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed.");
    }
  }

  useEffect(() => {
    void loadAdminData();
  }, []);

  if (selectedClient) {
    return (
      <main className="shell admin-shell">
        <section className="admin-header">
          <div>
            <button className="button secondary back-button" type="button" onClick={() => setSelectedClient(null)}>
              <ArrowLeft size={18} aria-hidden="true" />
              Back to Admin
            </button>
            <p className="eyebrow">Client Detail</p>
            <h1>{selectedClient.email ?? "Unknown Client"}</h1>
          </div>
          <div className="button-row">
            <button className="button secondary" type="button" onClick={() => selectedClient.email && openClient(selectedClient)} disabled={loading}>
              <RefreshCw size={18} aria-hidden="true" />
              Refresh
            </button>
            <button className="button danger" type="button" onClick={handleDeleteClient}>
              <Trash2 size={18} aria-hidden="true" />
              Delete Client
            </button>
          </div>
        </section>

        {error && <p className="error-text">{error}</p>}

        <section className="metric-grid">
          <div><span>Verified</span><strong>{selectedClient.verified ? "Yes" : "No"}</strong></div>
          <div><span>Participants</span><strong>{selectedClient.participant_count}</strong></div>
          <div><span>Sessions</span><strong>{clientSessions.length}</strong></div>
          <div><span>Submitted</span><strong>{selectedClient.submitted_session_count}</strong></div>
          <div><span>Clips</span><strong>{clientClips.length}</strong></div>
          <div><span>Segments</span><strong>{selectedClient.segment_count}</strong></div>
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
                </tr>
              </thead>
              <tbody>
                {clientSessions.length === 0 && (
                  <tr><td colSpan={6}>No sessions for this client.</td></tr>
                )}
                {clientSessions.map((session) => (
                  <tr key={session.session_id}>
                    <td>{session.batch_id}</td>
                    <td>{session.session_id}</td>
                    <td>{session.status}</td>
                    <td>{session.clip_count}</td>
                    <td>{new Date(session.created_at_utc).toLocaleString()}</td>
                    <td>{session.submitted_at_utc ? new Date(session.submitted_at_utc).toLocaleString() : "Not submitted"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="table-panel">
          <div className="section-title">
            <h2>Clips</h2>
            <span>{clientClips.length} clips</span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Clip</th>
                  <th>Prompt</th>
                  <th>Playback</th>
                  <th>Session</th>
                  <th>Status</th>
                  <th>Flags</th>
                  <th>Duration</th>
                  <th>Size</th>
                  <th>Created</th>
                  <th>Delete</th>
                </tr>
              </thead>
              <tbody>
                {clientClips.length === 0 && (
                  <tr><td colSpan={10}>No clips for this client.</td></tr>
                )}
                {clientClips.map((clip) => (
                  <tr key={clip.clip_id}>
                    <td>{clip.clip_id}</td>
                    <td>{clip.prompt_id}</td>
                    <td><audio className="table-audio" controls src={getAdminClipAudioUrl(clip.clip_id)} /></td>
                    <td>{clip.session_id}</td>
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
          <button className="button secondary" type="button" onClick={loadAdminData} disabled={loading}>
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
          <div><span>Participants</span><strong>{summary.participants}</strong></div>
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
          <h2>Clients</h2>
          <span>{clients.length} accounts</span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Email</th>
                <th>Verified</th>
                <th>Participants</th>
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
                  <td colSpan={8}>No clients yet.</td>
                </tr>
              )}
              {clients.map((client) => (
                <tr key={client.email ?? "unknown"} className="clickable-row" onClick={() => openClient(client)}>
                  <td>{client.email ?? "Unknown"}</td>
                  <td>{client.verified ? "Yes" : "No"}</td>
                  <td>{client.participant_count}</td>
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
