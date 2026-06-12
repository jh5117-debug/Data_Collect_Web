import { useState } from "react";
import { AccountPage } from "./pages/AccountPage";
import { AdminPage } from "./pages/AdminPage";
import { ConsentPage } from "./pages/ConsentPage";
import { LoginPage } from "./pages/LoginPage";
import { MicTestPage } from "./pages/MicTestPage";
import { ParticipantPage } from "./pages/ParticipantPage";
import { RecordingPage } from "./pages/RecordingPage";
import { SummaryPage } from "./pages/SummaryPage";
import { WelcomePage } from "./pages/WelcomePage";
import type { AuthSession, ParticipantMetadata, Prompt, RecordingProgressState, RecordingStats } from "./types";

type FlowPage = "welcome" | "login" | "account" | "consent" | "participant" | "mic-test" | "recording" | "summary";

const EMPTY_STATS: RecordingStats = {
  completedPrompts: 0,
  uploadedClips: 0,
  failedUploads: 0,
  qcWarnings: 0,
  generatedSegments: 0
};

const EMPTY_RECORDING_PROGRESS: RecordingProgressState = {
  currentPromptIndex: 0,
  uploadedRows: [],
  failedUploads: 0
};

const AUTH_STORAGE_KEY = "vigil_recorder_auth_session";

function loadStoredAuth(): AuthSession | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AuthSession;
    if (!parsed.email || !parsed.auth_token || new Date(parsed.expires_at_utc).getTime() <= Date.now()) {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      return null;
    }
    return parsed;
  } catch {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    return null;
  }
}

function saveAuth(auth: AuthSession) {
  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(auth));
}

export default function App() {
  if (window.location.pathname === "/admin") {
    return <AdminPage />;
  }

  const initialAuth = loadStoredAuth();
  const [page, setPage] = useState<FlowPage>(initialAuth ? "account" : "welcome");
  const [authSession, setAuthSession] = useState<AuthSession | null>(initialAuth);
  const userEmail = authSession?.email ?? "";
  const [participantMetadata, setParticipantMetadata] = useState<ParticipantMetadata | null>(null);
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [stats, setStats] = useState<RecordingStats>(EMPTY_STATS);
  const [recordingProgress, setRecordingProgress] = useState<RecordingProgressState>(EMPTY_RECORDING_PROGRESS);

  if (page === "welcome") {
    return <WelcomePage onStart={() => setPage("login")} />;
  }

  if (page === "login") {
    return (
      <LoginPage
        onBack={() => setPage("welcome")}
        onVerified={(auth) => {
          saveAuth(auth);
          setAuthSession(auth);
          setPage("account");
        }}
      />
    );
  }

  if (page === "account") {
    if (!authSession) {
      setPage("login");
      return null;
    }
    return (
      <AccountPage
        email={authSession.email}
        authToken={authSession.auth_token}
        onBack={() => setPage("login")}
        onStart={() => setPage("consent")}
      />
    );
  }

  if (page === "consent") {
    return <ConsentPage onBack={() => setPage("account")} onContinue={() => setPage("participant")} />;
  }

  if (page === "participant") {
    return (
      <ParticipantPage
        onBack={() => setPage("consent")}
        userEmail={userEmail}
        onReady={(metadata, nextPrompts) => {
          setParticipantMetadata(metadata);
          setPrompts(nextPrompts);
          setRecordingProgress(EMPTY_RECORDING_PROGRESS);
          setStats(EMPTY_STATS);
          setPage("mic-test");
        }}
      />
    );
  }

  if (page === "mic-test") {
    return (
      <MicTestPage
        onBack={() => setPage("participant")}
        onContinue={() => setPage("recording")}
      />
    );
  }

  if (page === "recording") {
    return (
      <RecordingPage
        prompts={prompts}
        progress={recordingProgress}
        onProgressChange={setRecordingProgress}
        onBack={() => setPage("mic-test")}
        onFinished={(nextStats) => {
          setStats(nextStats);
          setPage("summary");
        }}
      />
    );
  }

  return (
    <SummaryPage
      userEmail={userEmail}
      participantMetadata={participantMetadata}
      progress={recordingProgress}
      stats={stats}
      onProgressChange={setRecordingProgress}
      onBack={() => setPage("recording")}
      onSubmitted={() => setPage("account")}
    />
  );
}
