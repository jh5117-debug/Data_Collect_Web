import { ArrowRight, Loader2, Mail } from "lucide-react";
import { FormEvent, useState } from "react";
import { requestLoginCode, verifyLoginCode } from "../api";
import { BackButton } from "../components/BackButton";
import type { AuthSession } from "../types";

interface LoginPageProps {
  onBack: () => void;
  onVerified: (auth: AuthSession) => void;
}

const REMEMBERED_EMAILS_KEY = "vigil_recorder_remembered_emails";
const AUTH_STORAGE_KEY = "vigil_recorder_auth_session";

function loadRememberedEmails(): string[] {
  try {
    return JSON.parse(localStorage.getItem(REMEMBERED_EMAILS_KEY) ?? "[]") as string[];
  } catch {
    return [];
  }
}

function rememberEmail(email: string) {
  const normalized = email.trim().toLowerCase();
  const next = [normalized, ...loadRememberedEmails().filter((item) => item !== normalized)].slice(0, 5);
  localStorage.setItem(REMEMBERED_EMAILS_KEY, JSON.stringify(next));
}

function loadStoredAuthForEmail(email: string): AuthSession | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AuthSession;
    if (
      parsed.email !== email.trim().toLowerCase() ||
      !parsed.auth_token ||
      new Date(parsed.expires_at_utc).getTime() <= Date.now()
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function LoginPage({ onBack, onVerified }: LoginPageProps) {
  const [rememberedEmails] = useState(loadRememberedEmails);
  const [email, setEmail] = useState(rememberedEmails[0] ?? "");
  const [code, setCode] = useState("");
  const [codeRequested, setCodeRequested] = useState(false);
  const [devCode, setDevCode] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRequestCode(event: FormEvent) {
    event.preventDefault();
    await requestCodeForEmail(email);
  }

  async function requestCodeForEmail(nextEmail: string) {
    setLoading(true);
    setError(null);
    try {
      const result = await requestLoginCode(nextEmail);
      setEmail(nextEmail);
      setCodeRequested(true);
      setDevCode(result.dev_code ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not send login code.");
    } finally {
      setLoading(false);
    }
  }

  async function handleRecentEmail(nextEmail: string) {
    const storedAuth = loadStoredAuthForEmail(nextEmail);
    if (storedAuth) {
      onVerified(storedAuth);
      return;
    }
    setCode("");
    setDevCode(null);
    await requestCodeForEmail(nextEmail);
  }

  async function handleVerify(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await verifyLoginCode(email, code);
      rememberEmail(result.email);
      onVerified({
        email: result.email,
        auth_token: result.auth_token,
        expires_at_utc: result.expires_at_utc
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not verify login code.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell narrow">
      <section className="form-panel">
        <BackButton onClick={onBack} disabled={loading} />
        <p className="eyebrow">Participant Login</p>
        <h1>Email Login</h1>
        <p className="instruction">
          Use your email to access your own recording workspace and submission history.
        </p>

        {rememberedEmails.length > 0 && !codeRequested && (
          <div className="quick-login">
            <span>Recent emails</span>
            <div className="button-row compact-row">
              {rememberedEmails.map((item) => (
                <button className="button secondary" type="button" key={item} onClick={() => handleRecentEmail(item)} disabled={loading}>
                  {item}
                </button>
              ))}
            </div>
          </div>
        )}

        <form onSubmit={codeRequested ? handleVerify : handleRequestCode}>
          <label className="field">
            <span>Email</span>
            <input
              type="email"
              list="remembered-emails"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
              disabled={loading || codeRequested}
              placeholder="name@example.com"
            />
            <datalist id="remembered-emails">
              {rememberedEmails.map((item) => (
                <option key={item} value={item} />
              ))}
            </datalist>
          </label>

          {codeRequested && (
            <label className="field">
              <span>Login code</span>
              <input
                value={code}
                onChange={(event) => setCode(event.target.value)}
                required
                inputMode="numeric"
                placeholder="6-digit code"
              />
            </label>
          )}

          {devCode && (
            <p className="dev-code">
              Local dev code: <strong>{devCode}</strong>
            </p>
          )}

          {error && <p className="error-text">{error}</p>}

          <button className="button primary wide" type="submit" disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} aria-hidden="true" /> : codeRequested ? <ArrowRight size={18} aria-hidden="true" /> : <Mail size={18} aria-hidden="true" />}
            {codeRequested ? "Verify & Continue" : "Send Login Code"}
          </button>
        </form>
      </section>
    </main>
  );
}
