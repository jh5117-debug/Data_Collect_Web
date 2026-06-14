import { ArrowRight, Loader2, UserRound } from "lucide-react";
import { FormEvent, useState } from "react";
import { loginWithName } from "../api";
import { BackButton } from "../components/BackButton";
import type { AuthSession } from "../types";

interface LoginPageProps {
  onBack: () => void;
  onVerified: (auth: AuthSession) => void;
}

const REMEMBERED_NAMES_KEY = "vigil_recorder_remembered_names";
const AUTH_STORAGE_KEY = "vigil_recorder_auth_session";

function normalizeName(name: string): string {
  return name.trim().replace(/\s+/g, " ");
}

function loadRememberedNames(): string[] {
  try {
    return JSON.parse(localStorage.getItem(REMEMBERED_NAMES_KEY) ?? "[]") as string[];
  } catch {
    return [];
  }
}

function rememberName(name: string) {
  const normalized = normalizeName(name);
  const next = [normalized, ...loadRememberedNames().filter((item) => item !== normalized)].slice(0, 5);
  localStorage.setItem(REMEMBERED_NAMES_KEY, JSON.stringify(next));
}

function loadStoredAuthForName(name: string): AuthSession | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AuthSession;
    if (
      parsed.email !== normalizeName(name) ||
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
  const [rememberedNames] = useState(loadRememberedNames);
  const [name, setName] = useState(rememberedNames[0] ?? "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loginByName(nextName: string) {
    const normalized = normalizeName(nextName);
    setLoading(true);
    setError(null);
    try {
      const storedAuth = loadStoredAuthForName(normalized);
      if (storedAuth) {
        onVerified(storedAuth);
        return;
      }
      const result = await loginWithName(normalized);
      rememberName(result.name ?? result.email);
      onVerified({
        email: result.name ?? result.email,
        auth_token: result.auth_token,
        expires_at_utc: result.expires_at_utc
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not sign in.");
    } finally {
      setLoading(false);
    }
  }

  async function handleLogin(event: FormEvent) {
    event.preventDefault();
    await loginByName(name);
  }

  return (
    <main className="shell narrow">
      <section className="form-panel">
        <BackButton onClick={onBack} disabled={loading} />
        <p className="eyebrow">Participant Login</p>
        <h1>Name Login</h1>
        <p className="instruction">
          Enter your name to access your recording workspace and submission history.
        </p>

        {rememberedNames.length > 0 && (
          <div className="quick-login">
            <span>Recent names</span>
            <div className="button-row compact-row">
              {rememberedNames.map((item) => (
                <button className="button secondary" type="button" key={item} onClick={() => loginByName(item)} disabled={loading}>
                  {item}
                </button>
              ))}
            </div>
          </div>
        )}

        <form onSubmit={handleLogin}>
          <label className="field">
            <span>Name</span>
            <input
              type="text"
              list="remembered-names"
              value={name}
              onChange={(event) => setName(event.target.value)}
              required
              minLength={2}
              maxLength={80}
              disabled={loading}
              placeholder="Jia Huang"
              autoComplete="name"
            />
            <datalist id="remembered-names">
              {rememberedNames.map((item) => (
                <option key={item} value={item} />
              ))}
            </datalist>
          </label>

          {error && <p className="error-text">{error}</p>}

          <button className="button primary wide" type="submit" disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <UserRound size={18} aria-hidden="true" />}
            Continue
            {!loading && <ArrowRight size={18} aria-hidden="true" />}
          </button>
        </form>
      </section>
    </main>
  );
}
