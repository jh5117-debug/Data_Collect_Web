import { ArrowRight } from "lucide-react";

interface WelcomePageProps {
  onStart: () => void;
}

export function WelcomePage({ onStart }: WelcomePageProps) {
  return (
    <main className="shell welcome-screen">
      <section className="hero-panel">
        <p className="eyebrow">5-10 minutes</p>
        <h1>Vigil Voice Trigger Data Collection</h1>
        <p className="lede">
          Welcome to Vigil Recorder. This tool collects clean voice samples for the Vigil voice trigger
          system. Please speak naturally. Do not imitate another accent. Please record in a relatively
          quiet place if possible.
        </p>
        <button className="button primary wide" type="button" onClick={onStart}>
          Start
          <ArrowRight size={18} aria-hidden="true" />
        </button>
      </section>
    </main>
  );
}
