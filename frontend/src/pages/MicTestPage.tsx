import { ArrowRight } from "lucide-react";
import { useState } from "react";
import { AudioRecorder } from "../components/AudioRecorder";
import { BackButton } from "../components/BackButton";

interface MicTestPageProps {
  onBack: () => void;
  onContinue: () => void;
}

export function MicTestPage({ onBack, onContinue }: MicTestPageProps) {
  const [blob, setBlob] = useState<Blob | null>(null);

  return (
    <main className="shell narrow">
      <section className="form-panel">
        <BackButton onClick={onBack} />
        <p className="eyebrow">Microphone Test</p>
        <h1>Check Your Recording</h1>
        <p className="instruction">
          Please stay silent for one second, then say: This is a microphone test for Vigil data collection.
          This test is local only and is not uploaded.
        </p>
        <AudioRecorder onBlobChange={setBlob} />
        <div className="button-row">
          <button className="button primary" type="button" onClick={onContinue} disabled={!blob}>
            Continue
            <ArrowRight size={18} aria-hidden="true" />
          </button>
        </div>
      </section>
    </main>
  );
}
