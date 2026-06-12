import { ArrowRight } from "lucide-react";
import { useState } from "react";
import { BackButton } from "../components/BackButton";

interface ConsentPageProps {
  onBack: () => void;
  onContinue: () => void;
}

const CONSENT_ITEMS = [
  "I understand my voice will be recorded for internal research and model development.",
  "I will not include patient names, MRNs, phone numbers, addresses, or other patient-identifiable information.",
  "I understand my real name will not be used in audio filenames."
];

export function ConsentPage({ onBack, onContinue }: ConsentPageProps) {
  const [checked, setChecked] = useState<boolean[]>(CONSENT_ITEMS.map(() => false));
  const allChecked = checked.every(Boolean);

  return (
    <main className="shell narrow">
      <section className="form-panel">
        <BackButton onClick={onBack} />
        <h1>Consent and Privacy</h1>
        <div className="check-list">
          {CONSENT_ITEMS.map((item, index) => (
            <label className="check-row" key={item}>
              <input
                type="checkbox"
                checked={checked[index]}
                onChange={(event) => {
                  const next = [...checked];
                  next[index] = event.target.checked;
                  setChecked(next);
                }}
              />
              <span>{item}</span>
            </label>
          ))}
        </div>
        <button className="button primary wide" type="button" onClick={onContinue} disabled={!allChecked}>
          Continue
          <ArrowRight size={18} aria-hidden="true" />
        </button>
      </section>
    </main>
  );
}
