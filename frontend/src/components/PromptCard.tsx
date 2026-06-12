import type { Prompt } from "../types";

interface PromptCardProps {
  prompt: Prompt;
}

export function PromptCard({ prompt }: PromptCardProps) {
  return (
    <section className="prompt-panel" aria-labelledby="prompt-target">
      <p className="prompt-instruction">{prompt.instruction_text}</p>
      <h2 id="prompt-target" className="target-phrase">
        {prompt.target_phrase}
      </h2>
      {prompt.recording_mode === "repeat" && (
        <p className="prompt-meta">
          {prompt.target_repetition_count} repetitions, short pause between each
        </p>
      )}
    </section>
  );
}
