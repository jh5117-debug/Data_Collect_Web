import type { PromptGroupId } from "./types";

export interface PromptGroupConfig {
  id: PromptGroupId;
  title: string;
  instruction: string;
  inputLabel?: string;
  fixedTranscript?: string;
  examples: string[];
  contains_vigil: boolean;
  wake_intent: boolean;
  is_negative: boolean;
}

export const PROMPT_GROUPS: PromptGroupConfig[] = [
  {
    id: "P1_vigil_only",
    title: "Prompt 1 — VIGIL Only",
    instruction: 'Please say "VIGIL" once per recording. You can upload as many recordings as you like. The more the better.',
    fixedTranscript: "VIGIL",
    examples: [],
    contains_vigil: true,
    wake_intent: true,
    is_negative: false
  },
  {
    id: "P2_phrase_plus_vigil",
    title: "Prompt 2 — Phrase/Sentence + VIGIL",
    instruction:
      'Please say a phrase or sentence ending with or followed by "VIGIL" in one recording. You can upload as many recordings as you like, with the same or different phrases/sentences.',
    inputLabel: "Exact phrase/sentence you will say",
    examples: [
      "Hi VIGIL.",
      "Hey VIGIL.",
      "Hello VIGIL.",
      "Next, VIGIL.",
      "What's next, VIGIL?",
      "Am I doing it yet right, VIGIL?"
    ],
    contains_vigil: true,
    wake_intent: true,
    is_negative: false
  },
  {
    id: "P3_vigil_plus_phrase",
    title: "Prompt 3 — VIGIL + Phrase/Sentence",
    instruction:
      'Please say "VIGIL" plus a phrase or sentence in one recording. You can upload as many recordings as you like, with the same or different phrases/sentences.',
    inputLabel: "Exact phrase/sentence you will say",
    examples: [
      "VIGIL, next.",
      "VIGIL, go back.",
      "VIGIL, what's next?",
      "VIGIL, am I doing right?"
    ],
    contains_vigil: true,
    wake_intent: true,
    is_negative: false
  },
  {
    id: "P4_negative",
    title: "Prompt 4 — Negative Examples",
    instruction:
      'Please record confusing common words or sentences. These recordings should NOT wake up Vigil. Do not say the exact word "Vigil" in this section.',
    inputLabel: "Exact negative word or sentence you will say",
    examples: [
      "visual",
      "visuals",
      "visible",
      "digital",
      "individual",
      "residual",
      "video",
      "vital",
      "vigilant",
      "This is a visual input.",
      "The video is clear.",
      "The image is visible.",
      "This is a digital system.",
      "The individual is moving.",
      "The vital signs are normal."
    ],
    contains_vigil: false,
    wake_intent: false,
    is_negative: true
  }
];

export function normalizeTranscript(value: string): string {
  return value.trim().replace(/\s+/g, " ").replace(/\bvigil\b/gi, "Vigil");
}

export function containsExactVigil(value: string): boolean {
  return /\bvigil\b/i.test(value);
}
