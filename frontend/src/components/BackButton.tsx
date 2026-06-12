import { ArrowLeft } from "lucide-react";

interface BackButtonProps {
  onClick: () => void;
  disabled?: boolean;
}

export function BackButton({ onClick, disabled = false }: BackButtonProps) {
  return (
    <button className="button secondary back-button" type="button" onClick={onClick} disabled={disabled}>
      <ArrowLeft size={18} aria-hidden="true" />
      Back
    </button>
  );
}
