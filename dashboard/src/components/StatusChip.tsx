import type { StepStatus } from "@/types/api";
import { statusColorClass, statusDotClass } from "@/lib/status";

export function StatusChip({ status, label }: { status: StepStatus; label?: string }) {
  return (
    <span className={`chip chip-muted ${statusColorClass(status)}`}>
      <span className={`dot ${statusDotClass(status)}`} />
      <span>{label ?? status}</span>
    </span>
  );
}
