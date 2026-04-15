import type { StepFamily, StepStatus } from "@/types/api";

/** Canonical status -> palette mapping. Values are tailwind color tokens. */
export function statusColorClass(status: StepStatus): string {
  switch (status) {
    case "completed":
      return "text-status-ok";
    case "failed":
      return "text-status-fail";
    case "running":
      return "text-status-run";
    case "reused":
      return "text-status-reuse";
    case "pending":
    case "blocked":
      return "text-ink-400";
    default:
      return "text-ink-300";
  }
}

export function statusDotClass(status: StepStatus): string {
  switch (status) {
    case "completed":
      return "bg-status-ok";
    case "failed":
      return "bg-status-fail";
    case "running":
      return "bg-status-run animate-pulse";
    case "reused":
      return "bg-status-reuse";
    case "blocked":
      return "bg-status-warn";
    default:
      return "bg-ink-500";
  }
}

export function statusBorderClass(status: StepStatus): string {
  switch (status) {
    case "completed":
      return "border-status-ok/40";
    case "failed":
      return "border-status-fail/60";
    case "running":
      return "border-status-run/60";
    case "reused":
      return "border-status-reuse/50";
    default:
      return "border-ink-700";
  }
}

/** Operation family -> color hint used on node labels + inspector chrome. */
export function familyAccent(family: StepFamily | null | undefined): {
  label: string;
  text: string;
  bar: string;
} {
  switch (family) {
    case "capture":
      return { label: "capture", text: "text-amber-300", bar: "bg-amber-500/70" };
    case "derive":
      return { label: "derive", text: "text-emerald-300", bar: "bg-emerald-500/70" };
    case "readout":
      return { label: "readout", text: "text-sky-300", bar: "bg-sky-500/70" };
    case "representation":
      return { label: "representation", text: "text-fuchsia-300", bar: "bg-fuchsia-500/70" };
    case "report":
      return { label: "report", text: "text-rose-300", bar: "bg-rose-500/70" };
    default:
      return { label: family ?? "—", text: "text-ink-300", bar: "bg-ink-500" };
  }
}
