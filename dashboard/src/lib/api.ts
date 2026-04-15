import type {
  DatasetPreview,
  LabelPreview,
  PromptPreview,
  ReportDetail,
  ResultPreview,
  RunDetail,
  RunsResponse,
  StepDetail,
  StepDetailListResponse,
} from "@/types/api";

const BASE = ""; // same-origin via Vite proxy; mounted static serves both paths.

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public url: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE}${path}`;
  const res = await fetch(url, {
    headers: { accept: "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = String((body as { detail: unknown }).detail);
      }
    } catch {
      /* swallow */
    }
    throw new ApiError(detail, res.status, url);
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, string | number | undefined | null>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  if (entries.length === 0) return "";
  const search = new URLSearchParams();
  for (const [k, v] of entries) search.set(k, String(v));
  return `?${search.toString()}`;
}

export const api = {
  listRuns: (filters: { status?: string; workflow_name?: string; limit?: number } = {}) =>
    req<RunsResponse>(`/api/runs${qs(filters)}`),

  getRun: (runId: string) => req<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`),

  getStep: (runId: string, stepName: string) =>
    req<StepDetail>(`/api/runs/${encodeURIComponent(runId)}/steps/${encodeURIComponent(stepName)}`),

  getAllSteps: (runId: string) =>
    req<StepDetailListResponse>(`/api/runs/${encodeURIComponent(runId)}/steps-detail`),

  getDatasetPreview: (runId: string, stepName: string, opts: { sample_size?: number; source_step?: string } = {}) =>
    req<DatasetPreview>(
      `/api/runs/${encodeURIComponent(runId)}/steps/${encodeURIComponent(stepName)}/dataset-preview${qs(opts)}`,
    ),

  getLabelPreview: (runId: string, stepName: string, opts: { source_step?: string } = {}) =>
    req<LabelPreview>(
      `/api/runs/${encodeURIComponent(runId)}/steps/${encodeURIComponent(stepName)}/label-preview${qs(opts)}`,
    ),

  getPromptPreview: (runId: string, stepName: string, opts: { example_keys?: string; max_examples?: number } = {}) =>
    req<PromptPreview>(
      `/api/runs/${encodeURIComponent(runId)}/steps/${encodeURIComponent(stepName)}/prompt-preview${qs(opts)}`,
    ),

  getStepResult: (runId: string, stepName: string) =>
    req<ResultPreview>(
      `/api/runs/${encodeURIComponent(runId)}/steps/${encodeURIComponent(stepName)}/result`,
    ),

  getReport: (artifactId: string) => req<ReportDetail>(`/api/reports/${encodeURIComponent(artifactId)}`),

  reportAssetUrl: (artifactId: string, assetPath: string) =>
    `/api/reports/${encodeURIComponent(artifactId)}/assets/${assetPath.split("/").map(encodeURIComponent).join("/")}`,
};
