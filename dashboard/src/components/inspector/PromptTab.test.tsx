import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { PromptTab } from "./PromptTab";
import { renderWithProviders } from "@/test/harness";
import {
  promptPreviewDegradedFixture,
  promptPreviewWithSectionFixture,
} from "@/test/fixtures";
import { api } from "@/lib/api";

describe("PromptTab", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("highlights the selected section span and shows the selection sentence", async () => {
    vi.spyOn(api, "getPromptPreview").mockResolvedValue(promptPreviewWithSectionFixture);

    renderWithProviders(<PromptTab runId="r" stepName="probe" />);

    await waitFor(() =>
      expect(screen.getByText(/section STRATEGY -> tokens 8\.\.11/)).toBeInTheDocument(),
    );
    // The section label is rendered above the highlighted span. It appears in
    // both the sections overlay and the selection sentence; check at least
    // two occurrences.
    expect(screen.getAllByText(/STRATEGY/).length).toBeGreaterThanOrEqual(2);
    // Not degraded — no fallback banner.
    expect(screen.queryByText(/section-level fallback/)).not.toBeInTheDocument();
  });

  it("shows a degraded-state banner when tokenizer is unavailable", async () => {
    vi.spyOn(api, "getPromptPreview").mockResolvedValue(promptPreviewDegradedFixture);

    renderWithProviders(<PromptTab runId="r" stepName="probe" />);

    await waitFor(() =>
      expect(screen.getByText(/section-level fallback/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/section-level, not exact tokens/)).toBeInTheDocument();
  });

  it("renders a hard unresolved state without inventing spans", async () => {
    vi.spyOn(api, "getPromptPreview").mockResolvedValue({
      available: false,
      reason: "Section STRATEGY not present in prompt_metadata_builder output",
      degraded: false,
      examples: [],
    });

    renderWithProviders(<PromptTab runId="r" stepName="probe" />);

    await waitFor(() =>
      expect(screen.getByText(/prompt preview unresolved/)).toBeInTheDocument(),
    );
  });
});
