import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { Routes, Route } from "react-router-dom";
import { ReportGallery } from "./ReportGallery";
import { renderWithProviders } from "@/test/harness";
import { reportDetailFixture } from "@/test/fixtures";
import { api } from "@/lib/api";

describe("ReportGallery", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders figures, tables, and copied results from the report manifest", async () => {
    vi.spyOn(api, "getReport").mockResolvedValue(reportDetailFixture);

    renderWithProviders(
      <Routes>
        <Route path="/runs/:runId/reports/:artifactId" element={<ReportGallery />} />
      </Routes>,
      { initialEntries: ["/runs/run_alpha/reports/a_report"] },
    );

    await waitFor(() => expect(screen.getByText("Probe accuracy")).toBeInTheDocument());
    expect(screen.getByText("primary figures")).toBeInTheDocument();
    expect(screen.getByText("probe_results.json")).toBeInTheDocument();
    // Figure image src uses the asset URL helper.
    const img = screen.getByAltText("Probe accuracy") as HTMLImageElement;
    expect(img.src).toContain("/api/reports/a_report/assets/assets/probe.png");
  });
});
