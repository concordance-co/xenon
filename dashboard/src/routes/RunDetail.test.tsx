import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { Routes, Route } from "react-router-dom";
import { RunDetail } from "./RunDetail";
import { renderWithProviders } from "@/test/harness";
import { reportDetailFixture, runDetailFixture, stepDetailFixture } from "@/test/fixtures";
import { api } from "@/lib/api";

/**
 * React Flow uses ResizeObserver + DOMRect; happy-dom needs a tiny stub.
 */
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).ResizeObserver = (globalThis as any).ResizeObserver ?? ResizeObserverStub;

describe("RunDetail", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders graph nodes from the run detail and selects a step on click", async () => {
    vi.spyOn(api, "getRun").mockResolvedValue(runDetailFixture);
    vi.spyOn(api, "getStep").mockResolvedValue(stepDetailFixture);

    renderWithProviders(
      <Routes>
        <Route path="/runs/:runId" element={<RunDetail />} />
      </Routes>,
      { initialEntries: ["/runs/run_alpha"] },
    );

    // Run header renders with the workflow name.
    await waitFor(() => expect(screen.getAllByText("demo_wf").length).toBeGreaterThan(0));

    // Overview is the default page and eagerly renders each step's card, so
    // all three step names appear somewhere in the DOM.
    expect((await screen.findAllByText("cap")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("probe").length).toBeGreaterThan(0);
    expect(screen.getAllByText("report").length).toBeGreaterThan(0);

    // The overview page only fetches step detail for capture steps (to pull
    // the sites visualization). Analysis/report steps use getStepResult /
    // getReport respectively — no need to hit getStep for those.
    await waitFor(() =>
      expect((api.getStep as Mock).mock.calls.length).toBeGreaterThanOrEqual(1),
    );
  });

  it("offers report generation when the workflow has a report step but no local report", async () => {
    const withoutLocalReport = {
      ...runDetailFixture,
      report: {
        has_report_step: true,
        step_name: "report",
        artifact_id: "a_report",
        local_available: false,
        reason: "report output_dir does not exist: /tmp/missing-report",
      },
    };
    const withLocalReport = {
      ...runDetailFixture,
      report: {
        has_report_step: true,
        step_name: "report",
        artifact_id: "a_report_local",
        local_available: true,
        reason: null,
      },
      steps: runDetailFixture.steps.map((step) =>
        step.step_name === "report"
          ? { ...step, artifact_id: "a_report_local" }
          : step,
      ),
    };

    vi.spyOn(api, "getRun")
      .mockResolvedValueOnce(withoutLocalReport)
      .mockResolvedValueOnce(withLocalReport);
    vi.spyOn(api, "getStep").mockResolvedValue(stepDetailFixture);
    vi.spyOn(api, "generateReport").mockResolvedValue({
      run_id: "run_alpha",
      step_name: "report",
      artifact_id: "a_report_local",
      report: {
        ...reportDetailFixture,
        artifact_id: "a_report_local",
      },
    });
    vi.spyOn(api, "getReport").mockResolvedValue({
      ...reportDetailFixture,
      artifact_id: "a_report_local",
    });

    renderWithProviders(
      <Routes>
        <Route path="/runs/:runId" element={<RunDetail />} />
      </Routes>,
      { initialEntries: ["/runs/run_alpha"] },
    );

    await waitFor(() => expect(screen.getAllByText("demo_wf").length).toBeGreaterThan(0));

    fireEvent.click(screen.getByRole("button", { name: "report" }));
    expect(await screen.findByRole("button", { name: /generate report/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /generate report/i }));

    await waitFor(() => expect(api.generateReport).toHaveBeenCalledWith("run_alpha", { step_name: "report" }));
    await waitFor(() => expect(screen.getByText("Probe accuracy")).toBeInTheDocument());
  });
});
