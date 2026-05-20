import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RunsIndex } from "./RunsIndex";
import { renderWithProviders } from "@/test/harness";
import { runsFixture } from "@/test/fixtures";
import { api } from "@/lib/api";

describe("RunsIndex", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("groups runs by workflow and expands to show individual runs", async () => {
    const spy = vi.spyOn(api, "listRuns").mockResolvedValue(runsFixture);

    renderWithProviders(<RunsIndex />);

    // Wait for the workflow group header to appear (both test runs share
    // "demo_wf" so they merge into one group row).
    await waitFor(() =>
      expect(screen.getAllByText("demo_wf").length).toBeGreaterThan(0),
    );
    expect(spy).toHaveBeenCalled();

    // Group summary shows run count.
    expect(screen.getByText(/2 runs/)).toBeInTheDocument();

    // Individual run IDs are behind the toggle — click to expand.
    expect(screen.queryByText("run_alpha")).not.toBeInTheDocument();
    await userEvent.click(screen.getByText("demo_wf"));

    // Now the table with individual rows is visible.
    expect(await screen.findByText("run_alpha")).toBeInTheDocument();
    expect(screen.getByText("run_beta")).toBeInTheDocument();
    expect(screen.getAllByText("completed").length).toBeGreaterThan(0);
    expect(screen.getAllByText("failed").length).toBeGreaterThan(0);
    expect(screen.getByText(/out of memory/)).toBeInTheDocument();
  });

  it("passes status filter to the API when changed", async () => {
    const spy = vi
      .spyOn(api, "listRuns")
      .mockResolvedValue({ runs: [] })
      .mockResolvedValueOnce(runsFixture);

    renderWithProviders(<RunsIndex />);
    await waitFor(() =>
      expect(screen.getAllByText("demo_wf").length).toBeGreaterThan(0),
    );

    const select = screen.getByLabelText("status") as HTMLSelectElement;
    await userEvent.selectOptions(select, "failed");

    await waitFor(() => {
      const call = spy.mock.calls.at(-1)?.[0];
      expect(call).toMatchObject({ status: "failed" });
    });
  });
});
