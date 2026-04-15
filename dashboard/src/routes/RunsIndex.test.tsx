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

  it("renders the runs table with status chips and report flag", async () => {
    const spy = vi.spyOn(api, "listRuns").mockResolvedValue(runsFixture);

    renderWithProviders(<RunsIndex />);

    await waitFor(() => expect(screen.getByText("run_alpha")).toBeInTheDocument());
    expect(spy).toHaveBeenCalled();
    expect(screen.getByText("run_beta")).toBeInTheDocument();
    // Status chips render as the status text.
    expect(screen.getAllByText("completed").length).toBeGreaterThan(0);
    expect(screen.getAllByText("failed").length).toBeGreaterThan(0);
    // has_report chip for alpha.
    expect(screen.getByText("yes")).toBeInTheDocument();
    // Error text for beta is surfaced.
    expect(screen.getByText(/out of memory/)).toBeInTheDocument();
  });

  it("passes status filter to the API when changed", async () => {
    const spy = vi
      .spyOn(api, "listRuns")
      .mockResolvedValue({ runs: [] })
      .mockResolvedValueOnce(runsFixture);

    renderWithProviders(<RunsIndex />);
    await waitFor(() => expect(screen.getByText("run_alpha")).toBeInTheDocument());

    const select = screen.getByLabelText("status") as HTMLSelectElement;
    await userEvent.selectOptions(select, "failed");

    await waitFor(() => {
      const call = spy.mock.calls.at(-1)?.[0];
      expect(call).toMatchObject({ status: "failed" });
    });
  });
});
