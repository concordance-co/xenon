import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { DatasetTab } from "./DatasetTab";
import { LabelsTab } from "./LabelsTab";
import { renderWithProviders } from "@/test/harness";
import { datasetPreviewFixture, labelPreviewFixture } from "@/test/fixtures";
import { api } from "@/lib/api";

describe("DatasetTab", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders source identity and sampled rows", async () => {
    vi.spyOn(api, "getDatasetPreview").mockResolvedValue(datasetPreviewFixture);
    renderWithProviders(<DatasetTab runId="r" stepName="cap" />);
    await waitFor(() => expect(screen.getByText("ex0")).toBeInTheDocument());
    expect(screen.getByText("ex1")).toBeInTheDocument();
    // Source kind chip
    expect(screen.getByText("memory")).toBeInTheDocument();
  });

  it("shows the unavailable banner with dataset options", async () => {
    vi.spyOn(api, "getDatasetPreview").mockResolvedValue({
      available: false,
      reason: "multiple upstream capture datasets; select one",
      rows: [],
      dataset_options: [
        { step_name: "cap_a", label: "cap_a (capture)" },
        { step_name: "cap_b", label: "cap_b (capture)" },
      ],
    });
    renderWithProviders(<DatasetTab runId="r" stepName="probe" />);
    await waitFor(() =>
      expect(screen.getByText(/multiple upstream capture datasets/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/cap_a \(capture\)/)).toBeInTheDocument();
  });
});

describe("LabelsTab", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders label distributions as bar rows", async () => {
    vi.spyOn(api, "getLabelPreview").mockResolvedValue(labelPreviewFixture);
    renderWithProviders(<LabelsTab runId="r" stepName="cap" />);
    await waitFor(() =>
      expect(screen.getByText(/label · class/)).toBeInTheDocument(),
    );
    expect(screen.getByText("pos")).toBeInTheDocument();
    expect(screen.getByText("neg")).toBeInTheDocument();
  });
});
