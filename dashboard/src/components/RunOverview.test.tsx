import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { RunOverview } from "./RunOverview";
import { renderWithProviders } from "@/test/harness";
import { runDetailFixture } from "@/test/fixtures";
import { api } from "@/lib/api";
import type { RunDetail, ResultPreview } from "@/types/api";

describe("RunOverview", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders geometry results as a projection preview instead of the generic metric fallback", async () => {
    const detail: RunDetail = {
      ...runDetailFixture,
      nodes: [
        {
          ...runDetailFixture.nodes[1],
          id: "geometry_step",
          step_name: "geometry_step",
          spec_kind: "geometry",
          family: "representation",
          artifact_id: "a_geometry",
          artifact_kind: "geometry",
          reused: false,
        },
      ],
      steps: [
        {
          ...runDetailFixture.steps[1],
          step_name: "geometry_step",
          spec_kind: "geometry",
          family: "representation",
          artifact_id: "a_geometry",
          artifact_kind: "geometry",
          reused_from_run_id: null,
          reused_from_artifact_id: null,
        },
      ],
      report: null,
    };

    const geometryResult: ResultPreview = {
      available: true,
      payload: {
        kind: "geometry_result",
        method: "pca",
        layers: [
          {
            layer: 8,
            component_count: 3,
            components: [
              [0.05, 0.1, 0.0],
              [0.1, 0.05, 0.0],
              [0.75, 0.85, 0.0],
              [0.8, 0.75, 0.0],
            ],
            explained_variance_ratio: [0.22, 0.19, 0.13],
            example_count: 4,
            labels: ["alpha", "alpha", "beta", "beta"],
            selected_example_keys: ["a", "b", "c", "d"],
          },
          {
            layer: 16,
            component_count: 3,
            components: [
              [0.1, 0.2, 0.0],
              [0.2, 0.1, 0.0],
              [0.8, 0.9, 0.0],
              [0.9, 0.8, 0.0],
            ],
            explained_variance_ratio: [0.41, 0.29, 0.12],
            example_count: 4,
            labels: ["alpha", "alpha", "beta", "beta"],
            selected_example_keys: ["a", "b", "c", "d"],
          },
        ],
        summary: {
          method: "pca",
          example_count: 4,
          layer_count: 1,
        },
      },
      headline: {
        method: "pca",
        example_count: 4,
      },
      tables: [],
      truncated: false,
    };

    vi.spyOn(api, "getStepResult").mockResolvedValue(geometryResult);

    renderWithProviders(<RunOverview detail={detail} />);

    expect(await screen.findByText(/pc1 41\.0% × pc2 29\.0%/i)).toBeInTheDocument();
    expect(screen.getAllByText(/layer 16/i).length).toBeGreaterThan(0);
    expect(screen.getByText("alpha")).toBeInTheDocument();
    expect(screen.getByText("beta")).toBeInTheDocument();
    expect(screen.getByLabelText(/geometry layer/i)).toBeInTheDocument();
  });
});
