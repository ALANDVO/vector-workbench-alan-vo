import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ActivationGraphViewer } from "../components/ActivationGraphViewer.tsx";
import { apiClient } from "../api/client.ts";

vi.mock("../api/client.ts", () => ({
  apiClient: {
    compareActivations: vi.fn(),
  },
}));

describe("ActivationGraphViewer Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders activation perplexity comparison reproducing Issue #588", async () => {
    vi.mocked(apiClient.compareActivations).mockResolvedValue({
      num_layers: 30,
      relu_sqr_perplexity: 17.11,
      silu_perplexity: 99.82,
      perplexity_delta: 82.71,
      root_cause_explanation: "Squared ReLU restores official perplexity baseline.",
      relu_sqr_stats: [
        { layer: 0, norm: 0.44, sparsity: 0.25, output_norm: 0.5 },
      ],
      silu_stats: [
        { layer: 0, norm: 0.35, sparsity: 0.0, output_norm: 0.4 },
      ],
    });

    render(<ActivationGraphViewer />);

    await waitFor(() => {
      expect(screen.getByText(/PPL 17.11/i)).toBeDefined();
      expect(screen.getByText(/PPL 99.82/i)).toBeDefined();
      expect(screen.getByText(/DEGRADED/i)).toBeDefined();
      expect(screen.getByText(/Squared ReLU restores official perplexity/i)).toBeDefined();
    });
  });
});
