import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { InferencePlayground } from "../components/InferencePlayground.tsx";
import { apiClient } from "../api/client.ts";

vi.mock("../api/client.ts", () => ({
  apiClient: {
    getModels: vi.fn(),
    loadModel: vi.fn(),
    createCompletion: vi.fn(),
  },
}));

describe("InferencePlayground Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.getModels).mockResolvedValue([
      {
        id: "bitnet-b1.58-2b-i2s",
        name: "BitNet-b1.58-2B-4T (i2_s)",
        parameter_count: 2000000000,
        quant_type: "ternary-1.58bit",
        memory_mb: 480,
        is_loaded: true,
        context_length: 2048,
        description: "Official reference model",
      },
    ]);
  });

  it("renders in-memory inference heading and model controls", async () => {
    render(<InferencePlayground />);
    expect(screen.getByText(/In-Memory Persistent Inference Engine/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText(/Resident in RAM/i)).toBeDefined();
    });
  });

  it("triggers completion generation and renders output", async () => {
    vi.mocked(apiClient.createCompletion).mockResolvedValue({
      id: "cmpl-1",
      object: "text_completion",
      created: 1234567890,
      model: "bitnet-b1.58-2b-i2s",
      choices: [{ index: 0, text: "Generated ternary response text.", finish_reason: "stop" }],
      usage: { prompt_tokens: 5, completion_tokens: 4, total_tokens: 9 },
      latency_ms: 12.5,
      tokens_per_second: 320.0,
    });

    render(<InferencePlayground />);
    await waitFor(() => expect(screen.getByText(/BitNet-b1.58-2B-4T/i)).toBeDefined());

    const submitBtn = screen.getByRole("button", { name: /Run In-Memory Completion/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText(/Generated ternary response text./i)).toBeDefined();
      expect(screen.getByText(/12.5 ms/i)).toBeDefined();
    });
  });
});
