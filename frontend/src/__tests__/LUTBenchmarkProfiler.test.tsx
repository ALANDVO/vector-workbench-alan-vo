import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { LUTBenchmarkProfiler } from "../components/LUTBenchmarkProfiler.tsx";
import { apiClient } from "../api/client.ts";
// Mock the API client
vi.mock("../api/client.ts", () => ({
  apiClient: {
    runLUTKernelBenchmark: vi.fn(),
  },
}));

describe("LUTBenchmarkProfiler", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the component with input controls and run button", () => {
    render(<LUTBenchmarkProfiler />);

    expect(screen.getByText("LUT Kernel Profiler (Issue #569)")).toBeInTheDocument();
    const inputs = screen.getAllByDisplayValue("128");
    expect(screen.getByRole("button", { name: /Profiling/i })).toBeInTheDocument();
  });

  it("displays benchmark results after successful API call", async () => {
    const mockResult = {
      scalar_result: {
        group_size: 1,
        dim_m: 128,
        dim_n: 128,
        dim_k: 128,
        additions_count: 10000,
        multiplications_count: 0,
        lut_lookups: 0,
        latency_microseconds: 5000,
        memory_bandwidth_mb_s: 100.5,
        operational_intensity: 2.5,
        speedup_vs_scalar: 1.0,
      },
      lut3_result: {
        group_size: 3,
        dim_m: 128,
        dim_n: 128,
        dim_k: 128,
        additions_count: 5000,
        multiplications_count: 0,
        lut_lookups: 2000,
        latency_microseconds: 3000,
        memory_bandwidth_mb_s: 150.2,
        operational_intensity: 3.8,
        speedup_vs_scalar: 1.67,
      },
      lut5_result: {
        group_size: 5,
        dim_m: 128,
        dim_n: 128,
        dim_k: 128,
        additions_count: 2000,
        multiplications_count: 0,
        lut_lookups: 5000,
        latency_microseconds: 2000,
        memory_bandwidth_mb_s: 200.1,
        operational_intensity: 5.2,
        speedup_vs_scalar: 2.5,
      },
      speedup_3ary_vs_scalar: 1.67,
      speedup_5ary_vs_scalar: 2.5,
      speedup_5ary_vs_3ary: 1.5,
      bandwidth_scalar_mb_s: 100.5,
      bandwidth_lut3_mb_s: 150.2,
      bandwidth_lut5_mb_s: 200.1,
      operational_intensity_scalar: 2.5,
      operational_intensity_lut3: 3.8,
      operational_intensity_lut5: 5.2,
    };

    (apiClient.runLUTKernelBenchmark as vi.Mock).mockResolvedValue(mockResult);

    render(<LUTBenchmarkProfiler />);

    await waitFor(() => {
      expect(screen.getByText("1.67x")).toBeInTheDocument();
      expect(screen.getByText("2.50x")).toBeInTheDocument();
      expect(screen.getByText("1.50x")).toBeInTheDocument();
    });
  });

  it("shows error message when API call fails", async () => {
    (apiClient.runLUTKernelBenchmark as vi.Mock).mockRejectedValue(
      new Error("Network error")
    );

    render(<LUTBenchmarkProfiler />);

    await waitFor(() => {
      expect(screen.getByText("Network error")).toBeInTheDocument();
    });
  });
});