import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { DiagnosticsMatrix } from "../components/DiagnosticsMatrix.tsx";
import { apiClient } from "../api/client.ts";

vi.mock("../api/client.ts", () => ({
  apiClient: {
    evaluateCompiler: vi.fn(),
  },
}));

describe("DiagnosticsMatrix Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders compiler hazard report and remediation advice", async () => {
    vi.mocked(apiClient.evaluateCompiler).mockResolvedValue({
      compiler: "clang",
      version: "20.1.0",
      arch: "x86_64",
      safe_to_compile: false,
      hazards: [
        {
          issue_code: "CLANG20-CONST-CORRECTNESS",
          severity: "error",
          title: "Clang 20 Const Pointer Qualifier Discard (Issue #480)",
          description: "Clang 20 strictly enforces const qualifiers.",
          recommended_fix: "Patch ggml-bitnet-mad.cpp line 811: change to const int8_t *.",
        },
      ],
      recommended_cmake_flags: {
        CMAKE_BUILD_TYPE: "Release",
        LLAMA_BITNET: "ON",
      },
    });

    render(<DiagnosticsMatrix />);

    await waitFor(() => {
      expect(screen.getByText(/BUILD TRAPS DETECTED/i)).toBeDefined();
      expect(screen.getByText(/CLANG20-CONST-CORRECTNESS/i)).toBeDefined();
      expect(screen.getByText(/Patch ggml-bitnet-mad.cpp/i)).toBeDefined();
    });
  });
});
