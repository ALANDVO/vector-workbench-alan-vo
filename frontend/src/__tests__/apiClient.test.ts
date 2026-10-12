import { describe, expect, it, vi, beforeEach } from "vitest";
import { apiClient, authStorage } from "../api/client.ts";

describe("API Client & Auth Storage", () => {
  beforeEach(() => {
    authStorage.setToken(null);
    vi.restoreAllMocks();
  });

  it("sets and retrieves in-memory tokens without localStorage persistence", () => {
    expect(authStorage.getToken()).toBeNull();
    authStorage.setToken("mock-jwt-token-abc");
    expect(authStorage.getToken()).toBe("mock-jwt-token-abc");
  });

  it("attaches Bearer authorization header to outgoing requests", async () => {
    authStorage.setToken("valid-test-bearer-token");

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ data: [] }),
    });
    globalThis.fetch = mockFetch;

    await apiClient.getModels();

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [, options] = mockFetch.mock.calls[0];
    const headers = options.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer valid-test-bearer-token");
  });

  it("handles API error responses cleanly", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 403,
      statusText: "Forbidden",
      json: async () => ({ detail: "Operation requires 'operator' role" }),
    });
    globalThis.fetch = mockFetch;

    await expect(apiClient.getModels()).rejects.toThrow("Operation requires 'operator' role");
  });
});
