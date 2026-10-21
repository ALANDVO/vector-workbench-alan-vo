import {
  ActivationCompareResponse,
  AdvisoryReportResponse,
  AuditLogResponse,
  BenchmarkRun,
  ChatCompletionResponse,
  ChatMessage,
  CompilerDiagnosticResponse,
  CompletionResponse,
  LUTBenchmarkResponse,
  ModelInfo,
  QuantizeResponse,
  UserProfile,
} from "./types.ts";

let inMemoryToken: string | null = null;

export const authStorage = {
  getToken: (): string | null => inMemoryToken,
  setToken: (token: string | null) => {
    inMemoryToken = token;
  },
  savePkceVerifier: (verifier: string) => {
    sessionStorage.setItem("pkce_verifier", verifier);
  },
  getPkceVerifier: (): string | null => {
    return sessionStorage.getItem("pkce_verifier");
  },
  clearPkceVerifier: () => {
    sessionStorage.removeItem("pkce_verifier");
  },
};

const BASE_URL = "/api/v1";

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers || {});
  headers.set("Content-Type", "application/json");

  const token = authStorage.getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errMessage = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errBody = await response.json();
      if (errBody.detail) {
        errMessage = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
      }
    } catch {
      // ignore json parse error on non-json error responses
    }
    throw new Error(errMessage);
  }

  return response.json() as Promise<T>;
}

export const apiClient = {
  loginWithDemo: async (role: "viewer" | "operator" | "admin" = "operator"): Promise<string> => {
    const res = await request<{ access_token: string }>("/auth/demo-token", {
      method: "POST",
      body: JSON.stringify({ role }),
    });
    authStorage.setToken(res.access_token);
    return res.access_token;
  },

  getProfile: async (): Promise<UserProfile> => {
    return request<UserProfile>("/auth/me");
  },

  getModels: async (): Promise<ModelInfo[]> => {
    const res = await request<{ data: ModelInfo[] }>("/models");
    return res.data;
  },

  loadModel: async (modelId: string): Promise<ModelInfo> => {
    return request<ModelInfo>("/models/load", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId, prewarm_tensors: true }),
    });
  },

  unloadModel: async (modelId: string): Promise<{ message: string }> => {
    return request<{ message: string }>(`/models/unload?model_id=${encodeURIComponent(modelId)}`, {
      method: "POST",
    });
  },

  createCompletion: async (
    model: string,
    prompt: string,
    maxTokens: number = 64,
    activation: "relu_sqr" | "silu" = "relu_sqr"
  ): Promise<CompletionResponse> => {
    return request<CompletionResponse>("/completions", {
      method: "POST",
      body: JSON.stringify({
        model,
        prompt,
        max_tokens: maxTokens,
        activation,
      }),
    });
  },

  createChatCompletion: async (
    model: string,
    messages: ChatMessage[],
    maxTokens: number = 64,
    activation: "relu_sqr" | "silu" = "relu_sqr"
  ): Promise<ChatCompletionResponse> => {
    return request<ChatCompletionResponse>("/chat/completions", {
      method: "POST",
      body: JSON.stringify({
        model,
        messages,
        max_tokens: maxTokens,
        activation,
      }),
    });
  },

  quantizeMatrix: async (
    rows: number = 64,
    cols: number = 64,
    seed: number = 42,
    lutGroup: number = 3
  ): Promise<QuantizeResponse> => {
    return request<QuantizeResponse>("/quantize", {
      method: "POST",
      body: JSON.stringify({ rows, cols, seed, lut_group_size: lutGroup }),
    });
  },

  compareActivations: async (
    numLayers: number = 30,
    hiddenDim: number = 64
  ): Promise<ActivationCompareResponse> => {
    return request<ActivationCompareResponse>("/benchmarks/activation-compare", {
      method: "POST",
      body: JSON.stringify({ num_layers: numLayers, hidden_dim: hiddenDim }),
    });
  },

  runBenchmark: async (
    type: "perplexity" | "quantization_fidelity" | "throughput",
    matrixDim: number = 128,
    numLayers: number = 30
  ) => {
    return request("/benchmarks/run", {
      method: "POST",
      body: JSON.stringify({
        benchmark_type: type,
        matrix_dim: matrixDim,
        num_layers: numLayers,
      }),
    });
  },
  getBenchmarkHistory: async (): Promise<BenchmarkRun[]> => {
    const res = await request<{ runs: BenchmarkRun[] }>("/benchmarks/history");
    return res.runs;
  },

  runLUTKernelBenchmark: async (dimM: number = 128, dimN: number = 128, dimK: number = 128): Promise<LUTBenchmarkResponse> => {
    return request<LUTBenchmarkResponse>("/benchmarks/lut-kernels", {
      method: "POST",
      body: JSON.stringify({ dim_m: dimM, dim_n: dimN, dim_k: dimK }),
    });
  },

  evaluateCompiler: async (
    compilerName: string,
    compilerVersion: string,
    targetArch: "x86_64" | "aarch64",
    cflags: string[] = []
  ): Promise<CompilerDiagnosticResponse> => {
    return request<CompilerDiagnosticResponse>("/diagnostics/evaluate", {
      method: "POST",
      body: JSON.stringify({
        compiler_name: compilerName,
        compiler_version: compilerVersion,
        target_arch: targetArch,
        cflags,
      }),
    });
  },

  getAuditLogs: async (page: number = 1, pageSize: number = 20, action?: string): Promise<AuditLogResponse> => {
    let url = `/audit/logs?page=${page}&page_size=${pageSize}`;
    if (action) {
      url += `&action=${encodeURIComponent(action)}`;
    }
    return request<AuditLogResponse>(url);
  },

  generateAdvisory: async (customContext: string = ""): Promise<AdvisoryReportResponse> => {
    return request<AdvisoryReportResponse>("/advisory/analyze", {
      method: "POST",
      body: JSON.stringify({ custom_context: customContext }),
    });
  },
};
