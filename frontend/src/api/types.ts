export interface ModelInfo {
  id: string;
  name: string;
  parameter_count: number;
  quant_type: string;
  memory_mb: number;
  is_loaded: boolean;
  context_length: number;
  description: string;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: ChatMessage;
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  latency_ms: number;
  tokens_per_second: number;
}

export interface CompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    text: string;
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  latency_ms: number;
  tokens_per_second: number;
}

export interface ActivationStats {
  layer: number;
  norm: number;
  sparsity: number;
  output_norm: number;
}

export interface ActivationCompareResponse {
  num_layers: number;
  relu_sqr_perplexity: number;
  silu_perplexity: number;
  perplexity_delta: number;
  root_cause_explanation: string;
  relu_sqr_stats: ActivationStats[];
  silu_stats: ActivationStats[];
}

export interface QuantizeResponse {
  rows: number;
  cols: number;
  scale: number;
  sparsity: number;
  original_bytes: number;
  quantized_bytes: number;
  compression_ratio: number;
  mse: number;
  rmse: number;
  cosine_similarity: number;
  sample_weights: number[][];
}

export interface BenchmarkRun {
  id: number;
  timestamp: string;
  benchmark_type: string;
  metric_name: string;
  metric_value: number;
  details: Record<string, any>;
}

export interface HazardItem {
  issue_code: string;
  severity: "error" | "warning" | "info";
  title: string;
  description: string;
  recommended_fix: string;
}

export interface CompilerDiagnosticResponse {
  compiler: string;
  version: string;
  arch: string;
  safe_to_compile: boolean;
  hazards: HazardItem[];
  recommended_cmake_flags: Record<string, string>;
}

export interface AuditLog {
  id: number;
  timestamp: string;
  user_id: string;
  action: string;
  resource: string;
  status: string;
  details: string;
}

export interface AuditLogResponse {
  total: number;
  page: number;
  page_size: number;
  data: AuditLog[];
}

export interface AdvisoryReportResponse {
  provider: string;
  model: string;
  is_advisory_only: boolean;
  grounded_findings: string[];
  analysis_text: string;
}

export interface UserProfile {
  subject: string;
  username: string;
  email: string;
  roles: string[];
}
