"""Pydantic schemas for API requests, responses, and domain entities."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information regarding a registered 1-bit or quantized model."""

    model_config = {"protected_namespaces": ()}

    id: str
    name: str
    parameter_count: int
    quant_type: str
    memory_mb: float
    is_loaded: bool
    context_length: int = 2048
    description: str = ""


class ModelListResponse(BaseModel):
    """List of available models in the registry."""

    data: List[ModelInfo]


class ModelLoadRequest(BaseModel):
    """Request to keep model memory-resident in RAM."""

    model_config = {"protected_namespaces": ()}

    model_id: str
    prewarm_tensors: bool = True


class ChatMessage(BaseModel):
    """Single message in a conversational thread."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "bitnet-b1.58-2b-i2s"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    activation: Literal["relu_sqr", "silu"] = "relu_sqr"


class ChatChoice(BaseModel):
    """Chat completion choice structure."""

    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    """Token usage counter."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: CompletionUsage
    latency_ms: float
    tokens_per_second: float


class CompletionRequest(BaseModel):
    """Text completion request."""

    model: str = "bitnet-b1.58-2b-i2s"
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    activation: Literal["relu_sqr", "silu"] = "relu_sqr"


class CompletionChoice(BaseModel):
    """Text completion choice."""

    index: int
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    """Text completion response with latency metrics."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage
    latency_ms: float
    tokens_per_second: float


class TernaryQuantizeRequest(BaseModel):
    """Request to quantize a continuous tensor and inspect metrics."""

    rows: int = Field(default=64, ge=2, le=512)
    cols: int = Field(default=64, ge=2, le=512)
    seed: int = Field(default=42)
    lut_group_size: int = Field(default=3, ge=1, le=5)


class TernaryQuantizeResponse(BaseModel):
    """Metrics and fidelity analysis of ternary quantization."""

    rows: int
    cols: int
    scale: float
    sparsity: float
    original_bytes: int
    quantized_bytes: int
    compression_ratio: float
    mse: float
    rmse: float
    cosine_similarity: float
    sample_weights: List[List[int]]


class ActivationCompareRequest(BaseModel):
    """Request to compare Squared ReLU vs SiLU across layers."""

    num_layers: int = Field(default=30, ge=1, le=48)
    hidden_dim: int = Field(default=64, ge=16, le=256)
    intermediate_dim: int = Field(default=128, ge=32, le=512)
    seed: int = Field(default=42)


class ActivationCompareResponse(BaseModel):
    """Comparative report for Squared ReLU vs SiLU."""

    num_layers: int
    relu_sqr_perplexity: float
    silu_perplexity: float
    perplexity_delta: float
    root_cause_explanation: str
    relu_sqr_stats: List[Dict[str, Any]]
    silu_stats: List[Dict[str, Any]]


class BenchmarkRunRequest(BaseModel):
    """Request to execute a benchmark suite."""

    benchmark_type: Literal["perplexity", "quantization_fidelity", "throughput", "lut_kernels"]
    matrix_dim: int = Field(default=128, ge=32, le=512)
    num_layers: int = Field(default=30, ge=1, le=48)


class LUTBenchmarkRequest(BaseModel):
    """Request to run LUT kernel benchmark suite."""

    dim_m: int = Field(default=128, ge=16, le=512)
    dim_n: int = Field(default=128, ge=16, le=512)
    dim_k: int = Field(default=128, ge=16, le=512)


class LUTKernelResultSchema(BaseModel):
    """Single LUT kernel benchmark result."""

    group_size: int
    dim_m: int
    dim_n: int
    dim_k: int
    additions_count: int
    multiplications_count: int
    lut_lookups: int
    latency_microseconds: float
    memory_bandwidth_mb_s: float
    operational_intensity: float
    speedup_vs_scalar: float


class LUTBenchmarkResponse(BaseModel):
    """Complete LUT benchmark comparison report."""

    scalar_result: LUTKernelResultSchema
    lut3_result: LUTKernelResultSchema
    lut5_result: LUTKernelResultSchema
    speedup_3ary_vs_scalar: float
    speedup_5ary_vs_scalar: float
    speedup_5ary_vs_3ary: float
    bandwidth_scalar_mb_s: float
    bandwidth_lut3_mb_s: float
    bandwidth_lut5_mb_s: float
    operational_intensity_scalar: float
    operational_intensity_lut3: float
    operational_intensity_lut5: float


class BenchmarkRunResponse(BaseModel):
    """Benchmark outcome record."""

    id: int
    benchmark_type: str
    metric_name: str
    metric_value: float
    details: Dict[str, Any]


class CompilerDiagnosticRequest(BaseModel):
    """Request to analyze compiler matrix and build options."""

    compiler_name: str = "clang"
    compiler_version: str = "20.1.0"
    target_arch: Literal["x86_64", "aarch64"] = "x86_64"
    cflags: List[str] = Field(default_factory=list)


class HazardItem(BaseModel):
    """Identified build hazard or compiler incompatibility."""

    issue_code: str
    severity: Literal["error", "warning", "info"]
    title: str
    description: str
    recommended_fix: str


class CompilerDiagnosticResponse(BaseModel):
    """Diagnostics evaluation output."""

    compiler: str
    version: str
    arch: str
    safe_to_compile: bool
    hazards: List[HazardItem]
    recommended_cmake_flags: Dict[str, str]


class AdvisoryReportRequest(BaseModel):
    """Request for opt-in advisory analysis."""

    benchmark_id: Optional[int] = None
    custom_context: str = ""


class AdvisoryReportResponse(BaseModel):
    """Advisory analysis report."""

    provider: str
    model: str
    is_advisory_only: bool = True
    grounded_findings: List[str]
    analysis_text: str


class AuditLogResponse(BaseModel):
    """Paginated audit logs response."""

    total: int
    page: int
    page_size: int
    data: List[Dict[str, Any]]


class TokenResponse(BaseModel):
    """OAuth/OIDC token response envelope."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    roles: List[str]
