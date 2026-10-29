# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

feature-evaluation-cli-enhancement

## Roadmap

### feature-core-inference-engine — Deterministic 1-bit / 1.58-bit Ternary Matrix Engine, FFN Squared ReLU Graph, and In-Memory Server

Status: done

Acceptance:
- Quantize continuous weights into ternary {-1, 0, +1} values with deterministic per-tensor scaling
- Implement FFN activation forward graph comparing Squared ReLU and SiLU with perplexity metrics
- Provide in-memory model execution and persistent chat completion API without reloading weights
- Support token-based role authentication with secure local demo fallback

Implementation: `backend/app/main.py`, `backend/app/core/ternary_engine.py`, `backend/app/core/activation_graph.py`, `backend/app/services/inference_service.py`, `backend/app/api/inference.py`, `backend/app/api/auth.py`, `frontend/src/App.tsx`, `frontend/src/components/InferencePlayground.tsx`

Tests: `backend/tests/test_ternary_engine.py`, `backend/tests/test_activation_graph.py`, `backend/tests/test_inference_api.py`, `frontend/src/__tests__/InferencePlayground.test.tsx`

### feature-build-diagnostics — Compiler & Environment Matrix Diagnostic Suite

Status: done

Acceptance:
- Matrix scanner identifies compiler flags, Clang 20 const-pointer compatibility, and ARM vector type hazards
- Diagnostic reporting exports environment verification recipes

Implementation: `backend/app/services/diagnostics_service.py`, `backend/app/api/diagnostics.py`, `backend/tests/test_benchmarks_and_diagnostics.py`, `frontend/src/components/DiagnosticsMatrix.tsx`

Tests: `backend/tests/test_benchmarks_and_diagnostics.py::test_evaluate_compiler_matrix_clang20_hazard`, `backend/tests/test_benchmarks_and_diagnostics.py::test_evaluate_compiler_matrix_arm_hazard`

### feature-lut-kernel-benchmarks — AVX-512 & NEON 5-Activation Ternary Look-Up Table Kernel Profiler

Status: done

Acceptance:
- Benchmarks multi-activation LUT packing throughput against standard scalar ternary dot products
- Generates hardware memory bandwidth and operational intensity comparison charts

Implementation: `backend/app/services/lut_benchmark_service.py`, `backend/app/api/benchmarks.py`, `backend/tests/test_lut_benchmarks.py`, `frontend/src/components/LUTBenchmarkProfiler.tsx`, `frontend/src/__tests__/LUTBenchmarkProfiler.test.tsx`

Tests: `backend/tests/test_lut_benchmarks.py`

### feature-evaluation-cli-enhancement — Enhanced Evaluation CLI with LUT Kernel Benchmark Support and JSON Export

Status: in_progress

Acceptance:
- CLI supports running LUT kernel benchmarks (scalar, 3-ary, 5-ary) via command line
- CLI supports JSON output export for CI/CD integration
- CLI includes all benchmark types in --benchmark all option

Implementation: `backend/app/cli.py`

Tests: `backend/tests/test_benchmarks_and_diagnostics.py`

### feature-llm-advisory-eval — Multi-Provider Opt-In LLM Advisory Analysis & Evaluation CLI

Status: planned

Acceptance:
- Deterministic evaluation CLI measures quantization perplexity across custom test corpora
- Optional provider adapter formats evaluation reports using OpenAI/Anthropic/Gemini/Ollama endpoints

