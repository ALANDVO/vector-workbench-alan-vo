# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

feature-core-inference-engine

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

Status: planned

Acceptance:
- Matrix scanner identifies compiler flags, Clang 20 const-pointer compatibility, and ARM vector type hazards
- Diagnostic reporting exports environment verification recipes

### feature-lut-kernel-benchmarks — AVX-512 & NEON 5-Activation Ternary Look-Up Table Kernel Profiler

Status: planned

Acceptance:
- Benchmarks multi-activation LUT packing throughput against standard scalar ternary dot products
- Generates hardware memory bandwidth and operational intensity comparison charts

### feature-llm-advisory-eval — Multi-Provider Opt-In LLM Advisory Analysis & Evaluation CLI

Status: planned

Acceptance:
- Deterministic evaluation CLI measures quantization perplexity across custom test corpora
- Optional provider adapter formats evaluation reports using OpenAI/Anthropic/Gemini/Ollama endpoints

