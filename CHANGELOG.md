# Changelog

## 1.5.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.4.1 — work-in-progress checkpoint

### Added
- `llm-advise` subcommand to the Evaluation CLI for multi-provider LLM advisory analysis
- Perplexity calibration benchmark (Squared ReLU vs SiLU) with deterministic offline heuristic
- Optional provider adapter formats (OpenAI/Anthropic/Gemini/Ollama) for evaluation reports
- JSON output export for CI/CD integration including advisory report fields

### Changed
- Updated `backend/app/cli.py` with new `llm-advise` subcommand and `--provider` option

## 1.4.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.3.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.2.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.1.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.0.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

All notable changes to Vector Workbench are documented in this file.

## [1.0.0] - 2026-10-12

### Added
- Core 1-bit / 1.58-bit ternary matrix quantization engine with AbsMax activation scaling.
- FFN activation forward graph comparing Squared ReLU (`relu(x)²`) and SiLU activations, reproducing the 17.1 vs 99.8 perplexity divergence across multi-layer transformers.
- In-memory model server keeping weights memory-resident with chat/completion API and latency instrumentation.
- Role-based authorization supporting OIDC Bearer tokens with viewer, operator, and admin permissions.
- Local demo authentication provider gated to localhost development environments.
- Comprehensive test suite covering matrix quantization, FFN graph numerics, in-memory inference, and auth boundaries.
- Docker compose configuration with Keycloak SSO integration and persistent storage.

Maintainer: Alan Vo <alanvo@gmail.com>

---

Alan Vo · [alanvo.net](https://alanvo.net) · [X: @alanvoasi](https://x.com/alanvoasi)
