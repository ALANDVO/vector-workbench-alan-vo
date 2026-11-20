"""CLI entrypoint for reproducible AI/ML evaluation benchmarks."""

import argparse
import asyncio
import json
import sys
from app.services.benchmark_service import benchmark_service
from app.services.lut_benchmark_service import lut_benchmark_service
from app.services.llm_advisory import llm_advisory_service


def main():
    """Execute reproducible evaluation benchmarks via command line."""
    parser = argparse.ArgumentParser(
        description="Vector Workbench AI/ML Evaluation CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # evaluate subcommand: run benchmark suite
    eval_parser = subparsers.add_parser("evaluate", help="Run reproducible evaluation benchmarks")
    eval_parser.add_argument(
        "--benchmark",
        choices=["perplexity", "quantization", "throughput", "lut", "lut-scalar", "lut-3ary", "lut-5ary", "all"],
        default="all",
        help="Benchmark suite to execute",
    )
    eval_parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Matrix dimension for quantization and throughput benchmarks",
    )
    eval_parser.add_argument(
        "--layers",
        type=int,
        default=30,
        help="Number of transformer layers for perplexity evaluation",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path for CI/CD integration",
    )
    eval_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (useful with --output)",
    )

    # llm-advise subcommand: run perplexity evaluation and generate advisory report
    advise_parser = subparsers.add_parser(
        "llm-advise",
        help="Run perplexity evaluation and generate LLM advisory report",
    )
    advise_parser.add_argument(
        "--layers",
        type=int,
        default=30,
        help="Number of transformer layers for perplexity evaluation",
    )
    advise_parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path for CI/CD integration (includes advisory report)",
    )
    advise_parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "gemini", "ollama", "offline"],
        default="offline",
        help="LLM provider for advisory report (default: offline)",
    )
    advise_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (useful with --output)",
    )

    args = parser.parse_args()

    results = {}

    if args.command == "evaluate":
        if args.benchmark in ("perplexity", "all"):
            if not args.quiet:
                print(f"Running Perplexity Calibration Benchmark ({args.layers} layers)...")
            results["perplexity"] = benchmark_service.run_perplexity_benchmark(num_layers=args.layers)

        if args.benchmark in ("quantization", "all"):
            if not args.quiet:
                print(f"Running Quantization Fidelity Benchmark ({args.dim}x{args.dim})...")
            results["quantization"] = benchmark_service.run_quantization_benchmark(dim=args.dim)

        if args.benchmark in ("throughput", "all"):
            if not args.quiet:
                print(f"Running Integer Throughput Benchmark ({args.dim}x{args.dim})...")
            results["throughput"] = benchmark_service.run_throughput_benchmark(dim=args.dim)

        if args.benchmark in ("lut-scalar", "all"):
            if not args.quiet:
                print(f"Running LUT Scalar Ternary GEMM Benchmark ({args.dim}x{args.dim}x{args.dim})...")
            results["lut_scalar"] = lut_benchmark_service.benchmark_scalar_ternary_gemm(
                dim_m=args.dim, dim_n=args.dim, dim_k=args.dim
            ).__dict__

        if args.benchmark in ("lut-3ary", "all"):
            if not args.quiet:
                print(f"Running LUT 3-ary Kernel Benchmark ({args.dim}x{args.dim}x{args.dim})...")
            results["lut_3ary"] = lut_benchmark_service.benchmark_lut3_kernel(
                dim_m=args.dim, dim_n=args.dim, dim_k=args.dim
            ).__dict__

        if args.benchmark in ("lut-5ary", "all"):
            if not args.quiet:
                print(f"Running LUT 5-ary Kernel Benchmark ({args.dim}x{args.dim}x{args.dim})...")
            results["lut_5ary"] = lut_benchmark_service.benchmark_lut5_kernel(
                dim_m=args.dim, dim_n=args.dim, dim_k=args.dim
            ).__dict__

        if args.benchmark in ("lut", "all"):
            if not args.quiet:
                print(f"Running LUT Kernel Benchmark Suite ({args.dim}x{args.dim}x{args.dim})...")
            results["lut"] = lut_benchmark_service.run_lut_benchmark_suite(
                dim_m=args.dim, dim_n=args.dim, dim_k=args.dim
            ).__dict__

        if not args.quiet:
            print("\n================== EVALUATION RESULTS ==================")
            print(json.dumps(results, indent=2))
            print("========================================================\n")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            if not args.quiet:
                print(f"Results written to {args.output}")

    elif args.command == "llm-advise":
        # Run perplexity evaluation (synchronous)
        if not args.quiet:
            print(f"Running Perplexity Calibration Benchmark ({args.layers} layers)...")
        ppl_results = benchmark_service.run_perplexity_benchmark(num_layers=args.layers)
        results["perplexity"] = ppl_results

        # Generate LLM advisory report (async)
        if not args.quiet:
            print(f"Generating LLM advisory report using provider='{args.provider}'...")

        benchmark_summary = {
            "metric": "Perplexity Calibration",
            "relu_sqr_ppl": ppl_results["relu_sqr_perplexity"],
            "silu_ppl": ppl_results["silu_perplexity"],
            "norm_drift": ppl_results["relu_sqr_norm_drift"],
            "compression": "8.0x (1.58-bit packed)",
        }

        advisory_report = asyncio.run(
            llm_advisory_service.generate_advisory_report(
                benchmark_summary=benchmark_summary,
                custom_context="",
            )
        )

        results["advisory_report"] = {
            "provider": advisory_report.provider,
            "model": advisory_report.model,
            "is_advisory_only": advisory_report.is_advisory_only,
            "grounded_findings": advisory_report.grounded_findings,
            "analysis_text": advisory_report.analysis_text,
        }

        if not args.quiet:
            print("\n================== ADVISORY REPORT ==================")
            print(f"Provider: {advisory_report.provider}")
            print(f"Model: {advisory_report.model}")
            print(f"Is Advisory Only: {advisory_report.is_advisory_only}")
            print(f"\nGrounded Findings:")
            for finding in advisory_report.grounded_findings:
                print(f"  - {finding}")
            print(f"\nAnalysis:\n{advisory_report.analysis_text}")
            print("====================================================\n")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            if not args.quiet:
                print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
