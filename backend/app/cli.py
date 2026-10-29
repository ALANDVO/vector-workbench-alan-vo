"""CLI entrypoint for reproducible AI/ML evaluation benchmarks."""

import argparse
import json
import sys
from app.services.benchmark_service import benchmark_service
from app.services.lut_benchmark_service import lut_benchmark_service


def main():
    """Execute reproducible evaluation benchmarks via command line."""
    parser = argparse.ArgumentParser(
        description="Vector Workbench AI/ML Evaluation CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run reproducible evaluation benchmarks")
    eval_parser.add_argument(
        "--benchmark",
        choices=["perplexity", "quantization", "throughput", "lut", "all"],
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

    args = parser.parse_args()

    results = {}
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

    if args.benchmark in ("lut", "all"):
        if not args.quiet:
            print(f"Running LUT Kernel Benchmark ({args.dim}x{args.dim}x{args.dim})...")
        results["lut"] = lut_benchmark_service.run_lut_benchmark_suite(
            dim_m=args.dim, dim_n=args.dim, dim_k=args.dim
        )

    if not args.quiet:
        print("\n================== EVALUATION RESULTS ==================")
        print(json.dumps(results, indent=2))
        print("========================================================\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
