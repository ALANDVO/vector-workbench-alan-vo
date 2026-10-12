"""CLI entrypoint for reproducible AI/ML evaluation benchmarks."""

import argparse
import json
import sys
from app.services.benchmark_service import benchmark_service


def main():
    """Execute reproducible evaluation benchmarks via command line."""
    parser = argparse.ArgumentParser(
        description="Vector Workbench AI/ML Evaluation CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run reproducible evaluation benchmarks")
    eval_parser.add_argument(
        "--benchmark",
        choices=["perplexity", "quantization", "throughput", "all"],
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

    args = parser.parse_args()

    results = {}
    if args.benchmark in ("perplexity", "all"):
        print(f"Running Perplexity Calibration Benchmark ({args.layers} layers)...")
        results["perplexity"] = benchmark_service.run_perplexity_benchmark(num_layers=args.layers)

    if args.benchmark in ("quantization", "all"):
        print(f"Running Quantization Fidelity Benchmark ({args.dim}x{args.dim})...")
        results["quantization"] = benchmark_service.run_quantization_benchmark(dim=args.dim)

    if args.benchmark in ("throughput", "all"):
        print(f"Running Integer Throughput Benchmark ({args.dim}x{args.dim})...")
        results["throughput"] = benchmark_service.run_throughput_benchmark(dim=args.dim)

    print("\n================== EVALUATION RESULTS ==================")
    print(json.dumps(results, indent=2))
    print("========================================================\n")


if __name__ == "__main__":
    main()
