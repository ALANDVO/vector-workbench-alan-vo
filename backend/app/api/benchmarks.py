"""Benchmark and numerical calibration endpoints."""

from typing import Annotated, Any, Dict, List
from fastapi import APIRouter, Depends

from app.core.activation_graph import ActivationGraph
from app.core.security import UserPrincipal, get_current_user, require_role
from app.models.database import db
from app.models.schemas import (
    ActivationCompareRequest,
    ActivationCompareResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
)
from app.services.benchmark_service import benchmark_service

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks & Calibration"])


@router.post("/activation-compare", response_model=ActivationCompareResponse)
def compare_activations(
    req: ActivationCompareRequest,
    user: Annotated[UserPrincipal, Depends(require_role("operator"))],
):
    """Compare multi-layer transformer FFN activation: Squared ReLU vs SiLU (Issue #588)."""
    relu_report = ActivationGraph.simulate_calibration(
        num_layers=req.num_layers,
        activation="relu_sqr",
        hidden_dim=req.hidden_dim,
        intermediate_dim=req.intermediate_dim,
        seed=req.seed,
    )
    silu_report = ActivationGraph.simulate_calibration(
        num_layers=req.num_layers,
        activation="silu",
        hidden_dim=req.hidden_dim,
        intermediate_dim=req.intermediate_dim,
        seed=req.seed,
    )

    relu_stats = [
        {
            "layer": s.layer_index,
            "norm": s.activation_norm,
            "sparsity": s.sparsity_ratio,
            "output_norm": s.output_norm,
        }
        for s in relu_report.layer_stats
    ]
    silu_stats = [
        {
            "layer": s.layer_index,
            "norm": s.activation_norm,
            "sparsity": s.sparsity_ratio,
            "output_norm": s.output_norm,
        }
        for s in silu_report.layer_stats
    ]

    delta = round(silu_report.calculated_perplexity - relu_report.calculated_perplexity, 2)

    return ActivationCompareResponse(
        num_layers=req.num_layers,
        relu_sqr_perplexity=relu_report.calculated_perplexity,
        silu_perplexity=silu_report.calculated_perplexity,
        perplexity_delta=delta,
        root_cause_explanation=(
            "BitNet b1.58 relies on Squared ReLU (relu(x)²) in the FFN sub-layer. "
            "Upstream mistakenly configured LLM_FFN_SILU. SiLU shifts activations into "
            "sub-norm compounding scales over 30 layers, exploding perplexity from 17.1 to 99.8. "
            "Restoring LLM_FFN_RELU_SQR resolves the degradation across all CPU backends."
        ),
        relu_sqr_stats=relu_stats,
        silu_stats=silu_stats,
    )


@router.post("/run", response_model=BenchmarkRunResponse)
def run_benchmark(
    req: BenchmarkRunRequest,
    user: Annotated[UserPrincipal, Depends(require_role("operator"))],
):
    """Execute a selected benchmark suite and record results."""
    if req.benchmark_type == "perplexity":
        details = benchmark_service.run_perplexity_benchmark(num_layers=req.num_layers)
        metric_name = "perplexity_gap"
        metric_val = details["perplexity_gap"]
    elif req.benchmark_type == "throughput":
        details = benchmark_service.run_throughput_benchmark(dim=req.matrix_dim)
        metric_name = "speedup_5ary_vs_3ary"
        metric_val = details["speedup_5ary_vs_3ary"]
    else:
        details = benchmark_service.run_quantization_benchmark(dim=req.matrix_dim)
        metric_name = "cosine_similarity"
        metric_val = details["cosine_similarity"]

    history = db.get_benchmark_runs(limit=1)
    record_id = history[0]["id"] if history else 1

    return BenchmarkRunResponse(
        id=record_id,
        benchmark_type=req.benchmark_type,
        metric_name=metric_name,
        metric_value=metric_val,
        details=details,
    )


@router.get("/history")
def get_benchmark_history(
    user: Annotated[UserPrincipal, Depends(get_current_user)],
):
    """Retrieve historical benchmark experiment records."""
    runs = db.get_benchmark_runs(limit=25)
    return {"runs": runs}
