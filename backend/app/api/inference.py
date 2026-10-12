"""Inference endpoints implementing OpenAI-compatible schema and model management."""

import math
import random
import time
from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import UserPrincipal, get_current_user, require_role
from app.core.ternary_engine import TernaryEngine
from app.models.database import db
from app.models.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
    ModelInfo,
    ModelListResponse,
    ModelLoadRequest,
    TernaryQuantizeRequest,
    TernaryQuantizeResponse,
)
from app.services.inference_service import inference_service

router = APIRouter(prefix="", tags=["Inference & Models"])


@router.get("/models", response_model=ModelListResponse)
def list_models(
    user: Annotated[UserPrincipal, Depends(get_current_user)]
):
    """List available registered models."""
    models = inference_service.list_models()
    return ModelListResponse(data=models)


@router.post("/models/load", response_model=ModelInfo)
def load_model(
    req: ModelLoadRequest,
    user: Annotated[UserPrincipal, Depends(require_role("operator"))],
):
    """Keep model resident in memory (Issue #206)."""
    info = inference_service.load_model(req.model_id)
    db.log_audit(
        user_id=user.subject,
        action="MODEL_LOAD",
        resource=f"/api/v1/models/{req.model_id}",
        status="SUCCESS",
        details=f"Loaded {req.model_id} into RAM",
    )
    return info


@router.post("/models/unload")
def unload_model(
    model_id: str,
    user: Annotated[UserPrincipal, Depends(require_role("admin"))],
):
    """Unload model from RAM to reclaim memory."""
    success = inference_service.unload_model(model_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} is not currently resident in memory",
        )
    db.log_audit(
        user_id=user.subject,
        action="MODEL_UNLOAD",
        resource=f"/api/v1/models/{model_id}",
        status="SUCCESS",
        details=f"Unloaded {model_id} from RAM",
    )
    return {"message": f"Model {model_id} unloaded successfully"}


@router.post("/completions", response_model=CompletionResponse)
def create_completion(
    req: CompletionRequest,
    user: Annotated[UserPrincipal, Depends(require_role("operator"))],
):
    """Execute text completion using in-memory model."""
    text, tokens, latency_ms, tps = inference_service.generate_completion(
        model_id=req.model,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        activation=req.activation,
    )

    db.record_inference_session(
        model_id=req.model,
        prompt=req.prompt,
        completion=text,
        tokens=tokens,
        latency_ms=latency_ms,
        user_id=user.subject,
    )

    prompt_tokens = max(1, len(req.prompt.split()))

    return CompletionResponse(
        id=f"cmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=req.model,
        choices=[CompletionChoice(index=0, text=text, finish_reason="stop")],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=tokens,
            total_tokens=prompt_tokens + tokens,
        ),
        latency_ms=latency_ms,
        tokens_per_second=tps,
    )


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def create_chat_completion(
    req: ChatCompletionRequest,
    user: Annotated[UserPrincipal, Depends(require_role("operator"))],
):
    """Execute OpenAI-compatible chat completion."""
    msg, tokens, latency_ms, tps = inference_service.generate_chat(
        model_id=req.model,
        messages=req.messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        activation=req.activation,
    )

    full_prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    db.record_inference_session(
        model_id=req.model,
        prompt=full_prompt,
        completion=msg.content,
        tokens=tokens,
        latency_ms=latency_ms,
        user_id=user.subject,
    )

    prompt_tokens = max(1, sum(len(m.content.split()) for m in req.messages))

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=req.model,
        choices=[ChatChoice(index=0, message=msg, finish_reason="stop")],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=tokens,
            total_tokens=prompt_tokens + tokens,
        ),
        latency_ms=latency_ms,
        tokens_per_second=tps,
    )


@router.post("/quantize", response_model=TernaryQuantizeResponse)
def quantize_tensor(
    req: TernaryQuantizeRequest,
    user: Annotated[UserPrincipal, Depends(require_role("operator"))],
):
    """Quantize continuous weights matrix into {-1, 0, +1} and report metrics."""
    rng = random.Random(req.seed)
    # Generate Gaussian matrix
    matrix = [
        [(rng.gauss(0.0, 1.0) / math.sqrt(req.cols)) for _ in range(req.cols)]
        for _ in range(req.rows)
    ]

    quant = TernaryEngine.quantize_weights(matrix)
    metrics = TernaryEngine.calculate_fidelity_metrics(matrix, quant)

    # Return sample slice of weights
    sample_rows = min(8, req.rows)
    sample_cols = min(8, req.cols)
    sample = [quant.weights[r][:sample_cols] for r in range(sample_rows)]

    return TernaryQuantizeResponse(
        rows=quant.rows,
        cols=quant.cols,
        scale=round(quant.scale, 6),
        sparsity=round(quant.sparsity, 4),
        original_bytes=quant.original_bytes,
        quantized_bytes=quant.quantized_bytes,
        compression_ratio=round(quant.compression_ratio, 2),
        mse=metrics["mse"],
        rmse=metrics["rmse"],
        cosine_similarity=metrics["cosine_similarity"],
        sample_weights=sample,
    )
