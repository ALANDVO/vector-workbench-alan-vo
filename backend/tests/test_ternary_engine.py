"""Unit tests for TernaryEngine quantization and integer matrix arithmetic."""

import pytest
from app.core.ternary_engine import TernaryEngine


def test_quantize_weights_structure_and_scale():
    matrix = [
        [0.8, -0.4, 0.05],
        [-0.9, 0.02, 0.75],
    ]
    quant = TernaryEngine.quantize_weights(matrix)

    assert quant.rows == 2
    assert quant.cols == 3
    assert quant.scale > 0.0

    # Ensure all quantized weights belong to {-1, 0, 1}
    for row in quant.weights:
        for val in row:
            assert val in (-1, 0, 1)

    assert quant.compression_ratio > 1.0


def test_quantize_empty_matrix_raises():
    with pytest.raises(ValueError):
        TernaryEngine.quantize_weights([])


def test_dequantize_fidelity():
    matrix = [
        [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0],
    ]
    quant = TernaryEngine.quantize_weights(matrix)
    recon = TernaryEngine.dequantize_weights(quant)

    assert len(recon) == 2
    assert len(recon[0]) == 3
    # Check reconstruction preserves signs
    assert recon[0][0] > 0
    assert recon[0][1] < 0
    assert recon[0][2] == 0


def test_matmul_ternary_additions_only():
    w_mat = [
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0, -1.0],
    ]
    quant = TernaryEngine.quantize_weights(w_mat)

    x = [[2.0, 3.0, 4.0, 5.0]]
    res = TernaryEngine.matmul_ternary(x, quant, lut_group_size=3)

    assert len(res.output) == 1
    assert len(res.output[0]) == 2
    assert res.additions_count > 0
    assert res.subtractions_count > 0
    assert res.multiplications_count == 2  # Only output scaling
    assert res.latency_microseconds >= 0.0


def test_matmul_dimension_mismatch():
    quant = TernaryEngine.quantize_weights([[1.0, -1.0]])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        TernaryEngine.matmul_ternary([[1.0, 2.0, 3.0]], quant)


def test_quantize_activations_absmax():
    acts = [[-10.0, 0.0, 5.0, 10.0]]
    quant_act, scale = TernaryEngine.quantize_activations(acts, bits=8)

    assert quant_act[0][0] == -127
    assert quant_act[0][1] == 0
    assert quant_act[0][3] == 127
    assert scale > 0.0


def test_fidelity_metrics_cosine_similarity():
    matrix = [
        [0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ]
    quant = TernaryEngine.quantize_weights(matrix)
    metrics = TernaryEngine.calculate_fidelity_metrics(matrix, quant)

    assert metrics["cosine_similarity"] > 0.9
    assert metrics["mse"] >= 0.0
    assert 0.0 <= metrics["sparsity"] <= 1.0
