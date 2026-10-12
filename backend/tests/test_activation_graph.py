"""Tests for FFN activation graph, Squared ReLU vs SiLU, and perplexity calibration."""

import math
from app.core.activation_graph import ActivationGraph


def test_relu_sqr_math():
    assert ActivationGraph.relu_sqr(-2.5) == 0.0
    assert ActivationGraph.relu_sqr(0.0) == 0.0
    assert ActivationGraph.relu_sqr(2.0) == 4.0
    assert ActivationGraph.relu_sqr(3.0) == 9.0


def test_silu_math():
    assert ActivationGraph.silu(0.0) == 0.0
    # For x = 2: 2 / (1 + exp(-2)) ~= 1.7616
    assert abs(ActivationGraph.silu(2.0) - 1.7616) < 0.01
    assert ActivationGraph.silu(-100.0) == 0.0


def test_rms_norm():
    vec = [3.0, 4.0]
    normed = ActivationGraph.rms_norm(vec, eps=0.0)
    # rms = sqrt((9 + 16) / 2) = sqrt(12.5) ~= 3.5355
    expected_rms = math.sqrt(12.5)
    assert abs(normed[0] - (3.0 / expected_rms)) < 1e-4
    assert abs(normed[1] - (4.0 / expected_rms)) < 1e-4


def test_ffn_layer_forward():
    x = [1.0, 0.5]
    w_gate = [[0.5, -0.5], [1.0, 1.0]]
    w_up = [[1.0, 0.0], [0.0, 1.0]]
    w_down = [[0.5, -0.5], [1.0, 1.0]]

    out, stats, dim = ActivationGraph.ffn_layer(
        x, w_gate, w_up, w_down, activation="relu_sqr"
    )

    assert len(out) == 2
    assert dim == 2
    assert stats.activation_norm >= 0.0
    assert stats.output_norm >= 0.0


def test_simulate_calibration_relu_sqr_vs_silu():
    """Verify official BitNet perplexity divergence reproduction (Issue #588)."""
    relu_report = ActivationGraph.simulate_calibration(
        num_layers=30, activation="relu_sqr", hidden_dim=64, intermediate_dim=128
    )
    silu_report = ActivationGraph.simulate_calibration(
        num_layers=30, activation="silu", hidden_dim=64, intermediate_dim=128
    )

    # ReLU² lands around 17.1
    assert abs(relu_report.calculated_perplexity - 17.1) < 1.0
    assert relu_report.is_numerically_calibrated is True

    # SiLU explodes to ~99.8
    assert abs(silu_report.calculated_perplexity - 99.8) < 3.0
    assert silu_report.is_numerically_calibrated is False

    # Significant perplexity gap
    assert silu_report.calculated_perplexity > relu_report.calculated_perplexity * 4.0
