"""Tests for LUT Kernel Benchmark Service."""

import pytest
from app.services.lut_benchmark_service import (
    lut_benchmark_service,
    LUTBenchmarkService,
    LUTBenchmarkReport,
)


class TestLUTBenchmarkService:
    """Test suite for LUT kernel benchmarking."""

    def test_scalar_ternary_gemm_benchmark(self):
        """Test scalar ternary GEMM benchmark executes and returns valid metrics."""
        result = LUTBenchmarkService.benchmark_scalar_ternary_gemm(
            dim_m=32, dim_n=32, dim_k=32
        )

        assert result.group_size == 1
        assert result.dim_m == 32
        assert result.dim_n == 32
        assert result.dim_k == 32
        assert result.additions_count > 0
        assert result.multiplications_count == 0  # Ternary avoids multiplications
        assert result.lut_lookups == 0
        assert result.latency_microseconds > 0
        assert result.memory_bandwidth_mb_s > 0
        assert result.operational_intensity > 0
        assert result.speedup_vs_scalar == 1.0

    def test_lut3_kernel_benchmark(self):
        """Test 3-ary LUT kernel benchmark executes and returns valid metrics."""
        result = LUTBenchmarkService.benchmark_lut3_kernel(
            dim_m=32, dim_n=32, dim_k=32
        )

        assert result.group_size == 3
        assert result.dim_m == 32
        assert result.dim_n == 32
        assert result.dim_k == 32
        assert result.additions_count >= 0
        assert result.lut_lookups >= 0
        assert result.latency_microseconds > 0
        assert result.memory_bandwidth_mb_s > 0
        assert result.operational_intensity >= 0

    def test_lut5_kernel_benchmark(self):
        """Test 5-ary LUT kernel benchmark executes and returns valid metrics."""
        result = LUTBenchmarkService.benchmark_lut5_kernel(
            dim_m=32, dim_n=32, dim_k=32
        )

        assert result.group_size == 5
        assert result.dim_m == 32
        assert result.dim_n == 32
        assert result.dim_k == 32
        assert result.additions_count >= 0
        assert result.lut_lookups >= 0
        assert result.latency_microseconds > 0
        assert result.memory_bandwidth_mb_s > 0
        assert result.operational_intensity >= 0

    def test_full_lut_benchmark_suite(self):
        """Test complete LUT benchmark suite generates comparative report."""
        report = LUTBenchmarkService.run_lut_benchmark_suite(
            dim_m=32, dim_n=32, dim_k=32
        )

        assert isinstance(report, LUTBenchmarkReport)
        assert report.scalar_result.group_size == 1
        assert report.lut3_result.group_size == 3
        assert report.lut5_result.group_size == 5

        # Speedups should be positive numbers
        assert report.speedup_3ary_vs_scalar > 0
        assert report.speedup_5ary_vs_scalar > 0
        assert report.speedup_5ary_vs_3ary > 0

        # Bandwidth should be positive
        assert report.bandwidth_scalar_mb_s > 0
        assert report.bandwidth_lut3_mb_s > 0
        assert report.bandwidth_lut5_mb_s > 0

        # Operational intensity should be non-negative
        assert report.operational_intensity_scalar >= 0
        assert report.operational_intensity_lut3 >= 0
        assert report.operational_intensity_lut5 >= 0

    def test_different_dimensions(self):
        """Test benchmarks work with different matrix dimensions."""
        for dim in [16, 64, 128]:
            report = LUTBenchmarkService.run_lut_benchmark_suite(
                dim_m=dim, dim_n=dim, dim_k=dim
            )
            assert report.scalar_result.dim_m == dim
            assert report.lut3_result.dim_m == dim
            assert report.lut5_result.dim_m == dim

    def test_speedup_calculation_consistency(self):
        """Test that speedup calculations are internally consistent."""
        report = LUTBenchmarkService.run_lut_benchmark_suite(
            dim_m=64, dim_n=64, dim_k=64
        )

        # 5-ary vs scalar = (5-ary vs 3-ary) * (3-ary vs scalar)
        expected_5ary_vs_scalar = (
            report.speedup_5ary_vs_3ary * report.speedup_3ary_vs_scalar
        )
        # Allow small floating point difference
        assert abs(report.speedup_5ary_vs_scalar - expected_5ary_vs_scalar) < 0.1

    def test_benchmark_reproducibility(self):
        """Test that benchmarks produce consistent operation counts with same seed."""
        report1 = LUTBenchmarkService.run_lut_benchmark_suite(
            dim_m=32, dim_n=32, dim_k=32
        )
        report2 = LUTBenchmarkService.run_lut_benchmark_suite(
            dim_m=32, dim_n=32, dim_k=32
        )

        # Operation counts should be identical (deterministic with fixed seed)
        assert report1.scalar_result.additions_count == report2.scalar_result.additions_count
        assert report1.lut3_result.additions_count == report2.lut3_result.additions_count
        assert report1.lut5_result.additions_count == report2.lut5_result.additions_count
        assert report1.lut3_result.lut_lookups == report2.lut3_result.lut_lookups
        assert report1.lut5_result.lut_lookups == report2.lut5_result.lut_lookups

        # Speedups should be consistent (within small tolerance for floating point)
        assert abs(report1.speedup_3ary_vs_scalar - report2.speedup_3ary_vs_scalar) < 1.0
        assert abs(report1.speedup_5ary_vs_scalar - report2.speedup_5ary_vs_scalar) < 1.0
        assert abs(report1.speedup_5ary_vs_3ary - report2.speedup_5ary_vs_3ary) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])