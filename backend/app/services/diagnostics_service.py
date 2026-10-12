"""Compiler and build environment diagnostics.

Identifies known build failures and compiler traps reported in public issues
(Issues #180, #185, #334, #480) and generates validated CMake build configurations.
"""

import platform
import shutil
import subprocess
from typing import Dict, List
from app.models.schemas import CompilerDiagnosticResponse, HazardItem


class DiagnosticsService:
    """Evaluates host compiler toolchains and flags against known build traps."""

    @staticmethod
    def detect_host_environment() -> Dict[str, str]:
        """Detect underlying host compiler and architecture."""
        cc = shutil.which("clang") or shutil.which("gcc") or "unknown"
        version_str = "unknown"
        if cc != "unknown":
            try:
                res = subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=5)
                first_line = res.stdout.splitlines()[0] if res.stdout else ""
                version_str = first_line
            except Exception:
                version_str = "detected"

        return {
            "compiler": cc,
            "version": version_str,
            "machine": platform.machine(),
            "system": platform.system(),
        }

    @classmethod
    def evaluate_compiler_matrix(
        cls,
        compiler_name: str = "clang",
        compiler_version: str = "20.1.0",
        target_arch: str = "x86_64",
        cflags: List[str] = None,
    ) -> CompilerDiagnosticResponse:
        """Scan compiler, version, and architecture for known BitNet build traps."""
        cflags = cflags or []
        hazards: List[HazardItem] = []

        # Check Issue #480: Clang 20 const-correctness error in ggml-bitnet-mad.cpp
        if "clang" in compiler_name.lower():
            major_version = 0
            try:
                major_version = int(compiler_version.split(".")[0])
            except (ValueError, IndexError):
                pass

            if major_version >= 20:
                hazards.append(
                    HazardItem(
                        issue_code="CLANG20-CONST-CORRECTNESS",
                        severity="error",
                        title="Clang 20 Const Pointer Qualifier Discard (Issue #480)",
                        description=(
                            "Clang 20 strictly enforces const qualifiers. Assigning 'const int8_t*' "
                            "to 'int8_t* y_col' in ggml-bitnet-mad.cpp:811 causes build failure."
                        ),
                        recommended_fix=(
                            "Patch ggml-bitnet-mad.cpp line 811: change 'int8_t * y_col' to 'const int8_t * y_col', "
                            "or pin compiler toolchain to Clang 18/19."
                        ),
                    )
                )

        # Check Issue #180: Unrecognized warning flags -Wunreachable-code-break
        for flag in cflags:
            if "unreachable-code-break" in flag or "unreachable-code-return" in flag:
                hazards.append(
                    HazardItem(
                        issue_code="CMAKE-UNRECOGNIZED-FLAG",
                        severity="error",
                        title="Unrecognized Command-Line Flag (Issue #180)",
                        description=(
                            f"Flag '{flag}' is not supported across modern Clang/GCC releases "
                            "and causes CMake build aborts."
                        ),
                        recommended_fix="Remove unreachable-code variants from CMakeLists.txt CFLAGS.",
                    )
                )

        # Check Issue #185: ARM Neon type conversion error
        if target_arch in ("aarch64", "arm64"):
            hazards.append(
                HazardItem(
                    issue_code="ARM-NEON-TYPE-CONVERSION",
                    severity="error",
                    title="ARM NEON int16x8_t to const int8x16_t Cast Error (Issue #185)",
                    description=(
                        "In bitnet-lut-kernels.h line 190, vdupq_n_s16(0) returns int16x8_t which "
                        "cannot implicitly convert to const int8x16_t on GCC/Clang ARM backends."
                    ),
                    recommended_fix="Use vreinterpretq_s8_s16(vdupq_n_s16(0)) or vdupq_n_s8(0).",
                )
            )

        # Check Issue #334: Missing chrono header
        hazards.append(
            HazardItem(
                issue_code="STD-CHRONO-MISSING-HEADER",
                severity="warning",
                title="Missing <chrono> Include in Common Utilities (Issue #334)",
                description=(
                    "Building llama.cpp common/common.cpp without explicit #include <chrono> "
                    "fails on modern C++20 toolchains."
                ),
                recommended_fix="Add '#include <chrono>' to the top of 3rdparty/llama.cpp/common/common.cpp.",
            )
        )

        cmake_flags = {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_COMPILER": "clang",
            "CMAKE_CXX_COMPILER": "clang++",
            "LLAMA_BITNET": "ON",
        }
        if target_arch in ("aarch64", "arm64"):
            cmake_flags["LLAMA_ARM_TL1"] = "ON"
        else:
            cmake_flags["LLAMA_AVX512"] = "ON"

        safe = not any(h.severity == "error" for h in hazards)

        return CompilerDiagnosticResponse(
            compiler=compiler_name,
            version=compiler_version,
            arch=target_arch,
            safe_to_compile=safe,
            hazards=hazards,
            recommended_cmake_flags=cmake_flags,
        )


diagnostics_service = DiagnosticsService()
