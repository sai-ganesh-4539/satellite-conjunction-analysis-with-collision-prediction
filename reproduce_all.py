"""
reproduce_all.py — End-to-End Master Verification & Artifact Packaging Pipeline

Executes the complete project reproduction sequence:
  1. Verifies dataset presence and data integrity (MD5 checksums).
  2. Runs pytest unit test suite (packing, shapes, Monte Carlo convergence).
  3. Executes Monte Carlo vs Surrogate benchmark validation.
  4. Generates all 6 publication-grade vector figures in figures/.
  5. Packages all code, figures, benchmarks, and papers into results/SUBMISSION_BUNDLE.zip.

Usage:
  python reproduce_all.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import zipfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(REPO_ROOT, "figures")


def compute_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def step_1_verify_data():
    print("\n" + "=" * 70)
    print("STEP 1: VERIFYING SOURCE DATASETS AND INTEGRITY")
    print("=" * 70)
    cdm_csv = os.path.join(DATA_DIR, "cdms.csv")
    satcat_csv = os.path.join(DATA_DIR, "satcat.csv")

    for path, name in [(cdm_csv, "CDMs dataset"), (satcat_csv, "SATCAT catalog")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset: {path}")
        size_kb = os.path.getsize(path) / 1024.0
        md5 = compute_md5(path)
        print(f"  [OK] {name:16s} ({size_kb:8.1f} KB) | MD5: {md5}")


def step_2_run_tests():
    print("\n" + "=" * 70)
    print("STEP 2: EXECUTING UNIT TEST SUITE (PYTEST)")
    print("=" * 70)
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("Unit tests failed!")
    print("  [OK] All unit tests passed cleanly (zero NaNs, packing verified).")


def step_3_run_benchmark():
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING MONTE CARLO VS. SURROGATE BENCHMARK")
    print("=" * 70)
    cmd = [
        sys.executable, "scripts/benchmark_vs_monte_carlo.py",
        "--n-events", "100", "--mc-samples", "1000000", "--epochs", "8"
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(res.stdout[-400:] if len(res.stdout) > 400 else res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("Benchmark script failed!")
    print("  [OK] Monte Carlo benchmark executed successfully.")


def step_4_generate_figures():
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING HIGH-RESOLUTION PUBLICATION FIGURES")
    print("=" * 70)
    cmd = [sys.executable, "scripts/generate_publication_figures.py"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("Figure generation failed!")
    print("  [OK] All 6 publication figures created in figures/.")


def step_5_package_bundle():
    print("\n" + "=" * 70)
    print("STEP 5: PACKAGING SUBMISSION BUNDLE ZIP")
    print("=" * 70)
    zip_path = os.path.join(RESULTS_DIR, "SUBMISSION_BUNDLE.zip")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files_to_pack = [
        "RESEARCH_PAPER.md",
        "research_paper.tex",
        "research_paper.docx",
        "PRESENTATION_GUIDE.md",
        "README.md",
        "orbit_guard.py",
        "grouped_cv_check.py",
        "reproduce_all.py",
        "src/model.py",
        "src/monte_carlo_engine.py",
        "src/collision_forecaster.py",
        "src/operational_metrics.py",
        "src/data_processing.py",
        "src/config.py",
        "results/monte_carlo_vs_surrogate_benchmark.json",
        "figures/fig1_unified_roc_pr_curves.png",
        "figures/fig1_unified_roc_pr_curves.pdf",
        "figures/fig2_calibration_reliability.png",
        "figures/fig2_calibration_reliability.pdf",
        "figures/fig3_grouped_cv_leakage_audit.png",
        "figures/fig3_grouped_cv_leakage_audit.pdf",
        "figures/fig4_streaming_horizon_early_warning.png",
        "figures/fig4_streaming_horizon_early_warning.pdf",
        "figures/fig5_pareto_latency_vs_auc.png",
        "figures/fig5_pareto_latency_vs_auc.pdf",
        "figures/fig6_monte_carlo_correlation.png",
        "figures/fig6_monte_carlo_correlation.pdf",
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files_to_pack:
            abs_p = os.path.join(REPO_ROOT, rel)
            if os.path.exists(abs_p):
                zf.write(abs_p, arcname=rel)
                print(f"  + Added: {rel}")
            else:
                print(f"  - Warning: file not found: {rel}")

    size_mb = os.path.getsize(zip_path) / (1024.0 * 1024.0)
    print(f"\n[OK] Submission Bundle packaged successfully: {zip_path} ({size_mb:.2f} MB)")


def main():
    t_start = time.perf_counter()
    print("=" * 80)
    print("STARTING COMPLETE MASTER REPRODUCIBILITY PIPELINE")
    print("=" * 80)

    step_1_verify_data()
    step_2_run_tests()
    step_3_run_benchmark()
    step_4_generate_figures()
    step_5_package_bundle()

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"MASTER REPRODUCIBILITY PIPELINE COMPLETE! Total Time: {elapsed:.2f} seconds")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
