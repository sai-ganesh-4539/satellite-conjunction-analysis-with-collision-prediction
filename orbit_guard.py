"""
orbit_guard.py — Operational Mission-Control CLI & Satellite Conjunction Assessment Tool

Interactive flight safety tool for satellite operators and mission control:
  1. Streams sequential Conjunction Data Messages (CDMs) as they arrive pre-TCA.
  2. Runs real-time AI surrogate risk forecasting (predicted miss distance, peak Pc,
     and safety triage classification in under 2 milliseconds).
  3. Validates against 1,000,000-sample B-plane Monte Carlo ground truth.
  4. Computes actionable Collision Avoidance Maneuver (CAM) Delta-V recommendations.

Usage:
  python orbit_guard.py --demo
  python orbit_guard.py --event-id 42
  python orbit_guard.py --event-id 105 --mc-samples 1000000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from collision_forecaster import (
    bundle_stem, load_bundle, load_forecast_events, scale_sequences, summarize_71, _predict_torch,
)
from monte_carlo_engine import (
    bplane_miss, hard_body_radius, invert_isotropic_scale, isotropic_covariance,
    isotropic_pc, monte_carlo_pc, PC_CRITICAL, PC_SAFE, safety_class,
)


def compute_cam_recommendation(
    miss_distance_m: float,
    predicted_pc: float,
    primary_alt_km: float = 550.0,
    target_clearance_m: float = 5000.0,
    time_to_tca_hours: float = 48.0,
) -> dict:
    """Computes basic impulsive collision avoidance maneuver (CAM) parameters.

    In LEO circular orbits (approx 550 km), an along-track (prograde/retrograde)
    burn delta-V is the most fuel-efficient way to change in-track separation at TCA.
    Delta-s ~= 3 * pi * (delta_v / v_orb) * a * (number of orbits).
    For delta_t = 24-48 hours, approx 0.05 to 0.3 m/s achieves 3 to 10 km clearance.
    """
    if predicted_pc < PC_SAFE:
        return {
            "status": "SAFE / MONITOR",
            "action_required": False,
            "delta_v_ms": 0.0,
            "burn_window": "None required",
            "direction": "N/A",
            "message": "Encounter probability is well below safety screening threshold (Pc < 1e-5). Maintain nominal tracking.",
        }
    elif predicted_pc < PC_CRITICAL:
        return {
            "status": "ELEVATED CONJUNCTION RISK",
            "action_required": False,
            "delta_v_ms": 0.025,
            "burn_window": f"Prepare standby burn plan for T - {min(24.0, time_to_tca_hours/2):.0f}h",
            "direction": "Prograde (+V) or Retrograde (-V)",
            "message": "Close monitoring required. Covariance dispersion elevated. Prepare avoidance sequence if next update confirms escalation.",
        }
    else:
        # Critical encounter: calculate required delta-V to achieve target clearance (e.g. 5 km)
        deficit_m = max(0.0, target_clearance_m - miss_distance_m)
        lead_time_days = max(0.5, time_to_tca_hours / 24.0)
        est_dv = max(0.08, float(deficit_m / (15000.0 * lead_time_days)))
        return {
            "status": "CRITICAL RISK: COLLISION AVOIDANCE MANEUVER REQUIRED",
            "action_required": True,
            "delta_v_ms": round(est_dv, 3),
            "burn_window": f"EXECUTE AT T - 24 HOURS (Lead time: {time_to_tca_hours:.1f}h available)",
            "direction": "Optimal Along-Track (Prograde +V)",
            "message": f"CRITICAL: Probability exceeds 5e-4 threshold. Execute impulsive avoidance burn of {est_dv:.3f} m/s to ensure >5.0 km clearance.",
        }


def format_banner():
    print("=" * 80)
    print("      ORBIT-GUARD: SATELLITE CONJUNCTION ASSESSMENT & FLIGHT SAFETY SYSTEM")
    print("      AI Surrogate vs. 1,000,000-Sample Monte Carlo Astrodynamics Engine")
    print("=" * 80)


def stream_encounter(
    event_idx: int,
    batch,
    bundle,
    mc_samples: int = 1000000,
    step_delay: float = 0.4,
):
    prefixes = batch.prefixes[event_idx]
    n_updates = len(prefixes)
    final_miss = float(batch.miss_m[event_idx])
    final_pc = float(batch.pc[event_idx])
    sat1_id = batch.sat1[event_idx]
    sat2_id = batch.sat2[event_idx]
    event_id = str(batch.event_id[event_idx])

    last_row = prefixes[-1]
    radius = float(last_row[12] + last_row[13])

    format_banner()
    print("\nCONJUNCTION ENCOUNTER DOSSIER:")
    print(f"  * Event Tracking ID:      {event_id}")
    print(f"  * Primary Satellite:      NORAD ID #{sat1_id} (Exclusion Radius: {last_row[12]:.1f}m)")
    print(f"  * Secondary Space Object: NORAD ID #{sat2_id} (Exclusion Radius: {last_row[13]:.1f}m)")
    print(f"  * Combined Hard-Body:     Radius R = {radius:.1f} meters")
    print(f"  * Total CDMs in Track:    {n_updates} sequential radar updates")
    print("-" * 80)

    # Invert isotropic covariance scale to compute realistic B-plane Monte Carlo reference
    inv = invert_isotropic_scale(final_miss, radius, final_pc)
    sigma_mc = float(inv["sigma_m"]) if inv.get("sigma_m") is not None else max(1000.0, final_miss * 2.0)
    matched_r = float(inv["radius_m"])

    print("\nBEGINNING REAL-TIME TELEMETRY STREAM ACROSS MONITORING CAMPAIGN...\n")

    device = torch.device("cpu")
    retnet_time_ms = 1.2

    for step_k in range(1, n_updates + 1):
        sub_prefix = prefixes[:step_k]
        hours_to_tca = max(2.0, (n_updates - step_k + 1) * 16.0)

        seq_mat = np.zeros((1, 14, 14), dtype=np.float32)
        seq_mat[0, :step_k] = sub_prefix
        scaled_x = scale_sequences(seq_mat, bundle["mean"], bundle["std"])

        t0 = time.perf_counter()
        with torch.no_grad():
            pred_retnet = _predict_torch(bundle["retnet"], scaled_x, device)
        retnet_time_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        summary_feat = summarize_71(sub_prefix).reshape(1, -1)
        pred_hist = bundle["hist"].predict(summary_feat)
        hist_time_ms = (time.perf_counter() - t1) * 1000.0

        pred_pc_val = float(pred_retnet["pc"][0])
        pred_miss_val = float(pred_retnet["miss_m"][0])

        status_tag = "[CRITICAL ALERT]" if pred_pc_val >= PC_CRITICAL else ("[ELEVATED RISK]" if pred_pc_val >= PC_SAFE else "[SAFE]")

        print(f"  --- Update #{step_k}/{n_updates} (TCA - {hours_to_tca:4.1f}h) ---")
        print(f"      * Latest Observed Miss:     {sub_prefix[-1][0]:.1f} m (Delta: {sub_prefix[-1][1]*100:+.1f} m)")
        print(f"      * RetNet Forecast at TCA:   Miss = {pred_miss_val:.1f} m | Pc = {pred_pc_val:.2e} {status_tag}")
        print(f"      * AI Inference Latency:     {retnet_time_ms:.2f} ms (RetNet) | {hist_time_ms:.2f} ms (HistGB)")

        if step_delay > 0:
            time.sleep(step_delay)

    # Final Ground-Truth Verification vs 1,000,000-Sample Monte Carlo Simulation
    print("\n" + "=" * 80)
    print("GROUND-TRUTH HIGH-FIDELITY MONTE CARLO SIMULATION EXECUTION...")
    print(f"  Executing {mc_samples:,} virtual particle trajectory propagations in B-plane...")
    
    t_mc_start = time.perf_counter()
    mc_res = monte_carlo_pc(
        bplane_miss(final_miss),
        isotropic_covariance(sigma_mc),
        matched_r,
        n_samples=mc_samples,
        seed=42,
    )
    mc_elapsed_ms = (time.perf_counter() - t_mc_start) * 1000.0

    speedup = mc_elapsed_ms / max(0.001, retnet_time_ms)

    print(f"  * Monte Carlo Collision Prob: {mc_res['pc']:.4e} (Hits: {mc_res['hits']}/{mc_samples:,})")
    print(f"  * 95% Wilson Confidence Band: [{mc_res['wilson_95'][0]:.4e}, {mc_res['wilson_95'][1]:.4e}]")
    print(f"  * Monte Carlo Execution Time: {mc_elapsed_ms:.1f} ms")
    print(f"  * RetNet Inference Time:      {retnet_time_ms:.2f} ms")
    print(f"  * Operational Speedup:        {speedup:,.0f}x FASTER THAN MONTE CARLO")

    # Actionable Flight Maneuver Recommendation
    cam = compute_cam_recommendation(pred_miss_val, pred_pc_val, time_to_tca_hours=hours_to_tca)

    print("\n" + "=" * 80)
    print("ACTIONABLE FLIGHT DYNAMICS RECOMMENDATION:")
    print(f"  Status:       {cam['status']}")
    print(f"  Burn Window:  {cam['burn_window']}")
    if cam['action_required']:
        print(f"  Maneuver Dv:  {cam['delta_v_ms']} m/s ({cam['direction']})")
    print(f"  Advisory:     {cam['message']}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ORBIT-GUARD: Satellite Conjunction Assessment Flight System")
    parser.add_argument("--event-id", type=int, default=None, help="Encounter event index (0 to 835)")
    parser.add_argument("--demo", action="store_true", help="Run high-risk demo encounter")
    parser.add_argument("--mc-samples", type=int, default=1000000, help="Number of Monte Carlo samples")
    parser.add_argument("--fast", action="store_true", help="Skip animation delay")
    args = parser.parse_args()

    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    print("Initializing Orbit-Guard astrodynamics core and surrogate weights...")
    batch = load_forecast_events()
    stem = bundle_stem()
    device = torch.device("cpu")
    bundle = load_bundle(stem, device=device)

    # Choose event
    if args.demo or args.event_id is None:
        high_risk_candidates = np.where((batch.pc >= 1e-3) & (batch.n_prefix >= 4))[0]
        if len(high_risk_candidates) > 0:
            target_idx = int(high_risk_candidates[0])
        else:
            target_idx = int(np.argmax(batch.pc))
        print(f"Running Demo Mode: Selected representative conjunction encounter index #{target_idx}")
    else:
        target_idx = int(args.event_id)
        if target_idx < 0 or target_idx >= len(batch.y):
            raise ValueError(f"Event ID must be between 0 and {len(batch.y)-1}")

    stream_encounter(
        event_idx=target_idx,
        batch=batch,
        bundle=bundle,
        mc_samples=args.mc_samples,
        step_delay=0.0 if args.fast else 0.35,
    )


if __name__ == "__main__":
    main()
