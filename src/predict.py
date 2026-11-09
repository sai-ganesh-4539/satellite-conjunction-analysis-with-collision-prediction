"""
predict.py

Single-CDM inference: predict collision probability for new CDM data.
Useful for live deployment or quick demos.

Usage:
    python src/predict.py --cdm_id 12345
    python src/predict.py --min_rng 0.05 --pc 0.001
"""

import argparse
import numpy as np
import torch
from model import RetNetForCollisionPrediction


def predict_single(
    min_rng=0.1,
    pc=1e-5,
    sat1_rcs=1.0,
    sat2_rcs=1.0,
    sat1_excl=10.0,
    sat2_excl=10.0,
    emergency=0,
    sat1_type=0,
    sat2_type=0,
    hours_to_tca=48.0,
    hours_since_created=0.0,
    hour_of_day=12.0,
    day_of_week=2.0,
    days_until_tca=2.0,
    checkpoint="checkpoints/best_retnet.pt",
    scaler_path="data/processed/scaler.pt",
):
    """Predict collision probability from a single CDM's features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", 14)
    model = RetNetForCollisionPrediction(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Build feature vector (14 features)
    features = np.array([
        min_rng, pc, sat1_rcs, sat2_rcs, sat1_excl, sat2_excl, emergency,
        sat1_type, sat2_type,
        hours_to_tca, hours_since_created, hour_of_day, day_of_week, days_until_tca,
    ], dtype=np.float32)

    # Normalize
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = torch.load(scaler_path, weights_only=False)
        features = scaler.transform(features.reshape(1, -1)).flatten()
    except Exception:
        print("  Warning: Could not load scaler, using raw features")

    # Create sequence: single CDM at last position, rest padded
    X = np.zeros((1, 25, len(features)), dtype=np.float32)
    X[0, -1, :] = features  # put at last timestep

    X_tensor = torch.FloatTensor(X).to(device)

    # Predict
    with torch.no_grad():
        logit = model(X_tensor).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    # Output
    print(f"\n{'='*50}")
    print(f"  CDM Prediction Result")
    print(f"{'='*50}")
    print(f"  Input:  MIN_RNG={min_rng} km, PC={pc:.2e}")
    print(f"  Output: Collision Probability = {prob:.6f}")
    print(f"  Logit:  {logit:.4f}")
    if prob > 0.5:
        print(f"  *** HIGH RISK — Maneuver recommended ***")
    else:
        print(f"  Low risk — No maneuver needed")
    print(f"{'='*50}")

    return prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_rng", type=float, default=0.1)
    parser.add_argument("--pc", type=float, default=1e-5)
    parser.add_argument("--sat1_rcs", type=float, default=1.0)
    parser.add_argument("--sat2_rcs", type=float, default=1.0)
    parser.add_argument("--sat1_excl", type=float, default=10.0)
    parser.add_argument("--sat2_excl", type=float, default=10.0)
    parser.add_argument("--emergency", type=int, default=0)
    parser.add_argument("--hours_to_tca", type=float, default=48.0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_retnet.pt")
    args = parser.parse_args()

    predict_single(
        min_rng=args.min_rng, pc=args.pc,
        sat1_rcs=args.sat1_rcs, sat2_rcs=args.sat2_rcs,
        sat1_excl=args.sat1_excl, sat2_excl=args.sat2_excl,
        emergency=args.emergency, hours_to_tca=args.hours_to_tca,
        checkpoint=args.checkpoint,
    )
