"""Metrics for this recipe, in pure MLX. Imported by both training variants.

One implementation, used identically by train_manual.py and train_nn.py, so the two
rows of results.md are actually comparable. All metrics are computed in dollars:
predictions are converted out of standardised space before they get here.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

RECIPE_DIR = Path(__file__).resolve().parent


def regression_metrics(y_true, y_pred):
    """RMSE, MAE and R^2. Both inputs must be on the same (raw) scale."""
    resid = y_true - y_pred
    rmse = mx.sqrt(mx.mean(mx.square(resid)))
    mae = mx.mean(mx.abs(resid))
    ss_res = mx.sum(mx.square(resid))
    ss_tot = mx.sum(mx.square(y_true - mx.mean(y_true)))
    r2 = 1.0 - ss_res / ss_tot
    mx.eval(rmse, mae, r2)
    return {"rmse": rmse.item(), "mae": mae.item(), "r2": r2.item()}


def load_config(path=None):
    import yaml

    path = Path(path) if path else RECIPE_DIR / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    """Re-score a saved checkpoint. Training already reports metrics; this exists so a
    checkpoint can be evaluated on its own."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["manual", "nn"], required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import data_pipeline

    cfg = load_config(args.config)
    data = data_pipeline.load(cfg)
    ckpt = RECIPE_DIR / "checkpoints" / f"{args.variant}.safetensors"

    if args.variant == "manual":
        w = mx.load(str(ckpt))["w"]
        y_pred = data["y_scaler"].inverse(data["X_test"] @ w)
    else:
        import mlx.nn as nn

        model = nn.Linear(len(data["feature_names"]), 1)
        model.load_weights(str(ckpt))
        y_pred = data["y_scaler"].inverse(model(data["X_test"]).squeeze(-1))

    print(json.dumps(regression_metrics(data["y_test"], y_pred), indent=2))


if __name__ == "__main__":
    main()
