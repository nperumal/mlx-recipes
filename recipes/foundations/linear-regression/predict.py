"""Price one house with a trained model.

The example in config.yaml stays inside the range the model was fit on: every binary
column is 0 or 1 and every continuous value sits within the observed min/max. A linear
model will happily return a number for a nine-bedroom house on a two-acre lot, but it
would be pure extrapolation and the recipe should not teach that.
"""

import argparse
from pathlib import Path

import data_pipeline
import mlx.core as mx
from evaluate import load_config

RECIPE_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["manual", "nn"], default="nn")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = data_pipeline.load(cfg)
    names = data["feature_names"]

    example = cfg["predict"]["example"]
    missing = [n for n in names if n not in example]
    if missing:
        raise SystemExit(f"config predict.example is missing features: {missing}")

    x = mx.array([[float(example[n]) for n in names]], dtype=mx.float32)
    x = data["x_scaler"].transform(x)

    ckpt = RECIPE_DIR / "checkpoints" / f"{args.variant}.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt} -- run `make train-{args.variant}` first")

    if args.variant == "manual":
        w = mx.load(str(ckpt))["w"]
        pred = data["y_scaler"].inverse(x @ w)
    else:
        import mlx.nn as nn

        model = nn.Linear(len(names), 1)
        model.load_weights(str(ckpt))
        pred = data["y_scaler"].inverse(model(x).squeeze(-1))

    mx.eval(pred)
    print("Features:")
    for n in names:
        print(f"  {n:<10} {example[n]}")
    print(f"\nPredicted price ({args.variant}): ${pred.item():,.0f}")


if __name__ == "__main__":
    main()
