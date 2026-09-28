"""Classify one penguin with the trained softmax model.

Prints the full probability vector rather than just the winning label -- a classifier
that says "Chinstrap" is much less useful than one that says "Chinstrap, 0.62".
"""

import argparse
from pathlib import Path

import data_pipeline
import mlx.core as mx
import mlx.nn as nn
from evaluate import load_config

RECIPE_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = data_pipeline.load(cfg, cfg["softmax"]["classes"])
    names = data["feature_names"]

    example = cfg["predict"]["example"]
    missing = [n for n in names if n not in example]
    if missing:
        raise SystemExit(f"config predict.example is missing features: {missing}")

    ckpt = RECIPE_DIR / "checkpoints" / "softmax.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt} -- run `make train-softmax` first")

    x = mx.array([[float(example[n]) for n in names]], dtype=mx.float32)
    x = data["scaler"].transform(x)

    model = nn.Linear(len(names), len(data["classes"]))
    model.load_weights(str(ckpt))
    probs = mx.softmax(model(x), axis=-1)[0]
    mx.eval(probs)

    print("Measurements:")
    for n in names:
        print(f"  {n:<20} {example[n]}")
    print("\nP(species):")
    for name, p in zip(data["classes"], probs.tolist(), strict=True):
        bar = "#" * int(round(p * 40))
        print(f"  {name:<10} {p:6.3f}  {bar}")


if __name__ == "__main__":
    main()
