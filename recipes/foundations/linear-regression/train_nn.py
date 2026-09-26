"""The same linear regression through mlx.nn and mlx.optimizers.

Identical data, identical loss, identical learning rate and iteration count as
train_manual.py. The differences are exactly two:

  1. the model is an nn.Linear module rather than a bare weight vector, so it carries
     a bias term -- which converges to approximately zero here, because the
     standardiser centres both X and y;
  2. the update is delegated to an optimizer object, which is what makes switching
     from SGD to Adam a one-line change.

That second point is the argument for the abstraction, and config.yaml lets you try it.
"""

import argparse
from pathlib import Path

import data_pipeline
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import runner
from evaluate import load_config, regression_metrics

RECIPE_DIR = Path(__file__).resolve().parent


def build_optimizer(name, lr):
    if name == "sgd":
        return optim.SGD(learning_rate=lr)
    if name == "adam":
        return optim.Adam(learning_rate=lr)
    raise SystemExit(f"unknown optimizer '{name}' (expected sgd or adam)")


def loss_fn(model, inputs, targets):
    predictions = model(inputs).squeeze(-1)
    return 0.5 * mx.mean(mx.square(predictions - targets))


def train(cfg, data):
    tcfg = cfg["train"]
    X, y = data["X_train"], data["y_train"]

    mx.random.seed(cfg["seed"])
    model = nn.Linear(X.shape[1], 1)
    optimizer = build_optimizer(tcfg["optimizer"], tcfg["learning_rate"])
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    mx.eval(model.parameters())

    for _ in range(tcfg["warmup_iters"]):
        _, grad = loss_and_grad(model, X, y)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state)

    # Reset to a cold model and optimizer for the timed run.
    mx.random.seed(cfg["seed"])
    model = nn.Linear(X.shape[1], 1)
    optimizer = build_optimizer(tcfg["optimizer"], tcfg["learning_rate"])
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    mx.eval(model.parameters())

    losses = []
    with runner.Benchmark() as bench:
        for _ in range(tcfg["num_iters"]):
            loss, grad = loss_and_grad(model, X, y)
            optimizer.update(model, grad)
            # No .item() inside the loop -- see the note in runner.Benchmark.
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(loss)

    loss_history = [float(v.item()) for v in losses]
    return model, bench, loss_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = runner.select_device(args.device)

    data = data_pipeline.load(cfg)
    model, bench, loss_history = train(cfg, data)

    y_pred = data["y_scaler"].inverse(model(data["X_test"]).squeeze(-1))
    metrics = regression_metrics(data["y_test"], y_pred)

    ckpt_dir = RECIPE_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    model.save_weights(str(ckpt_dir / "nn.safetensors"))

    record, _ = runner.write_run("nn", device, cfg, metrics, bench, loss_history)
    runner.report(record)
    print(f"  bias after training: {model.bias.item():.6f}  (expected near zero)")


if __name__ == "__main__":
    main()
