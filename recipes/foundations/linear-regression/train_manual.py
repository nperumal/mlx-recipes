"""Linear regression with mx.grad and a hand-written gradient descent step.

No nn.Module, no optimizer object: just a weight vector, an explicit loss, and
w <- w - lr * grad. Compare with train_nn.py, which does the same thing through
mlx.nn and mlx.optimizers.

There is no intercept here, and that is correct rather than an oversight: both X and
y are centred by the standardiser, so the least-squares intercept is zero.
"""

import argparse
from pathlib import Path

import data_pipeline
import mlx.core as mx
import runner
from evaluate import load_config, regression_metrics

RECIPE_DIR = Path(__file__).resolve().parent


def make_loss(X, y):
    def loss_fn(w):
        return 0.5 * mx.mean(mx.square(X @ w - y))

    return loss_fn


def train(cfg, data):
    tcfg = cfg["train"]
    if tcfg["optimizer"] != "sgd":
        raise SystemExit(
            f"train_manual.py implements plain SGD only; config asks for "
            f"'{tcfg['optimizer']}'. Use train_nn.py for other optimizers -- that is "
            f"precisely the payoff of the mlx.optimizers abstraction."
        )

    X, y = data["X_train"], data["y_train"]
    loss_fn = make_loss(X, y)
    loss_and_grad = mx.value_and_grad(loss_fn)

    mx.random.seed(cfg["seed"])
    w = 1e-6 * mx.random.normal((X.shape[1],))
    mx.eval(w)

    lr = tcfg["learning_rate"]

    # Warm up outside the timed window so kernel compilation is not counted.
    for _ in range(tcfg["warmup_iters"]):
        _, grad = loss_and_grad(w)
        w = w - lr * grad
        mx.eval(w)

    # Re-initialise so the timed run starts from the same place a cold run would.
    mx.random.seed(cfg["seed"])
    w = 1e-6 * mx.random.normal((X.shape[1],))
    mx.eval(w)

    losses = []
    with runner.Benchmark() as bench:
        for _ in range(tcfg["num_iters"]):
            loss, grad = loss_and_grad(w)
            w = w - lr * grad
            # Evaluate both so the graph stays shallow, but do NOT call .item():
            # a host sync every iteration would dominate the measurement.
            mx.eval(w, loss)
            losses.append(loss)

    loss_history = [float(v.item()) for v in losses]
    return w, bench, loss_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = runner.select_device(args.device)

    data = data_pipeline.load(cfg)
    w, bench, loss_history = train(cfg, data)

    y_pred = data["y_scaler"].inverse(data["X_test"] @ w)
    metrics = regression_metrics(data["y_test"], y_pred)

    ckpt_dir = RECIPE_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    mx.save_safetensors(str(ckpt_dir / "manual.safetensors"), {"w": w})

    record, _ = runner.write_run("manual", device, cfg, metrics, bench, loss_history)
    runner.report(record)


if __name__ == "__main__":
    main()
