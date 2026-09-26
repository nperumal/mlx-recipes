"""Linear regression with mx.grad and a hand-written gradient descent step.

No nn.Module, no optimizer object: just a weight vector, an explicit loss, and
w <- w - lr * grad. Compare with train_nn.py, which does the same thing through
mlx.nn and mlx.optimizers.

There is no intercept here, and that is correct rather than an oversight: both X and
y are centred by the standardiser, so the least-squares intercept is zero.
"""

import argparse
import time
from pathlib import Path

import data_pipeline
import mlx.core as mx
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
    loss_and_grad = mx.value_and_grad(make_loss(X, y))
    lr = tcfg["learning_rate"]

    mx.random.seed(cfg["seed"])
    w = 1e-6 * mx.random.normal((X.shape[1],))
    mx.eval(w)

    losses = []
    tic = time.perf_counter()
    for _ in range(tcfg["num_iters"]):
        loss, grad = loss_and_grad(w)
        w = w - lr * grad
        # mx.eval is what actually runs the computation. MLX is lazy: without this
        # the loop would only build an ever-deeper graph and nothing would execute.
        # Note what is NOT here -- a .item() call. That would copy the loss to the
        # host on every iteration, which is both slower and unnecessary.
        mx.eval(w, loss)
        losses.append(loss)
    seconds = time.perf_counter() - tic

    # Now that the loop is finished, converting to Python floats is free of consequence.
    history = [float(v.item()) for v in losses]
    return w, seconds, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    data = data_pipeline.load(cfg)
    w, seconds, history = train(cfg, data)

    y_pred = data["y_scaler"].inverse(data["X_test"] @ w)
    m = regression_metrics(data["y_test"], y_pred)

    ckpt_dir = RECIPE_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    mx.save_safetensors(str(ckpt_dir / "manual.safetensors"), {"w": w})

    print(f"manual  RMSE ${m['rmse']:,.0f}  MAE ${m['mae']:,.0f}  R2 {m['r2']:.4f}")
    print(
        f"        final train loss {history[-1]:.5f}, "
        f"{cfg['train']['num_iters']} iterations in {seconds:.2f}s on {args.device}"
    )


if __name__ == "__main__":
    main()
