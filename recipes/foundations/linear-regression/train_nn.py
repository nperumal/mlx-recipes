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
import time
from pathlib import Path

import data_pipeline
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
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

    losses = []
    tic = time.perf_counter()
    for _ in range(tcfg["num_iters"]):
        loss, grad = loss_and_grad(model, X, y)
        optimizer.update(model, grad)
        # Same reasoning as train_manual.py: force the computation, but do not pull
        # the value back to the host inside the loop. The optimizer carries state of
        # its own (momentum, moments), so that has to be evaluated too.
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(loss)
    seconds = time.perf_counter() - tic

    history = [float(v.item()) for v in losses]
    return model, seconds, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    data = data_pipeline.load(cfg)
    model, seconds, history = train(cfg, data)

    y_pred = data["y_scaler"].inverse(model(data["X_test"]).squeeze(-1))
    m = regression_metrics(data["y_test"], y_pred)

    ckpt_dir = RECIPE_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    model.save_weights(str(ckpt_dir / "nn.safetensors"))

    print(f"nn      RMSE ${m['rmse']:,.0f}  MAE ${m['mae']:,.0f}  R2 {m['r2']:.4f}")
    print(
        f"        final train loss {history[-1]:.5f}, "
        f"{cfg['train']['num_iters']} iterations in {seconds:.2f}s on {args.device}"
    )
    print(f"        bias after training: {model.bias.item():.6f}  (expected near zero)")


if __name__ == "__main__":
    main()
