"""The same model generalised to three classes, through mlx.nn and mlx.optimizers.

Binary logistic regression is softmax regression with K = 2, so this is the same
model rather than a different one. What changes:

    w  (d,)   becomes  W  (d, K)          -- nn.Linear(d, K) instead of (d, 1)
    sigmoid            becomes  softmax
    BCE                becomes  categorical cross-entropy

and the gradient is, once again, (p - onehot(y)) -- the third time that identity
shows up across these two recipes.
"""

import argparse
import time
from pathlib import Path

import data_pipeline
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from evaluate import load_config, report_multiclass

RECIPE_DIR = Path(__file__).resolve().parent


def build_optimizer(name, lr):
    if name == "sgd":
        return optim.SGD(learning_rate=lr)
    if name == "adam":
        return optim.Adam(learning_rate=lr)
    raise SystemExit(f"unknown optimizer '{name}' (expected sgd or adam)")


def cross_entropy(model, X, y):
    """Categorical cross-entropy written out, for the same reason as bce_with_logits.

    Naively you would softmax the logits and take the log of the chosen class, but
    exponentiating raw logits overflows. The identity

        -log softmax(z)_y = logsumexp(z) - z_y

    avoids that: mx.logsumexp subtracts the row maximum internally, which cancels in
    the softmax ratio and keeps every exponent at or below zero.

    mlx.nn.losses.cross_entropy does exactly this and is what you would use in real
    code. It is spelled out here because the point of the recipe is the mechanism.
    """
    z = model(X)
    chosen = mx.take_along_axis(z, y[:, None], axis=-1).squeeze(-1)
    return mx.mean(mx.logsumexp(z, axis=-1) - chosen)


def train(cfg, data):
    tcfg = cfg["train"]
    X, y = data["X_train"], data["y_train"]

    mx.random.seed(cfg["seed"])
    model = nn.Linear(X.shape[1], len(data["classes"]))
    optimizer = build_optimizer(tcfg["optimizer"], tcfg["learning_rate"])
    loss_and_grad = nn.value_and_grad(model, cross_entropy)
    mx.eval(model.parameters())

    losses = []
    tic = time.perf_counter()
    for _ in range(tcfg["num_iters"]):
        loss, grads = loss_and_grad(model, X, y)
        optimizer.update(model, grads)
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

    data = data_pipeline.load(cfg, cfg["softmax"]["classes"])
    model, seconds, history = train(cfg, data)

    logits = model(data["X_test"])
    mx.eval(logits)

    ckpt_dir = RECIPE_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    model.save_weights(str(ckpt_dir / "softmax.safetensors"))

    print(f"softmax  {len(data['classes'])} classes: {', '.join(data['classes'])}")
    print(
        f"        {data['n_train']} train / {data['n_test']} test, "
        f"class counts {data['class_counts']}, {data['n_skipped']} rows dropped for "
        f"missing values"
    )
    report_multiclass(data["y_test"], logits, data["classes"])
    print(
        f"  final train loss {history[-1]:.5f}, "
        f"{cfg['train']['num_iters']} iterations in {seconds:.2f}s on {args.device}"
    )


if __name__ == "__main__":
    main()
