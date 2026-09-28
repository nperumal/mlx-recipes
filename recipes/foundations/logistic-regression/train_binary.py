"""Binary logistic regression from first principles: mx.grad on a weight vector.

Adelie vs Chinstrap. No nn.Module and no optimizer object, so the gradient MLX
computes can be checked against the one you would derive by hand -- and the tests do
exactly that.

The model is linear in log-odds:

    z = w . x + b                     the logit
    p = sigmoid(z) = P(Chinstrap|x)

and the loss is the negative log-likelihood of a Bernoulli outcome. Its derivative
with respect to z collapses to (p - y), so the parameter gradient is

    dL/dw = X^T (p - y) / n

which is the same shape as the least-squares gradient in the linear-regression
recipe. The training loop is, deliberately, almost the same code.
"""

import argparse
import time
from pathlib import Path

import data_pipeline
import mlx.core as mx
from evaluate import load_config, report_binary

RECIPE_DIR = Path(__file__).resolve().parent


def logits(params, X):
    return X @ params["w"] + params["b"]


def bce_with_logits(z, y):
    """Binary cross-entropy computed from the logits, never from the probability.

    The textbook form -[y log p + (1-y) log(1-p)] overflows: at z = -50 the sigmoid
    underflows to exactly 0 in float32 and log(0) is -inf. Rewriting in terms of z,

        -log(sigmoid(z))   = log(1 + e^-z) = logaddexp(0, -z)
        -log(1-sigmoid(z)) = log(1 + e^+z) = logaddexp(0, +z)

    gives the same number with no overflow and no clipping hacks.
    """
    zero = mx.zeros_like(z)
    return mx.mean(y * mx.logaddexp(zero, -z) + (1 - y) * mx.logaddexp(zero, z))


def make_loss(X, y, l2):
    def loss_fn(params):
        loss = bce_with_logits(logits(params, X), y)
        if l2 > 0:
            # Ridge penalty. Without it, perfectly separable classes have no finite
            # maximum-likelihood solution and ||w|| grows without bound.
            loss = loss + l2 * mx.sum(params["w"] * params["w"])
        return loss

    return loss_fn


def train(cfg, data):
    tcfg = cfg["train"]
    X = data["X_train"]
    y = data["y_train"].astype(mx.float32)

    mx.random.seed(cfg["seed"])
    params = {"w": mx.zeros((X.shape[1],)), "b": mx.zeros((1,))}
    loss_and_grad = mx.value_and_grad(make_loss(X, y, tcfg["l2"]))
    lr = tcfg["learning_rate"]

    losses = []
    tic = time.perf_counter()
    for _ in range(tcfg["num_iters"]):
        loss, grads = loss_and_grad(params)
        params = {k: v - lr * grads[k] for k, v in params.items()}
        # Force the computation without copying it to the host. See the
        # linear-regression recipe's Gotchas for why .item() does not belong here.
        mx.eval(params, loss)
        losses.append(loss)
    seconds = time.perf_counter() - tic

    history = [float(v.item()) for v in losses]
    return params, seconds, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    data = data_pipeline.load(cfg, cfg["binary"]["classes"])
    params, seconds, history = train(cfg, data)

    probs = mx.sigmoid(logits(params, data["X_test"]))
    mx.eval(probs)

    ckpt_dir = RECIPE_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    mx.save_safetensors(str(ckpt_dir / "binary.safetensors"), params)

    negative, positive = data["classes"]
    print(f"binary  {negative} vs {positive}  (positive class: {positive})")
    print(
        f"        {data['n_train']} train / {data['n_test']} test, "
        f"class counts {data['class_counts']}, {data['n_skipped']} rows dropped for "
        f"missing values"
    )
    report_binary(data["y_test"], probs, data["classes"], cfg["binary"]["threshold"])
    print(
        f"  final train loss {history[-1]:.5f}, "
        f"{cfg['train']['num_iters']} iterations in {seconds:.2f}s on {args.device}"
    )
    print(f"  ||w|| = {mx.sqrt(mx.sum(params['w'] ** 2)).item():.3f}")
    print("  coefficients (per standard deviation, in log-odds):")
    for name, value in zip(data["feature_names"], params["w"].tolist(), strict=True):
        print(f"    {name:<20}{value:+.3f}   odds x{pow(2.718281828, value):.2f}")


if __name__ == "__main__":
    main()
