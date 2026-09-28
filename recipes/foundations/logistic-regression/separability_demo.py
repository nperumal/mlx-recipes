"""Show that the unpenalised maximum-likelihood estimate does not exist here.

When two classes are linearly separable, the likelihood has no maximum: the optimizer
can always lower the loss a little further by scaling w up, pushing fitted
probabilities closer to exactly 0 and 1. The parameters run away and never converge.

What makes this worth a script of its own is that **the metrics do not show it**. Test
accuracy and ROC-AUC are identical with and without the penalty. The only visible
symptom is the norm of the weight vector, which nothing in a standard classification
report prints.

Run it with `make separability`. It reproduces the two claims in the README: that all
three species pairs are separable, and the table of ||w|| against iteration count.
"""

import argparse
import copy
import itertools
from pathlib import Path

import data_pipeline
import mlx.core as mx
import train_binary
from evaluate import load_config

RECIPE_DIR = Path(__file__).resolve().parent
ITERATIONS = (500, 2000, 8000, 20000)


def fit(cfg, data, iters, l2):
    """Train one model and report what the weights did."""
    trial = copy.deepcopy(cfg)
    trial["train"]["num_iters"] = iters
    trial["train"]["l2"] = l2
    params, _, history = train_binary.train(trial, data)

    norm = mx.sqrt(mx.sum(params["w"] ** 2))
    predicted = (mx.sigmoid(train_binary.logits(params, data["X_train"])) >= 0.5).astype(mx.int32)
    train_accuracy = mx.mean((predicted == data["y_train"]).astype(mx.float32))
    mx.eval(norm, train_accuracy)
    return norm.item(), history[-1], train_accuracy.item()


def check_all_pairs(cfg, species, iters):
    print(f"Separability of each species pair ({iters:,} iterations, no penalty)")
    print(f"  {'pair':<24}{'train acc':>10}{'train loss':>13}{'||w||':>9}")
    for a, b in itertools.combinations(species, 2):
        data = data_pipeline.load(cfg, [a, b])
        norm, loss, accuracy = fit(cfg, data, iters, 0.0)
        print(f"  {a + ' vs ' + b:<24}{accuracy:>10.4f}{loss:>13.2e}{norm:>9.2f}")
    print(
        "\n  Training accuracy of 1.0000 with the loss still falling means the classes\n"
        "  are separable: there is a hyperplane with every point on its correct side,\n"
        "  so the fit can keep sharpening it forever.\n"
    )


def penalty_table(cfg, pair, l2):
    data = data_pipeline.load(cfg, pair)
    print(f"||w|| against iterations, {pair[0]} vs {pair[1]}")
    print(f"  {'iterations':>11}{'l2 = 0.0':>24}{f'l2 = {l2}':>24}")
    for iters in ITERATIONS:
        cells = []
        for penalty in (0.0, l2):
            norm, loss, _ = fit(cfg, data, iters, penalty)
            cells.append(f"{norm:8.2f}  (loss {loss:.1e})")
        print(f"  {iters:>11,}{cells[0]:>24}{cells[1]:>24}")
    print(
        "\n  The unpenalised column never settles. The penalised one is stationary\n"
        "  after a couple of thousand iterations, because the l2 term grows faster\n"
        "  than the likelihood improves."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--iters",
        type=int,
        default=ITERATIONS[-1],
        help="iterations for the separability check (default: %(default)s)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    check_all_pairs(cfg, cfg["softmax"]["classes"], args.iters)
    penalty_table(cfg, cfg["binary"]["classes"], cfg["train"]["l2"] or 0.01)


if __name__ == "__main__":
    main()
