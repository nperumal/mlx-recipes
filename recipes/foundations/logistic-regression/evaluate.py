"""Classification metrics, in pure MLX. Imported by both training scripts.

Nothing here loops over samples in Python. The confusion matrix is a matrix product
of one-hot indicators, and the whole ROC curve falls out of one argsort and two
cumulative sums -- which is a better demonstration of the framework than the models
themselves.
"""

import argparse
from pathlib import Path

import mlx.core as mx

RECIPE_DIR = Path(__file__).resolve().parent


def load_config(path=None):
    import yaml

    path = Path(path) if path else RECIPE_DIR / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def accuracy(y_true, y_pred):
    return mx.mean((y_true == y_pred).astype(mx.float32))


def confusion_matrix(y_true, y_pred, n_classes):
    """Rows are the true class, columns the predicted one.

    One-hot both label vectors and multiply: (K, n) @ (n, K) counts every pair in a
    single matmul, no loop over classes or samples.
    """
    k = mx.arange(n_classes)
    true_onehot = (y_true[:, None] == k[None, :]).astype(mx.float32)
    pred_onehot = (y_pred[:, None] == k[None, :]).astype(mx.float32)
    return true_onehot.T @ pred_onehot


def precision_recall_f1(cm):
    """Per-class precision, recall and F1 from a confusion matrix."""
    n_classes = cm.shape[0]
    k = mx.arange(n_classes)
    eye = (k[:, None] == k[None, :]).astype(mx.float32)
    tp = mx.sum(cm * eye, axis=1)
    predicted = mx.sum(cm, axis=0)
    actual = mx.sum(cm, axis=1)

    precision = mx.where(predicted > 0, tp / mx.maximum(predicted, 1.0), mx.zeros_like(tp))
    recall = mx.where(actual > 0, tp / mx.maximum(actual, 1.0), mx.zeros_like(tp))
    denom = precision + recall
    f1 = mx.where(denom > 0, 2 * precision * recall / mx.maximum(denom, 1e-12), mx.zeros_like(tp))
    return precision, recall, f1


def roc_curve(y_true, scores):
    """False- and true-positive rates at every threshold, vectorised.

    Sort by descending score, then walk down the list treating the top i items as
    predicted-positive. The running count of real positives is the true-positive
    count, and the running count of negatives is the false-positive count -- so two
    cumulative sums give the entire curve.
    """
    order = mx.argsort(-scores)
    y = y_true.astype(mx.float32)[order]
    tp = mx.cumsum(y)
    fp = mx.cumsum(1.0 - y)
    n_pos, n_neg = tp[-1], fp[-1]
    if n_pos.item() == 0 or n_neg.item() == 0:
        raise ValueError("ROC is undefined when one class is absent from the labels")
    zero = mx.zeros((1,))
    return (
        mx.concatenate([zero, fp / n_neg]),
        mx.concatenate([zero, tp / n_pos]),
    )


def roc_auc(y_true, scores):
    """Area under the ROC curve, by the trapezoid rule over the curve above.

    Equivalently, and more usefully: the probability that a randomly chosen positive
    is scored above a randomly chosen negative. 0.5 is coin-flip ranking, 1.0 is
    perfect. It says nothing about calibration or about any particular threshold.
    """
    fpr, tpr = roc_curve(y_true, scores)
    widths = fpr[1:] - fpr[:-1]
    heights = (tpr[1:] + tpr[:-1]) / 2
    return mx.sum(widths * heights)


def best_f1_threshold(y_true, scores):
    """The threshold maximising F1, and that F1.

    0.5 is a convention, not a result: it is optimal only when the two kinds of error
    cost the same. Every cut point is scored here in one pass, no Python loop.
    """
    order = mx.argsort(-scores)
    sorted_scores = scores[order]
    y = y_true.astype(mx.float32)[order]
    tp = mx.cumsum(y)
    taken = mx.arange(1, y.shape[0] + 1).astype(mx.float32)
    n_pos = tp[-1]

    precision = tp / taken
    recall = tp / n_pos
    denom = precision + recall
    f1 = mx.where(denom > 0, 2 * precision * recall / mx.maximum(denom, 1e-12), mx.zeros_like(tp))
    best = mx.argmax(f1)
    return sorted_scores[best], f1[best]


def report_binary(y_true, probs, classes, threshold):
    y_pred = (probs >= threshold).astype(mx.int32)
    cm = confusion_matrix(y_true, y_pred, 2)
    precision, recall, f1 = precision_recall_f1(cm)
    auc = roc_auc(y_true, probs)
    best_t, best_f1 = best_f1_threshold(y_true, probs)
    mx.eval(cm, precision, recall, f1, auc, best_t, best_f1)

    positive = classes[1]
    print(f"  accuracy          {accuracy(y_true, y_pred).item():.4f}  (threshold {threshold})")
    print(f"  ROC-AUC           {auc.item():.4f}   (threshold-free)")
    print(f"  precision [{positive}]  {precision[1].item():.4f}")
    print(f"  recall    [{positive}]  {recall[1].item():.4f}")
    print(f"  F1        [{positive}]  {f1[1].item():.4f}")
    print(f"  best F1 {best_f1.item():.4f} at threshold {best_t.item():.3f}")
    print_confusion(cm, classes)


def report_multiclass(y_true, logits, classes):
    y_pred = mx.argmax(logits, axis=-1).astype(mx.int32)
    cm = confusion_matrix(y_true, y_pred, len(classes))
    precision, recall, f1 = precision_recall_f1(cm)
    mx.eval(cm, precision, recall, f1)

    print(f"  accuracy          {accuracy(y_true, y_pred).item():.4f}")
    print(f"  macro F1          {mx.mean(f1).item():.4f}")
    for i, name in enumerate(classes):
        print(
            f"  {name:<10} precision {precision[i].item():.4f}  "
            f"recall {recall[i].item():.4f}  F1 {f1[i].item():.4f}"
        )
    print_confusion(cm, classes)


def print_confusion(cm, classes):
    width = max(len(c) for c in classes) + 2
    print("  confusion (rows = true, cols = predicted)")
    print(" " * (width + 4) + "".join(f"{c[:8]:>10}" for c in classes))
    for i, name in enumerate(classes):
        counts = "".join(f"{int(cm[i, j].item()):>10}" for j in range(len(classes)))
        print(f"    {name:<{width}}{counts}")


def main():
    ap = argparse.ArgumentParser(description="Re-score a saved checkpoint.")
    ap.add_argument("--variant", choices=["binary", "softmax"], required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import data_pipeline

    cfg = load_config(args.config)
    ckpt = RECIPE_DIR / "checkpoints" / f"{args.variant}.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt} -- run `make train-{args.variant}` first")

    if args.variant == "binary":
        data = data_pipeline.load(cfg, cfg["binary"]["classes"])
        p = mx.load(str(ckpt))
        probs = mx.sigmoid(data["X_test"] @ p["w"] + p["b"])
        report_binary(data["y_test"], probs, data["classes"], cfg["binary"]["threshold"])
    else:
        import mlx.nn as nn

        data = data_pipeline.load(cfg, cfg["softmax"]["classes"])
        model = nn.Linear(len(data["feature_names"]), len(data["classes"]))
        model.load_weights(str(ckpt))
        report_multiclass(data["y_test"], model(data["X_test"]), data["classes"])


if __name__ == "__main__":
    main()
