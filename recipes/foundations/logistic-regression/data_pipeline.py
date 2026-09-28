"""Load, filter, encode, split and standardise the Palmer Penguins data.

Shared by both training scripts so they see identical inputs.

Two things here that the linear-regression recipe did not have to deal with: the
label is a string that needs mapping to an integer class index, and the CSV has
missing values.
"""

from pathlib import Path

import mlx.core as mx
import mlx.data as dx

RECIPE_DIR = Path(__file__).resolve().parent

# How this CSV spells "no value". An empty field arrives from mlx.data as a
# zero-length byte buffer, which float() will not accept.
MISSING = {"", "NA", "NaN", "nan", "."}


def _text(value):
    """mlx.data hands back each CSV field as a byte buffer, not a string."""
    return value.tobytes().decode("utf-8").strip()


def _number(value):
    """Parse a numeric field, or None when it is missing."""
    s = _text(value)
    return None if s in MISSING else float(s)


def load_raw(csv_path, feature_names, label_name):
    """Read the CSV, dropping any row with a missing feature or label.

    Two of the 344 penguins are missing every measurement. Dropping them is the
    honest minimum; imputation would be a different recipe.
    """
    stream = dx.stream_csv_reader(str(csv_path), sep=",", quote='"')
    features, labels, skipped = [], [], 0
    for row in stream:
        values = [_number(row[name]) for name in feature_names]
        label = _text(row[label_name])
        if label in MISSING or any(v is None for v in values):
            skipped += 1
            continue
        features.append(values)
        labels.append(label)
    return features, labels, skipped


class Standardizer:
    """Zero-mean, unit-variance scaling, fit on training data only."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, a, axis=0):
        std = mx.std(a, axis=axis)
        std = mx.where(std == 0, mx.ones_like(std), std)
        return cls(mx.mean(a, axis=axis), std)

    def transform(self, a):
        return (a - self.mean) / self.std


def load(cfg, classes):
    """Return the splits for a given list of species.

    `classes` fixes the label encoding: index 0 is classes[0], and so on. For the
    binary problem that makes classes[1] the positive class.
    """
    dcfg = cfg["data"]
    features, labels, skipped = load_raw(RECIPE_DIR / dcfg["csv"], dcfg["features"], dcfg["label"])

    index_of = {name: i for i, name in enumerate(classes)}
    kept = [(f, index_of[lab]) for f, lab in zip(features, labels, strict=True) if lab in index_of]
    if not kept:
        raise SystemExit(f"no rows matched classes {classes}")

    X = mx.array([f for f, _ in kept], dtype=mx.float32)
    y = mx.array([c for _, c in kept], dtype=mx.int32)

    n = X.shape[0]
    if dcfg.get("shuffle", True):
        mx.random.seed(cfg["seed"])
        perm = mx.random.permutation(n)
        X, y = X[perm], y[perm]

    n_train = int(dcfg["train_split"] * n)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    scaler = Standardizer.fit(X_train)
    counts = [int(mx.sum(y == i).item()) for i in range(len(classes))]

    return {
        "X_train": scaler.transform(X_train),
        "X_test": scaler.transform(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "classes": list(classes),
        "class_counts": counts,
        "feature_names": dcfg["features"],
        "n_train": n_train,
        "n_test": n - n_train,
        "n_skipped": skipped,
    }
