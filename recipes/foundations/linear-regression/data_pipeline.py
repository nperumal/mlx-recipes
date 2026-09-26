"""Load, shuffle, split and standardise the Windsor housing data.

Shared by both training variants so they see byte-identical inputs. The only intended
difference between train_manual.py and train_nn.py is the training core; if the data
path differed too, the comparison in results.md would mean nothing.

Pure MLX, loaded through mlx.data.
"""

from pathlib import Path

import mlx.core as mx
import mlx.data as dx

RECIPE_DIR = Path(__file__).resolve().parent


def _number(value):
    """Decode one CSV field.

    mlx.data hands back each field as a uint8 buffer rather than a Python string, so
    every value needs an explicit decode. This is the main ergonomic wrinkle of using
    mlx.data for CSV work and it is discussed in the README's Gotchas section.
    """
    return float(value.tobytes().decode("utf-8"))


def load_raw(csv_path, feature_names, label_name):
    """Read the CSV into two MLX arrays: X (n, d) and y (n,), both float32."""
    stream = dx.stream_csv_reader(str(csv_path), sep=",", quote='"')
    features, labels = [], []
    for row in stream:
        features.append([_number(row[name]) for name in feature_names])
        labels.append(_number(row[label_name]))
    X = mx.array(features, dtype=mx.float32)
    y = mx.array(labels, dtype=mx.float32)
    return X, y


class Standardizer:
    """Zero-mean, unit-variance scaling, fit on training data only."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, a, axis=0):
        std = mx.std(a, axis=axis)
        # A constant column would divide by zero. Leave it alone instead.
        std = mx.where(std == 0, mx.ones_like(std), std)
        return cls(mx.mean(a, axis=axis), std)

    def transform(self, a):
        return (a - self.mean) / self.std

    def inverse(self, a):
        return a * self.std + self.mean


def load(cfg):
    """Return the prepared splits and the fitted scalers.

    y_train is standardised because that is what the models are fit against.
    y_test is deliberately left in raw dollars: predictions are converted back before
    scoring, so every metric in this recipe is reported in dollars and both variants
    are measured the same way.
    """
    dcfg = cfg["data"]
    csv_path = RECIPE_DIR / dcfg["csv"]
    X, y = load_raw(csv_path, dcfg["features"], dcfg["label"])

    n = X.shape[0]
    if dcfg.get("shuffle", True):
        mx.random.seed(cfg["seed"])
        perm = mx.random.permutation(n)
        X, y = X[perm], y[perm]

    n_train = int(dcfg["train_split"] * n)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    x_scaler = Standardizer.fit(X_train)
    y_scaler = Standardizer.fit(y_train)

    return {
        "X_train": x_scaler.transform(X_train),
        "X_test": x_scaler.transform(X_test),
        "y_train": y_scaler.transform(y_train),
        "y_test": y_test,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_names": dcfg["features"],
        "n_train": n_train,
        "n_test": n - n_train,
    }
