"""Correctness tests for the linear-regression recipe.

Four kinds, per CONTRIBUTING.md section 4, none of which need an external library:
synthetic recovery, cross-path agreement, gradient check, known answer.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_manual  # noqa: E402
import train_nn  # noqa: E402
from data_pipeline import Standardizer  # noqa: E402
from evaluate import regression_metrics  # noqa: E402

SEED = 0


def synthetic(n=2000, d=5, noise=0.01, seed=SEED):
    """y = X w* + noise, with w* returned so it can be recovered."""
    mx.random.seed(seed)
    X = mx.random.normal((n, d))
    w_true = mx.random.normal((d,))
    y = X @ w_true + noise * mx.random.normal((n,))
    mx.eval(X, y, w_true)
    return X, y, w_true


def fit_manual(X, y, lr=0.1, iters=2000):
    loss_and_grad = mx.value_and_grad(train_manual.make_loss(X, y))
    w = mx.zeros((X.shape[1],))
    for _ in range(iters):
        _, grad = loss_and_grad(w)
        w = w - lr * grad
        mx.eval(w)
    return w


# --- synthetic recovery ---------------------------------------------------------


def test_recovers_known_coefficients():
    X, y, w_true = synthetic()
    w = fit_manual(X, y)
    assert mx.max(mx.abs(w - w_true)).item() < 1e-2


# --- cross-path agreement -------------------------------------------------------


def test_manual_and_nn_agree():
    """The two variants are the same model; with the same optimizer and enough
    iterations they must land in the same place."""
    X, y, _ = synthetic(n=1000, d=4)

    w_manual = fit_manual(X, y, lr=0.1, iters=3000)

    mx.random.seed(SEED)
    model = nn.Linear(X.shape[1], 1)
    optimizer = optim.SGD(learning_rate=0.1)
    loss_and_grad = nn.value_and_grad(model, train_nn.loss_fn)
    for _ in range(3000):
        _, grad = loss_and_grad(model, X, y)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state)

    w_nn = model.weight.squeeze(0)
    mx.eval(w_nn)
    assert mx.max(mx.abs(w_manual - w_nn)).item() < 1e-2
    # X and y are centred, so nn.Linear's bias should have collapsed to ~0.
    assert abs(model.bias.item()) < 1e-2


# --- gradient check -------------------------------------------------------------


def test_gradient_matches_finite_differences():
    X, y, _ = synthetic(n=200, d=3)
    loss_fn = train_manual.make_loss(X, y)
    grad_fn = mx.grad(loss_fn)

    w = mx.array([0.3, -0.7, 1.1])
    analytic = grad_fn(w)
    mx.eval(analytic)

    eps = 1e-3
    for i in range(w.shape[0]):
        # One-hot bump built without in-place assignment.
        bump = eps * (mx.arange(w.shape[0]) == i).astype(mx.float32)
        numeric = (loss_fn(w + bump) - loss_fn(w - bump)) / (2 * eps)
        mx.eval(numeric)
        assert abs(numeric.item() - analytic[i].item()) < 1e-2


# --- known answer ---------------------------------------------------------------


def test_exact_fit_on_tiny_problem():
    """y = 2 * x exactly; one feature, no noise. The fit must find 2."""
    X = mx.array([[1.0], [2.0], [3.0], [4.0]])
    y = mx.array([2.0, 4.0, 6.0, 8.0])
    w = fit_manual(X, y, lr=0.05, iters=2000)
    assert abs(w[0].item() - 2.0) < 1e-3


def test_metrics_on_perfect_prediction():
    y = mx.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["r2"] == pytest.approx(1.0, abs=1e-6)


# --- scaler ---------------------------------------------------------------------


def test_standardizer_roundtrip():
    a = mx.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    s = Standardizer.fit(a)
    back = s.inverse(s.transform(a))
    mx.eval(back)
    assert mx.max(mx.abs(back - a)).item() < 1e-4


def test_standardizer_survives_constant_column():
    a = mx.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    s = Standardizer.fit(a)
    out = s.transform(a)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item())
