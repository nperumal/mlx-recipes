"""Correctness tests for the logistic-regression recipe.

Per CONTRIBUTING.md section 4: synthetic recovery, cross-path agreement, gradient
check, known answer. Nothing here needs a library outside MLX.

The gradient tests are the interesting ones -- they check the identity the README
derives, that dL/dz collapses to (p - y), against what mx.grad actually computes.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_binary  # noqa: E402
import train_softmax  # noqa: E402
from data_pipeline import Standardizer  # noqa: E402
from evaluate import (  # noqa: E402
    accuracy,
    best_f1_threshold,
    confusion_matrix,
    precision_recall_f1,
    roc_auc,
)

SEED = 0


def synthetic_binary(n=4000, d=4, seed=SEED):
    """y ~ Bernoulli(sigmoid(Xw* + b*)), with the true parameters returned."""
    mx.random.seed(seed)
    X = mx.random.normal((n, d))
    w_true = mx.array([1.5, -2.0, 0.5, 1.0])
    b_true = mx.array([0.3])
    p = mx.sigmoid(X @ w_true + b_true)
    y = (mx.random.uniform(shape=(n,)) < p).astype(mx.float32)
    mx.eval(X, y)
    return X, y, w_true, b_true


def fit(X, y, lr=0.5, iters=3000, l2=0.0):
    params = {"w": mx.zeros((X.shape[1],)), "b": mx.zeros((1,))}
    loss_and_grad = mx.value_and_grad(train_binary.make_loss(X, y, l2))
    for _ in range(iters):
        _, g = loss_and_grad(params)
        params = {k: v - lr * g[k] for k, v in params.items()}
        mx.eval(params)
    return params


# --- synthetic recovery ---------------------------------------------------------


def test_recovers_known_coefficients():
    X, y, w_true, b_true = synthetic_binary()
    params = fit(X, y)
    assert mx.max(mx.abs(params["w"] - w_true)).item() < 0.2
    assert abs(params["b"].item() - b_true.item()) < 0.2


# --- the gradient identity from the README --------------------------------------


def test_gradient_equals_p_minus_y():
    """dL/dw must equal X^T (p - y) / n, and dL/db the mean of (p - y)."""
    X, y, _, _ = synthetic_binary(n=500)
    params = {"w": mx.array([0.4, -0.3, 0.2, 0.1]), "b": mx.array([0.05])}

    _, grads = mx.value_and_grad(train_binary.make_loss(X, y, 0.0))(params)
    p = mx.sigmoid(train_binary.logits(params, X))
    expected_w = X.T @ (p - y) / X.shape[0]
    expected_b = mx.mean(p - y)
    mx.eval(grads, expected_w, expected_b)

    assert mx.max(mx.abs(grads["w"] - expected_w)).item() < 1e-5
    assert abs(grads["b"].item() - expected_b.item()) < 1e-5


def test_gradient_matches_finite_differences():
    X, y, _, _ = synthetic_binary(n=300)
    loss_fn = train_binary.make_loss(X, y, 0.0)
    params = {"w": mx.array([0.3, -0.7, 1.1, 0.2]), "b": mx.array([0.1])}
    _, grads = mx.value_and_grad(loss_fn)(params)
    mx.eval(grads)

    eps = 1e-3
    for i in range(params["w"].shape[0]):
        bump = eps * (mx.arange(params["w"].shape[0]) == i).astype(mx.float32)
        up = loss_fn({"w": params["w"] + bump, "b": params["b"]})
        down = loss_fn({"w": params["w"] - bump, "b": params["b"]})
        numeric = ((up - down) / (2 * eps)).item()
        assert abs(numeric - grads["w"][i].item()) < 1e-2


# --- numerical stability --------------------------------------------------------


def test_loss_survives_extreme_logits():
    """The naive formula overflows here; the logaddexp form must not."""
    z = mx.array([-500.0, -100.0, 0.0, 100.0, 500.0])
    for target in (0.0, 1.0):
        y = mx.full(z.shape, target)
        loss = train_binary.bce_with_logits(z, y)
        mx.eval(loss)
        assert mx.isfinite(loss).item(), f"loss not finite for y={target}"

    naive = -mx.mean(mx.log(mx.sigmoid(mx.array([-500.0]))))
    mx.eval(naive)
    assert not mx.isfinite(naive).item(), "naive form was expected to overflow"


def test_loss_matches_textbook_form_in_safe_range():
    z = mx.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = mx.array([0.0, 1.0, 1.0, 0.0, 1.0])
    p = mx.sigmoid(z)
    textbook = -mx.mean(y * mx.log(p) + (1 - y) * mx.log(1 - p))
    stable = train_binary.bce_with_logits(z, y)
    mx.eval(textbook, stable)
    assert abs(textbook.item() - stable.item()) < 1e-6


# --- cross-path agreement: softmax with K=2 is binary logistic -------------------


def test_softmax_with_two_classes_matches_binary():
    """Softmax regression with K=2 is the same model as binary logistic regression,
    so the two must reach the same decisions."""
    X, y, _, _ = synthetic_binary(n=1500, d=4)
    params = fit(X, y, lr=0.5, iters=2000)
    binary_pred = (mx.sigmoid(train_binary.logits(params, X)) >= 0.5).astype(mx.int32)

    mx.random.seed(SEED)
    model = nn.Linear(X.shape[1], 2)
    optimizer = optim.SGD(learning_rate=0.5)
    loss_and_grad = nn.value_and_grad(model, train_softmax.cross_entropy)
    y_int = y.astype(mx.int32)
    for _ in range(2000):
        _, g = loss_and_grad(model, X, y_int)
        optimizer.update(model, g)
        mx.eval(model.parameters(), optimizer.state)
    softmax_pred = mx.argmax(model(X), axis=-1).astype(mx.int32)

    agreement = mx.mean((binary_pred == softmax_pred).astype(mx.float32))
    mx.eval(agreement)
    assert agreement.item() > 0.99


def test_cross_entropy_matches_manual_softmax():
    mx.random.seed(SEED)
    model = nn.Linear(3, 4)
    X = mx.random.normal((10, 3))
    y = mx.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=mx.int32)

    z = model(X)
    manual = -mx.mean(mx.log(mx.take_along_axis(mx.softmax(z, axis=-1), y[:, None], axis=-1)))
    stable = train_softmax.cross_entropy(model, X, y)
    mx.eval(manual, stable)
    assert abs(manual.item() - stable.item()) < 1e-5


# --- known answers: metrics -----------------------------------------------------


def test_confusion_matrix_known_answer():
    y_true = mx.array([0, 0, 1, 1, 2])
    y_pred = mx.array([0, 1, 1, 1, 2])
    cm = confusion_matrix(y_true, y_pred, 3)
    mx.eval(cm)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]


def test_precision_recall_known_answer():
    cm = mx.array([[2.0, 1.0], [1.0, 3.0]])
    precision, recall, f1 = precision_recall_f1(cm)
    mx.eval(precision, recall, f1)
    # class 1: tp=3, predicted=4, actual=4
    assert precision[1].item() == pytest.approx(0.75, abs=1e-6)
    assert recall[1].item() == pytest.approx(0.75, abs=1e-6)
    assert f1[1].item() == pytest.approx(0.75, abs=1e-6)


def test_roc_auc_known_answers():
    """AUC is the probability that a random positive outranks a random negative, so
    each case below can be checked by counting positive/negative pairs by hand."""
    y = mx.array([0, 0, 1, 1])  # samples 2 and 3 are the positives
    cases = [
        # every positive outranks every negative -> 4/4
        (mx.array([0.1, 0.2, 0.8, 0.9]), 1.0),
        # every positive is outranked -> 0/4
        (mx.array([0.9, 0.8, 0.2, 0.1]), 0.0),
        # positives 0.2 and 0.8 vs negatives 0.1 and 0.9 -> 2/4
        (mx.array([0.1, 0.9, 0.2, 0.8]), 0.5),
        # positives 0.2 and 0.9 vs negatives 0.1 and 0.8 -> 3/4
        (mx.array([0.1, 0.8, 0.2, 0.9]), 0.75),
    ]
    for scores, expected in cases:
        auc = roc_auc(y, scores)
        mx.eval(auc)
        assert auc.item() == pytest.approx(expected, abs=1e-6)


def test_best_f1_threshold_prefers_a_perfect_split():
    y = mx.array([0, 0, 1, 1])
    scores = mx.array([0.1, 0.2, 0.8, 0.9])
    threshold, f1 = best_f1_threshold(y, scores)
    mx.eval(threshold, f1)
    assert f1.item() == pytest.approx(1.0, abs=1e-6)
    assert threshold.item() == pytest.approx(0.8, abs=1e-6)


def test_accuracy_known_answer():
    assert accuracy(mx.array([0, 1, 1, 0]), mx.array([0, 1, 0, 0])).item() == pytest.approx(0.75)


# --- scaler ---------------------------------------------------------------------


def test_standardizer_survives_constant_column():
    a = mx.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    out = Standardizer.fit(a).transform(a)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item())
