# Perceptron, by hand and by gradient

A single neuron learning the NAND gate, twice. First with Rosenblatt's 1958 update
rule and a hard step function; then with a sigmoid, a loss function and automatic
differentiation.

These are a **progression, not a comparison**. The first file is how this was done
before backpropagation existed. The second is what replaced it, and the reason for
the replacement is printed at the top of its output.

> **This is a warm-up, not a full recipe.** It has no dataset, no `config.yaml`, no
> `Makefile` and no tests — see [CONTRIBUTING.md](../../../CONTRIBUTING.md) section 3.
> The four rows of the truth table are the entire input space, and the hyperparameters
> are part of the explanation rather than something to tune, so the usual recipe
> scaffolding would bury a sixty-line demonstration rather than support it.

## What it teaches

- **A learning algorithm does not need a loss function.** `perceptron.py` has
  neither a loss nor a gradient anywhere in it. It adjusts the weights whenever a
  prediction is wrong, and stops when none are. That is the whole algorithm.
- **Why the step function was abandoned.** Its derivative is zero everywhere, so
  there is no signal to backpropagate and no way to stack layers. `perceptron_nn.py`
  demonstrates this with `mx.grad` rather than asserting it.
- **What a decision boundary is.** With two inputs the model draws a line, and both
  files print it. Seeing `x2 = -x1 + 1` separate `(1,1)` from the rest makes
  "linearly separable" concrete.
- **Convergence is not the same as confidence.** The perceptron rule halts the moment
  every point is on the right side. Gradient descent keeps going. Same boundary,
  very different weight magnitudes — see Results.
- **`mx.grad`, `nn.Module`, `nn.value_and_grad` and `mlx.optimizers`** on the smallest
  problem that can carry them.

## Running it

```sh
pip install -r requirements.txt
python perceptron.py
python perceptron_nn.py
```

No dataset to download: the four rows of the truth table are the entire input space.

## Results

Apple silicon, MLX 0.32.x, seed 0. Both are deterministic.

**`perceptron.py`** — the 1958 rule:

```
epoch  1: mistakes 1/4  w = [+0.00, +0.00]  b = +0.00
epoch  2: mistakes 3/4  w = [-0.10, -0.10]  b = -0.10
epoch  3: mistakes 1/4  w = [+0.00, +0.00]  b = +0.20
epoch  4: mistakes 0/4  w = [-0.10, -0.10]  b = +0.10

converged after 4 epochs
  w = [-0.10, -0.10]   b = +0.10
  boundary: x2 = -1.00 * x1 +1.00
```

The mistake count goes 1 → 3 → 1 → 0. The perceptron rule does not descend
monotonically, because there is no quantity it is descending.

**`perceptron_nn.py`** — gradient descent:

```
d/dx step(x)    [0.0, 0.0, 0.0, 0.0, 0.0]   <- no signal
d/dx sigmoid(x) [0.105, 0.235, 0.25, 0.235, 0.105]

  loss 0.081891
  w = [-4.23, -4.23]   b = +6.54
  boundary: x2 = -1.00 * x1 +1.55
```

**Same slope, 40x the magnitude.** Both found the same separating direction. The
perceptron rule stopped as soon as the four points were classified; gradient descent
kept pushing the probabilities toward 0 and 1, which inflates `|w|` without moving
the line. That is the separability divergence from the
[logistic-regression](../logistic-regression/) recipe, visible on four points.

## The part that is historically important

NAND is linearly separable, so one neuron suffices. XOR is not: no straight line puts
`{(0,1), (1,0)}` on one side and `{(0,0), (1,1)}` on the other. A single neuron trained
on XOR converges to a loss of exactly `ln 2 = 0.6931` and predicts 0.5 on all four
inputs — not struggling, but provably out of capacity.

Minsky and Papert published that limitation in 1969 and it emptied the field for most
of a decade. The fix is one hidden layer, which needs backpropagation, which needs a
differentiable activation — which is why the step function had to go.

## Gotchas

**A scalar bias is not a one-element array.** This bit the first version of
`perceptron.py`:

```python
b = mx.array([0.0])  # shape (1,)
errors = y_pred - y  # shape (4,)
b -= lr * errors  # broadcasts to (4,) -- b now has one bias PER ROW
```

MLX raises nothing. The model quietly acquires four biases, memorises one answer per
training row, and leaves `w` at `[0.0, 0.0]` — while printing a perfect match between
predicted and actual. The fix is `b -= lr * mx.sum(errors)`, and `b` is declared as
`mx.array(0.0)` so the shape cannot drift.

**`nn.Linear` returns `(batch, 1)`.** Targets are usually `(batch,)`, so subtracting
them broadcasts to `(batch, batch)` and the loss silently becomes a mean over all
cross-pairs. Hence `.squeeze(-1)` in `Perceptron.__call__`.

**`mx.grad` through a step function returns zeros, not an error.** Nothing warns you
that your network cannot learn; it simply never changes.

## Requirements

Apple silicon, macOS 14.0 or later, native arm64 Python 3.10+. No `mlx-data` here —
the data is four rows written inline.

## Data

The NAND truth table:

| x1 | x2 | NAND |
|---|---|---|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Four rows is the complete input space, so there is no train/test split and nothing to
generalise to. This is function fitting, not learning in the statistical sense — which
is exactly why it is a clear place to watch a training rule work.
