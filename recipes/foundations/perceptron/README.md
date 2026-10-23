# Perceptron, and what it could not do

Four files walking from 1958 to 1986, on two logic gates.

| file | model | gradients |
|---|---|---|
| `perceptron.py` | one neuron, NAND | none at all — Rosenblatt's update rule |
| `perceptron_nn.py` | one neuron, NAND | `mx.grad` via `nn.value_and_grad` |
| `mlp.py` | two layers, XOR | backpropagation written out by hand |
| `mlp_nn.py` | two layers, XOR | `mx.grad` again, plus the single-neuron failure |

They are a **progression, not a comparison**. Read them in that order and the
argument builds: a learning rule with no gradients at all; the gradient version that
replaced it; the problem a single neuron provably cannot solve; and the hidden layer
that solves it, derived longhand and then by autograd.

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
- **The chain rule, visible.** `mlp.py` writes backpropagation out: two deltas, two
  transposes, four parameter updates. `mlp_nn.py` replaces that entire method with one
  call to `nn.value_and_grad`. Reading them side by side shows exactly what autograd
  is doing for you.
- **`mx.grad`, `nn.Module`, `nn.value_and_grad` and `mlx.optimizers`** on the smallest
  problems that can carry them.

## Running it

```sh
pip install -r requirements.txt
python perceptron.py      # NAND, the 1958 rule
python perceptron_nn.py   # NAND, by gradient descent
python mlp.py             # XOR, backprop by hand
python mlp_nn.py          # XOR, by autograd
```

No dataset to download: the four rows of the truth table are the entire input space.

## Results

Seed 0 throughout, and every run is deterministic — the same four commands reproduce
these exactly.

<!-- Numbers below were produced on a CPU build of MLX 0.32.3. Re-run the four
     scripts on Apple silicon and replace this comment with the chip and MLX
     version, as the other recipes do. -->

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

## XOR: the limitation, demonstrated

NAND is linearly separable, so one neuron suffices. XOR is not: no straight line puts
`{(0,1), (1,0)}` on one side and `{(0,0), (1,1)}` on the other.

`mlp_nn.py` trains a single neuron on it first:

```
single neuron (the model perceptron.py trains)
  loss 0.6931   accuracy 2/4
   0   0 |   0    0.5000    1   XX
   0   1 |   1    0.5000    1   ok
   1   0 |   1    0.5000    1   ok
   1   1 |   0    0.5000    1   XX
```

`0.6931` is `ln 2` exactly, and the probability is `0.5000` on all four inputs. The
model is not struggling — it is predicting the base rate because it has nothing
better. There is no line to find.

Minsky and Papert published that limitation in 1969 and it emptied the field for most
of a decade. The fix is one hidden layer, which needs backpropagation, which needs a
differentiable activation — which is why the step function had to go.

Add the hidden layer and the same optimizer, on the same data, for the same number of
steps:

```
two layers, 4 hidden units
  loss 0.0000   accuracy 4/4
   0   0 |   0    0.0000    0   ok
   0   1 |   1    1.0000    1   ok
   1   0 |   1    1.0000    1   ok
   1   1 |   0    0.0000    0   ok
```

`mlp.py` reaches the same place with backpropagation written by hand — sigmoid
activations, two hidden units, MSE, 10,000 epochs:

```
  epoch     1: MSE 0.250016
  epoch  2000: MSE 0.001894
  epoch 10000: MSE 0.000251
```

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

**MLX has no `mx.dot`.** It is `mx.matmul` or the `@` operator. `mx.dot` is a NumPy
habit and fails with `AttributeError`, which is the easy case — the harder one is
`mx.zeros(1, n)`, where NumPy's two-argument shape is a `TypeError` in MLX because
`zeros` wants a tuple.

**XOR with two hidden units is genuinely fragile.** `mlp_nn.py` uses four units and
`tanh`, and those are measured choices rather than conventional ones. Accuracy over
five seeds, 3,000 Adam steps at lr 0.1:

| hidden | activation | accuracy per seed |
|---|---|---|
| 2 | tanh | fails on 5 of 8 seeds |
| 4 | tanh | **4/4 on every seed** |
| 4 | sigmoid | `[3, 4, 3, 4, 3]` |
| 4 | relu | `[2, 3, 3, 4, 3]` |

Two hidden units is exactly enough capacity for XOR and has a well-known local
minimum next to the solution. Sigmoid's gradient peaks at 0.25 and stalls in flat
regions; ReLU is worse here, because with only four training points a unit that goes
negative on all of them has zero gradient forever and never recovers.

`mlp.py` *does* solve it at width 2 — with sigmoid, MSE, a learning rate of 1.0 and
10,000 epochs. The two files disagree about what works, which is the real lesson:
with four data points, the interaction between activation, loss, optimizer and width
dominates, and no single one of them is the answer.

## Requirements

Apple silicon, macOS 14.0 or later, native arm64 Python 3.10+. No `mlx-data` here —
the data is four rows written inline.

## Data

The NAND truth table (`perceptron.py`, `perceptron_nn.py`):

| x1 | x2 | NAND |
|---|---|---|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR (`mlp.py`, `mlp_nn.py`) is the same four inputs with targets `1, 0, 0, 1` flipped
to `0, 1, 1, 0`.

Four rows is the complete input space, so there is no train/test split and nothing to
generalise to. This is function fitting, not learning in the statistical sense — which
is exactly why it is a clear place to watch a training rule work.
