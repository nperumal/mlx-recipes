# Logistic regression, binary and softmax

Classifying penguin species from four body measurements. First as a two-class problem
with `mx.grad` on a bare weight vector, then generalised to all three species with
`mlx.nn` and softmax.

These two scripts are a **progression, not a comparison**. `train_binary.py` is
hand-rolled so the math is visible; `train_softmax.py` shows the same model expressed
through the framework once you trust it.

## What it teaches

- **Logistic regression is linear in log-odds**, not in probability. The model says
  `log(p / (1−p)) = w·x + b`, which is also how to read a coefficient: `exp(w_j)` is an
  odds ratio per standard deviation of feature `j`.
- **Why cross-entropy and not squared error.** `MSE ∘ sigmoid` is non-convex and its
  gradient carries a factor of `σ′(z) = p(1−p)`, which vanishes exactly when the model
  is confidently wrong. Cross-entropy cancels that factor.
- **The gradient is `p − y`.** Both losses here differentiate to *prediction minus
  target*, so `∂L/∂w = Xᵀ(p − y)/n` — the same shape as the least-squares gradient in
  the [linear-regression](../linear-regression/) recipe. The training loop is almost
  the same code; only the prediction function changed.
- **Computing losses from logits, never from probabilities.** `mx.logaddexp` and
  `mx.logsumexp` exist for this. See Gotchas.
- **Vectorised classification metrics.** The confusion matrix is one matmul of one-hot
  indicators; the entire ROC curve is one `argsort` and two `cumsum`s. No Python loop
  over samples anywhere in `evaluate.py` — that file is arguably a better MLX
  demonstration than either model.
- **A threshold is a separate decision from a fit.** See below.

## Running it

```sh
pip install -r requirements.txt
make all            # both variants
make separability   # the diverging-weights demonstration below
make predict        # classify one penguin
make test           # 13 correctness tests, no extra dependencies
```

Add `DEVICE=cpu` to run on the CPU instead of the GPU.

## Results

Apple M2 Max, macOS 14.4.1, MLX 0.32.2. The shuffle, the split and the
initialisation are all seeded from `seed: 0` in `config.yaml`, so `make all`
reproduces these numbers exactly.

```
binary  Adelie vs Chinstrap  (positive class: Chinstrap)
  accuracy   0.9773    ROC-AUC 1.0000
  best F1 1.0000 at threshold 0.266

softmax  Adelie, Chinstrap, Gentoo
  accuracy   0.9565    macro F1 0.9550
  Gentoo     precision 1.0000  recall 1.0000
```

Both error modes are Adelie↔Chinstrap; Gentoo is never confused with anything. That
matches the biology — Gentoo are substantially larger — and it is why the binary half
uses the Adelie/Chinstrap pair.

## The threshold is not the model

The binary run says ROC-AUC **1.0000** and accuracy **0.9773** at the same time. Those
are not in tension, and the gap is the whole lesson.

AUC 1.0 means the model ranks *every* Chinstrap above *every* Adelie — the probabilities
are perfectly ordered. The single misclassification comes from cutting that ranking at
`p = 0.5`, which is simply the wrong place to cut. Moving the threshold to `0.266`
gives F1 = 1.0.

So: fitting produces a ranking. Choosing a threshold is a second, separate decision,
driven by what a false positive costs relative to a false negative. Reporting accuracy
alone hides that decision inside a default nobody chose.

## Separability, and why `l2` is not optional

**All three species pairs are linearly separable in these four measurements.** That
sounds like good news and is in fact a problem: when the classes are separable, the
maximum-likelihood estimate *does not exist*. The optimizer can always lower the loss
by scaling `w` up, pushing fitted probabilities closer to exactly 0 and 1, so `‖w‖`
grows without bound and never converges.

Run `make separability` to reproduce both claims. For Adelie vs Chinstrap:

| iterations | `l2 = 0.0` — ‖w‖ (train loss) | `l2 = 0.01` — ‖w‖ (train loss) |
|---:|---:|---:|
| 500 | 3.65 (8.9e-02) | 2.60 (2.0e-01) |
| 2,000 | 5.90 (4.6e-02) | 2.73 (2.0e-01) |
| 8,000 | 9.09 (2.6e-02) | 2.73 (2.0e-01) |
| 20,000 | 12.01 (1.7e-02) | 2.73 (2.0e-01) |

The unpenalised column never settles. The penalised one is stationary after 2,000
iterations. The same command shows all three pairs reaching 1.0000 training accuracy
with the loss still falling, which is what "separable" looks like from the inside.

The part worth internalising: **the metrics never told you.** Test accuracy and AUC are
identical either way. Nothing in a standard report reveals that the parameters are
running away — you have to look at `‖w‖`, which is why `train_binary.py` prints it. The
ridge penalty adds a `λ‖w‖²` term that grows faster than the likelihood improves,
which restores a unique finite solution.

## How it works

`data_pipeline.py` reads the CSV through `mlx.data`, drops rows with missing
measurements, maps species names to class indices, shuffles under `seed`, splits 80/20
and standardises on the training split only.

`train_binary.py` — parameters as a plain dict, which `mx.value_and_grad`
differentiates as a pytree:

```python
params = {"w": mx.zeros((d,)), "b": mx.zeros((1,))}
loss_and_grad = mx.value_and_grad(make_loss(X, y, l2))
for _ in range(num_iters):
    loss, grads = loss_and_grad(params)
    params = {k: v - lr * grads[k] for k, v in params.items()}
    mx.eval(params, loss)
```

`train_softmax.py` — the same model with `w (d,)` widened to `W (d, K)`, sigmoid
replaced by softmax, and the update delegated to `mlx.optimizers`.

## Gotchas

**Never compute the loss from the probability.** The textbook expression
`−[y·log(p) + (1−y)·log(1−p)]` overflows: at `z = −50` the sigmoid underflows to
exactly `0.0` in float32 and `log(0)` is `−inf`. Rewrite in terms of the logit, where
`−log σ(z) = log(1 + e^(−z)) = logaddexp(0, −z)`:

```python
zero = mx.zeros_like(z)
loss = mx.mean(y * mx.logaddexp(zero, -z) + (1 - y) * mx.logaddexp(zero, z))
```

Same number, no overflow, no clipping. The multiclass equivalent is
`logsumexp(z) − z_y`, which `mx.logsumexp` makes stable by subtracting the row maximum
internally. A test in `tests/` asserts the naive form really does overflow while this
one stays finite — worth reading.

**An empty CSV field is a zero-length byte buffer.** `mlx.data` hands back every field
as bytes, so a missing value decodes to `""` and `float("")` raises. `_number()` in
`data_pipeline.py` returns `None` for those and the loader drops the row. Two of the
344 penguins are missing every measurement.

**`mx.unique` does not exist**, which matters if you write ROC code that wants distinct
thresholds. The implementation here sidesteps it: sorting by score and taking cumulative
sums of the label vector gives the full curve without ever needing unique values.

**Class order defines the encoding.** `binary.classes: [Adelie, Chinstrap]` makes
Chinstrap class 1, the positive class. Swap them and precision/recall swap meaning.

## Configuration

Everything lives in [config.yaml](config.yaml).

| Key | Default | Meaning |
|---|---|---|
| `seed` | `0` | Seeds the shuffle and initialisation |
| `data.train_split` | `0.8` | Fraction used for training |
| `binary.classes` | `[Adelie, Chinstrap]` | The two species, and the label encoding |
| `binary.threshold` | `0.5` | Decision cut; see the threshold section |
| `softmax.classes` | all three | Class order for the multiclass problem |
| `train.optimizer` | `sgd` | `sgd` or `adam` (softmax variant) |
| `train.learning_rate` | `0.1` | |
| `train.num_iters` | `500` | Full-batch iterations |
| `train.l2` | `0.01` | Ridge penalty — read the separability section before setting it to 0 |

## Requirements

Apple silicon, macOS 14.0 or later, and a **native arm64** Python 3.10 or later. If pip
reports `No matching distribution found for mlx`:

```sh
python3 -VV                                             # must be 3.10+
python3 -c "import platform; print(platform.machine())" # must print arm64
sw_vers -productVersion                                 # must be 14.0+
```

## Data

`data/penguins.csv` — the Palmer Archipelago penguin data collected by Dr. Kristen
Gorman at Palmer Station, Antarctica LTER, distributed in the
[palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) package. **CC0**
(public domain). 344 rows, 15 KB, committed rather than downloaded.

Four measurements — `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`,
`body_mass_g` — predict `species`: Adelie (152), Gentoo (124), Chinstrap (68). Two rows
have no measurements and are dropped, leaving 342.

Horst AS, Hill AP, Gorman KB (2020). *palmerpenguins: Palmer Archipelago (Antarctica)
penguin data.* R package version 0.1.0.
