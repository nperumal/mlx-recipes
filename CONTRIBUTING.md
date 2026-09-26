# Contributing

Thanks for considering a recipe. This file is the contract — a recipe that follows it
will be reviewed quickly; one that doesn't will get the same three comments every time.

## 1. The purity rule

The whole point of this repo is to show what MLX can do. Code that hides MLX behind
another library defeats that. So:

**Zone 1 — the math path: 100% MLX, no exceptions.**
Fitting, prediction, losses, gradients, metrics, train/test splits, scaling,
cross-validation, sampling. All `mlx.core` / `mlx.nn` / `mlx.optimizers`.
No scikit-learn. No NumPy. No SciPy.

**Zone 2 — I/O and plotting: exempt.**
Reading a CSV with the standard library `csv` module and drawing a loss curve with
matplotlib is fine. Being pure here produces awkward code and demonstrates nothing.

**Zone 3 — tests: an external reference is allowed, but only as an optional dev
dependency.** See §4. A user who clones a recipe and runs it must never install
scikit-learn.

If MLX is missing something you need, implement it in the recipe and write a short
section in the README explaining how. `mx.unique` and a quantile function
don't exist, for instance, and working around those is genuinely useful content — not a
reason to reach for NumPy.

## 2. Recipe layout

Copy `_template/`, which mirrors
[`recipes/foundations/linear-regression/`](recipes/foundations/linear-regression/) — the
reference implementation. Read that recipe before writing a new one.

```
recipes/<tier>/<recipe-name>/
├── README.md          # goal, what it teaches, results table, gotchas
├── config.yaml        # every hyperparameter
├── data_pipeline.py   # loading, splitting, scaling
├── train.py           # or train_<variant>.py when comparing implementations
├── evaluate.py        # the recipe's metrics, in MLX
├── predict.py         # or serve.py in the llm tier
├── Makefile           # make all runs it end to end
├── requirements.txt   # pinned
├── data/              # small committed dataset, or gitignored downloads
└── tests/
    └── test_correctness.py
```

**Recipes are standalone and self-contained.** A reader should understand one folder
without opening any other, and there is currently no shared library to import from.
That is deliberate: shared utilities will be extracted once two recipes demonstrably
need the same thing, rather than guessed at in advance. If you find yourself wanting a
`common/` module while writing the second recipe in the repo, say so in the PR and we
will pull it out of both.

## 3. No magic numbers

Every hyperparameter lives in `config.yaml`. Learning rates, batch sizes, seeds,
regularization strengths, split ratios.

**Keep measurement out of the foundations tier.** A training script may print one
timing line; it should not carry a benchmarking framework. These models are small, the
numbers teach nothing about the model, and the apparatus buries the code a reader came
for. If you do quote a number in a README, say which machine produced it. Recipes in
the deep-learning and llm tiers are a different matter — there, performance is a real
question and measuring it properly is part of the recipe.

## 4. Testing

Four kinds of test, none of which need an external library:

- **Synthetic recovery** — generate data from known parameters, assert you recover them
  within tolerance. The strongest test available for a classical model.
- **Cross-path agreement** — where a recipe has two routes to the same answer (closed
  form and gradient descent, say), assert they agree.
- **Gradient check** — finite differences against `mx.grad`. Validates the autograd
  usage directly.
- **Known-answer** — a tiny case worked out by hand.

You may additionally add a parity test against an external reference. It must be marked
`@pytest.mark.reference` and skip cleanly when the reference isn't installed, and the
reference must appear only in `requirements-dev.txt`.

## 5. Data and size

No weights and no checkpoints in git, ever.

Datasets: a reference dataset under about 1 MB may be committed inside the recipe at
`recipes/<tier>/<name>/data/`, provided the README states its origin and licence.
Making a reader download a 20 KB CSV before they can run an example is a worse first
experience than carrying it in the repo. Anything larger is downloaded by
`prepare_data.py` into a gitignored `data/` directory and checksum-verified.

Committed assets are otherwise limited to small generated plots.

## 6. Before opening a PR

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest` passes without dev extras installed
- [ ] `make all` runs clean from a fresh clone on Apple silicon
- [ ] README states what the recipe teaches, not just what it does
- [ ] README has a Gotchas section, and it is specific
- [ ] No NumPy or scikit-learn imports outside `tests/` and plotting
