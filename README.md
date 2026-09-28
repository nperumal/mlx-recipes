# mlx-recipes

End-to-end machine learning recipes for [MLX](https://github.com/ml-explore/mlx), Apple's
array framework for Apple silicon.

## What this is

[mlx-examples](https://github.com/ml-explore/mlx-examples) gives you standalone model
implementations. [mlx-lm](https://github.com/ml-explore/mlx-lm) gives you a full-featured
package for running and fine-tuning LLMs. This repo covers the connective tissue that
neither is aimed at: data preparation, evaluation harnesses, honest benchmarks, and the
whole path from a raw dataset to something you can actually run.

Every recipe here:

- runs end to end with a single command
- keeps every hyperparameter in `config.yaml` — no magic numbers buried in code
- is readable top to bottom: the model and the training loop are the content, not
  something you have to find between layers of scaffolding
- says plainly where MLX surprised us, in a Gotchas section — the laziness of
  `mx.eval`, missing ops, dtype defaults

## Recipes

| Recipe | What it teaches |
|---|---|
| [foundations/linear-regression](recipes/foundations/linear-regression/) | `mx.grad` and a hand-written update vs `nn.Linear` + `mlx.optimizers` — the same fit at two levels of abstraction, and when MLX actually computes anything |
| [foundations/logistic-regression](recipes/foundations/logistic-regression/) | binary then softmax classification — the `p − y` gradient, computing losses from logits with `logaddexp`/`logsumexp`, vectorised confusion matrices and ROC, and why separable classes have no maximum-likelihood solution |

## Pure MLX

The modelling path is 100% MLX. Fits, gradients, metrics, splits and scaling are all
`mlx.core` / `mlx.nn` / `mlx.optimizers` — no scikit-learn, no NumPy, no SciPy. The point
of the repo is to show what the framework can do, so anything that would hide MLX behind
another library is out. File I/O and plotting are exempt, and tests may use an external
reference as an optional dev dependency. [CONTRIBUTING.md](CONTRIBUTING.md) has the full
rule, and CI enforces it.

## Requirements

Apple silicon, macOS 14.0 or later, and a **native arm64** Python 3.10 or later. If pip
reports `No matching distribution found for mlx`, check the environment rather than the
version pin:

```sh
python3 -VV                                             # must be 3.10+
python3 -c "import platform; print(platform.machine())" # must print arm64
sw_vers -productVersion                                 # must be 14.0+
```

An x86_64 Python under Rosetta will not resolve `mlx`.

## Running a recipe

Recipes are standalone. Each one has its own dependencies and its own Makefile:

```sh
cd recipes/foundations/linear-regression
pip install -r requirements.txt
make all
```

## Roadmap

Recipes are organised into tiers, each with its own spine. Only the first tier has
anything in it today; the rest describe where this is going, not work that exists.

Performance measurement is deliberately absent from the foundations tier. These models
are small enough that timing them teaches nothing about the model, and benchmark
scaffolding crowds out the code a reader came to see. It belongs in the later tiers,
where "how fast, and how much memory" is a question people genuinely have.

- **`foundations/`** — data → train → eval → predict. Tabular and classical models.
  These are not here because MLX is the best tool for fitting a linear regression; for
  small tabular problems it usually isn't, and the recipes say so. They are here because
  they are the best on-ramp to the framework: the model is something you already
  understand, so all of your attention goes to `mx.array`, `mx.grad`, lazy evaluation,
  device placement and `float32` defaults. Next up: a closed-form least squares
  recipe covering `mx.linalg`, QR vs Cholesky and `float32` conditioning.
- **`deep-learning/`** — data → train → eval → export. Real networks, real training
  loops, checkpointing and schedules.
- **`llm/`** — data → train → eval → quantize → serve. The only tier where the full
  pipeline applies, leaning on mlx-lm rather than reimplementing it.

Shared utilities will appear once a second recipe actually needs them, extracted from two
working examples rather than guessed at in advance.

## Contributing

New recipes are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) is the contract, and
[`_template/`](_template/) is the skeleton to copy — it mirrors the linear-regression
recipe, which is the reference implementation.

## Related

- [mlx](https://github.com/ml-explore/mlx) — the framework
- [mlx-examples](https://github.com/ml-explore/mlx-examples) — standalone model implementations
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLMs with MLX
- [mlx-data](https://github.com/ml-explore/mlx-data) — the data loader used here
- [mlx-learn](https://pypi.org/project/mlx-learn/) — a scikit-learn-style API on MLX
  (alpha). Complementary: that is a library, this is recipes and measurements.

## License

MIT — see [LICENSE](LICENSE).
