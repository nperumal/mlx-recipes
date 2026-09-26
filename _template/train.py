"""Fit the model and write a checkpoint into checkpoints/ (gitignored).

Name this train_<variant>.py if the recipe compares two implementations.

Contract:
    - Reads every hyperparameter from config.yaml.
    - Pure MLX in the math path. No NumPy, no scikit-learn.
    - Seeds explicitly so a run is reproducible.
    - Runs untimed warmup iterations before the benchmark window opens.
    - Inside the timed loop: mx.eval on parameters and loss, never .item().
    - Writes a run record via runner.write_run for make_results.py.
"""
