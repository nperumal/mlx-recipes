"""Fit the model and write a checkpoint into checkpoints/ (gitignored).

Name this train_<variant>.py if the recipe compares two implementations.

Contract:
    - Reads every hyperparameter from config.yaml.
    - Pure MLX in the math path. No NumPy, no scikit-learn.
    - Seeds explicitly so a run is reproducible.
    - Calls mx.eval inside the training loop, and does NOT call .item() there --
      MLX is lazy, so mx.eval is what runs the computation, while .item() copies
      back to the host on every iteration for no reason. Collect loss arrays and
      convert them after the loop.
    - Prints the metrics and a single timing line. Nothing more elaborate: this is a
      teaching repo, and measurement apparatus crowds out the thing being taught.
"""
