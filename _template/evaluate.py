"""Metrics for this recipe, in pure MLX.

One implementation, imported by every training variant, so the variants are directly
comparable. Metrics are computed on the original scale: if the target was standardised
for training, predictions are converted back before they reach here.

Not implemented -- see recipes/foundations/linear-regression/evaluate.py for a worked
version. Planned shape:

    def <task>_metrics(y_true, y_pred): ...   # returns a plain dict of floats
    def load_config(path=None): ...           # reads config.yaml
    def main(): ...                           # re-score a saved checkpoint

Everything vectorised. A Python loop over samples is a bug.
"""
