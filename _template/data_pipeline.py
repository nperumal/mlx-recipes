"""Loading, splitting and scaling. Shared by every training variant in the recipe.

If the recipe compares two implementations, they must import this so they see identical
inputs -- otherwise the comparison in results.md means nothing.

Not implemented -- see recipes/foundations/linear-regression/data_pipeline.py. Planned
shape:

    def load_raw(csv_path, feature_names, label_name): ...
    class Standardizer: ...    # fit on train only; transform / inverse
    def load(cfg): ...         # returns splits, scalers, and metadata

Rules: shuffle before splitting, seeded from cfg["seed"]; fit scalers on the training
split only; keep the test target on its original scale so metrics are reported in real
units.
"""
