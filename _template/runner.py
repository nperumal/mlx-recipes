"""Device selection, timing, peak memory, and the per-run record make_results.py reads.

Kept out of the training scripts so that reading train.py shows the training loop and
nothing else.

Not implemented -- see recipes/foundations/linear-regression/runner.py. Planned shape:

    def select_device(name): ...     # mx.set_default_device(mx.gpu | mx.cpu)
    def environment(device): ...     # platform, python, mlx version
    class Benchmark: ...             # context manager: wall clock + peak memory
    def write_run(...): ...          # one JSON record per (variant, device)
    def report(record): ...          # one human-readable line

MLX is lazy, and that is the whole difficulty here. A timer that stops without forcing
evaluation measures graph construction, not compute. Calling .item() inside the training
loop forces a host sync every iteration and turns a throughput measurement into a
measurement of synchronisation. Call mx.eval on the parameters and the loss each step,
collect the loss arrays, and convert them to Python floats only after the timer stops.
"""
