"""Shared plumbing for the two training variants: device selection, timing, memory,
and writing the per-run record that make_results.py turns into results.md.

Kept out of the training scripts so that reading train_manual.py or train_nn.py shows
the training loop and nothing else.
"""

import contextlib
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx

RECIPE_DIR = Path(__file__).resolve().parent


def select_device(name):
    mx.set_default_device(mx.gpu if name == "gpu" else mx.cpu)
    return name


def _sysctl(key):
    """Read a macOS sysctl value, or None anywhere it is unavailable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _mlx_version():
    # The version lives on mlx.core, not on the mlx package.
    v = getattr(mx, "__version__", None)
    if v:
        return v
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mlx")
    except PackageNotFoundError:
        return "unknown"


def environment(device):
    """Everything needed to make a benchmark number interpretable.

    The chip matters more than the architecture: 'arm64' is true of an M1 and an M3
    Max alike, and a throughput number is meaningless without knowing which.
    """
    chip = None
    memory_gb = None
    if platform.system() == "Darwin":
        chip = _sysctl("machdep.cpu.brand_string")
        mem = _sysctl("hw.memsize")
        if mem and mem.isdigit():
            memory_gb = round(int(mem) / (1024**3))

    return {
        "device": device,
        "chip": chip or platform.processor() or platform.machine(),
        "memory_gb": memory_gb,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx": _mlx_version(),
    }


class Benchmark:
    """Wall clock and peak memory around the training loop.

    MLX is lazy. A timer that stops without forcing evaluation measures graph
    construction rather than compute, so the loop must call mx.eval before the clock
    stops. Equally, calling .item() inside the loop forces a host sync every iteration
    and turns a throughput measurement into a measurement of synchronisation.
    """

    def __init__(self):
        self.seconds = None
        self.peak_bytes = None

    def __enter__(self):
        with contextlib.suppress(AttributeError):
            mx.reset_peak_memory()
        self._tic = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._tic
        try:
            self.peak_bytes = mx.get_peak_memory()
        except AttributeError:
            self.peak_bytes = None
        return False


def write_run(variant, device, cfg, metrics, bench, loss_history):
    """Write one run record. make_results.py aggregates these into results.md."""
    out_dir = RECIPE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    record = {
        "variant": variant,
        "env": environment(device),
        "config": {
            "optimizer": cfg["train"]["optimizer"],
            "learning_rate": cfg["train"]["learning_rate"],
            "num_iters": cfg["train"]["num_iters"],
            "seed": cfg["seed"],
        },
        "metrics": metrics,
        "timing": {
            "seconds": bench.seconds,
            "iters_per_second": cfg["train"]["num_iters"] / bench.seconds
            if bench.seconds
            else None,
        },
        "peak_memory_bytes": bench.peak_bytes,
        "final_train_loss": loss_history[-1] if loss_history else None,
    }
    path = out_dir / f"{variant}-{device}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return record, path


def report(record):
    m, t = record["metrics"], record["timing"]
    peak = record["peak_memory_bytes"]
    peak_s = f"{peak / 1e6:.1f} MB" if peak else "n/a"
    print(
        f"[{record['variant']} / {record['env']['device']}] "
        f"RMSE ${m['rmse']:,.0f}  MAE ${m['mae']:,.0f}  R2 {m['r2']:.4f}  "
        f"{t['seconds']:.3f}s  {t['iters_per_second']:,.0f} it/s  peak {peak_s}"
    )
