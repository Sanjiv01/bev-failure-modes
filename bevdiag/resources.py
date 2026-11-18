"""Resource budget: use ~90% of the machine, leave ~10% headroom so the laptop stays responsive.

Every entry point calls :func:`apply_budget` once before doing heavy work.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

BUDGET_FRACTION = float(os.environ.get("BEVDIAG_BUDGET", "0.9"))


@dataclass(frozen=True)
class Budget:
    cpu_threads: int
    loader_workers: int
    gpu_mem_fraction: float
    ram_bytes: int


def compute_budget(fraction: float = BUDGET_FRACTION) -> Budget:
    import psutil

    logical = os.cpu_count() or 4
    threads = max(1, int(logical * fraction))  # 16 -> 14
    # image decoding workers share the thread budget with torch intra-op threads
    workers = max(0, min(8, threads // 2))
    ram = int(psutil.virtual_memory().total * fraction)
    return Budget(cpu_threads=threads, loader_workers=workers,
                  gpu_mem_fraction=fraction, ram_bytes=ram)


def apply_budget(fraction: float = BUDGET_FRACTION, verbose: bool = True) -> Budget:
    b = compute_budget(fraction)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(b.cpu_threads))
    import torch

    torch.set_num_threads(b.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(b.gpu_mem_fraction, 0)
    try:  # lower process priority a notch so the desktop stays snappy
        import psutil

        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(5)
    except Exception:
        pass
    if verbose:
        print(f"[budget] {fraction:.0%}: {b.cpu_threads} CPU threads, {b.loader_workers} loader workers, "
              f"GPU mem cap {b.gpu_mem_fraction:.0%}")
    return b
