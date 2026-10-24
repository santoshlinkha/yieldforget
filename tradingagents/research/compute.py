"""Which machine this is, and whether a job runs on the CPU or the GPU.

``hardware()`` reports the operating system, CPU, memory, GPU, VRAM, driver and
the installed numerical frameworks, as found now: nothing is assumed from an
older description.

``choose_device(setting, rows)``: ``cpu`` and ``cuda`` are obeyed (``cuda`` falls
back to the CPU, with a note, when no usable GPU is present); ``auto`` uses the
GPU only for boosted-tree jobs big enough to gain from it. On this machine's
benchmark (an RTX 3050 Ti Laptop, 4 GB) that meant 200,000 rows or more; smaller
jobs are faster on the CPU once data transfer is counted.

``gpu_slot()`` lets one process at a time use the GPU, through a lock file, so
two jobs cannot exhaust its memory together. A job that fails on the GPU is
rerun on the CPU; it never falls back to a paid remote service.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

AUTO_GPU_ROWS = 200_000


def _nvidia() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, total, used, driver, cap = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "vram_mb": int(total), "vram_used_mb": int(used), "driver": driver,
                     "compute_capability": cap})
    return gpus


def _ram_gb() -> float | None:
    try:
        if os.name == "nt":
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEM()
            m.dwLength = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 2 ** 30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except (OSError, ValueError, AttributeError):
        return None


def _version(module: str) -> str | None:
    try:
        mod = __import__(module)
        return getattr(mod, "__version__", "installed")
    except ImportError:
        return None


def hardware() -> dict:
    cpu = platform.processor()
    if os.name == "nt":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=15).stdout.strip() or cpu
    frameworks = {m: _version(m) for m in ("numpy", "pandas", "sklearn", "xgboost", "arch", "torch")}
    xgb_cuda = None
    if frameworks["xgboost"]:
        import xgboost
        xgb_cuda = bool(xgboost.build_info().get("USE_CUDA"))
    torch_cuda = None
    if frameworks["torch"]:
        import torch
        torch_cuda = {"available": torch.cuda.is_available(), "cuda": torch.version.cuda}
    return {"os": f"{platform.system()} {platform.release()} ({platform.version()})", "python": sys.version.split()[0],
            "cpu": cpu, "cores": os.cpu_count(), "ram_gb": _ram_gb(), "gpus": _nvidia(),
            "frameworks": frameworks, "xgboost_cuda_build": xgb_cuda, "torch_cuda": torch_cuda}


def cuda_usable() -> bool:
    if not _nvidia():
        return False
    try:
        import xgboost
        return bool(xgboost.build_info().get("USE_CUDA"))
    except ImportError:
        return False


def choose_device(setting: str, rows: int) -> tuple[str, str]:
    """``(device, why)`` for a boosted-tree job of ``rows`` rows."""
    setting = (setting or "auto").lower()
    if setting == "cpu":
        return "cpu", "set to CPU"
    usable = cuda_usable()
    if setting in ("gpu", "cuda"):
        return ("cuda", "set to GPU") if usable else ("cpu", "set to GPU, but no usable CUDA GPU was found")
    if usable and rows >= AUTO_GPU_ROWS:
        return "cuda", f"auto: {rows:,} rows is enough to gain from the GPU"
    return "cpu", "auto: small job, the CPU is as fast" if usable else "auto: no usable CUDA GPU"


@contextlib.contextmanager
def gpu_slot(folder: Path, timeout: float = 600.0):
    """Hold the machine's single GPU slot (a lock file) for the duration; waits up to ``timeout``."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "gpu.lock"
    start = time.time()
    handle = open(path, "a+b")  # noqa: SIM115 (held for the whole block)
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() - start > timeout:
                    raise TimeoutError("another job has held the GPU too long") from None
                time.sleep(0.5)
        yield
    finally:
        with contextlib.suppress(OSError):
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def run_on(device: str, folder: Path, fit_cpu, fit_gpu):
    """Run ``fit_gpu`` holding the GPU slot, or ``fit_cpu``; a GPU failure reruns on the CPU.
    Returns ``(result, device used, note)``."""
    if device != "cuda":
        return fit_cpu(), "cpu", None
    try:
        with gpu_slot(folder):
            return fit_gpu(), "cuda", None
    except Exception as exc:  # out of memory, driver trouble: the CPU, never a paid service
        return fit_cpu(), "cpu", f"GPU run failed ({type(exc).__name__}: {str(exc)[:120]}); ran on the CPU"
