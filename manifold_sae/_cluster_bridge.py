"""Cluster runtime bridges.

Shared shims used by the LLM experiment drivers so each driver doesn't
carry its own copy.
"""

from __future__ import annotations

import importlib


def bypass_gamfit_cuda_check() -> None:
    """Make gamfit's pre-flight CUDA-conflict check a no-op.

    Cluster nodes inevitably map two CUDA stacks into the process: the
    pip-bundled ``nvidia/`` wheels under the venv, AND a system CUDA
    install (usually ``/usr/local/cuda-*``). gamfit's safety check
    refuses to load Rust on dual-stack because the worst-case scenario
    (a cublas handle freed by the wrong implementation) is genuinely
    dangerous in theory.

    In practice cudarc's ``culib()`` is a process-wide ``OnceLock``, so
    every cuBLAS symbol gam ever resolves goes through one and only
    one library instance — the other mapping is dead weight that gam
    never calls into. The catastrophe condition cannot trigger.

    Upstream gam already downgraded the Rust-side check to a warning
    (see ``gam/src/gpu/runtime.rs::warn_cuda_library_conflicts``). The
    Python-side check still raises on PyPI 0.1.100, hence this shim.
    Drop it when a future gamfit release lands with the Python-side
    fix too.

    The shim must run BEFORE any gamfit module imports the Rust
    extension, so the experiment drivers call this at module load
    immediately after the stdlib imports.
    """
    try:
        import gamfit._cuda as _gc
    except ImportError:
        return  # gamfit not installed; nothing to patch

    def _no_conflicts():
        return {
            "platform": "linux",
            "mapped": {},
            "conflicts": {},
            "packaged_nvidia_roots": [],
            "packaged_complete_stacks": [],
            "system_complete_stacks": [],
        }

    _gc.cuda_diagnostics = _no_conflicts
    _gc.assert_no_cuda_library_conflicts = lambda context: None
    for mod_name in ("gamfit._binding", "gamfit.torch._reml", "gamfit._api"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "assert_no_cuda_library_conflicts"):
                mod.assert_no_cuda_library_conflicts = lambda context: None
            if hasattr(mod, "cuda_diagnostics"):
                mod.cuda_diagnostics = _no_conflicts
        except ImportError:
            pass
    try:
        import gamfit._binding as _gb
        if hasattr(_gb.rust_module, "cache_clear"):
            _gb.rust_module.cache_clear()
    except ImportError:
        pass


def require_cuda_if_env(env_var: str = "MSAE_REQUIRE_CUDA") -> None:
    """Raise immediately if the cluster job requested GPUs but CUDA
    isn't visible to torch. Saves hours of wasted compute on the
    silent-CPU-fallback failure mode.

    Set ``MSAE_REQUIRE_CUDA=1`` to enable. Job submitters do this
    automatically when ``gpus > 0``.
    """
    import os

    import torch

    if os.environ.get(env_var) != "1":
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        f"{env_var}=1 but torch.cuda.is_available()=False "
        f"(torch.version.cuda={torch.version.cuda!r}). Likely a "
        f"torch/driver mismatch — pin torch in pyproject.toml to "
        f"match the host driver, or unset {env_var} for a deliberate "
        f"CPU run."
    )
