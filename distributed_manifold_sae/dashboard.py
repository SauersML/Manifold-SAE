"""TensorBoard logging wrapper. Soft-dependency on tensorboard."""

from __future__ import annotations

from pathlib import Path


def build_writer(log_dir: str):
    """Return a SummaryWriter, or a no-op stub if tensorboard not installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return _NoopWriter()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


class _NoopWriter:
    def add_scalar(self, *a, **kw): pass
    def add_histogram(self, *a, **kw): pass
    def add_image(self, *a, **kw): pass
    def close(self): pass
