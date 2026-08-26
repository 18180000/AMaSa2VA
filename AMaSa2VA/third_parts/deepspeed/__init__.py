"""Minimal deepspeed stub for Sa2VA VQA inference imports.

This is only used to satisfy xtuner import-time references during inference.
It is not a functional DeepSpeed implementation.
"""

from contextlib import ContextDecorator

__version__ = "0.0.0-stub"


class _GatheredParameters(ContextDecorator):
    def __init__(self, params=None, modifier_rank=0):
        self.params = params
        self.modifier_rank = modifier_rank

    def __enter__(self):
        return self.params

    def __exit__(self, exc_type, exc, tb):
        return False


class _Zero:
    GatheredParameters = _GatheredParameters


zero = _Zero()
