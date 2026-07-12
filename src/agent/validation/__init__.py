"""Output-validation layer: run policy/safety checks on a turn's output and emit a rating."""

from .graph import validation_graph

__all__ = ["validation_graph"]
