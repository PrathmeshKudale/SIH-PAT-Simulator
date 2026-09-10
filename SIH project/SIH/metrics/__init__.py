"""
Metrics module: structured logging (CSV/JSON), quantitative metric calculation,
and batch scenario benchmarking.
"""

from metrics.logger import MetricsLogger, SwitchEvent, TrackLossEvent
from metrics.calculator import MetricsCalculator

def __getattr__(name: str):
    if name == "BatchScenarioRunner":
        from metrics.batch_runner import BatchScenarioRunner
        return BatchScenarioRunner
    raise AttributeError(f"module 'metrics' has no attribute '{name}'")

__all__ = [
    "MetricsLogger",
    "SwitchEvent",
    "TrackLossEvent",
    "MetricsCalculator",
    "BatchScenarioRunner",
]
