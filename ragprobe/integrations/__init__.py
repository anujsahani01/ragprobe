"""Integrations — optional connectors to external platforms."""

try:
    from ragprobe.integrations.mlflow_logger import log_to_mlflow, log_agent_to_mlflow
    __all__ = ["log_to_mlflow", "log_agent_to_mlflow"]
except ImportError:
    # MLflow not installed — that's fine, it's optional
    __all__ = []
