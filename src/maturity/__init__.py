"""maturity -- 模块二：成熟度评估与问题发现。

公开入口：``analyze`` / ``analyze_graph`` 与 ``Problem`` 数据契约。
"""
from __future__ import annotations

from .analyzer import AnalysisResult, analyze, analyze_graph
from .rules import RULES, RuleThresholds
from .schema import (
    DIMENSION_LABELS,
    Dimension,
    Evidence,
    Problem,
    Severity,
    Verification,
    load_problems,
    save_problems,
)
from .view import GraphView

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "analyze_graph",
    "AnalysisResult",
    "GraphView",
    "RULES",
    "RuleThresholds",
    "Problem",
    "Evidence",
    "Verification",
    "Dimension",
    "Severity",
    "DIMENSION_LABELS",
    "save_problems",
    "load_problems",
]
