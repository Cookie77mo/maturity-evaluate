"""模块二数据契约：问题（Problem）与证据（Evidence）。

模块二唯一对外产物是「问题清单」，模块三按 ``fix_targets`` 定位修复目标。
所有结构都必须可直接序列化为 JSON，便于 JSONL 落盘与跨模块传递。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class Dimension(str, Enum):
    """成熟度维度。后续可在此扩展，指标库按维度归类。"""

    RELIABILITY = "reliability"
    OBSERVABILITY = "observability"
    MAINTAINABILITY = "maintainability"
    CONSISTENCY = "consistency"
    CONFIGURABILITY = "configurability"
    SECURITY = "security"


DIMENSION_LABELS: dict[str, str] = {
    Dimension.RELIABILITY.value: "可靠性",
    Dimension.OBSERVABILITY.value: "可观测性",
    Dimension.MAINTAINABILITY.value: "可维护性",
    Dimension.CONSISTENCY.value: "一致性",
    Dimension.CONFIGURABILITY.value: "可演进/配置",
    Dimension.SECURITY.value: "安全性",
}


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"low": 1, "medium": 2, "high": 3}[self.value]


def make_problem_id(indicator: str, entities: Iterable[str]) -> str:
    """确定性 ID：同一指标 + 同一实体集合 → 同一 ID，重复分析可去重。"""
    base = indicator + "|" + "|".join(sorted(str(e) for e in entities))
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


@dataclass
class Evidence:
    """问题的证据链，直接由知识图谱的 provenance 派生。"""

    nodes: list[str] = field(default_factory=list)              # 关联节点 id
    edges: list[dict[str, str]] = field(default_factory=list)    # {source,target,relation}
    paths: list[list[str]] = field(default_factory=list)         # 节点 id 路径（调用链/环）
    quotes: list[dict[str, Any]] = field(default_factory=list)   # provenance 原文摘录

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "paths": self.paths,
            "quotes": self.quotes,
        }


@dataclass
class Verification:
    """问题的验证状态：静态推断 → 动态实验证实/推翻。"""

    status: str = "unverified"   # unverified | confirmed | refuted
    method: str = ""             # 计划或已执行的实验（如 pod-kill）
    observed: str = ""           # 实验观测到的事实

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "method": self.method, "observed": self.observed}


@dataclass
class Problem:
    dimension: str
    indicator: str
    severity: str
    title: str
    statement: str
    entities: list[str] = field(default_factory=list)
    evidence: Evidence = field(default_factory=Evidence)
    detection_method: str = "rule"      # rule | agent | injection
    confidence: float = 0.9
    suggestion: str = ""
    fix_targets: list[str] = field(default_factory=list)
    verification: Verification = field(default_factory=Verification)
    id: str = ""

    def __post_init__(self) -> None:
        # 允许传入 Dimension / Severity 枚举，序列化时统一成字符串。
        if isinstance(self.dimension, Dimension):
            self.dimension = self.dimension.value
        if isinstance(self.severity, Severity):
            self.severity = self.severity.value
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence 必须位于 [0,1]：{self.confidence}")
        if not self.id:
            self.id = make_problem_id(self.indicator, self.entities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.id,
            "dimension": self.dimension,
            "dimension_label": DIMENSION_LABELS.get(self.dimension, self.dimension),
            "indicator": self.indicator,
            "severity": self.severity,
            "title": self.title,
            "statement": self.statement,
            "entities": self.entities,
            "evidence": self.evidence.to_dict(),
            "detection": {"method": self.detection_method, "confidence": round(self.confidence, 4)},
            "suggestion": self.suggestion,
            "fix_targets": self.fix_targets,
            "verification": self.verification.to_dict(),
        }


def save_problems(problems: Iterable[Problem], path: str | Path) -> Path:
    """写成 JSONL：每行一个问题，方便模块三逐条消费。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for problem in problems:
            fh.write(json.dumps(problem.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_problems(path: str | Path) -> list[dict[str, Any]]:
    """读回 JSONL（保持 dict 形态，供报告/回填等下游使用）。"""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
