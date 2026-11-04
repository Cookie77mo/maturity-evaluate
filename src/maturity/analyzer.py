"""分析编排：载入图谱 → 跑全部规则 → 去重排序 → 输出问题清单。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from kg_builder.graph.knowledge_graph import KnowledgeGraph
from kg_builder.graph.serialization import load_graph

from .rules import RULES, RuleThresholds
from .schema import Problem, Severity, save_problems
from .view import GraphView


@dataclass
class AnalysisResult:
    problems: list[Problem] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    def by_severity(self) -> dict[str, int]:
        counts = {"high": 0, "medium": 0, "low": 0}
        for problem in self.problems:
            counts[problem.severity] = counts.get(problem.severity, 0) + 1
        return counts

    def by_dimension(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for problem in self.problems:
            counts[problem.dimension] = counts.get(problem.dimension, 0) + 1
        return counts

    def summary_text(self, top: int = 0) -> str:
        severity = self.by_severity()
        dimension = self.by_dimension()
        lines = [
            f"问题总数: {len(self.problems)}  "
            f"(high={severity['high']}, medium={severity['medium']}, low={severity['low']})",
            "按维度: " + (", ".join(f"{k}={v}" for k, v in sorted(dimension.items())) or "无"),
        ]
        if not self.coverage.get("has_deployment"):
            lines.append("提示: 图谱中没有部署数据，副本/探针类规则未生效。")
        if not self.coverage.get("has_runtime"):
            lines.append("提示: 图谱中没有运行时指标，可观测性仅能报告采集缺口。")
        if top:
            lines.append("")
            for problem in self.problems[:top]:
                lines.append(
                    f"[{problem.severity:6}] {problem.indicator}: {problem.title} "
                    f"({problem.id})"
                )
        return "\n".join(lines)


def analyze_graph(
    graph: KnowledgeGraph,
    thresholds: Optional[RuleThresholds] = None,
    min_severity: str = "low",
) -> AnalysisResult:
    view = GraphView(graph)
    cfg = thresholds or RuleThresholds()

    problems: dict[str, Problem] = {}
    for rule in RULES:
        for problem in rule(view, cfg):
            existing = problems.get(problem.id)
            # 同一问题被多条规则命中时保留置信度更高的一条
            if existing is None or problem.confidence > existing.confidence:
                problems[problem.id] = problem

    floor = Severity(min_severity).rank
    kept = [p for p in problems.values() if Severity(p.severity).rank >= floor]
    kept.sort(key=lambda p: (-Severity(p.severity).rank, -p.confidence, p.dimension, p.indicator))
    return AnalysisResult(problems=kept, coverage=view.coverage())


def analyze(
    graph_path: str | Path,
    out_path: str | Path = "out/problems.jsonl",
    thresholds: Optional[RuleThresholds] = None,
    min_severity: str = "low",
) -> AnalysisResult:
    """载入模块一产出的 kg.json，分析并写出 problems.jsonl。"""
    graph = load_graph(graph_path)
    result = analyze_graph(graph, thresholds=thresholds, min_severity=min_severity)
    save_problems(result.problems, out_path)
    return result
