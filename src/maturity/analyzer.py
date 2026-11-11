"""
分析编排：载入图谱 -> 跑全部规则 -> 去重排序 -> 输出问题清单。

analyzer 是模块二的"调度中心"，本身不包含任何判定逻辑，只负责：
  1. 把模块一的 KG 包成 GraphView
  2. 依次执行 RULES 列表里的每条规则，收集所有 Problem
  3. 按 problem_id 去重（同一问题被多条规则命中时留置信度高的）
  4. 按 min_severity 过滤，按严重度+置信度排序
  5. 写成 JSONL 文件
规则本身的判定逻辑全部在 rules.py，analyzer 不关心。
"""
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
    """分析结果容器。

    problems : 去重排序后的问题列表
    coverage : 图谱数据覆盖情况（用于判断哪些规则因缺数据被跳过）
    by_severity() / by_dimension() 供报告生成和终端摘要使用。
    """
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
        """生成终端可读的摘要文本。

        包含：问题总数、按严重度分布、按维度分布、数据覆盖提示。
        top>0 时额外列出前 N 条问题的一行摘要。
        """
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


#输入内存里的 KG 对象，由单元测试、已经持有图谱的调用方调用
def analyze_graph(
    graph: KnowledgeGraph,
    thresholds: Optional[RuleThresholds] = None,
    min_severity: str = "low",
) -> AnalysisResult:
    """对一张知识图谱执行全部规则，返回去重排序后的问题列表。

    这是模块二的核心编排函数，分三步：
      1) 跑规则：遍历 RULES 列表，每条规则独立执行，收集所有 Problem
      2) 去重：按 problem.id（SHA1(indicator+entities)）去重，
         同一 ID 被多条规则命中时保留 confidence 更高的那条
      3) 过滤+排序：按 min_severity 过滤，
         排序键 = (严重度降序, 置信度降序, 维度, 指标ID)
    """
    view = GraphView(graph)
    cfg = thresholds or RuleThresholds()

    # 用 dict 存 {problem_id: Problem}，天然去重
    problems: dict[str, Problem] = {}
    for rule in RULES:
        for problem in rule(view, cfg):
            existing = problems.get(problem.id)
            # 同一问题被多条规则命中时保留置信度更高的一条
            if existing is None or problem.confidence > existing.confidence:
                problems[problem.id] = problem

    # min_severity 过滤：只保留 rank >= floor 的问题
    floor = Severity(min_severity).rank
    kept = [p for p in problems.values() if Severity(p.severity).rank >= floor]
    # 排序：严重度高的在前，同严重度按置信度，再按维度和指标ID（保证输出稳定）
    kept.sort(key=lambda p: (-Severity(p.severity).rank, -p.confidence, p.dimension, p.indicator))
    return AnalysisResult(problems=kept, coverage=view.coverage())

#输入文件路径，载入 kg.json → 调 analyze_graph → 写 problems.jsonl
def analyze(
    graph_path: str | Path,
    out_path: str | Path = "out/problems.jsonl",
    thresholds: Optional[RuleThresholds] = None,
    min_severity: str = "low",
) -> AnalysisResult:
    """对外入口：载入模块一的 kg.json，分析后写出 problems.jsonl。

    CLI 和未来的 Agent/注入层都通过这个函数触发分析。
    graph_path 是模块一 kgb build 的产出物。
    """
    graph = load_graph(graph_path)
    result = analyze_graph(graph, thresholds=thresholds, min_severity=min_severity)
    save_problems(result.problems, out_path)
    return result
