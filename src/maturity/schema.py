"""
模块二数据契约：问题（Problem）与证据（Evidence）。

为什么单独抽一个 schema.py？
  模块二唯一对外产物是「问题清单」，模块三按 ``fix_targets`` 定位修复目标。
  所有结构都必须可直接序列化为 JSON，便于 JSONL 落盘与跨模块传递。
  把数据结构集中在这里，规则层只负责"判"，不操心"怎么存/怎么传"。

核心设计：
  - Problem 是一条成熟度问题的完整描述（维度、指标、严重度、证据、修复建议）
  - Evidence 是证据链，直接由知识图谱的 provenance 派生
  - Verification 是验证状态机，预留给故障注入层（静态推断 -> 动态证实/推翻）
  - problem_id 用 SHA1(indicator + sorted(entities)) 生成，保证同一问题可去重
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class Dimension(str, Enum):
    """成熟度维度。

    对齐 ISO/IEC 25010 质量特性的子集，是问题分类的顶层标签。
    后续要加维度（如性能、兼容性）只在此枚举里加一行，指标库按维度归类。
    继承 str 是为了序列化时直接是字符串，不用额外转换。
    """

    RELIABILITY = "reliability"
    OBSERVABILITY = "observability"
    MAINTAINABILITY = "maintainability"
    CONSISTENCY = "consistency"
    CONFIGURABILITY = "configurability"
    SECURITY = "security"


# 维度英文 -> 中文标签，报告里显示用
DIMENSION_LABELS: dict[str, str] = {
    Dimension.RELIABILITY.value: "可靠性",
    Dimension.OBSERVABILITY.value: "可观测性",
    Dimension.MAINTAINABILITY.value: "可维护性",
    Dimension.CONSISTENCY.value: "一致性",
    Dimension.CONFIGURABILITY.value: "可演进/配置",
    Dimension.SECURITY.value: "安全性",
}


class Severity(str, Enum):
    """严重度三档。

    high   : 必须修，直接影响可用性/数据安全
    medium : 建议修，影响可维护性/一致性
    low    : 可选修，多为信息性提示或采集缺口

    rank 属性用于排序和过滤（high=3 > medium=2 > low=1）。
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"low": 1, "medium": 2, "high": 3}[self.value]


def make_problem_id(indicator: str, entities: Iterable[str]) -> str:
    """生成确定性的问题 ID。

    设计意图：同一指标 + 同一实体集合 -> 同一 ID。
    这样多次分析同一图谱（或多条规则命中同一问题）时，analyzer 可以按 ID 去重，
    而不会产出重复的问题条目。
    取 SHA1 前 12 位，够唯一又不长。
    """
    base = indicator + "|" + "|".join(sorted(str(e) for e in entities))
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


@dataclass
class Evidence:
    """问题的证据链，直接由知识图谱的 provenance 派生。

    四个字段从不同角度支撑"为什么判定这是个问题"：
      nodes  : 问题涉及哪些节点（服务/配置/文档的 id）
      edges  : 问题涉及哪些边（{source, target, relation}）
      paths  : 关键路径（调用链、依赖环的节点序列）
      quotes : provenance 原文摘录（代码片段、文档段落、置信度）
    模块三或人工复核时，看 evidence 就能追溯判定依据。
    """

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
    """问题的验证状态机。

    当前规则层产出的问题都是 unverified（静态推断）。
    故障注入层（待实现）会主动注入故障，把问题升级为 confirmed（证实）或 refuted（推翻），
    并把观测结果写入 observed。这是"静态分析 -> 动态验证"闭环的关键字段。
    """

    status: str = "unverified"   # unverified | confirmed | refuted
    method: str = ""             # 计划或已执行的实验（如 pod-kill）
    observed: str = ""           # 实验观测到的事实

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "method": self.method, "observed": self.observed}


@dataclass
class Problem:
    """一条成熟度问题的完整描述。

    字段说明：
      dimension        : 所属维度（reliability/maintainability/...）
      indicator        : 指标 ID（single_replica / single_point_of_failure / ...）
      severity         : 严重度（high/medium/low）
      title            : 一句话标题（报告里直接显示）
      statement        : 详细陈述（为什么是问题、影响）
      entities         : 涉及的节点 id 列表
      evidence         : 证据链
      detection_method : 检测方式（rule=规则 / agent=LLM / injection=注入）
      confidence       : 置信度 [0,1]，来自规则硬编码或 provenance
      suggestion       : 修复建议（自然语言）
      fix_targets      : 修复目标文件路径列表（模块三据此定位）
      verification     : 验证状态
      id               : 确定性 ID（去重用）
    """
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
        # 允许规则层传入 Dimension / Severity 枚举对象，这里统一转成字符串，
        # 保证序列化（to_dict）时输出的是纯字符串，下游不用处理枚举类型
        if isinstance(self.dimension, Dimension):
            self.dimension = self.dimension.value
        if isinstance(self.severity, Severity):
            self.severity = self.severity.value
        # 置信度必须在 [0,1]，越界直接抛错（防止规则层写错）
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence 必须位于 [0,1]：{self.confidence}")
        # 如果调用方没传 id，自动用 indicator+entities 生成确定性 id
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
    """把问题清单写成 JSONL 文件。

    为什么用 JSONL 而不是 JSON 数组？
      - 每行一个问题，模块三可以逐行流式读取，不用一次性 load 整个大文件
      - 方便增量追加（后续 Agent 层、注入层产出的问题可以直接 append）
      - grep/awk 等命令行工具能直接处理
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for problem in problems:
            fh.write(json.dumps(problem.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_problems(path: str | Path) -> list[dict[str, Any]]:
    """读回 JSONL，返回 dict 列表（不是 Problem 对象）。

    故意不还原成 Problem 实例：下游（报告生成、回填图谱）只需要 dict，
    还原成 dataclass 反而增加开销。如果要做校验再单独处理。
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
