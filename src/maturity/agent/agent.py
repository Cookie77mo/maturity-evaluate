"""Agent 语义研判层：子图裁剪 -> Rubric 驱动 -> 工具化查询 -> 证据校验 -> Problem。

与规则层的关系：

  规则层（rules.py）负责确定性、可复现的图算法判定，置信度由代码固定；
  Agent 层负责规则判不了的语义问题（文档与实现是否一致、两个名字不同的
  实体是否同一对象、探针配置是否合理……）。

  两者产出同一种 Problem，靠 ``detection_method`` 区分（rule / agent），
  最终由 analyzer 合并去重。

当前状态：骨架。Rubric 需要确认成熟度指标定义后填充；没有 Rubric 的指标
一律跳过，不会编造问题。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from ..schema import Evidence, Problem
from ..view import GraphView
from .context import ContextBuilder, SubgraphContext
from .evidence import CitationResult, CitationValidator
from .rubric import Rubric, RubricLibrary
from .tools import GraphTools, ToolSpec

# 单轮最多执行几个工具调用，防止模型一次请求几十个查询
MAX_TOOL_CALLS = 3

SYSTEM_PROMPT = """你是微服务成熟度评审员。你的唯一任务是：根据给出的知识图谱子图和评判标准，判断是否构成成熟度问题。

硬性规则：
1. 只能引用上下文中真实出现过的 id，禁止编造任何 id 或证据。
2. 每条结论必须至少给出一个 node_ids 或 edges，否则不要输出这条结论。
3. 证据不足时返回 {"findings": []}，不要猜测。
4. 只输出 JSON 对象，不要输出解释性文字、Markdown 代码块或额外说明。

输出格式（二选一）：

需要更多信息时：
{"tool_calls": [{"name": "工具名", "args": {...}}]}

给出结论时：
{"findings": [{"statement": "一句话说明问题", "node_ids": ["真实节点id"], "edges": [{"source": "id", "target": "id", "relation": "CALLS"}], "confidence": 0.0}]}
"""


@dataclass
class AgentConfig:
    """Agent 运行参数。

    hops            : 子图裁剪的跳数
    max_nodes       : 裁剪后的节点上限
    max_quotes      : 每个节点最多渲染几条 provenance
    max_tool_rounds : 最多允许几轮工具查询（每轮最多 MAX_TOOL_CALLS 个调用）
    max_confidence  : Agent 结论的置信度上限。语义推断不应高于确定性规则，
                      否则去重时会盖掉规则层更可靠的结论。
    max_entities    : 单条问题最多列几个实体
    """

    hops: int = 2
    max_nodes: int = 40
    max_quotes: int = 2
    max_tool_rounds: int = 2
    max_confidence: float = 0.85
    max_entities: int = 10


def _clamp_confidence(value: Any, fallback: float = 0.7) -> float:
    """把模型返回的置信度收敛到 [0, 1]，非法值退回 fallback。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, number))


def build_user_prompt(
    rubric: Rubric,
    context: SubgraphContext,
    coverage: dict[str, Any],
    observations: list[dict[str, Any]],
    specs: list[ToolSpec],
    max_tool_rounds: int,
) -> str:
    """拼装单次研判的 user prompt。"""
    tool_lines = [
        f"- {spec.name}: {spec.description} 参数: {json.dumps(spec.args, ensure_ascii=False)}"
        for spec in specs
    ]
    parts = [
        "## 指标",
        f"indicator: {rubric.indicator}",
        f"维度: {rubric.dimension}",
        f"要回答的判断: {rubric.question}",
        f"合格（不报问题）: {rubric.pass_criteria}",
        f"不合格（报问题）: {rubric.fail_criteria}",
        "",
        "## 图谱数据覆盖",
        json.dumps(coverage, ensure_ascii=False),
        "",
        "## 可用工具",
        *tool_lines,
        f"最多可以查 {max_tool_rounds} 轮；不需要查就直接给 findings。",
    ]
    if observations:
        parts.extend(["", "## 已执行的工具结果"])
        for item in observations:
            parts.append(json.dumps(item, ensure_ascii=False))
    parts.extend(["", context.text])
    return "\n".join(parts)


class SemanticAgent:
    """单个指标的语义研判器。

    一次 :meth:`judge` 只处理一个 indicator。多个指标请分别调用 ——
    一个指标一个 Rubric，混在一起会让模型在"该按哪条标准判"上摇摆。
    """

    def __init__(
        self,
        view: GraphView,
        *,
        llm: Any = None,
        rubrics: Optional[RubricLibrary] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.view = view
        self.llm = llm
        self.rubrics = rubrics or RubricLibrary.empty()
        self.config = config or AgentConfig()
        self.tools = GraphTools(view)
        self.builder = ContextBuilder(
            view,
            hops=self.config.hops,
            max_nodes=self.config.max_nodes,
            max_quotes=self.config.max_quotes,
        )

    # ------------------------------------------------------------------ 主流程
    def judge(self, indicator: str, seed_ids: list[str]) -> list[Problem]:
        """研判一个指标。没有 Rubric 或没有有效种子时返回空列表。"""
        rubric = self.rubrics.get(indicator)
        if rubric is None:
            return []
        seeds = [node_id for node_id in dict.fromkeys(seed_ids) if self.view.node(node_id)]
        if not seeds:
            return []

        context = self.builder.around(seeds)
        # 收紧到"模型真正看到过的节点和边"，超出上下文的一律视为幻觉
        validator = CitationValidator(
            self.view,
            allowed_nodes=set(context.node_ids),
            allowed_edges=set(context.edges),
        )

        problems: list[Problem] = []
        for finding in self._ask(rubric, context):
            citation = validator.check(
                finding.get("node_ids") or (),
                finding.get("edges") or (),
            )
            if not citation.ok:
                # 没有任何有效证据 -> 丢弃，不产出"空口"问题
                continue
            problems.append(self._to_problem(rubric, finding, citation))
        return problems

    # ------------------------------------------------------------------ 调用模型
    def _ask(self, rubric: Rubric, context: SubgraphContext) -> list[dict[str, Any]]:
        if self.llm is None:
            raise RuntimeError(
                "SemanticAgent 需要 LLM 客户端；请用 build_llm_client() 构造后传入。"
            )

        coverage = self.view.coverage()
        specs = self.tools.specs()
        observations: list[dict[str, Any]] = []
        payload: dict[str, Any] = {}

        for round_no in range(self.config.max_tool_rounds + 1):
            prompt = build_user_prompt(
                rubric,
                context,
                coverage,
                observations,
                specs,
                self.config.max_tool_rounds,
            )
            payload = self.llm.complete_json(SYSTEM_PROMPT, prompt) or {}
            calls = payload.get("tool_calls") or []
            if not calls or round_no >= self.config.max_tool_rounds:
                break
            observations.extend(self._run_tools(calls))

        findings = payload.get("findings") or []
        return [item for item in findings if isinstance(item, dict)]

    def _run_tools(self, calls: list[Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in calls[:MAX_TOOL_CALLS]:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            args = call.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            results.append(
                {
                    "tool": name,
                    "args": args,
                    "result": self.tools.call(name, args),
                }
            )
        return results

    # ------------------------------------------------------------------ 产出
    def _to_problem(
        self,
        rubric: Rubric,
        finding: dict[str, Any],
        citation: CitationResult,
    ) -> Problem:
        entities = list(dict.fromkeys(citation.nodes))[: self.config.max_entities]
        confidence = min(
            _clamp_confidence(finding.get("confidence")),
            self.config.max_confidence,
        )
        statement = str(finding.get("statement") or "").strip() or rubric.fail_criteria
        return Problem(
            dimension=rubric.dimension,
            indicator=rubric.indicator,
            severity=rubric.severity,
            title=rubric.title,
            statement=statement,
            entities=entities,
            evidence=Evidence(
                nodes=entities,
                edges=citation.edges,
                paths=[],
                quotes=self.view.node_quotes(entities, limit=self.config.max_quotes * 2),
            ),
            detection_method="agent",
            confidence=confidence,
            suggestion=rubric.suggestion,
            fix_targets=self.view.file_targets(entities),
        )


def build_llm_client(config_file: Optional[str] = None) -> Any:
    """复用模块一的 ``llm`` 配置段构造 LLM 客户端。

    只读 ``kg_builder.config``，不修改模块一的任何代码：配置文件格式沿用
    现有的 ``llm:`` 段（model / base_url / api_key ...）。
    """
    from kg_builder.config import LLMSettings, load_config_file
    from kg_builder.enrichment.llm_client import create_llm_client

    data = load_config_file(config_file) if config_file else {}
    raw = data.get("llm") if isinstance(data, dict) else None
    settings = LLMSettings(**raw) if isinstance(raw, dict) else LLMSettings()
    return create_llm_client(settings)
