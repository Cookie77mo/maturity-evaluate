"""Agent 层基础设施单元测试。

全部使用假 LLM，不产生任何真实模型调用，因此可以离线运行。
"""
from __future__ import annotations

from kg_builder.graph.knowledge_graph import KnowledgeGraph
from kg_builder.graph.model import Provenance

from maturity.agent import (
    AgentConfig,
    CitationValidator,
    ContextBuilder,
    GraphTools,
    Rubric,
    RubricLibrary,
    SemanticAgent,
)
from maturity.view import GraphView


def _prov(source_type: str = "source_code", source_id: str = "src/x.py") -> Provenance:
    return Provenance(source_type=source_type, source_id=source_id, confidence=0.9)


def _graph():
    """A --CALLS--> B --CALLS--> C，其中 C 只有文档来源（模拟"文档有、代码无"）。"""
    graph = KnowledgeGraph()
    a = graph.upsert_node("Service", "order-service", provenance=[_prov(source_id="src/order.py")])
    b = graph.upsert_node("Service", "inventory-service", provenance=[_prov(source_id="src/inv.py")])
    c = graph.upsert_node(
        "Service", "pay-service",
        provenance=[_prov(source_type="document", source_id="docs/arch.md")],
    )
    graph.upsert_edge(a.id, b.id, "CALLS", provenance=[_prov()])
    graph.upsert_edge(b.id, c.id, "CALLS", provenance=[_prov()])
    return graph, a, b, c


def _rubric() -> Rubric:
    return Rubric(
        indicator="doc_code_mismatch",
        dimension="consistency",
        title="文档描述的服务缺少代码实现",
        question="文档描述的服务是否能在代码侧找到证据？",
        pass_criteria="存在 source_code 来源的 provenance",
        fail_criteria="只有文档来源的 provenance",
        severity="medium",
        suggestion="核对文档与实现",
    )


class FakeLLM:
    """按预设顺序返回 payload 的假模型，并记录收到过的 prompt。"""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    @property
    def available(self) -> bool:
        return True

    def complete_json(self, system: str, user: str):
        self.calls.append({"system": system, "user": user})
        return self._payloads.pop(0)


# ------------------------------------------------------------------ 子图裁剪
def test_context_keeps_seed_and_one_hop_neighbors():
    graph, a, b, c = _graph()
    context = ContextBuilder(GraphView(graph), hops=1, max_nodes=10).around([a.id])
    assert context.seeds == [a.id]
    assert a.id in context.node_ids
    assert b.id in context.node_ids
    assert c.id not in context.node_ids, "2 跳外的节点不应进入 1 跳上下文"


def test_context_respects_max_nodes_and_marks_truncated():
    graph, a, _b, _c = _graph()
    context = ContextBuilder(GraphView(graph), hops=2, max_nodes=2).around([a.id])
    assert len(context.node_ids) == 2
    assert context.truncated is True


def test_context_never_renders_dangling_edges():
    graph, a, _b, _c = _graph()
    # 只保留 1 个节点时，A--CALLS-->B 的 B 端被裁掉，这条边必须一起消失
    context = ContextBuilder(GraphView(graph), hops=2, max_nodes=1).around([a.id])
    assert context.edges == []
    assert a.id in context.text


def test_context_ignores_unknown_seeds():
    graph, _a, _b, _c = _graph()
    context = ContextBuilder(GraphView(graph)).around(["not-a-node"])
    assert context.node_ids == []
    assert context.seeds == []


# ------------------------------------------------------------------ 工具接口
def test_tools_report_unknown_tool_without_raising():
    graph, _a, _b, _c = _graph()
    result = GraphTools(GraphView(graph)).call("no_such_tool", {})
    assert result["ok"] is False
    assert "available" in result


def test_tools_find_nodes_by_name_fragment():
    graph, _a, b, _c = _graph()
    result = GraphTools(GraphView(graph)).call("find_nodes", {"name_contains": "inventory"})
    assert result["ok"] is True
    assert result["data"]["count"] == 1
    assert result["data"]["nodes"][0]["id"] == b.id


def test_tools_find_paths_between_services():
    graph, a, b, c = _graph()
    result = GraphTools(GraphView(graph)).call(
        "paths_between", {"source": a.id, "target": c.id, "max_hops": 4}
    )
    assert result["ok"] is True
    assert result["data"]["paths"] == [[a.id, b.id, c.id]]


def test_tools_accept_no_arguments():
    graph, _a, _b, _c = _graph()
    result = GraphTools(GraphView(graph)).call("coverage", {})
    assert result["ok"] is True
    assert result["data"]["has_service"] is True


# ------------------------------------------------------------------ 证据校验
def test_citation_drops_hallucinated_node():
    graph, a, b, _c = _graph()
    validator = CitationValidator(GraphView(graph), allowed_nodes={a.id, b.id})
    result = validator.check(["ghost-id", a.id], [])
    assert result.nodes == [a.id]
    assert "ghost-id" in result.invalid
    assert result.ok is True


def test_citation_rejects_edge_outside_allowed_set():
    graph, a, b, _c = _graph()
    validator = CitationValidator(
        GraphView(graph),
        allowed_nodes={a.id},
        allowed_edges=set(),  # 上下文里一条边都没有
    )
    result = validator.check([], [{"source": a.id, "target": b.id, "relation": "CALLS"}])
    assert result.ok is False
    assert result.invalid


def test_citation_rejects_malformed_edge_reference():
    graph, _a, _b, _c = _graph()
    validator = CitationValidator(GraphView(graph))
    result = validator.check([], [{"source": "x"}])
    assert result.ok is False


# ------------------------------------------------------------------ 编排
def test_agent_emits_problem_with_valid_citation():
    graph, _a, _b, c = _graph()
    llm = FakeLLM([
        {
            "findings": [
                {
                    "statement": "文档描述了 pay-service，但代码侧没有对应实现",
                    "node_ids": [c.id],
                    "confidence": 0.99,
                }
            ]
        }
    ])
    agent = SemanticAgent(
        GraphView(graph),
        llm=llm,
        rubrics=RubricLibrary([_rubric()]),
    )
    problems = agent.judge("doc_code_mismatch", [c.id])

    assert len(problems) == 1
    problem = problems[0]
    assert problem.detection_method == "agent"
    assert c.id in problem.evidence.nodes
    assert problem.confidence <= agent.config.max_confidence, "Agent 置信度必须被封顶"
    assert problem.fix_targets, "应从 provenance 提取出修复目标文件"


def test_agent_drops_finding_without_valid_evidence():
    graph, _a, _b, c = _graph()
    llm = FakeLLM([{"findings": [{"statement": "无证据的猜测", "node_ids": ["ghost-id"], "confidence": 0.9}]}])
    agent = SemanticAgent(GraphView(graph), llm=llm, rubrics=RubricLibrary([_rubric()]))
    assert agent.judge("doc_code_mismatch", [c.id]) == []


def test_agent_skips_indicator_without_rubric():
    graph, a, _b, _c = _graph()
    agent = SemanticAgent(GraphView(graph), llm=FakeLLM([]), rubrics=RubricLibrary.empty())
    assert agent.judge("doc_code_mismatch", [a.id]) == []


def test_agent_runs_tool_round_before_findings():
    graph, _a, _b, c = _graph()
    llm = FakeLLM([
        {"tool_calls": [{"name": "get_node", "args": {"node_id": c.id}}]},
        {"findings": [{"statement": "证据充分", "node_ids": [c.id], "confidence": 0.7}]},
    ])
    agent = SemanticAgent(
        GraphView(graph),
        llm=llm,
        rubrics=RubricLibrary([_rubric()]),
        config=AgentConfig(max_tool_rounds=2),
    )
    problems = agent.judge("doc_code_mismatch", [c.id])

    assert len(llm.calls) == 2, "第一轮查工具，第二轮才出结论"
    assert "已执行的工具结果" in llm.calls[1]["user"]
    assert len(problems) == 1


def test_agent_stops_after_max_tool_rounds():
    graph, _a, _b, c = _graph()
    # 永远只要求查工具，不产出 findings
    loop_payload = {"tool_calls": [{"name": "coverage", "args": {}}]}
    llm = FakeLLM([loop_payload, loop_payload, loop_payload, loop_payload])
    agent = SemanticAgent(
        GraphView(graph),
        llm=llm,
        rubrics=RubricLibrary([_rubric()]),
        config=AgentConfig(max_tool_rounds=2),
    )
    assert agent.judge("doc_code_mismatch", [c.id]) == []
    assert len(llm.calls) == 3, "max_tool_rounds=2 -> 最多 3 次调用后必须收口"
