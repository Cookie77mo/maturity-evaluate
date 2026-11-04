"""模块二规则层单元测试。"""
from __future__ import annotations

from kg_builder.graph.knowledge_graph import KnowledgeGraph
from kg_builder.graph.model import Provenance

from maturity.analyzer import analyze_graph


def _prov(source_type: str, source_id: str, evidence: str = "") -> Provenance:
    return Provenance(source_type=source_type, source_id=source_id, evidence=evidence, confidence=0.9)


def _service(graph: KnowledgeGraph, name: str, *, core: bool = False):
    attributes = {"semantic": {"is_core_business": core}} if core else {}
    return graph.upsert_node("Service", name, attributes=attributes, provenance=[_prov("source_code", f"src/{name}.py")])


def _indicators(result) -> set[str]:
    return {problem.indicator for problem in result.problems}


def test_empty_graph_reports_only_runtime_coverage_gap():
    result = analyze_graph(KnowledgeGraph())
    assert _indicators(result) == {"runtime_coverage_missing"}
    assert result.problems[0].dimension == "observability"
    assert result.problems[0].entities == []


def test_dependency_cycle_detected():
    graph = KnowledgeGraph()
    a, b, c = (_service(graph, name) for name in ("A", "B", "C"))
    for src, dst in ((a, b), (b, c), (c, a)):
        graph.upsert_edge(src.id, dst.id, "CALLS", provenance=[_prov("source_code", "src/calls.py")])
    result = analyze_graph(graph)
    cycle = [p for p in result.problems if p.indicator == "dependency_cycle"]
    assert cycle, "应检测到服务依赖环"
    assert cycle[0].evidence.paths


def test_articulation_point_is_single_point_of_failure():
    graph = KnowledgeGraph()
    a, b, c = (_service(graph, name) for name in ("A", "B", "C"))
    graph.upsert_edge(a.id, b.id, "CALLS", provenance=[_prov("source_code", "src/calls.py")])
    graph.upsert_edge(b.id, c.id, "CALLS", provenance=[_prov("source_code", "src/calls.py")])
    result = analyze_graph(graph)
    spof = [p for p in result.problems if p.indicator == "single_point_of_failure"]
    assert len(spof) == 1
    assert spof[0].entities == [b.id]


def test_doc_only_service_is_inconsistency():
    graph = KnowledgeGraph()
    doc = graph.upsert_node("Document", "architecture.md", provenance=[_prov("document", "docs/architecture.md")])
    ghost = graph.upsert_node("Service", "GhostService", provenance=[_prov("document", "docs/architecture.md")])
    graph.upsert_edge(doc.id, ghost.id, "DEPENDS_ON", provenance=[_prov("document", "docs/architecture.md")])
    result = analyze_graph(graph)
    mismatch = [p for p in result.problems if p.indicator == "doc_code_mismatch"]
    assert mismatch and mismatch[0].evidence.quotes


def test_single_replica_is_high_for_core_service():
    graph = KnowledgeGraph()
    service = _service(graph, "PaymentService", core=True)
    deployment = graph.upsert_node(
        "Deployment", "payment", attributes={"replicas": 1},
        provenance=[_prov("deployment", "prod/default/deployment/payment")],
    )
    graph.upsert_edge(service.id, deployment.id, "DEPLOYED_AS",
                      attributes={"replicas": 1}, provenance=[_prov("deployment", "prod/default/deployment/payment")])
    result = analyze_graph(graph)
    replica = [p for p in result.problems if p.indicator == "single_replica"]
    assert replica and replica[0].severity == "high"
    assert service.id in replica[0].entities


def test_config_value_conflict_detected():
    graph = KnowledgeGraph()
    config = graph.upsert_node(
        "Config", "payment.timeout",
        provenance=[
            _prov("document", "config/a.yaml", "payment.timeout=30"),
            _prov("document", "config/b.yaml", "payment.timeout=60"),
        ],
    )
    result = analyze_graph(graph)
    conflict = [p for p in result.problems if p.indicator == "config_value_conflict"]
    assert conflict and "30" in conflict[0].statement and "60" in conflict[0].statement


def test_min_severity_filters_problems():
    graph = KnowledgeGraph()
    _service(graph, "LonelyService")
    result = analyze_graph(graph, min_severity="high")
    assert result.problems == []
