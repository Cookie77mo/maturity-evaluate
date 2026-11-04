"""知识图谱只读视图：给规则层提供稳定、语义化的访问接口。

不修改 ``KnowledgeGraph``，只做查询与派生（服务子图、证据摘录、文件定位）。
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

import networkx as nx

from kg_builder.graph.knowledge_graph import KnowledgeGraph
from kg_builder.graph.model import KGEdge, KGNode

SERVICE_RELATIONS = ("CALLS", "DEPENDS_ON")

# provenance.source_id 里只有这些后缀才视为"可修复的文件"
_FILE_SUFFIXES = (
    ".py", ".pyi", ".java", ".go", ".js", ".jsx", ".ts", ".tsx", ".rb", ".php",
    ".cs", ".kt", ".scala", ".c", ".cpp", ".rs",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".properties", ".md", ".markdown",
    ".txt", ".xml", ".sql", ".dockerfile",
)


def _strip_line(source_id: str) -> str:
    """把 ``path.py:12`` 变成 ``path.py``；其它形式原样返回。"""
    return re.sub(r":\d+(?:-\d+)?$", "", source_id)


class GraphView:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        self._nodes: dict[str, KGNode] = {n.id: n for n in graph.nodes}

    # ------------------------------------------------------------------ 基础
    @property
    def nodes(self) -> list[KGNode]:
        return list(self._nodes.values())

    def node(self, node_id: str) -> Optional[KGNode]:
        return self._nodes.get(node_id)

    def find(self, node_type: str) -> list[KGNode]:
        return [n for n in self._nodes.values() if n.type == node_type]

    def services(self) -> list[KGNode]:
        return self.find("Service")

    def out_edges(self, node_id: str, relations: Optional[Iterable[str]] = None) -> list[KGEdge]:
        rels = set(relations) if relations else None
        return [e for e in self.graph.edges if e.source == node_id and (rels is None or e.relation in rels)]

    def in_edges(self, node_id: str, relations: Optional[Iterable[str]] = None) -> list[KGEdge]:
        rels = set(relations) if relations else None
        return [e for e in self.graph.edges if e.target == node_id and (rels is None or e.relation in rels)]

    # ------------------------------------------------------------------ 语义
    def semantic(self, node: KGNode) -> dict[str, Any]:
        value = node.attributes.get("semantic")
        return value if isinstance(value, dict) else {}

    def is_core(self, node: KGNode) -> bool:
        if bool(self.semantic(node).get("is_core_business")):
            return True
        return bool(node.attributes.get("is_core_business"))

    # -------------------------------------------------------------- 服务子图
    def service_graph(self, relations: Iterable[str] = SERVICE_RELATIONS) -> nx.DiGraph:
        """只含 Service→Service 边的有向图，供图算法使用。"""
        rels = set(relations)
        graph = nx.DiGraph()
        for service in self.services():
            graph.add_node(service.id, name=service.name)
        for edge in self.graph.edges:
            if edge.relation not in rels:
                continue
            if edge.source not in graph or edge.target not in graph:
                continue
            graph.add_edge(edge.source, edge.target, relation=edge.relation)
        return graph

    # ------------------------------------------------------------------ 证据
    def quotes(self, provs: Iterable[Any], limit: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for prov in list(provs)[:limit]:
            out.append({
                "source_type": prov.source_type,
                "source_id": prov.source_id,
                "evidence": prov.evidence,
                "confidence": round(float(prov.confidence), 4),
            })
        return out

    def node_quotes(self, node_ids: Iterable[str], limit: int = 8) -> list[dict[str, Any]]:
        quotes: list[dict[str, Any]] = []
        for node_id in node_ids:
            node = self.node(node_id)
            if node is not None:
                quotes.extend(self.quotes(node.provenance, limit=3))
                if len(quotes) >= limit:
                    return quotes[:limit]
        return quotes

    def edge_ref(self, edge: KGEdge) -> dict[str, str]:
        return {"source": edge.source, "target": edge.target, "relation": edge.relation}

    # -------------------------------------------------------------- 修复目标
    def file_targets(self, node_ids: Iterable[str]) -> list[str]:
        """从 provenance 中提取真实文件路径，作为模块三的修复锚点。"""
        found: set[str] = set()
        for node_id in node_ids:
            node = self.node(node_id)
            if node is None:
                continue
            for prov in node.provenance:
                source_id = _strip_line(prov.source_id or "")
                low = source_id.lower()
                if "://" in source_id:
                    continue
                if not low.endswith(_FILE_SUFFIXES):
                    continue
                found.add(source_id)
        return sorted(found)

    # ------------------------------------------------------------------ 覆盖
    def coverage(self) -> dict[str, Any]:
        stats = self.graph.stats()
        by_type = stats.get("nodes_by_type", {})
        return {
            "node_count": stats.get("node_count", 0),
            "edge_count": stats.get("edge_count", 0),
            "nodes_by_type": by_type,
            "has_service": by_type.get("Service", 0) > 0,
            "has_deployment": by_type.get("Deployment", 0) > 0 or by_type.get("Pod", 0) > 0,
            "has_runtime": by_type.get("Metric", 0) > 0,
            "has_history": "history" in stats.get("provenance_by_source", {}),
        }
