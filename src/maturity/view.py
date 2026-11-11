"""
知识图谱只读视图：给规则层提供稳定、语义化的访问接口。

为什么需要这一层？
  模块一的 KnowledgeGraph 是底层结构（nodes/edges/provenance 的原始 list），
  如果让 10 条规则各自去遍历原始结构，一旦模块一改内部字段名，所有规则都要跟着改。
  GraphView 把常用查询封装成方法，规则层只调用这些方法，实现解耦。

设计原则：只读，不修改 KnowledgeGraph；只做查询与派生（子图、证据、文件定位）。
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
    """对 KnowledgeGraph 的只读封装。

    内部用 _nodes 字典建 id->节点 的索引，O(1) 查找。
    所有方法都是"读"操作，不会改原始图谱。
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        # 建索引：id -> KGNode，避免每次 find 都遍历全部节点
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
        """抽取只含 Service->Service 的有向子图，供图算法使用。

        为什么要单独抽子图？
          networkx 的图算法（强连通分量、割点、最长路径）只认 networkx 的图对象，
          不认识模块一的 KGEdge。所以要把 Service 间的 CALLS/DEPENDS_ON 关系
          转成一个干净的 nx.DiGraph。

        为什么只取 CALLS/DEPENDS_ON？
          依赖环、单点故障、调用链长度这些指标，只关心服务间的调用/依赖拓扑，
          CONFIGURED_BY（读配置）、READS/WRITES（读写数据库）不属于"服务调用关系"，
          混进来会污染图算法的结果。
        """
        rels = set(relations)
        graph = nx.DiGraph()
        for service in self.services():
            graph.add_node(service.id, name=service.name)
        for edge in self.graph.edges:
            if edge.relation not in rels:
                continue
            # 只保留两端都是 Service 的边（文档->服务、服务->部署等非服务间边跳过）
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
        """从 provenance 中提取真实文件路径，作为模块三的修复锚点。

        模块三拿到问题后，要知道改哪个文件。
        来源是节点的 provenance.source_id，比如 "src/order_service.py:42"。
        过滤逻辑：
          - 剥掉行号后缀（:42 或 :42-50），只留文件路径
          - 跳过带 :// 的 URL（不是本地文件）
          - 只保留代码/配置/文档后缀（_FILE_SUFFIXES），过滤掉非文件来源
        返回排序去重后的文件路径列表。
        """
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
        """统计图谱的数据覆盖情况。

        规则层用它来判断哪些规则该跑、哪些只报采集缺口：
          - has_deployment=False -> 副本数、探针类规则跳过（没部署数据，判了也是瞎猜）
          - has_runtime=False    -> 可观测性规则只报一条"缺少运行时数据"，不逐服务刷
          - has_service=False    -> 整个分析没意义（连服务都没采集到）
        """
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
