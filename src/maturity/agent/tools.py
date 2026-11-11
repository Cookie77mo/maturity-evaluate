"""工具化查询接口：把 GraphView 的只读能力暴露成 LLM 可按名调用的 JSON 接口。

为什么需要工具，而不是把全图塞进 prompt：

  裁剪后的上下文只包含 2 跳邻居。如果模型怀疑"这两个名字不同但功能相同的
  服务其实是同一个"，它需要主动去全图里查有没有同名/近名节点 —— 这是
  上下文之外的查询需求，只能靠工具满足。

约定：

  - 全部只读，绝不修改图谱；
  - 返回统一结构 ``{"ok": bool, "data": ...}`` 或 ``{"ok": False, "error": ...}``；
  - 参数错误不抛异常，让模型有机会自己纠正；
  - 单次结果条数有上限，防止一个查询把上下文撑爆。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx

from ..view import GraphView

# 单次工具调用的最大返回条数
MAX_RESULT = 20

# 只有这两类关系被视为"服务间的服务调用拓扑"
SERVICE_RELATIONS = ("CALLS", "DEPENDS_ON")


@dataclass(frozen=True)
class ToolSpec:
    """工具描述，直接序列化给模型看。"""

    name: str
    description: str
    args: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "args": self.args}


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_node",
        description="按 id 取单个节点的完整信息（属性 + provenance）。",
        args={"node_id": "节点 id"},
    ),
    ToolSpec(
        name="find_nodes",
        description="按名称片段模糊查找节点，用于确认是否存在同名/近名的重复实体。",
        args={"name_contains": "名称片段，可空", "type": "节点类型，如 Service，可空"},
    ),
    ToolSpec(
        name="neighbors",
        description="取某节点的直接邻居（入向/出向/双向）。",
        args={"node_id": "节点 id", "direction": "in | out | both，默认 both", "relations": "关系类型列表，可空"},
    ),
    ToolSpec(
        name="edges_between",
        description="查询两个节点之间是否存在边，用于确认依赖方向。",
        args={"source": "起点 id", "target": "终点 id"},
    ),
    ToolSpec(
        name="paths_between",
        description="查询两个服务之间的调用路径，用于确认故障传播链路。",
        args={"source": "起点 id", "target": "终点 id", "max_hops": "最大跳数，默认 4"},
    ),
    ToolSpec(
        name="coverage",
        description="查看图谱的数据覆盖情况（是否有部署/运行时/历史数据）。",
        args={},
    ),
)


class GraphTools:
    """工具集实现。所有方法都通过 :meth:`call` 以名字分发。"""

    def __init__(self, view: GraphView) -> None:
        self.view = view
        self._nx_graph: Optional[nx.DiGraph] = None

    # ------------------------------------------------------------------ 分发
    def specs(self) -> list[ToolSpec]:
        return list(TOOL_SPECS)

    def call(self, name: str, args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """按名字调用工具。

        未知工具和参数错误都返回结构化错误，不抛异常 —— 模型看到错误信息
        后可以换一种调用方式重试。
        """
        args = args or {}
        if not isinstance(args, dict):
            return {"ok": False, "error": "args 必须是对象"}
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {
                "ok": False,
                "error": f"未知工具: {name}",
                "available": [spec.name for spec in TOOL_SPECS],
            }
        try:
            return {"ok": True, "data": handler(**args)}
        except TypeError as exc:
            return {"ok": False, "error": f"参数不匹配: {exc}"}

    # ------------------------------------------------------------------ 工具
    def _tool_get_node(self, node_id: str) -> dict[str, Any]:
        node = self.view.node(str(node_id))
        if node is None:
            return {"found": False, "node_id": node_id}
        return {
            "found": True,
            "id": node.id,
            "type": node.type,
            "name": node.name,
            "attributes": node.attributes,
            "provenance": [
                {
                    "source_type": prov.source_type,
                    "source_id": prov.source_id,
                    "confidence": round(float(prov.confidence), 2),
                    "evidence": prov.evidence[:200],
                }
                for prov in node.provenance[:5]
            ],
        }

    def _tool_find_nodes(self, name_contains: str = "", type: str = "") -> dict[str, Any]:
        needle = str(name_contains or "").strip().lower()
        wanted_type = str(type or "").strip()
        matches: list[dict[str, str]] = []
        for node in self.view.nodes:
            if wanted_type and node.type != wanted_type:
                continue
            if needle:
                haystack = f"{node.name} {node.canonical_name} {node.id}".lower()
                if needle not in haystack:
                    continue
            matches.append({"id": node.id, "type": node.type, "name": node.name})
            if len(matches) >= MAX_RESULT:
                break
        return {"count": len(matches), "nodes": matches}

    def _tool_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        relations: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        node_id = str(node_id)
        if self.view.node(node_id) is None:
            return {"found": False, "node_id": node_id}
        rels = set(relations) if relations else None
        direction = str(direction or "both").lower()
        found: list[dict[str, str]] = []
        for edge in self.view.graph.edges:
            if rels is not None and edge.relation not in rels:
                continue
            if direction in ("out", "both") and edge.source == node_id:
                found.append({"relation": edge.relation, "direction": "out", "neighbor": edge.target})
            if direction in ("in", "both") and edge.target == node_id:
                found.append({"relation": edge.relation, "direction": "in", "neighbor": edge.source})
            if len(found) >= MAX_RESULT:
                break
        return {"found": True, "count": len(found), "neighbors": found}

    def _tool_edges_between(self, source: str, target: str) -> dict[str, Any]:
        source, target = str(source), str(target)
        relations = [
            edge.relation
            for edge in self.view.graph.edges
            if edge.source == source and edge.target == target
        ]
        return {"source": source, "target": target, "relations": relations}

    def _tool_paths_between(
        self,
        source: str,
        target: str,
        max_hops: int = 4,
    ) -> dict[str, Any]:
        source, target = str(source), str(target)
        graph = self._service_nx()
        if source not in graph or target not in graph:
            return {"count": 0, "paths": [], "note": "起点或终点不在服务调用子图中"}
        cutoff = max(1, int(max_hops))
        paths: list[list[str]] = []
        try:
            for path in nx.all_simple_paths(graph, source, target, cutoff=cutoff):
                paths.append(list(path))
                if len(paths) >= MAX_RESULT:
                    break
        except nx.NetworkXNoPath:
            pass
        return {"count": len(paths), "paths": paths}

    def _tool_coverage(self) -> dict[str, Any]:
        return self.view.coverage()

    # ------------------------------------------------------------------ 内部
    def _service_nx(self) -> nx.DiGraph:
        """惰性构建只含 CALLS / DEPENDS_ON 的有向图，用于路径查询。"""
        if self._nx_graph is None:
            graph = nx.DiGraph()
            for node in self.view.graph.nodes:
                graph.add_node(node.id)
            for edge in self.view.graph.edges:
                if edge.relation in SERVICE_RELATIONS:
                    graph.add_edge(edge.source, edge.target)
            self._nx_graph = graph
        return self._nx_graph
