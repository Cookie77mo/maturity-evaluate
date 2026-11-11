"""证据引用强制校验（fail-closed）。

LLM 最大的风险不是答错，而是**编造证据**：引用一个不存在的 node_id，
或者凭空写一段"代码里其实没有"的 quote。模块三会按 evidence 去定位文件，
假证据会直接导致改错地方。

策略是默认拒绝：

  1. 每条结论必须至少引用 1 个真实存在的 node 或 edge，否则整条丢弃；
  2. 引用的 id 不在本次裁剪出的上下文里 -> 视为幻觉，丢弃该引用
     （``allowed_nodes`` / ``allowed_edges`` 就是用来做这个收紧的）；
  3. 引用全部无效的结论直接不算产出，由调用方决定丢弃。

注意这是"结构性校验"，只保证引用指向真实对象，不保证模型对这条证据的
解读是对的。后者只能靠人工复核或注入层实验。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..view import GraphView


@dataclass
class CitationResult:
    """一次校验的结果。

    nodes   : 通过校验的节点 id
    edges   : 通过校验的边引用 {source, target, relation}
    invalid : 被丢弃的原始引用（保留原文，便于事后调 prompt）
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """有没有至少一条有效证据。没有就说明这条结论是空口断言。"""
        return bool(self.nodes or self.edges)


class CitationValidator:
    """把模型给出的引用和真实图谱比对。

    allowed_nodes / allowed_edges 为 None 时表示"全图都允许"；
    SemanticAgent 会传入本次裁剪的节点集合，只允许引用它真正看到过的东西。
    """

    def __init__(
        self,
        view: GraphView,
        *,
        allowed_nodes: Optional[Iterable[str]] = None,
        allowed_edges: Optional[Iterable[tuple[str, str, str]]] = None,
    ) -> None:
        self.view = view
        self.allowed_nodes = set(allowed_nodes) if allowed_nodes is not None else None
        self.allowed_edges = set(allowed_edges) if allowed_edges is not None else None
        # 全量索引，O(1) 判断引用是否真实存在
        self._node_ids: set[str] = {node.id for node in view.graph.nodes}
        self._edge_index: set[tuple[str, str, str]] = {edge.key for edge in view.graph.edges}

    # ------------------------------------------------------------------ 校验
    def check(
        self,
        node_ids: Iterable[Any] = (),
        edges: Iterable[Any] = (),
    ) -> CitationResult:
        result = CitationResult()

        for raw in node_ids:
            node_id = str(raw)
            if self._node_ok(node_id):
                if node_id not in result.nodes:
                    result.nodes.append(node_id)
            else:
                result.invalid.append(node_id)

        for raw in edges:
            key = self._edge_key(raw)
            if key is not None and self._edge_ok(key):
                ref = {"source": key[0], "target": key[1], "relation": key[2]}
                if ref not in result.edges:
                    result.edges.append(ref)
            else:
                result.invalid.append(str(raw))

        return result

    # ------------------------------------------------------------------ 内部
    def _node_ok(self, node_id: str) -> bool:
        if node_id not in self._node_ids:
            return False
        return self.allowed_nodes is None or node_id in self.allowed_nodes

    def _edge_ok(self, key: tuple[str, str, str]) -> bool:
        source, target, _relation = key
        if source not in self._node_ids or target not in self._node_ids:
            return False
        if key not in self._edge_index:
            return False
        return self.allowed_edges is None or key in self.allowed_edges

    @staticmethod
    def _edge_key(raw: Any) -> Optional[tuple[str, str, str]]:
        """把模型给的边引用规范化成三元组；格式不对就返回 None（当作无效）。"""
        if not isinstance(raw, dict):
            return None
        source = str(raw.get("source") or "").strip()
        target = str(raw.get("target") or "").strip()
        relation = str(raw.get("relation") or "").strip()
        if not (source and target and relation):
            return None
        return (source, target, relation)
