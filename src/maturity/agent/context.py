"""子图裁剪：把候选对象周围的 k 跳邻居切出来，作为 Agent 的上下文。

为什么不能把整张图谱喂给 LLM：

  1. 成本与窗口：令牌随节点数线性增长，而判断一个指标通常只需要 2 跳内的证据；
  2. 注意力稀释：无关节点越多，模型越容易引用与问题无关的实体；
  3. 规模：图谱从几十个节点涨到几百个之后，整图根本塞不进上下文窗口。

裁剪策略：

  1. 种子节点必然保留 —— 它们是被规则或人工挑出来的嫌疑对象；
  2. 从种子出发做有界 BFS，按跳数由近及远填充，直到 ``max_nodes``；
  3. 只保留两端都入选的边，避免渲染出指向不存在节点的"悬空边"
     （悬空边会诱导模型去引用一个它根本看不到的节点）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

from kg_builder.graph.model import KGEdge, KGNode

from ..view import GraphView

# 只把这些属性渲染进上下文。attributes 里可能有很深的 dict（如完整的 resources），
# 全量输出既费令牌，也容易让模型抓错重点。
_ATTR_KEYS = (
    "is_core_business",
    "replicas",
    "readiness_probe",
    "liveness_probe",
    "status",
    "restart_count",
    "qos_class",
    "value",
)

# 单条 provenance 的 evidence 文本截断长度
_EVIDENCE_LIMIT = 120


def node_summary(node: KGNode) -> str:
    """把节点压成一行摘要。

    id 必须完整输出 —— LLM 要靠它精确引用，证据校验也是按完整 id 比对的。
    """
    parts = [f"id={node.id}", f"type={node.type}", f"name={node.name}"]
    for key in _ATTR_KEYS:
        if key in node.attributes:
            parts.append(f"{key}={node.attributes[key]}")
    semantic = node.attributes.get("semantic")
    if isinstance(semantic, dict) and "is_core_business" in semantic:
        parts.append(f"is_core_business={semantic['is_core_business']}")
    resources = node.attributes.get("resources")
    if isinstance(resources, dict):
        # 只暴露"有没有 limits"，不铺开完整的 requests/limits 数值
        parts.append(f"limits={bool(resources.get('limits'))}")
    return " ".join(parts)


@dataclass
class SubgraphContext:
    """一次裁剪的结果。

    seeds     : 种子节点 id（渲染时用 ``*`` 标出，提示模型这是关注点）
    node_ids  : 入选的全部节点 id（BFS 顺序，近的在前）
    edges     : 入选边的 (source, target, relation) 三元组
    text      : 渲染后的文本，直接拼进 prompt
    truncated : 是否因为 max_nodes 被截断（截断说明上下文可能不完整）
    """

    seeds: list[str]
    node_ids: list[str]
    edges: list[tuple[str, str, str]]
    text: str
    truncated: bool = False


class ContextBuilder:
    """按跳数裁剪子图并渲染成文本。

    邻居索引在构造时建一次，避免每次 BFS 都全量扫描 edges。
    """

    def __init__(
        self,
        view: GraphView,
        *,
        hops: int = 2,
        max_nodes: int = 40,
        max_quotes: int = 2,
    ) -> None:
        self.view = view
        self.hops = max(1, int(hops))
        self.max_nodes = max(1, int(max_nodes))
        self.max_quotes = max(0, int(max_quotes))
        # 无向邻接表：判断一个服务是不是问题，上下游都可能是证据
        self._adj: dict[str, list[str]] = {}
        for edge in view.graph.edges:
            self._adj.setdefault(edge.source, []).append(edge.target)
            if edge.target != edge.source:
                self._adj.setdefault(edge.target, []).append(edge.source)

    # ------------------------------------------------------------------ 裁剪
    def around(
        self,
        seed_ids: Iterable[str],
        *,
        hops: Optional[int] = None,
        max_nodes: Optional[int] = None,
    ) -> SubgraphContext:
        """从种子节点做有限跳数、有限节点数的广度优先扩展。"""
        # dict.fromkeys 去重并保持顺序；只接受真实存在的 id
        seeds = [sid for sid in dict.fromkeys(seed_ids) if self.view.node(sid) is not None]
        limit_hops = self.hops if hops is None else max(1, int(hops))
        limit_nodes = self.max_nodes if max_nodes is None else max(1, int(max_nodes))

        dist: dict[str, int] = {sid: 0 for sid in seeds}
        order: list[str] = list(seeds)
        queue: deque[str] = deque(seeds)
        while queue:
            current = queue.popleft()
            if dist[current] >= limit_hops:
                continue
            for nxt in self._adj.get(current, ()):
                if nxt in dist:
                    continue
                dist[nxt] = dist[current] + 1
                order.append(nxt)
                queue.append(nxt)

        selected = order[:limit_nodes]
        truncated = len(order) > limit_nodes
        chosen = set(selected)

        # 只保留两端都入选的边
        edge_objs: list[KGEdge] = [
            edge
            for edge in self.view.graph.edges
            if edge.source in chosen and edge.target in chosen
        ]

        return SubgraphContext(
            seeds=seeds,
            node_ids=selected,
            edges=[edge.key for edge in edge_objs],
            text=self._render(seeds, selected, edge_objs),
            truncated=truncated,
        )

    # ------------------------------------------------------------------ 渲染
    def _render(
        self,
        seeds: list[str],
        selected: list[str],
        edge_objs: list[KGEdge],
    ) -> str:
        seed_set = set(seeds)
        lines: list[str] = ["## 子图节点（引用时只能使用这里出现的 id）"]
        for node_id in selected:
            node = self.view.node(node_id)
            if node is None:  # 理论上不会发生，防御一下
                continue
            marker = "* " if node_id in seed_set else "- "
            lines.append(marker + node_summary(node))
            for prov in node.provenance[: self.max_quotes]:
                snippet = f" | {prov.evidence[:_EVIDENCE_LIMIT]}" if prov.evidence else ""
                lines.append(
                    f"    · {prov.source_type}: {prov.source_id} "
                    f"conf={float(prov.confidence):.2f}{snippet}"
                )

        lines.append("")
        lines.append("## 子图关系")
        if not edge_objs:
            lines.append("- （无）")
        for edge in edge_objs:
            conf = max((float(p.confidence) for p in edge.provenance), default=1.0)
            lines.append(f"- {edge.source} --{edge.relation}--> {edge.target} conf={conf:.2f}")
        return "\n".join(lines)
