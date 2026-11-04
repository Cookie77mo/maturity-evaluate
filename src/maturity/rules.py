"""确定性图规则层（无 LLM）。

每条规则只读 ``GraphView``，输出带证据链的 ``Problem``。
数据缺失时规则**不臆造问题**（例如没有 Deployment 节点就不判副本数），
覆盖缺口由 ``coverage`` / 可观测性规则单独报告。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import networkx as nx

from .schema import Dimension, Evidence, Problem, Severity
from .view import GraphView


@dataclass
class RuleThresholds:
    """可被 CLI / 配置文件覆盖的阈值。"""

    fan_in: int = 5                 # 服务被依赖次数超过该值 → 热点
    max_chain: int = 5              # 服务调用链最大跳数
    min_replicas: int = 2           # 部署副本数下限
    max_entities: int = 10          # 单条问题最多列出的实体数


def _quotes(view: GraphView, node_ids: Iterable[str], limit: int = 8) -> list[dict]:
    return view.node_quotes(list(dict.fromkeys(node_ids)), limit=limit)


def _problem(
    view: GraphView,
    *,
    dimension: Dimension,
    indicator: str,
    severity: Severity,
    title: str,
    statement: str,
    entities: Iterable[str],
    edges: Iterable = (),
    paths: Iterable[Iterable[str]] = (),
    confidence: float = 0.9,
    suggestion: str = "",
    threshold_limit: int = 10,
) -> Problem:
    entity_list = list(dict.fromkeys(entities))[:threshold_limit]
    edge_list = list(edges)
    quote_nodes = entity_list + [e.source for e in edge_list] + [e.target for e in edge_list]
    return Problem(
        dimension=dimension.value,
        indicator=indicator,
        severity=severity.value,
        title=title,
        statement=statement,
        entities=entity_list,
        evidence=Evidence(
            nodes=entity_list,
            edges=[view.edge_ref(e) for e in edge_list],
            paths=[list(p) for p in paths],
            quotes=_quotes(view, quote_nodes),
        ),
        confidence=confidence,
        suggestion=suggestion,
        fix_targets=view.file_targets(entity_list),
    )


# ---------------------------------------------------------------- 可靠性/韧性
def rule_dependency_cycle(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    """服务依赖环：强连通分量 > 1 个节点即视为循环依赖。"""
    sg = view.service_graph()
    problems: list[Problem] = []
    for component in nx.strongly_connected_components(sg):
        if len(component) < 2:
            continue
        nodes = sorted(component)
        core = [n for n in nodes if (node := view.node(n)) and view.is_core(node)]
        edges = [e for e in view.graph.edges if e.relation in ("CALLS", "DEPENDS_ON")
                 and e.source in component and e.target in component]
        problems.append(_problem(
            view,
            dimension=Dimension.MAINTAINABILITY,
            indicator="dependency_cycle",
            severity=Severity.HIGH if core else Severity.MEDIUM,
            title="服务依赖存在环",
            statement=(
                f"{len(nodes)} 个服务互相依赖形成环：{', '.join(n.split('/')[1].split('#')[0] for n in nodes)}；"
                "任一环节故障会沿环传播，且无法独立发布。"
            ),
            entities=nodes,
            edges=edges,
            paths=[sorted(component)],
            confidence=0.95,
            suggestion="梳理调用方向，抽离公共能力或引入异步解耦，打断环。",
            threshold_limit=cfg.max_entities,
        ))
    return problems


def rule_single_point_of_failure(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    """服务的割点 = 去掉它会使依赖图分裂，即单点故障。"""
    sg = view.service_graph()
    undirected = sg.to_undirected()
    if undirected.number_of_edges() == 0:
        return []
    problems: list[Problem] = []
    for node_id in nx.articulation_points(undirected):
        node = view.node(node_id)
        if node is None:
            continue
        rest = undirected.copy()
        rest.remove_node(node_id)
        pieces = nx.number_connected_components(rest)
        problems.append(_problem(
            view,
            dimension=Dimension.RELIABILITY,
            indicator="single_point_of_failure",
            severity=Severity.HIGH if view.is_core(node) else Severity.MEDIUM,
            title="服务是单点故障",
            statement=(
                f"{node.name} 是依赖图中的割点：移除后剩余服务会分裂成 {pieces} 个互不连通的子图，"
                "其不可用将直接阻断跨子图调用。"
            ),
            entities=[node_id],
            edges=view.out_edges(node_id, ("CALLS", "DEPENDS_ON"))
                   + view.in_edges(node_id, ("CALLS", "DEPENDS_ON")),
            confidence=0.95,
            suggestion="为该服务增加冗余实例/降级路径，或拆分其承担的桥接职责。",
            threshold_limit=cfg.max_entities,
        ))
    return problems


def rule_fan_in_hotspot(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    """被大量服务依赖的热点：变更影响面大。"""
    sg = view.service_graph()
    problems: list[Problem] = []
    for node_id, degree in sg.in_degree():
        if degree < cfg.fan_in:
            continue
        node = view.node(node_id)
        if node is None:
            continue
        problems.append(_problem(
            view,
            dimension=Dimension.MAINTAINABILITY,
            indicator="fan_in_hotspot",
            severity=Severity.MEDIUM,
            title="服务被依赖度过高",
            statement=(
                f"{node.name} 被 {degree} 个服务直接依赖（阈值 {cfg.fan_in}），"
                "任何接口或行为变更都会波及大量上游。"
            ),
            entities=[node_id],
            edges=view.in_edges(node_id, ("CALLS", "DEPENDS_ON")),
            confidence=1.0,
            suggestion="收敛接口契约、补兼容层或按职责拆分，降低变更影响面。",
            threshold_limit=cfg.max_entities,
        ))
    return problems


def rule_long_call_chain(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    """同步调用链过长：尾延迟累积、级联失败概率上升。"""
    sg = view.service_graph()
    if sg.number_of_nodes() < 2 or not nx.is_directed_acyclic_graph(sg):
        return []
    path = nx.dag_longest_path(sg)
    if len(path) - 1 < cfg.max_chain:
        return []
    names = [view.node(n).name for n in path if view.node(n)]
    problems: list[Problem] = []
    problems.append(_problem(
        view,
        dimension=Dimension.RELIABILITY,
        indicator="long_call_chain",
        severity=Severity.MEDIUM,
        title="同步调用链过长",
        statement=(
            f"最长服务调用链包含 {len(path)} 个服务（{' → '.join(names)}），"
            f"超过阈值 {cfg.max_chain} 跳；下游延迟与故障会逐跳放大。"
        ),
        entities=path,
        edges=[e for e in view.graph.edges
               if e.relation in ("CALLS", "DEPENDS_ON") and e.source in path and e.target in path],
        paths=[path],
        confidence=0.9,
        suggestion="评估链路中可并行/异步化的环节，为长链路设置超时与熔断。",
        threshold_limit=cfg.max_entities,
    ))
    return problems


def rule_service_without_entry(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    """既无入向调用、也无对外接口与部署的孤立服务（大概率死代码或漏采集）。"""
    sg = view.service_graph()
    problems: list[Problem] = []
    for service in view.services():
        exposes = view.out_edges(service.id, ("EXPOSES",))
        deployed = view.out_edges(service.id, ("DEPLOYED_AS",))
        if exposes or deployed:
            continue
        if sg.has_node(service.id) and (sg.in_degree(service.id) > 0 or sg.out_degree(service.id) > 0):
            continue
        problems.append(_problem(
            view,
            dimension=Dimension.MAINTAINABILITY,
            indicator="service_without_entry",
            severity=Severity.LOW,
            title="服务无任何入口与部署关系",
            statement=(
                f"{service.name} 没有被其它服务调用、没有 EXPOSES 接口、也没有 DEPLOYED_AS 部署关系，"
                "可能是死代码，或采集/文档缺失。"
            ),
            entities=[service.id],
            confidence=0.6,
            suggestion="确认服务是否仍在链路中；若已废弃则删除，否则补齐接口/部署信息。",
            threshold_limit=cfg.max_entities,
        ))
    return problems


# ------------------------------------------------------------------ 一致性
def rule_doc_code_mismatch(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    """文档声明了某服务/接口，但代码侧没有任何证据 → 文档与实现不一致。"""
    problems: list[Problem] = []
    for edge in view.graph.edges:
        if edge.relation not in ("DEPENDS_ON", "IMPLEMENTS"):
            continue
        source = view.node(edge.source)
        target = view.node(edge.target)
        if source is None or target is None:
            continue
        if source.type != "Document" or target.type not in ("Service", "API"):
            continue
        if any(p.source_type == "source_code" for p in target.provenance):
            continue
        problems.append(_problem(
            view,
            dimension=Dimension.CONSISTENCY,
            indicator="doc_code_mismatch",
            severity=Severity.MEDIUM,
            title="文档描述与代码不一致",
            statement=(
                f"文档 {source.name} 提到 {target.type} {target.name}，"
                "但代码采集结果中没有任何对应证据；可能是文档过期或实现缺失。"
            ),
            entities=[source.id, target.id],
            edges=[edge],
            confidence=0.7,
            suggestion="核对文档与实现，更新过期文档或补齐缺失实现。",
            threshold_limit=cfg.max_entities,
        ))
    return problems


_VALUE_RE = re.compile(r"=\s*(.*)$")


def rule_config_conflict(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    """同一配置键在多个来源/文件中取值不一致。"""
    problems: list[Problem] = []
    for node in view.find("Config"):
        values: dict[str, list[str]] = {}
        attr_value = node.attributes.get("value")
        if attr_value is not None:
            values.setdefault(str(attr_value), [])
        for prov in node.provenance:
            match = _VALUE_RE.search(prov.evidence or "")
            if not match:
                continue
            raw = match.group(1).strip()
            if raw:
                values.setdefault(raw, []).append(prov.source_id)
        if len(values) < 2:
            continue
        problems.append(_problem(
            view,
            dimension=Dimension.CONFIGURABILITY,
            indicator="config_value_conflict",
            severity=Severity.MEDIUM,
            title="同一配置键取值冲突",
            statement=(
                f"配置项 {node.name} 出现 {len(values)} 个不同取值："
                + "；".join(f"{value}（{len(sources) or 1} 处）" for value, sources in values.items())
                + "。以哪个为准不明确，环境行为可能漂移。"
            ),
            entities=[node.id],
            confidence=0.75,
            suggestion="确定唯一事实来源，删除重复定义或改为显式环境覆盖。",
            threshold_limit=cfg.max_entities,
        ))
    return problems


# ------------------------------------------------------------------ 可观测性
def rule_observability(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    """有 Metric 数据时判定核心服务缺监控；完全没有时只报采集缺口，避免逐个刷屏。"""
    metrics = view.find("Metric")
    observed = [e for e in view.graph.edges if e.relation == "OBSERVED_BY"]
    if not metrics and not observed:
        return [Problem(
            dimension=Dimension.OBSERVABILITY.value,
            indicator="runtime_coverage_missing",
            severity=Severity.LOW.value,
            title="缺少运行时观测数据",
            statement=(
                "图谱中没有 Metric 节点或 OBSERVED_BY 关系（RuntimeCollector 仍为 stub），"
                "可观测性维度无法逐服务评估。"
            ),
            confidence=0.95,
            suggestion="接入 Prometheus/日志/链路采集后重新分析，再启用逐服务可观测性规则。",
        )]
    problems: list[Problem] = []
    observed_sources = {e.source for e in observed}
    for service in view.services():
        if not view.is_core(service) or service.id in observed_sources:
            continue
        problems.append(_problem(
            view,
            dimension=Dimension.OBSERVABILITY,
            indicator="core_service_unobserved",
            severity=Severity.MEDIUM,
            title="核心服务缺少运行时监控",
            statement=f"核心链路服务 {service.name} 没有任何 OBSERVED_BY 指标，故障无法被及时发现。",
            entities=[service.id],
            confidence=0.85,
            suggestion="为该服务接入 RED/USE 指标与告警。",
            threshold_limit=cfg.max_entities,
        ))
    return problems


# ------------------------------------------------------------ 部署（有数据才判）
def rule_deployment_replicas(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    problems: list[Problem] = []
    for edge in view.graph.edges:
        if edge.relation != "DEPLOYED_AS":
            continue
        service = view.node(edge.source)
        deployment = view.node(edge.target)
        if service is None or deployment is None:
            continue
        replicas = edge.attributes.get("replicas", deployment.attributes.get("replicas"))
        if replicas is None:
            continue
        try:
            replicas_int = int(replicas)
        except (TypeError, ValueError):
            continue
        if replicas_int >= cfg.min_replicas:
            continue
        problems.append(_problem(
            view,
            dimension=Dimension.RELIABILITY,
            indicator="single_replica",
            severity=Severity.HIGH if view.is_core(service) else Severity.MEDIUM,
            title="部署副本数不足",
            statement=(
                f"{service.name} 仅部署 {replicas_int} 个副本（下限 {cfg.min_replicas}），"
                "单实例故障即导致该服务不可用，也无法滚动升级不断服。"
            ),
            entities=[service.id, deployment.id],
            edges=[edge],
            confidence=0.95,
            suggestion=f"将副本数提升到至少 {cfg.min_replicas}，并配置反亲和与滚动更新策略。",
            threshold_limit=cfg.max_entities,
        ))
    return problems


def rule_pod_health(view: GraphView, cfg: RuleThresholds) -> list[Problem]:
    """Pod 级探针/资源限制缺失（仅在有 Pod 节点时生效）。"""
    pods = view.find("Pod")
    if not pods:
        return []
    problems: list[Problem] = []
    missing_probe = [
        p for p in pods
        if p.attributes.get("readiness_probe") is False or p.attributes.get("liveness_probe") is False
    ]
    if missing_probe:
        problems.append(_problem(
            view,
            dimension=Dimension.RELIABILITY,
            indicator="pod_missing_probe",
            severity=Severity.MEDIUM,
            title="容器缺少健康探针",
            statement=f"{len(missing_probe)} 个 Pod 缺少 readiness 或 liveness 探针，故障实例无法被自动摘除/重启。",
            entities=[p.id for p in missing_probe],
            confidence=0.95,
            suggestion="补充 readinessProbe 与 livenessProbe，并与真实启动/健康路径对齐。",
            threshold_limit=cfg.max_entities,
        ))
    no_limits = [
        p for p in pods
        if not (isinstance(p.attributes.get("resources"), dict) and p.attributes["resources"].get("limits"))
    ]
    if no_limits:
        problems.append(_problem(
            view,
            dimension=Dimension.RELIABILITY,
            indicator="pod_missing_resource_limit",
            severity=Severity.LOW,
            title="容器未设置资源上限",
            statement=f"{len(no_limits)} 个 Pod 未设置 CPU/内存 limits，单实例异常可能拖垮节点。",
            entities=[p.id for p in no_limits],
            confidence=0.95,
            suggestion="为容器补充 requests/limits，并验证限流后仍满足性能目标。",
            threshold_limit=cfg.max_entities,
        ))
    return problems


RULES: list[Callable[[GraphView, RuleThresholds], list[Problem]]] = [
    rule_dependency_cycle,
    rule_single_point_of_failure,
    rule_fan_in_hotspot,
    rule_long_call_chain,
    rule_service_without_entry,
    rule_doc_code_mismatch,
    rule_config_conflict,
    rule_observability,
    rule_deployment_replicas,
    rule_pod_health,
]
