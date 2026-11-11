"""
命令行入口（mpe = maturity problem evaluator）。

用 click 库实现，提供 analyze 子命令。

可以通过用户在命令行输入去更改标准，无需修改原代码的阈值参数：
`--fan-in` 几个依赖算"热点"
`--max-chain` 调用链几跳算"过长"
`--min-replicas` 副本数下限
`--max-entities` 单条问题最多列几个实体
`--min-severity` 只看什么严重度以上

用法::

    mpe analyze out/kg.json -o out/problems.jsonl
    mpe analyze out/kg.json --min-severity high --fan-in 4 --top 5

Agent 层（语义研判）另有两条子命令：

    mpe agent-context out/kg.json --node <id>     # 预览子图裁剪结果（不调模型）
    mpe agent-judge out/kg.json --indicator X --node <id> --config config.local.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from kg_builder.graph.serialization import load_graph

from .agent import AgentConfig, RubricLibrary, SemanticAgent, build_llm_client
from .agent.context import ContextBuilder
from .analyzer import analyze
from .rules import RuleThresholds
from .schema import save_problems
from .view import GraphView

# 内置 Rubric 目录（随包分发）；不存在时视为"还没有任何官方指标定义"
_RUBRIC_DIR = Path(__file__).resolve().parent / "rubrics"


@click.group()
def cli() -> None:
    """Maturity Evaluation Module 2: assessment & issue discovery."""


@cli.command("analyze")
@click.argument("graph_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default="out/problems.jsonl", show_default=True, help="问题清单输出路径(JSONL)。")
@click.option("--min-severity", type=click.Choice(["high", "medium", "low"]), default="low",
              show_default=True, help="只输出不低于该严重度的问题。")
@click.option("--fan-in", type=int, default=5, show_default=True, help="扇入热点阈值。")
@click.option("--max-chain", type=int, default=5, show_default=True, help="同步调用链最大跳数。")
@click.option("--min-replicas", type=int, default=2, show_default=True, help="部署副本数下限。")
@click.option("--max-entities", type=int, default=10, show_default=True, help="单条问题最多实体数。")
@click.option("--top", type=int, default=5, show_default=True, help="终端展示前 N 条问题(0=不展示)。")
def analyze_cmd(graph_path, out, min_severity, fan_in, max_chain, min_replicas, max_entities, top):
    """对模块一的知识图谱做成熟度问题分析。

    把命令行参数组装成 RuleThresholds，调用 analyzer.analyze()，
    然后打印摘要 + 输出文件路径。
    """
    thresholds = RuleThresholds(
        fan_in=fan_in,
        max_chain=max_chain,
        min_replicas=min_replicas,
        max_entities=max_entities,
    )
    result = analyze(graph_path, out, thresholds=thresholds, min_severity=min_severity)
    click.echo(result.summary_text(top=top))
    click.echo(f"\n已写入: {out}")


# ----------------------------------------------------------------- Agent 层
@cli.command("agent-context")
@click.argument("graph_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--node", "node_ids", multiple=True, required=True, help="种子节点 id，可重复传入。")
@click.option("--hops", type=int, default=2, show_default=True, help="子图裁剪跳数。")
@click.option("--max-nodes", type=int, default=40, show_default=True, help="裁剪后的节点上限。")
def agent_context_cmd(graph_path, node_ids, hops, max_nodes):
    """预览子图裁剪结果（不调用 LLM，零成本）。

    用于确认"喂给模型的上下文长什么样"——上下文不对，后面 prompt 怎么调都没用。
    """
    view = GraphView(load_graph(graph_path))
    builder = ContextBuilder(view, hops=hops, max_nodes=max_nodes)
    context = builder.around(node_ids)
    if not context.node_ids:
        raise click.ClickException("种子节点在图中不存在，请检查 --node 参数。")
    click.echo(context.text)
    suffix = "（已按 max-nodes 截断）" if context.truncated else ""
    click.echo(f"\n共 {len(context.node_ids)} 个节点 / {len(context.edges)} 条边{suffix}")


@cli.command("agent-judge")
@click.argument("graph_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--indicator", required=True, help="要研判的指标 ID（需在 Rubric 中有定义）。")
@click.option("--node", "node_ids", multiple=True, required=True, help="种子节点 id，可重复传入。")
@click.option("--config", "config_file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="含 llm 段的 YAML 配置（沿用模块一的配置格式）。")
@click.option("--rubrics", "rubrics_path", type=click.Path(exists=True), default=None,
              help="Rubric 文件或目录，默认用内置 src/maturity/rubrics。")
@click.option("--max-tool-rounds", type=int, default=2, show_default=True, help="最多工具查询轮数。")
@click.option("-o", "--out", default="", help="问题输出路径（JSONL）；留空只打印。")
def agent_judge_cmd(graph_path, indicator, node_ids, config_file, rubrics_path, max_tool_rounds, out):
    """用一个 Rubric 研判指定指标（会真实调用 LLM）。"""
    rubrics = _load_rubrics(rubrics_path)
    if rubrics.get(indicator) is None:
        available = ", ".join(rubrics.indicators()) or "（空）"
        raise click.ClickException(
            f"没有指标 {indicator} 的 Rubric。当前可用: {available}。"
            "请补充 Rubric 后再运行 —— Agent 不会在没有判定标准时凭直觉产出问题。"
        )

    try:
        llm = build_llm_client(config_file)
    except Exception as exc:  # noqa: BLE001 - 统一转成用户可读错误
        raise click.ClickException(f"LLM 未配置或初始化失败: {exc}") from exc

    view = GraphView(load_graph(graph_path))
    agent = SemanticAgent(
        view,
        llm=llm,
        rubrics=rubrics,
        config=AgentConfig(max_tool_rounds=max_tool_rounds),
    )
    try:
        problems = agent.judge(indicator, list(node_ids))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"研判失败: {exc}") from exc

    if not problems:
        click.echo("未产出问题：可能证据不足被校验丢弃，或模型判定为合格。")
        return
    for problem in problems:
        click.echo(
            f"[{problem.severity:6}] {problem.indicator}: {problem.title} "
            f"({problem.id}) conf={problem.confidence:.2f}"
        )
        click.echo(f"    {problem.statement}")
    if out:
        save_problems(problems, out)
        click.echo(f"\n已写入: {out}")


def _load_rubrics(rubrics_path: Optional[str]) -> RubricLibrary:
    """优先用显式路径，其次用内置目录；都没有时返回空库。"""
    if rubrics_path:
        return RubricLibrary.load(rubrics_path)
    if _RUBRIC_DIR.is_dir():
        return RubricLibrary.load(_RUBRIC_DIR)
    return RubricLibrary.empty()


if __name__ == "__main__":
    cli()
