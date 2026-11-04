"""命令行入口（``mpe`` = maturity problem evaluator）。

用法::

    mpe analyze out/kg.json -o out/problems.jsonl
    mpe analyze out/kg.json --min-severity high --fan-in 4 --top 5
"""
from __future__ import annotations

import click

from .analyzer import analyze
from .rules import RuleThresholds


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
    """对模块一的知识图谱做成熟度问题分析。"""
    thresholds = RuleThresholds(
        fan_in=fan_in,
        max_chain=max_chain,
        min_replicas=min_replicas,
        max_entities=max_entities,
    )
    result = analyze(graph_path, out, thresholds=thresholds, min_severity=min_severity)
    click.echo(result.summary_text(top=top))
    click.echo(f"\n已写入: {out}")


if __name__ == "__main__":
    cli()
