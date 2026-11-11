"""Rubric：成熟度指标的判定标准，是 Agent 层唯一的判定依据。

为什么把判定标准外置成 YAML，而不是写死在 prompt 里：

  1. 成熟度指标的定义会随课题组讨论反复调整，改一次定义不该动一次代码；
  2. 一个指标一份 Rubric，评审时可以逐条对照"你到底按什么判的"；
  3. 没有 Rubric 的指标，Agent 直接跳过 —— 不允许凭直觉编问题。

YAML 结构见 ``src/maturity/rubrics/`` 下的示例。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ..schema import Dimension, Severity


@dataclass(frozen=True)
class Rubric:
    """一个指标的完整判定标准。

    indicator      : 指标 ID，与规则层命名保持一致（如 doc_code_mismatch）
    dimension      : 所属成熟度维度，必须是 Dimension 里的取值
    title          : 产出问题时的标题
    question       : 要模型回答的那一个判断句
    pass_criteria  : 什么情况算合格（不报问题）
    fail_criteria  : 什么情况算不合格（报问题）
    severity       : 命中时的严重度，由 Rubric 固定，不交给模型决定
    suggestion     : 修复建议
    """

    indicator: str
    dimension: str
    title: str
    question: str
    pass_criteria: str
    fail_criteria: str
    severity: str = "medium"
    suggestion: str = ""


def _parse_rubric(raw: dict[str, Any], source: Path) -> Rubric:
    """从一条 YAML 记录构造 Rubric，字段缺失或取值非法时直接报错。

    这里故意不设默认值兜底：写错的 Rubric 会让 Agent 判出错误结论，
    宁可启动时炸掉，也不要静默降级。
    """
    required = ("indicator", "dimension", "question", "pass_criteria", "fail_criteria")
    missing = [key for key in required if not str(raw.get(key) or "").strip()]
    if missing:
        raise ValueError(f"{source}: rubric 缺少必填字段 {', '.join(missing)}")

    dimension = str(raw["dimension"]).strip()
    severity = str(raw.get("severity", "medium")).strip()
    try:
        Dimension(dimension)
    except ValueError as exc:
        valid = ", ".join(item.value for item in Dimension)
        raise ValueError(f"{source}: 未知维度 {dimension!r}，可选值: {valid}") from exc
    try:
        Severity(severity)
    except ValueError as exc:
        valid = ", ".join(item.value for item in Severity)
        raise ValueError(f"{source}: 未知严重度 {severity!r}，可选值: {valid}") from exc

    return Rubric(
        indicator=str(raw["indicator"]).strip(),
        dimension=dimension,
        title=str(raw.get("title") or raw["indicator"]).strip(),
        question=str(raw["question"]).strip(),
        pass_criteria=str(raw["pass_criteria"]).strip(),
        fail_criteria=str(raw["fail_criteria"]).strip(),
        severity=severity,
        suggestion=str(raw.get("suggestion") or "").strip(),
    )


class RubricLibrary:
    """Rubric 集合，按 indicator 索引。"""

    def __init__(self, rubrics: Iterable[Rubric] = ()) -> None:
        self._by_indicator: dict[str, Rubric] = {}
        for rubric in rubrics:
            self._by_indicator[rubric.indicator] = rubric

    # ------------------------------------------------------------------ 构造
    @classmethod
    def empty(cls) -> "RubricLibrary":
        """空库：所有指标都会被跳过，用于还没有官方定义时的安全默认值。"""
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "RubricLibrary":
        """加载一个 YAML 文件或一个目录（目录下所有 *.yaml / *.yml）。"""
        import yaml  # PyYAML 是核心依赖

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"rubric 路径不存在: {target}")
        if target.is_dir():
            files = sorted([*target.glob("*.yaml"), *target.glob("*.yml")])
        else:
            files = [target]

        rubrics: list[Rubric] = []
        for file in files:
            data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            items = data.get("rubrics", []) if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                raise ValueError(f"{file}: 顶层应为 rubrics 列表")
            for raw in items:
                if not isinstance(raw, dict):
                    raise ValueError(f"{file}: rubrics 的每一项都必须是映射")
                rubrics.append(_parse_rubric(raw, file))
        return cls(rubrics)

    # ------------------------------------------------------------------ 查询
    def get(self, indicator: str) -> Optional[Rubric]:
        return self._by_indicator.get(indicator)

    def indicators(self) -> list[str]:
        return sorted(self._by_indicator)

    def __len__(self) -> int:
        return len(self._by_indicator)

    def __contains__(self, indicator: object) -> bool:
        return indicator in self._by_indicator
