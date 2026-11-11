"""Agent 语义研判层（骨架）。

依赖关系：
  - 复用模块二的 ``view`` / ``schema``；
  - 只读复用模块一的 ``config`` 与 ``LLMClient``，不修改模块一任何代码。

loader 顺序：context（裁剪） -> tools（查询） -> evidence（校验） -> rubric（标准） -> agent（编排）。
"""
from .agent import AgentConfig, SemanticAgent, build_llm_client, build_user_prompt
from .context import ContextBuilder, SubgraphContext, node_summary
from .evidence import CitationResult, CitationValidator
from .rubric import Rubric, RubricLibrary
from .tools import TOOL_SPECS, GraphTools, ToolSpec

__all__ = [
    "AgentConfig",
    "CitationResult",
    "CitationValidator",
    "ContextBuilder",
    "GraphTools",
    "Rubric",
    "RubricLibrary",
    "SemanticAgent",
    "SubgraphContext",
    "TOOL_SPECS",
    "ToolSpec",
    "build_llm_client",
    "build_user_prompt",
    "node_summary",
]
