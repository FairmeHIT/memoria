"""
procedural.py — Lightweight procedural memory (M-Flow inspired).

Extracts user preferences, routines, and procedures from message content
using deterministic pattern matching (no LLM needed). Stores them as
tagged memories in the existing FTS5 + embedding store.

Patterns:
  - preference: "I prefer/like/love/enjoy X", "I dislike/hate X"
  - routine:    "I always/usually/typically X", "Every [time]"
  - procedure:  "When X, Y", "If X, then Y", step-by-step instructions

Usage:
    from memoria.procedural import extract_procedural
    results = extract_procedural("I prefer coffee in the morning.")
    # → [ProceduralMemory(type='preference', entity='coffee', ...)]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

_ProcType = str  # "preference" | "routine" | "procedure"


@dataclass
class ProceduralMemory:
    """一条抽出的程序性记忆。"""

    proc_type: _ProcType
    """preference / routine / procedure"""
    entity: str
    """提取的主体（如 'coffee', 'morning walk'）"""
    trigger: str
    """触发条件 / 上下文（如 'in the morning'）"""
    statement: str
    """完整的陈述句，用于展示"""
    sentiment: str = "neutral"
    """positive / negative / neutral（仅 preference）"""
    confidence: float = 0.7
    """提取置信度 (0-1)"""
    steps: list[str] = field(default_factory=list)
    """步骤列表（仅 procedure）"""


# ═══════════════════════════════════════════════════════════════
# 模式匹配规则
# ═══════════════════════════════════════════════════════════════

# 偏好动词 + 名词
_PREFERENCE_POSITIVE = re.compile(
    r"\b(I|we)\s+(prefer|like|love|enjoy|admire|appreciate|fancy|"
    r"am\s+a\s+fan\s+of|am\s+fond\s+of|am\s+into|am\s+keen\s+on)\s+"
    r"(.+?)\s*[\.!]?\s*$",
    re.IGNORECASE,
)
_PREFERENCE_NEGATIVE = re.compile(
    r"\b(I|we)\s+(dislike|hate|detest|loathe|don'?t\s+like|don'?t\s+enjoy|"
    r"am\s+not\s+a\s+fan\s+of|can'?t\s+stand|am\s+not\s+into)\s+"
    r"(.+?)\s*[\.!]?\s*$",
    re.IGNORECASE,
)
_FAVORITE = re.compile(
    r"\b(.+?)\s+is\s+(my\s+)?favorite\b",
    re.IGNORECASE,
)
_STATED_PREFERENCE = re.compile(
    r"\b(I\s+would\s+(rather|prefer)\s+to\s+)?(.+?)\s+is\s+(better|"
    r"nicer|more\s+enjoyable|more\s+convenient)\s+(than\s+)?",
    re.IGNORECASE,
)
# 个人属性（健康/条件）："I am allergic to X", "I have Y"
_PERSONAL_FACT = re.compile(
    r"\bI\s+am\s+(allergic\s+to|sensitive\s+to|intolerant\s+of)\s+"
    r"(.+?)\s*[\.!]?\s*$",
    re.IGNORECASE,
)
_PERSONAL_HEALTH = re.compile(
    r"\bI\s+have\s+(a\s+)?(high\s+blood\s+pressure|diabetes|asthma|"
    r"allergy|condition|disease|syndrome|disorder|intolerance)\s*"
    r"(\.|!)?\s*$",
    re.IGNORECASE,
)

# 日常习惯
_ROUTINE_ALWAYS = re.compile(
    r"\bI\s+(always|usually|typically|often|sometimes|occasionally|"
    r"rarely|never|every\s+\w+|frequently|regularly)\s+(.+?)\s*[\.!]?\s*$",
    re.IGNORECASE,
)
_ROUTINE_TIME = re.compile(
    r"\b(Every\s+(morning|afternoon|evening|night|day|week|month|year)|"
    r"Each\s+(morning|afternoon|evening|night|day)|"
    r"Once\s+a\s+(day|week|month|year)|"
    r"Twice\s+a\s+(day|week|month)|"
    r"Before\s+(going\s+to\s+bed|leaving|work|school)|"
    r"After\s+(waking\s+up|dinner|lunch|breakfast|work|school)|"
    r"At\s+(night|dinner|lunch|breakfast|work))\b",
    re.IGNORECASE,
)

# 条件流程
_TRIGGERED_PROCEDURE = re.compile(
    r"\b(When|If|Whenever|Once)\s+(.+?),\s*(then\s+)?(.+?)\s*[\.!]?\s*$",
    re.IGNORECASE,
)
_STEPPED_PROCEDURE = re.compile(
    r"\bFirst[,\s]+(.+?)[,.\s]+(then|next|second|after\s+that|finally)[,\s]+(.+?)\s*[\.!]?\s*$",
    re.IGNORECASE,
)

# 祈使句（指令型）
_IMPERATIVE = re.compile(
    r"^\s*(Please\s+)?(Make\s+sure\s+to|Remember\s+to|Don'?t\s+forget\s+to|"
    r"Always\s+|Never\s+|Try\s+to)\s+(.+?)\s*[\.!]?\s*$",
    re.IGNORECASE,
)

# 时间标记（从句子中提取触发上下文）
_TIME_MARKER = re.compile(
    r"\b(in\s+the\s+(morning|afternoon|evening|night)|"
    r"at\s+(night|dinner|lunch|breakfast|midnight|noon|work|school)|"
    r"on\s+\w+days?|"
    r"during\s+(the\s+)?(\w+)|"
    r"when\s+(I|we)\s+(\w+)|"
    r"after\s+(\w+ing)|"
    r"before\s+(\w+ing))\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════
# 提取函数
# ═══════════════════════════════════════════════════════════════


def extract_procedural(text: str) -> list[ProceduralMemory]:
    """从消息文本中提取程序性记忆。

    Returns:
        空列表 = 未提取到程序性内容；否则返回提取到的记忆列表。
    """
    results: list[ProceduralMemory] = []
    _extract_preferences(text, results)
    _extract_routines(text, results)
    _extract_procedures(text, results)
    return results


def _extract_preferences(text: str, results: list[ProceduralMemory]) -> None:
    # 正面偏好: "I like coffee"
    m = _PREFERENCE_POSITIVE.search(text)
    if m:
        entity = m.group(3).strip()
        if entity:
            # 提取时间/上下文
            trigger = _extract_trigger(text)
            results.append(ProceduralMemory(
                proc_type="preference",
                entity=entity,
                trigger=trigger,
                statement=f"prefers {entity}",
                sentiment="positive",
                confidence=0.9,
            ))
            return

    # 负面偏好: "I dislike spicy food"
    m = _PREFERENCE_NEGATIVE.search(text)
    if m:
        entity = m.group(3).strip()
        if entity:
            trigger = _extract_trigger(text)
            results.append(ProceduralMemory(
                proc_type="preference",
                entity=entity,
                trigger=trigger,
                statement=f"dislikes {entity}",
                sentiment="negative",
                confidence=0.9,
            ))
            return

    # 最爱: "Coffee is my favorite"
    m = _FAVORITE.search(text)
    if m:
        entity = m.group(1).strip()
        if entity:
            trigger = _extract_trigger(text)
            results.append(ProceduralMemory(
                proc_type="preference",
                entity=entity,
                trigger=trigger,
                statement=f"favorite is {entity}",
                sentiment="positive",
                confidence=0.85,
            ))
            return

    # 个人属性: "I am allergic to peanuts"
    m = _PERSONAL_FACT.search(text)
    if m:
        entity = m.group(2).strip()
        if entity:
            results.append(ProceduralMemory(
                proc_type="preference",
                entity=entity,
                trigger="",
                statement=f"allergic to {entity}",
                sentiment="negative",
                confidence=0.8,
            ))
            return

    # 健康状况: "I have high blood pressure"
    m = _PERSONAL_HEALTH.search(text)
    if m:
        condition = m.group(2).strip()
        if condition:
            results.append(ProceduralMemory(
                proc_type="preference",
                entity=condition,
                trigger="",
                statement=f"has {condition}",
                sentiment="neutral",
                confidence=0.7,
            ))
            return


def _extract_routines(text: str, results: list[ProceduralMemory]) -> None:
    # "I always walk the dog"
    m = _ROUTINE_ALWAYS.search(text)
    if m:
        freq = m.group(1).strip()
        activity = m.group(2).strip()
        if activity:
            trigger = _extract_trigger(text)
            results.append(ProceduralMemory(
                proc_type="routine",
                entity=activity,
                trigger=trigger,
                statement=f"{freq} {activity}",
                sentiment="neutral",
                confidence=0.8,
            ))
            return


def _extract_procedures(text: str, results: list[ProceduralMemory]) -> None:
    # "When X, Y"
    m = _TRIGGERED_PROCEDURE.search(text)
    if m:
        condition = m.group(2).strip()
        action = m.group(4).strip()
        if condition and action:
            results.append(ProceduralMemory(
                proc_type="procedure",
                entity=action,
                trigger=condition,
                statement=f"when {condition}, {action}",
                confidence=0.75,
            ))
            return

    # "First X, then Y"
    m = _STEPPED_PROCEDURE.search(text)
    if m:
        step1 = m.group(1).strip()
        step2 = m.group(3).strip()
        if step1 and step2:
            results.append(ProceduralMemory(
                proc_type="procedure",
                entity=step1,
                trigger="",
                statement=f"{step1}, then {step2}",
                confidence=0.7,
                steps=[step1, step2],
            ))
            return

    # 祈使指令
    m = _IMPERATIVE.search(text)
    if m:
        instruction = m.group(2).strip()
        if instruction:
            trigger = _extract_trigger(text)
            results.append(ProceduralMemory(
                proc_type="procedure",
                entity=instruction[:40],
                trigger=trigger,
                statement=instruction,
                confidence=0.6,
            ))
            return


def _extract_trigger(text: str) -> str:
    """从文本中提取时间/上下文触发条件。"""
    m = _TIME_MARKER.search(text)
    if m:
        return m.group(0).strip()
    return ""


# ═══════════════════════════════════════════════════════════════
# 搜索 boosting
# ═══════════════════════════════════════════════════════════════

_PROCEDURAL_QUERY = re.compile(
    r"\b(how\s+(to|do|can|should|would)|"
    r"what\s+(is\s+the\s+(best|proper|right|correct)\s+way|"
    r"should\s+I|do\s+I|do\s+you\s+(prefer|like|recommend))|"
    r"prefer|preference|routine|habit|procedure|step|rule|"
    r"do\s+you\s+(like|enjoy|love|prefer)|"
    r"do\s+I\s+(like|enjoy|love|prefer|eat|drink|do|use)|"
    r"what\s+is\s+your\s+(favorite|routine|habit|preference)|"
    r"what\s+(activities|food|things|books|movies|music|hobbies)\s+(do\s+I|do\s+you))\b",
    re.IGNORECASE,
)


def is_procedural_query(query: str) -> bool:
    """判断查询是否涉及程序性记忆（偏好、习惯、流程）"""
    return bool(_PROCEDURAL_QUERY.search(query))


def procedural_boost(procedural: ProceduralMemory, query: str) -> float:
    """计算程序性记忆的 boosting 分数 (0~1)，用于 rerank 阶段叠加。

    匹配越强、置信度越高，boost 越大。
    """
    q = query.casefold()
    boost = 0.0
    # 主体匹配
    if procedural.entity.casefold() in q:
        boost += 0.3
    # 触发条件匹配
    if procedural.trigger and procedural.trigger.casefold() in q:
        boost += 0.2
    # 陈述匹配
    if procedural.statement.casefold() in q:
        boost += 0.2
    # 如果是偏好/习惯查询，按类型加 base
    if is_procedural_query(query) and procedural.proc_type in ("preference", "routine"):
        boost += 0.15
    return min(boost, 1.0) * procedural.confidence