"""Deterministic extraction of explicit, evidence-backed profile facts."""
from __future__ import annotations

import re
from dataclasses import dataclass


PREFERENCE_RE = re.compile(
    r"\bI\s+(?:(?P<negative>no\s+longer|do\s+not|don't)\s+)?"
    r"(?P<verb>prefer|like|love|enjoy)\s+(?P<value>.+)",
    flags=re.IGNORECASE,
)
EXCLUSIVE_FACT_PATTERNS = (
    ("location", re.compile(r"\bI\s+live\s+in\s+(?P<value>.+)", re.IGNORECASE)),
    ("employer", re.compile(r"\bI\s+work\s+(?:at|for)\s+(?P<value>.+)", re.IGNORECASE)),
    ("name", re.compile(r"\bmy\s+name\s+is\s+(?P<value>.+)", re.IGNORECASE)),
)
QUALIFIER_RE = re.compile(
    r"\s+(?:in|when|while|on|at|for|because|but)\b.*$", flags=re.IGNORECASE
)
WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Claim:
    """A minimal, rule-derived statement that always points back to one message."""

    predicate: str
    value_key: str
    polarity: int
    exclusive: bool = False


def extract_claims(content: str) -> tuple[Claim, ...]:
    """Extract explicit preferences and a narrow set of profile facts."""

    match = PREFERENCE_RE.search(content)
    if match is not None:
        value_key = _value_key(match.group("value"))
        if value_key:
            polarity = -1 if match.group("negative") else 1
            return (Claim(predicate="preference", value_key=value_key, polarity=polarity),)
    return _extract_exclusive_fact(content)


def _extract_exclusive_fact(content: str) -> tuple[Claim, ...]:
    for predicate, pattern in EXCLUSIVE_FACT_PATTERNS:
        match = pattern.search(content)
        if match is None:
            continue
        value_key = _value_key(match.group("value"))
        if value_key:
            return (Claim(predicate=predicate, value_key=value_key, polarity=1, exclusive=True),)
    return ()


def _value_key(value: str) -> str:
    unqualified = QUALIFIER_RE.sub("", value).strip().rstrip(".!?;:")
    return " ".join(WORD_RE.findall(unqualified.casefold()))
