from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Task:


    case_id: str
    query: str
    as_of: str
    language: str

    terminal_action: str
    gold_answer: str
    accepted_alternatives: list[str]
    required_clarifications: list[str]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str
    ok: bool


@dataclass
class Run:


    case_id: str
    model: str
    final_answer: str
    stop_reason: str
    tool_calls: list[ToolCall]


    trajectory: list[dict[str, Any]]
    model_turns: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Judgment:


    case_id: str
    correct: Optional[bool]
    reason: str


@dataclass
class UnitVerdict:


    unit_id: str
    verdict: str
    reason: str


@dataclass
class NumericValue:


    unit_id: str
    value: Optional[float]
    reason: str


@dataclass
class Computation:


    expression: str
    stated_result: str
    reason: str


@dataclass
class ExtraClaim:


    claim: str
    verdict: str
    reason: str


@dataclass
class UnitJudgment:


    case_id: str
    judge_model: str
    constraints: list[UnitVerdict]

    answer_points: list[UnitVerdict]
    facts: list[UnitVerdict]
    dependencies: list[UnitVerdict]
    numeric_values: list[NumericValue]
    computations: list[Computation]
    extra_claims: list[ExtraClaim]


@dataclass
class AnsAccSummary:


    model: str
    judge_model: str
    applicable: int
    unobservable: int
    undecided: int
    adjudicated: int
    correct: int
    ans_acc: Optional[float]
    coverage: float
