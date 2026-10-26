from __future__ import annotations

from dataclasses import dataclass

CONSTRAINT_KINDS = ("entity", "time", "period", "metric", "accounting_scope", "unit_currency", "assumption")
FACT_KINDS = ("fact", "arithmetic")
RELATIONS = ("entity", "time", "metric_scope", "unit_currency", "input_result_use", "evidential_support")


@dataclass
class ContextConstraint:
    kind: str
    requirement: str


@dataclass
class Fact:
    id: str
    kind: str
    statement: str


@dataclass
class Dependency:
    id: str
    relation: str
    prerequisites: list[str]
    requirement: str


@dataclass
class AnswerPoint:
    id: str
    statement: str


@dataclass
class NumericTarget:
    description: str
    value: float
    unit: str
    tolerance: float


@dataclass
class Annotation:
    case_id: str
    model: str
    context_constraints: list[ContextConstraint]
    facts: list[Fact]
    dependencies: list[Dependency]
    answer_points: list[AnswerPoint]
    numeric_targets: list[NumericTarget]
