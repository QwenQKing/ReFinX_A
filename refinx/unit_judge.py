from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict

from .data import load_records
from .annotation import (
    Annotation, AnswerPoint, ContextConstraint, Dependency, Fact, NumericTarget,
)
from .net import LLM_ENDPOINT, ROOT, load_key, post_json
from .schema import (
    Computation, ExtraClaim, NumericValue, Run, ToolCall, UnitJudgment, UnitVerdict,
)


TIMEOUT = 900

_VERDICTS = {
    "constraints": {"satisfied", "violated", "not_applicable"},
    "answer_points": {"supported", "unsupported", "missing"},
    "facts": {"correct", "incorrect", "not_observed"},
    "dependencies": {"valid", "invalid", "not_covered"},
    "extra_claims": {"supported", "unsupported"},
}


def _lines(items):
    return "\n".join(items) or "(none)"


def _trajectory(entries):
    lines = []
    for entry in entries:
        kind, step = entry["type"], entry["step"]
        if kind == "text":
            lines.append(f"[step {step}] agent: {entry['text']}")
        elif kind == "tool_call":
            arguments = json.dumps(entry["arguments"], ensure_ascii=False)
            lines.append(f"[step {step}] call {entry['name']} {arguments}")
        elif kind == "tool_result":

            lines.append(f"[step {step}] result: {entry['result']}")
    return _lines(lines)


def build_prompt(task, run, annotation, template):
    values = {
        "constraints": _lines(
            f"C{index} [{item.kind}] {item.requirement}"
            for index, item in enumerate(annotation.context_constraints, 1)
        ),
        "answer_points": _lines(
            f"{item.id} {item.statement}" for item in annotation.answer_points
        ),
        "facts": _lines(
            f"{item.id} [{item.kind}] {item.statement}" for item in annotation.facts
        ),
        "dependencies": _lines(
            f"{item.id} [{item.relation}] prerequisites "
            f"{','.join(item.prerequisites)}: {item.requirement}"
            for item in annotation.dependencies
        ),
        "numeric_targets": _lines(
            f"N{index} {item.description} ({item.unit})"
            for index, item in enumerate(annotation.numeric_targets, 1)
        ),
        "as_of": task["eval"]["as_of"],
        "query": task["eval"]["query"],
        "trajectory": _trajectory(run.trajectory),
        "final_answer": run.final_answer.strip() or "(no final response)",
    }

    return re.sub(
        r"\{(" + "|".join(values) + r")\}",
        lambda match: values[match.group(1)],
        template,
    )


def _check_items(items, fields, name):
    for item in items:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"{name} entries must have keys {sorted(fields)}")
        if not isinstance(item["reason"], str):
            raise ValueError(f"{name} reason must be a string")


def parse_judgment(text, case_id, annotation, model):
    if not isinstance(text, str):
        raise ValueError("judge content must be a string")
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if fence:
        text = fence.group(1)
    data = json.loads(text)
    expected = {
        "constraints": [
            f"C{i}" for i in range(1, len(annotation.context_constraints) + 1)
        ],
        "answer_points": [item.id for item in annotation.answer_points],
        "facts": [item.id for item in annotation.facts],
        "dependencies": [item.id for item in annotation.dependencies],
        "numeric_values": [
            f"N{i}" for i in range(1, len(annotation.numeric_targets) + 1)
        ],
    }
    if (
        not isinstance(data, dict)
        or set(data) != set(expected) | {"computations", "extra_claims"}
    ):
        raise ValueError("judgment must contain exactly the seven output keys")
    if any(not isinstance(items, list) for items in data.values()):
        raise ValueError("all judgment fields must be lists")
    for name, expected_ids in expected.items():
        field = "value" if name == "numeric_values" else "verdict"
        _check_items(data[name], {"unit_id", field, "reason"}, name)
        ids = [item["unit_id"] for item in data[name]]
        if (
            any(not isinstance(unit_id, str) for unit_id in ids)
            or len(ids) != len(set(ids))
            or set(ids) != set(expected_ids)
        ):
            raise ValueError(f"{name} ids must match the annotation exactly once")
    _check_items(
        data["computations"], {"expression", "stated_result", "reason"},
        "computations",
    )
    if len(data["computations"]) > 10:
        raise ValueError("computations must contain at most 10 entries")
    for item in data["computations"]:
        if not isinstance(item["expression"], str):
            raise ValueError("computations expression must be a string")
        if (
            not isinstance(item["stated_result"], str)
            or re.fullmatch(r"-?\d+(\.\d+)?", item["stated_result"]) is None
        ):
            raise ValueError("computations stated_result must be a decimal string")
    _check_items(
        data["extra_claims"], {"claim", "verdict", "reason"}, "extra_claims"
    )
    if len(data["extra_claims"]) > 5:
        raise ValueError("extra_claims must contain at most 5 entries")
    if any(not isinstance(item["claim"], str) for item in data["extra_claims"]):
        raise ValueError("extra_claims claim must be a string")
    for name, allowed in _VERDICTS.items():
        for item in data[name]:
            if not isinstance(item["verdict"], str) or item["verdict"] not in allowed:
                raise ValueError(f"invalid verdict in {name}")
    for item in data["numeric_values"]:
        value = item["value"]
        if value is not None and not (
            type(value) is int or (type(value) is float and math.isfinite(value))
        ):
            raise ValueError("numeric_values value must be a finite number or null")
    return UnitJudgment(
        case_id=case_id, judge_model=model,
        constraints=[UnitVerdict(**item) for item in data["constraints"]],
        answer_points=[UnitVerdict(**item) for item in data["answer_points"]],
        facts=[UnitVerdict(**item) for item in data["facts"]],
        dependencies=[UnitVerdict(**item) for item in data["dependencies"]],
        numeric_values=[NumericValue(**item) for item in data["numeric_values"]],
        computations=[Computation(**item) for item in data["computations"]],
        extra_claims=[ExtraClaim(**item) for item in data["extra_claims"]],
    )


def judge_run(task, run, annotation, model, template):
    if task is None:
        raise ValueError(f"case {run.case_id} is missing from task data")
    body = {
        "model": model, "temperature": 0, "max_tokens": 32768,
        "messages": [{
            "role": "user",
            "content": build_prompt(task, run, annotation, template),
        }],
    }
    headers = {"Authorization": "Bearer " + load_key("REFINX_LLM_API_KEY", "key.md")}
    for attempt in range(2):

        response = post_json(LLM_ENDPOINT, headers, body, timeout=TIMEOUT)
        try:
            content = response["choices"][0]["message"]["content"] or ""
            return parse_judgment(content, run.case_id, annotation, model)
        except (ValueError, KeyError, TypeError, IndexError):
            if attempt == 1:
                raise


def _read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _annotation_from_record(row):
    row = dict(row)
    row.setdefault("model", "")   
    for name, record_type in (
        ("context_constraints", ContextConstraint),
        ("facts", Fact),
        ("dependencies", Dependency),
        ("answer_points", AnswerPoint),
        ("numeric_targets", NumericTarget),
    ):
        row[name] = [record_type(**item) for item in row[name]]
    return Annotation(**row)


def _write_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Judge each visible run against its refined annotation.')
    parser.add_argument("--run", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--judge-model", default="deepseek-v4.1-flash")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")

    output = ROOT / "runs" / args.run
    judgments_path = output / "unit_judgments.jsonl"
    errors_path = output / "unit_judgments.errors.jsonl"
    annotations = {
        row["case_id"]: _annotation_from_record(row)
        for row in _read_jsonl(ROOT / "annotations" / (args.annotations + ".jsonl"))
    }
    tasks = {row["case_id"]: row for row in load_records(ROOT)}
    template = (ROOT / "refinx/prompts/unit_judge.md").read_text(encoding="utf-8")
    completed = (
        {row["case_id"] for row in _read_jsonl(judgments_path)}
        if args.resume and judgments_path.exists() else set()
    )
    judged = skipped_error = no_annotation = failed = 0
    pending = []

    latest = {row["case_id"]: row for row in _read_jsonl(output / "runs.jsonl")}
    for row in latest.values():
        if row["stop_reason"] == "error":
            skipped_error += 1
            continue
        if row["case_id"] not in annotations:
            no_annotation += 1
            continue
        if row["case_id"] in completed:
            continue
        row["tool_calls"] = [ToolCall(**item) for item in row["tool_calls"]]
        pending.append(Run(**row))

    with ExitStack() as stack:
        judgments_handle = stack.enter_context(
            judgments_path.open("a", encoding="utf-8")
        )
        errors_handle = stack.enter_context(errors_path.open("a", encoding="utf-8"))
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers))
        futures = {
            pool.submit(
                judge_run, tasks.get(run.case_id), run, annotations[run.case_id],
                args.judge_model, template,
            ): run.case_id
            for run in pending
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                judgment = future.result()
            except Exception as exc:
                _write_record(errors_handle, {
                    "case_id": case_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                failed += 1
            else:
                _write_record(judgments_handle, asdict(judgment))
                judged += 1
    print(
        f"judged={judged} skipped_error={skipped_error} "
        f"no_annotation={no_annotation} failed={failed}"
    )


if __name__ == "__main__":
    main()
