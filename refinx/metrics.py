from __future__ import annotations

import argparse
import json

from .data import load_records
from .net import ROOT
from .tools import evaluate_expression


METRICS = (
    "CtxSat", "ReqSat", "MainSupp", "FactAcc", "ArithAcc", "ReplayAcc", "NumAcc",
    "EM", "DepCov", "DepValid", "LCDF", "ExtraSupp", "RespSupp_main",
    "RespSupp_all", "CSG_main", "CSG_all", "AnsAcc",
)
LANGUAGE_METRICS = (
    "AnsAcc", "CtxSat", "FactAcc", "DepCov", "DepValid", "MainSupp",
)


class _Metric:
    def __init__(self):
        self.units = 0
        self.adjudicated = 0
        self.undecided = 0
        self.unobservable = 0
        self.positive = 0

    def add(self, success=None, *, count=1, unobservable=False):
        self.units += count
        if unobservable:
            self.unobservable += count
        elif success is None:
            self.undecided += count
        else:
            self.adjudicated += count
            if success:
                self.positive += count

    def result(self):
        return {
            "value": self.positive / self.adjudicated if self.adjudicated else None,
            "units": self.units,
            "adjudicated": self.adjudicated,
            "undecided": self.undecided,
            "unobservable": self.unobservable,
            "coverage": self.adjudicated / self.units if self.units else None,
        }


def question_language(query):
    characters = [character for character in query if not character.isspace()]
    cjk = sum("一" <= character <= "鿿" for character in characters)
    return "zh" if 5 * cjk > len(characters) else "en"


def _verdicts(items):
    return {item["unit_id"]: item["verdict"] for item in items}


def _replay_correct(computation):
    try:
        replayed = evaluate_expression(computation["expression"])
    except Exception:
        return None
    if type(replayed) not in (int, float):
        return None
    stated_result = computation["stated_result"]
    stated = float(stated_result)
    decimals = len(stated_result.partition(".")[2])


    tolerance = max(0.5 * 10 ** (-decimals), 0.001 * abs(stated))
    tolerance += 1e-9 * max(1, abs(stated))
    return abs(replayed - stated) <= tolerance


def _workflow_reliability(result):
    dimensions = {}
    for dimension, names in (
        ("D1", ("CtxSat", "ReqSat")),
        ("D2", ("DepValid", "DepCov")),
        ("D3", ("FactAcc", "ArithAcc", "ReplayAcc", "NumAcc", "EM", "AnsAcc")),
        ("D4", ("LCDF",)),
        ("D5", (
            "MainSupp", "ExtraSupp", "RespSupp_main", "RespSupp_all",
            "CSG_main", "CSG_all",
        )),
        ("D6", ("CER_1", "RD")),
    ):
        values = []
        for name in names:
            value = (
                result["D6"][name] if name in ("CER_1", "RD")
                else result[name]["value"]
            )
            if value is None:
                continue
            if name == "RD":
                value = 1 - abs(value)
            elif name in ("LCDF", "CSG_main", "CSG_all", "CER_1"):
                value = 1 - value
            values.append(value)
        dimensions[dimension] = sum(values) / len(values) if values else None
    available = [value for value in dimensions.values() if value is not None]
    return {
        "score": sum(available) / len(available) if available else None,
        "dimensions": dimensions,
    }


def aggregate(runs, judgments, unit_judgments, annotations, queries):


    metrics = {name: _Metric() for name in METRICS}
    by_language = {
        language: {name: _Metric() for name in LANGUAGE_METRICS}
        for language in ("zh", "en")
    }
    by_relation = {}
    d6 = {True: [0, 0], False: [0, 0]}

    for case_id, run in runs.items():
        language = question_language(queries[case_id])

        def record(name, success=None, *, count=1, unobservable=False):
            metrics[name].add(success, count=count, unobservable=unobservable)
            if name in LANGUAGE_METRICS:
                by_language[language][name].add(
                    success, count=count, unobservable=unobservable,
                )

        error = run["stop_reason"] == "error"
        correct = judgments.get(case_id, {}).get("correct")
        decided = correct is True or correct is False
        record("AnsAcc", correct if decided else None, unobservable=error)

        annotation = annotations.get(case_id)
        if annotation is None:
            continue
        judgment = unit_judgments.get(case_id)
        points = annotation["answer_points"]
        dependencies = annotation["dependencies"]
        for dependency in dependencies:
            by_relation.setdefault(dependency["relation"], _Metric())

        if error or judgment is None:


            candidates = {
                "CtxSat": len(annotation["context_constraints"]),
                "ReqSat": len(points),
                "MainSupp": len(points),
                "FactAcc": sum(fact["kind"] == "fact" for fact in annotation["facts"]),
                "ArithAcc": sum(
                    fact["kind"] == "arithmetic" for fact in annotation["facts"]
                ),
                "NumAcc": len(annotation["numeric_targets"]),
                "DepCov": len(dependencies),
                "DepValid": len(dependencies),
                "LCDF": len(dependencies),
                "RespSupp_main": int(bool(points)),
                "RespSupp_all": int(bool(points)),
            }
            for name, count in candidates.items():
                record(name, count=count, unobservable=error)
            for dependency in dependencies:
                by_relation[dependency["relation"]].add(unobservable=error)
            if judgment is not None:
                record(
                    "ExtraSupp", count=len(judgment["extra_claims"]),
                    unobservable=True,
                )
            continue

        constraints = _verdicts(judgment["constraints"])
        answers = _verdicts(judgment["answer_points"])
        facts = _verdicts(judgment["facts"])
        dependency_verdicts = _verdicts(judgment["dependencies"])
        numbers = {
            item["unit_id"]: item["value"] for item in judgment["numeric_values"]
        }
        extras = judgment["extra_claims"]

        for index, _ in enumerate(annotation["context_constraints"], 1):
            verdict = constraints[f"C{index}"]
            record(
                "CtxSat", verdict == "satisfied",
                unobservable=verdict == "not_applicable",
            )

        delivered = []
        for point in points:
            verdict = answers[point["id"]]
            given = verdict in ("supported", "unsupported")
            record("ReqSat", given)
            if given:
                delivered.append(verdict)
                record("MainSupp", verdict == "supported")

        for fact in annotation["facts"]:
            verdict = facts[fact["id"]]
            name = {"fact": "FactAcc", "arithmetic": "ArithAcc"}[fact["kind"]]
            record(
                name, verdict == "correct",
                unobservable=verdict == "not_observed",
            )

        for computation in judgment["computations"]:
            record("ReplayAcc", _replay_correct(computation))

        for index, target in enumerate(annotation["numeric_targets"], 1):
            value = numbers[f"N{index}"]
            tolerance = target["tolerance"] + 1e-9 * max(1, abs(target["value"]))
            record(
                "NumAcc",
                value is not None and abs(value - target["value"]) <= tolerance,
            )

        eligible_dependencies = []
        for dependency in dependencies:
            verdict = dependency_verdicts[dependency["id"]]
            covered = verdict in ("valid", "invalid")
            record("DepCov", covered)
            if covered:
                record("DepValid", verdict == "valid")
                by_relation[dependency["relation"]].add(verdict == "valid")
                if all(
                    facts.get(fact_id) == "correct"
                    for fact_id in dependency["prerequisites"]
                ):
                    eligible_dependencies.append(verdict)
                    record("LCDF", verdict == "invalid")

        for claim in extras:
            record("ExtraSupp", claim["verdict"] == "supported")
        main_supported = all(verdict == "supported" for verdict in delivered)
        extra_supported = all(claim["verdict"] == "supported" for claim in extras)
        if delivered:
            record("RespSupp_main", main_supported)
            record("RespSupp_all", main_supported and extra_supported)
        if correct is True:
            record("CSG_main", not main_supported)
            record("CSG_all", not (main_supported and extra_supported))


        if eligible_dependencies and decided:
            group = d6["invalid" in eligible_dependencies]
            group[0] += int(correct is False)
            group[1] += 1

    error_rates = {
        failed: errors / count if count else None
        for failed, (errors, count) in d6.items()
    }
    failed_rate = error_rates[True]
    not_failed_rate = error_rates[False]
    result = {name: metric.result() for name, metric in metrics.items()}
    result["DepValid_by_relation"] = {
        relation: metric.result() for relation, metric in sorted(by_relation.items())
    }
    result["D6"] = {
        "CER_1": failed_rate,
        "CER_0": not_failed_rate,
        "RD": (
            failed_rate - not_failed_rate
            if failed_rate is not None and not_failed_rate is not None else None
        ),
        "n_failed": d6[True][1],
        "n_not_failed": d6[False][1],
    }
    result["by_language"] = {
        language: {name: metric.result() for name, metric in group.items()}
        for language, group in by_language.items()
    }
    result["WRS"] = _workflow_reliability(result)
    return result


def _load_latest(path, *, optional=False):
    try:
        with path.open(encoding="utf-8") as handle:
            records = {}
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    records[row["case_id"]] = row
            return records
    except FileNotFoundError:
        if optional:
            return {}
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--annotations", required=True)
    args = parser.parse_args(argv)
    output = ROOT / "runs" / args.run
    runs = _load_latest(output / "runs.jsonl")
    judgments = _load_latest(output / "judgments.jsonl", optional=True)
    unit_judgments = _load_latest(output / "unit_judgments.jsonl", optional=True)
    annotations = _load_latest(ROOT / "annotations" / f"{args.annotations}.jsonl")
    queries = {row["case_id"]: row["eval"]["query"] for row in load_records(ROOT)}
    result = aggregate(runs, judgments, unit_judgments, annotations, queries)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (output / "metrics.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
