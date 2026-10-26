import io
import json
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from refinx import metrics
from refinx.run import evaluate
from refinx.schema import Run, Task
from refinx.tools import calculate, evaluate_expression


def _annotation(case_id, constraints, answers, facts, dependencies, numbers):
    return {
        "case_id": case_id,
        "model": "offline-annotator",
        "context_constraints": [
            {"kind": "entity", "requirement": f"constraint {index}"}
            for index in range(1, constraints + 1)
        ],
        "answer_points": [
            {"id": f"A{index}", "statement": f"answer {index}"}
            for index in range(1, answers + 1)
        ],
        "facts": [
            {"id": f"F{index}", "kind": kind, "statement": f"fact {index}"}
            for index, kind in enumerate(facts, 1)
        ],
        "dependencies": [
            {
                "id": f"D{index}", "relation": relation,
                "prerequisites": prerequisites, "requirement": "use the facts",
            }
            for index, (relation, prerequisites) in enumerate(dependencies, 1)
        ],
        "numeric_targets": [
            {
                "description": f"number {index}", "value": value,
                "unit": "USD", "tolerance": tolerance,
            }
            for index, (value, tolerance) in enumerate(numbers, 1)
        ],
    }


def _unit(
    case_id, constraints, answers, facts, dependencies, numbers, extras,
    computations=(),
):
    def verdicts(prefix, labels):
        return [
            {"unit_id": f"{prefix}{index}", "verdict": label, "reason": "offline"}
            for index, label in enumerate(labels, 1)
        ]

    return {
        "case_id": case_id,
        "judge_model": "offline-judge",
        "constraints": verdicts("C", constraints),
        "answer_points": verdicts("A", answers),
        "facts": verdicts("F", facts),
        "dependencies": verdicts("D", dependencies),
        "numeric_values": [
            {"unit_id": f"N{index}", "value": value, "reason": "offline"}
            for index, value in enumerate(numbers, 1)
        ],
        "computations": [
            {
                "expression": expression, "stated_result": stated_result,
                "reason": "offline",
            }
            for expression, stated_result in computations
        ],
        "extra_claims": [
            {"claim": f"extra {index}", "verdict": label, "reason": "offline"}
            for index, label in enumerate(extras, 1)
        ],
    }


def _run(case_id, stop_reason="final"):
    return asdict(Run(
        case_id=case_id, model="offline-agent", final_answer="answer",
        stop_reason=stop_reason, tool_calls=[], trajectory=[], model_turns=1,
        latency_s=0.0, prompt_tokens=0, completion_tokens=0,
    ))


class MetricsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "runs" / "demo"
        self.annotation_path = self.root / "annotations" / "refined.jsonl"
        self.data_path = (
            self.root / "paper/public-data/ReFInx_data/refinx_data.json"
        )
        self.annotations = [
            _annotation(
                "zh", 3, 3,
                ["fact", "fact", "fact", "arithmetic", "arithmetic", "arithmetic"],
                [
                    ("entity", ["F1"]), ("entity", ["F1"]),
                    ("time", ["F1", "F2"]), ("time", ["F1"]),
                    ("unit_currency", ["F1", "F3"]), ("input_result_use", ["F4"]),
                ],
                [(100.0, 0.1), (20.0, 0.0), (-10.0, 0.01)],
            ),
            _annotation(
                "en", 1, 2, ["fact", "arithmetic"],
                [
                    ("entity", ["F1", "F2"]), ("metric_scope", ["F1"]),
                    ("evidential_support", []),
                ],
                [(0.0, 0.0), (-200.0, 0.2)],
            ),
            _annotation(
                "error", 1, 1, ["fact", "arithmetic"],
                [("entity", ["F1"])], [(0.0, 0.0)],
            ),
            _annotation(
                "pending", 1, 1, ["fact", "arithmetic"],
                [("time", ["F1"])], [(0.0, 0.0)],
            ),
        ]
        self.runs = [
            _run("zh"), _run("en"), _run("error", "error"), _run("pending"),
        ]
        self.judgments = [
            {"case_id": case_id, "correct": correct, "reason": "offline"}
            for case_id, correct in (
                ("zh", True), ("en", False), ("error", True), ("pending", None),
            )
        ]
        self.units = [
            _unit(
                "zh", ["satisfied", "violated", "not_applicable"],
                ["supported", "unsupported", "missing"],
                [
                    "correct", "incorrect", "not_observed",
                    "correct", "incorrect", "not_observed",
                ],
                ["invalid", "valid", "invalid", "not_covered", "valid", "valid"],
                [100.10000005, None, -10.011], ["supported", "unsupported"],
            ),
            _unit(
                "en", ["satisfied"], ["supported", "missing"],
                ["correct", "correct"], ["valid", "not_covered", "valid"],
                [1e-9, -200.20000015], ["supported"],
            ),
            _unit(
                "error", ["satisfied"], ["supported"], ["correct", "correct"],
                ["valid"], [0.0], ["supported"],
            ),
        ]

        for judgment in self.units:
            for field in (
                "constraints", "answer_points", "facts", "dependencies",
                "numeric_values",
            ):
                judgment[field].reverse()
        self.data = [
            {
                "case_id": case_id, "classification": {"language": "en"},
                "eval": {"query": query},
            }
            for case_id, query in (
                ("zh", "这个公司的利润是多少？"), ("en", "What is the profit?"),
                ("error", "Which result failed?"),
                ("pending", "What is the missing result?"),
            )
        ]
        self._write_inputs()

    def _write_jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        path.write_text(text + "\n\n", encoding="utf-8")

    def _write_inputs(self):

        runs = deepcopy(self.runs)
        if runs:
            runs.insert(0, dict(runs[0], stop_reason="error"))
        annotations = deepcopy(self.annotations)
        if annotations:
            annotations.insert(0, dict(annotations[0], answer_points=[]))
        judgments = deepcopy(self.judgments)
        if judgments:
            judgments.insert(0, dict(judgments[0], correct=False))
        units = deepcopy(self.units)
        if units:
            stale = deepcopy(units[0])
            for number in stale["numeric_values"]:
                number["value"] = None
            units.insert(0, stale)
        self._write_jsonl(self.output / "runs.jsonl", runs)
        self._write_jsonl(self.output / "judgments.jsonl", judgments)
        self._write_jsonl(self.output / "unit_judgments.jsonl", units)
        self._write_jsonl(self.annotation_path, annotations)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        data = [dict(self.data[0], eval={"query": "An old English query."})]
        self.data_path.write_text(
            json.dumps(data + self.data, ensure_ascii=False), encoding="utf-8",
        )

    def _calculate(self):
        stdout = io.StringIO()
        with patch.object(metrics, "ROOT", self.root), redirect_stdout(stdout):
            metrics.main(["--run", "demo", "--annotations", "refined"])
        saved = (self.output / "metrics.json").read_text(encoding="utf-8")
        self.assertEqual(stdout.getvalue(), saved)
        return json.loads(saved)

    def assert_rate(self, actual, positive, units, adjudicated, undecided, unobservable):
        self.assertEqual(set(actual), {
            "value", "units", "adjudicated", "undecided", "unobservable", "coverage",
        })
        self.assertEqual(actual["units"], units)
        self.assertEqual(actual["adjudicated"], adjudicated)
        self.assertEqual(actual["undecided"], undecided)
        self.assertEqual(actual["unobservable"], unobservable)
        self.assertEqual(units, adjudicated + undecided + unobservable)
        if adjudicated:
            self.assertAlmostEqual(actual["value"], positive / adjudicated)
        else:
            self.assertIsNone(actual["value"])
        if units:
            self.assertAlmostEqual(actual["coverage"], adjudicated / units)
        else:
            self.assertIsNone(actual["coverage"])

    def test_metric_counts_and_latest_records(self):
        result = self._calculate()
        expected = {
            "CtxSat": (2, 6, 3, 1, 2),
            "ReqSat": (3, 7, 5, 1, 1),
            "MainSupp": (2, 5, 3, 1, 1),
            "FactAcc": (2, 6, 3, 1, 2),
            "ArithAcc": (2, 6, 3, 1, 2),
            "ReplayAcc": (0, 0, 0, 0, 0),
            "NumAcc": (3, 7, 5, 1, 1),
            "EM": (0, 0, 0, 0, 0),
            "DepCov": (7, 11, 9, 1, 1),
            "DepValid": (5, 9, 7, 1, 1),
            "LCDF": (1, 7, 5, 1, 1),
            "ExtraSupp": (2, 4, 3, 0, 1),
            "RespSupp_main": (1, 4, 2, 1, 1),
            "RespSupp_all": (1, 4, 2, 1, 1),
            "CSG_main": (1, 1, 1, 0, 0),
            "CSG_all": (1, 1, 1, 0, 0),
            "AnsAcc": (1, 4, 2, 1, 1),
        }
        self.assertEqual(
            set(result),
            set(expected) | {"D6", "WRS", "DepValid_by_relation", "by_language"},
        )
        for name, counts in expected.items():
            with self.subTest(metric=name):
                self.assert_rate(result[name], *counts)
        self.assertEqual(result["D6"], {
            "CER_1": 0.0, "CER_0": 1.0, "RD": -1.0,
            "n_failed": 1, "n_not_failed": 1,
        })

    def test_replay_tolerance_uses_stated_decimals_or_relative_floor(self):
        for expression, stated_result, correct in (
            ("(4.794 / 2.898 - 1) * 100", "65.4", True),
            ("65.5", "65.4", False),
            ("65.4", "65", True),
            ("5797.1", "5800", True),  
            ("5790", "5800", False),
            ("1.0009", "1.000", True),  
            ("1.0011", "1.000", False),
        ):
            with self.subTest(expression=expression, stated_result=stated_result):
                self.units[0]["computations"] = [{
                    "expression": expression, "stated_result": stated_result,
                    "reason": "offline",
                }]
                self._write_inputs()
                result = self._calculate()
                self.assert_rate(result["ReplayAcc"], int(correct), 1, 1, 0, 0)

    def test_replay_errors_and_complex_results_are_undecided(self):
        self.units[0]["computations"] = [
            {"expression": expression, "stated_result": "4", "reason": "offline"}
            for expression in (
                "1 / 0", "unknown + 1", "1 +", "(-1) ** 0.5", "2 + 2",
            )
        ]
        self._write_inputs()
        result = self._calculate()
        self.assert_rate(result["ReplayAcc"], 1, 5, 1, 4, 0)

    def test_replay_only_counts_recorded_nonerror_computations(self):
        computation = {
            "expression": "2 + 2", "stated_result": "4", "reason": "offline",
        }
        self.units[0]["computations"] = [computation]
        self.units[2]["computations"] = [computation, computation]
        self._write_inputs()
        self.assert_rate(self._calculate()["ReplayAcc"], 1, 1, 1, 0, 0)

    def test_relation_and_question_language_groups(self):
        result = self._calculate()
        relations = {
            "entity": (2, 4, 3, 0, 1),
            "time": (0, 2, 1, 1, 0),
            "metric_scope": (0, 0, 0, 0, 0),
            "unit_currency": (1, 1, 1, 0, 0),
            "input_result_use": (1, 1, 1, 0, 0),
            "evidential_support": (1, 1, 1, 0, 0),
        }
        self.assertEqual(set(result["DepValid_by_relation"]), set(relations))
        for relation, counts in relations.items():
            with self.subTest(relation=relation):
                self.assert_rate(result["DepValid_by_relation"][relation], *counts)
        languages = {
            "zh": {
                "AnsAcc": (1, 1, 1, 0, 0), "CtxSat": (1, 3, 2, 0, 1),
                "FactAcc": (1, 3, 2, 0, 1), "DepCov": (5, 6, 6, 0, 0),
                "DepValid": (3, 5, 5, 0, 0), "MainSupp": (1, 2, 2, 0, 0),
            },
            "en": {
                "AnsAcc": (0, 3, 1, 1, 1), "CtxSat": (1, 3, 1, 1, 1),
                "FactAcc": (1, 3, 1, 1, 1), "DepCov": (2, 5, 3, 1, 1),
                "DepValid": (2, 4, 2, 1, 1), "MainSupp": (1, 3, 1, 1, 1),
            },
        }
        self.assertEqual(set(result["by_language"]), set(languages))
        for language, expected in languages.items():
            self.assertEqual(set(result["by_language"][language]), set(expected))
            for name, counts in expected.items():
                with self.subTest(language=language, metric=name):
                    self.assert_rate(result["by_language"][language][name], *counts)

    def test_extra_claims_distinguish_main_and_all(self):
        for point in self.units[0]["answer_points"]:
            if point["unit_id"] == "A2":
                point["verdict"] = "supported"
        self._write_inputs()
        result = self._calculate()
        self.assert_rate(result["RespSupp_main"], 2, 4, 2, 1, 1)
        self.assert_rate(result["RespSupp_all"], 1, 4, 2, 1, 1)
        self.assert_rate(result["CSG_main"], 0, 1, 1, 0, 0)
        self.assert_rate(result["CSG_all"], 1, 1, 1, 0, 0)

    def test_undelivered_and_unused_units_are_excluded(self):
        for judgment in self.units[:2]:
            for point in judgment["answer_points"]:
                point["verdict"] = "missing"
            for dependency in judgment["dependencies"]:
                dependency["verdict"] = "not_covered"
        self._write_inputs()
        result = self._calculate()
        self.assert_rate(result["ReqSat"], 0, 7, 5, 1, 1)
        self.assert_rate(result["DepCov"], 0, 11, 9, 1, 1)
        for name in (
            "MainSupp", "DepValid", "LCDF", "RespSupp_main", "RespSupp_all",
        ):
            self.assert_rate(result[name], 0, 2, 0, 1, 1)
        self.assert_rate(result["CSG_main"], 0, 1, 1, 0, 0)
        self.assert_rate(result["CSG_all"], 1, 1, 1, 0, 0)
        self.assertEqual(result["D6"], {
            "CER_1": None, "CER_0": None, "RD": None,
            "n_failed": 0, "n_not_failed": 0,
        })

    def test_csg_requires_a_unit_judgment(self):
        self.judgments[3]["correct"] = True
        self._write_inputs()
        result = self._calculate()
        self.assert_rate(result["AnsAcc"], 2, 4, 3, 0, 1)
        self.assert_rate(result["CSG_main"], 1, 1, 1, 0, 0)
        self.assert_rate(result["CSG_all"], 1, 1, 1, 0, 0)
        self.assertEqual(result["D6"]["n_not_failed"], 1)

    def test_d6_requires_decided_ansacc_and_both_groups(self):
        self.judgments[1]["correct"] = None
        for correct in (True, False):
            with self.subTest(correct=correct):
                self.judgments[0]["correct"] = correct
                self._write_inputs()
                result = self._calculate()
                self.assertEqual(result["D6"], {
                    "CER_1": float(not correct), "CER_0": None, "RD": None,
                    "n_failed": 1, "n_not_failed": 0,
                })
                self.assertEqual(result["WRS"]["dimensions"]["D6"], float(correct))

    def test_d6_uses_only_lcdf_eligible_dependencies(self):

        for include_eligible, count, error_rate in (
            (False, 1, 1.0), (True, 2, 0.5),
        ):
            with self.subTest(include_eligible=include_eligible):
                for dependency in self.units[0]["dependencies"]:
                    if dependency["unit_id"] == "D3":
                        dependency["verdict"] = "invalid"
                    elif dependency["unit_id"] == "D2" and include_eligible:
                        dependency["verdict"] = "valid"
                    else:
                        dependency["verdict"] = "not_covered"
                self._write_inputs()
                result = self._calculate()
                self.assertEqual(result["D6"], {
                    "CER_1": None, "CER_0": error_rate, "RD": None,
                    "n_failed": 0, "n_not_failed": count,
                })

    def test_wrs_aligns_directions_and_weights_dimensions(self):
        result = self._calculate()
        expected = {
            "D1": 19 / 30, "D2": 47 / 63, "D3": 73 / 120,
            "D4": 4 / 5, "D5": 7 / 18, "D6": 1 / 2,
        }
        self.assertEqual(set(result["WRS"]), {"score", "dimensions"})
        self.assertEqual(set(result["WRS"]["dimensions"]), set(expected))
        for dimension, value in expected.items():
            with self.subTest(dimension=dimension):
                self.assertAlmostEqual(result["WRS"]["dimensions"][dimension], value)
        self.assertAlmostEqual(result["WRS"]["score"], sum(expected.values()) / 6)

    def test_wrs_aligns_positive_rd(self):
        self.judgments[0]["correct"] = False
        self.judgments[1]["correct"] = True
        self._write_inputs()
        result = self._calculate()
        self.assertEqual(result["D6"], {
            "CER_1": 1.0, "CER_0": 0.0, "RD": 1.0,
            "n_failed": 1, "n_not_failed": 1,
        })
        self.assertEqual(result["WRS"]["dimensions"]["D6"], 0.0)

    def test_wrs_skips_null_metrics_and_dimensions(self):
        result = metrics.aggregate(
            {"only": _run("only")},
            {"only": {"correct": False}},
            {"only": _unit(
                "only", [], [], ["correct"], [], [], [],
                computations=[("2 + 2", "5")],
            )},
            {"only": _annotation("only", 0, 0, ["fact"], [], [])},
            {"only": "Question"},
        )
        self.assert_rate(result["EM"], 0, 0, 0, 0, 0)
        for name in ("ArithAcc", "NumAcc", "EM"):
            self.assertIsNone(result[name]["value"])
        self.assertEqual(result["WRS"]["dimensions"], {
            "D1": None, "D2": None, "D3": 1 / 3,
            "D4": None, "D5": None, "D6": None,
        })
        self.assertAlmostEqual(result["WRS"]["score"], 1 / 3)

    def test_missing_judgment_files(self):
        (self.output / "judgments.jsonl").unlink()
        (self.output / "unit_judgments.jsonl").unlink()
        result = self._calculate()
        self.assert_rate(result["AnsAcc"], 0, 4, 0, 3, 1)
        self.assert_rate(result["CtxSat"], 0, 6, 0, 5, 1)
        self.assert_rate(result["MainSupp"], 0, 7, 0, 6, 1)
        self.assert_rate(result["DepValid"], 0, 11, 0, 10, 1)
        self.assert_rate(result["ExtraSupp"], 0, 0, 0, 0, 0)
        self.assert_rate(result["ReplayAcc"], 0, 0, 0, 0, 0)
        self.assert_rate(result["CSG_main"], 0, 0, 0, 0, 0)
        self.assertIsNone(result["WRS"]["score"])

    def test_unannotated_runs_only_contribute_to_ansacc(self):
        self.annotations = []
        self._write_inputs()
        result = self._calculate()
        self.assert_rate(result["AnsAcc"], 1, 4, 2, 1, 1)
        for name in metrics.METRICS:
            if name != "AnsAcc":
                self.assert_rate(result[name], 0, 0, 0, 0, 0)
        self.assertEqual(result["DepValid_by_relation"], {})

    def test_empty_run(self):
        self.runs = []
        self._write_inputs()
        result = self._calculate()
        for name in metrics.METRICS:
            self.assert_rate(result[name], 0, 0, 0, 0, 0)
        for group in result["by_language"].values():
            for metric in group.values():
                self.assert_rate(metric, 0, 0, 0, 0, 0)
        self.assertIsNone(result["D6"]["RD"])
        self.assertEqual(result["WRS"], {
            "score": None,
            "dimensions": dict.fromkeys(("D1", "D2", "D3", "D4", "D5", "D6")),
        })

    def test_language_threshold_ignores_whitespace(self):
        for query, expected in (
            ("中abcd", "en"), (" 中abc \t", "zh"), ("一鿿abc", "zh"),
            ("㐀abcd", "en"), ("", "en"), (" \t\r\n", "en"),
        ):
            with self.subTest(query=query):
                self.assertEqual(metrics.question_language(query), expected)


class ExpressionEvaluatorTests(unittest.TestCase):
    def test_numeric_results_and_calculate_error_text_are_preserved(self):
        for expression, expected in (
            ("2 ** 3 + 7 % 4", 11),
            ("round(sqrt(2), 3)", 1.414),
        ):
            with self.subTest(expression=expression):
                self.assertEqual(evaluate_expression(expression), expected)
                self.assertEqual(calculate(expression), str(expected))
        with self.assertRaisesRegex(ZeroDivisionError, "division by zero"):
            evaluate_expression("1 / 0")
        self.assertEqual(
            calculate("1 / 0"), "Error: ZeroDivisionError: division by zero",
        )


class EvaluateWhitespaceTests(unittest.TestCase):
    def test_whitespace_final_answer_does_not_call_judge(self):
        task = Task(
            case_id="whitespace", query="Question", as_of="", language="en",
            terminal_action="answer", gold_answer="42", accepted_alternatives=[],
            required_clarifications=[],
        )
        run = Run(**_run(task.case_id))
        for answer in ("", " \t\r\n", "\u3000 \n"):
            with self.subTest(answer=repr(answer)):
                run.final_answer = answer
                with patch("refinx.run.judge_answer") as judge:
                    returned, judgment = evaluate(
                        task, "offline-agent", "offline-judge", run=run,
                    )
                    judge.assert_not_called()
                self.assertIs(returned, run)
                self.assertEqual(judgment.case_id, task.case_id)
                self.assertIs(judgment.correct, False)
                self.assertEqual(judgment.reason, "no final answer")


if __name__ == "__main__":
    unittest.main()
