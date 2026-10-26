from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from refinx import unit_judge
from refinx.annotation import (
    Annotation, AnswerPoint, ContextConstraint, Dependency, Fact, NumericTarget,
)
from refinx.schema import (
    Computation, ExtraClaim, NumericValue, Run, ToolCall, UnitJudgment, UnitVerdict,
)

TEMPLATE = """{constraints}
{answer_points}
{facts}
{dependencies}
{numeric_targets}
{as_of}
{query}
{trajectory}
{final_answer}
{"numeric_values": [{"unit_id": "N1", "value": null, "reason": "..."}]}
"""
UNIT_FIELDS = (
    "constraints", "answer_points", "facts", "dependencies", "numeric_values",
)


def enter(test, cm):
    value = cm.__enter__()
    test.addCleanup(cm.__exit__, None, None, None)
    return value


def task(case_id="case-a"):
    return {
        "case_id": case_id,
        "eval": {
            "query": "What is the revenue growth?",
            "as_of": "2025-01-01T00:00:00Z",
        },
    }


def annotation(case_id="case-a"):
    return Annotation(
        case_id=case_id, model="offline-annotator",
        context_constraints=[
            ContextConstraint("period", "Use the reported period"),
            ContextConstraint("unit_currency", "Use CNY"),
        ],
        facts=[
            Fact("F2", "fact", "Revenue rose from 100 to 120."),
            Fact("F9", "arithmetic", "(120 / 100 - 1) * 100 = 20."),
        ],
        dependencies=[
            Dependency(
                "D3", "input_result_use", ["F2", "F9"],
                "Use both reported revenues",
            ),
        ],
        answer_points=[AnswerPoint("A7", "Revenue increased by 20%.")],
        numeric_targets=[
            NumericTarget("Revenue growth", 20, "%", 0.1),
            NumericTarget("Current revenue", 120, "CNY", 0.1),
        ],
    )


def run_record(case_id="case-a", stop_reason="final"):
    return Run(
        case_id=case_id, model="offline-agent",
        final_answer="Revenue increased by 20%; revenue was positive.",
        stop_reason=stop_reason,
        tool_calls=[
            ToolCall("search", {"query": "收入"}, "Revenue was 100, then 120.", True),
        ],
        trajectory=[
            {"step": 1, "type": "thinking", "text": "PRIVATE_THINKING"},
            {"step": 1, "type": "text", "text": "Compare the reported revenues."},
            {"step": 1, "type": "tool_call", "name": "search",
             "arguments": {"query": "收入"}},
            {"step": 1, "type": "tool_result", "name": "search",
             "result": "Revenue was 100, then 120."},
        ],
        model_turns=2, latency_s=1.0, prompt_tokens=10, completion_tokens=20,
    )


def unit(unit_id, verdict):
    return {"unit_id": unit_id, "verdict": verdict, "reason": "Visible evidence."}


def valid_judgment():
    return {
        "constraints": [unit("C1", "satisfied"), unit("C2", "not_applicable")],
        "answer_points": [unit("A7", "supported")],
        "facts": [unit("F2", "correct"), unit("F9", "not_observed")],
        "dependencies": [unit("D3", "not_covered")],
        "numeric_values": [
            {"unit_id": "N1", "value": 20, "reason": "The response states 20%."},
            {"unit_id": "N2", "value": None, "reason": "No revenue value is given."},
        ],
        "computations": [{
            "expression": "(120 / 100 - 1) * 100", "stated_result": "20",
            "reason": "The tool reported revenues of 100 and 120.",
        }],
        "extra_claims": [{
            "claim": "Revenue was positive.", "verdict": "supported",
            "reason": "The tool reported positive revenue.",
        }],
    }


def response(content=None):
    if content is None:
        content = json.dumps(valid_judgment())
    return {"choices": [{"message": {"content": content}}]}


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


class PromptTests(unittest.TestCase):
    def test_replacements_preserve_json_and_use_annotation_ids(self):
        prompt = unit_judge.build_prompt(task(), run_record(), annotation(), TEMPLATE)
        for name in (
            "constraints", "answer_points", "facts", "dependencies",
            "numeric_targets", "as_of", "query", "trajectory", "final_answer",
        ):
            self.assertNotIn("{" + name + "}", prompt)
        for text in (
            "C1 [period] Use the reported period",
            "C2 [unit_currency] Use CNY",
            "A7 Revenue increased by 20%.",
            "F2 [fact] Revenue rose from 100 to 120.",
            "F9 [arithmetic] (120 / 100 - 1) * 100 = 20.",
            "D3 [input_result_use] prerequisites F2,F9: Use both reported revenues",
            "N1 Revenue growth (%)",
            "N2 Current revenue (CNY)",
            task()["eval"]["as_of"],
            task()["eval"]["query"],
            run_record().final_answer,
            '{"numeric_values": [{"unit_id": "N1", "value": null, "reason": "..."}]}',
        ):
            self.assertIn(text, prompt)

    def test_trajectory_excludes_thinking_and_keeps_whole_tool_results(self):
        run = run_record()
        run.trajectory[-1]["result"] = "中" * 20000
        prompt = unit_judge.build_prompt(task(), run, annotation(), "{trajectory}")
        self.assertEqual(
            prompt,
            '[step 1] agent: Compare the reported revenues.\n'
            '[step 1] call search {"query": "收入"}\n'
            '[step 1] result: ' + "中" * 20000,
        )
        self.assertNotIn("thinking", prompt.lower())
        self.assertNotIn("PRIVATE_THINKING", prompt)

    def test_empty_lists_and_blank_final_answer(self):
        ann = annotation()
        for name in (
            "context_constraints", "answer_points", "facts",
            "dependencies", "numeric_targets",
        ):
            setattr(ann, name, [])
        run = run_record()
        run.final_answer = " \n\t "
        run.trajectory = []
        prompt = unit_judge.build_prompt(
            task(), run, ann,
            "{constraints}\n{answer_points}\n{facts}\n{dependencies}\n"
            "{numeric_targets}\n{trajectory}\n{final_answer}",
        )
        self.assertEqual(
            prompt, "\n".join(["(none)"] * 6 + ["(no final response)"])
        )

    def test_inserted_text_is_not_treated_as_more_template_tokens(self):
        row = task()
        row["eval"]["query"] = 'Keep {facts} and {"answer": 1} literal.'
        run = run_record()
        run.final_answer = "Keep {query} literal."
        self.assertEqual(
            unit_judge.build_prompt(
                row, run, annotation(), "{query}\n{final_answer}"
            ),
            row["eval"]["query"] + "\n" + run.final_answer,
        )


class ValidationTests(unittest.TestCase):
    def parse(self, data):
        return unit_judge.parse_judgment(
            json.dumps(data), "case-a", annotation(), "offline-judge"
        )

    def test_valid_result_builds_dataclasses(self):
        result = self.parse(valid_judgment())
        self.assertIsInstance(result, UnitJudgment)
        for name in UNIT_FIELDS[:-1]:
            self.assertTrue(
                all(isinstance(item, UnitVerdict) for item in getattr(result, name))
            )
        self.assertIsInstance(result.numeric_values[0], NumericValue)
        self.assertIsInstance(result.computations[0], Computation)
        self.assertIsInstance(result.extra_claims[0], ExtraClaim)
        self.assertEqual(asdict(result), {
            "case_id": "case-a", "judge_model": "offline-judge", **valid_judgment(),
        })

    def test_json_fences_are_removed(self):
        for fence in ("```", "```json", "```JSON"):
            with self.subTest(fence=fence):
                result = unit_judge.parse_judgment(
                    " \n" + fence + "\n" + json.dumps(valid_judgment()) + "\n```\n",
                    "case-a", annotation(), "offline-judge",
                )
                self.assertEqual(result.numeric_values[0].value, 20)

    def test_missing_extra_and_duplicate_unit_ids_are_rejected(self):
        for field in UNIT_FIELDS:
            for problem in ("missing", "extra", "duplicate"):
                with self.subTest(field=field, problem=problem):
                    data = valid_judgment()
                    if problem == "missing":
                        data[field].pop()
                    elif problem == "extra":
                        extra = deepcopy(data[field][0])
                        extra["unit_id"] = "unexpected"
                        data[field].append(extra)
                    elif len(data[field]) > 1:
                        data[field][-1] = deepcopy(data[field][0])
                    else:
                        data[field].append(deepcopy(data[field][0]))
                    with self.assertRaisesRegex(ValueError, "ids"):
                        self.parse(data)

    def test_invalid_verdicts_are_rejected(self):
        for field in UNIT_FIELDS[:-1] + ("extra_claims",):
            for label in ("unknown", None, True, ["supported"]):
                with self.subTest(field=field, label=label):
                    data = valid_judgment()
                    data[field][0]["verdict"] = label
                    with self.assertRaisesRegex(ValueError, "verdict"):
                        self.parse(data)

    def test_extra_claim_limit(self):
        data = valid_judgment()
        data["extra_claims"] *= 5
        self.assertEqual(len(self.parse(data).extra_claims), 5)
        data["extra_claims"].append(deepcopy(data["extra_claims"][0]))
        with self.assertRaisesRegex(ValueError, "at most 5"):
            self.parse(data)

    def test_computation_limit(self):
        data = valid_judgment()
        data["computations"] *= 10
        self.assertEqual(len(self.parse(data).computations), 10)
        data["computations"].append(deepcopy(data["computations"][0]))
        with self.assertRaisesRegex(ValueError, "at most 10"):
            self.parse(data)

    def test_computation_keys_and_expression_type(self):
        for missing in ("expression", "stated_result", "reason"):
            with self.subTest(missing=missing):
                data = valid_judgment()
                del data["computations"][0][missing]
                with self.assertRaisesRegex(ValueError, "keys"):
                    self.parse(data)
        data = valid_judgment()
        data["computations"][0]["unexpected"] = "extra"
        with self.assertRaisesRegex(ValueError, "keys"):
            self.parse(data)
        for expression in (None, 20, True, ["1 + 1"]):
            with self.subTest(expression=expression):
                data = valid_judgment()
                data["computations"][0]["expression"] = expression
                with self.assertRaisesRegex(ValueError, "expression"):
                    self.parse(data)

    def test_computation_stated_results_are_decimal_strings(self):
        for value in (
            "65.4%", "1,234", 65.4, None, True, "", "+1", ".5", "1.",
            "1e3", " 65", "65\n",
        ):
            with self.subTest(value=value):
                data = valid_judgment()
                data["computations"][0]["stated_result"] = value
                with self.assertRaisesRegex(ValueError, "stated_result"):
                    self.parse(data)
        for value in ("0", "65", "65.4", "-65.40", "001.00"):
            with self.subTest(value=value):
                data = valid_judgment()
                data["computations"][0]["stated_result"] = value
                self.assertEqual(
                    self.parse(data).computations[0].stated_result, value,
                )

    def test_numeric_values_reject_bool_and_nonfinite_numbers(self):
        for value in (True, False, "20", float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                data = valid_judgment()
                data["numeric_values"][0]["value"] = value
                with self.assertRaisesRegex(ValueError, "finite number or null"):
                    self.parse(data)
        for value in (None, 0, -2, 0.25):
            with self.subTest(value=value):
                data = valid_judgment()
                data["numeric_values"][0]["value"] = value
                self.assertEqual(self.parse(data).numeric_values[0].value, value)

    def test_reasons_and_claims_must_be_strings(self):
        for field in UNIT_FIELDS + ("computations", "extra_claims"):
            with self.subTest(field=field):
                data = valid_judgment()
                data[field][0]["reason"] = None
                with self.assertRaisesRegex(ValueError, "reason"):
                    self.parse(data)
        data = valid_judgment()
        data["extra_claims"][0]["claim"] = 123
        with self.assertRaisesRegex(ValueError, "claim"):
            self.parse(data)

    def test_top_level_keys_and_list_shapes(self):
        for value in ([], None, "text"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(value)
        data = valid_judgment()
        data["unexpected"] = []
        with self.assertRaises(ValueError):
            self.parse(data)
        for field in UNIT_FIELDS + ("computations", "extra_claims"):
            for problem in ("missing", "not_list", "not_object", "missing_reason"):
                with self.subTest(field=field, problem=problem):
                    data = valid_judgment()
                    if problem == "missing":
                        del data[field]
                    elif problem == "not_list":
                        data[field] = {}
                    elif problem == "not_object":
                        data[field][0] = "invalid"
                    else:
                        del data[field][0]["reason"]
                    with self.assertRaises(ValueError):
                        self.parse(data)

    def test_unit_order_is_not_significant_and_empty_lists_are_valid(self):
        data = valid_judgment()
        for field in UNIT_FIELDS:
            data[field].reverse()
        self.assertEqual(self.parse(data).facts[0].unit_id, "F9")
        empty = Annotation("case-a", "offline-annotator", [], [], [], [], [])
        result = unit_judge.parse_judgment(
            json.dumps({field: [] for field in data}),
            "case-a", empty, "offline-judge",
        )
        self.assertEqual(result.numeric_values, [])
        self.assertEqual(result.computations, [])
        data = valid_judgment()
        data["facts"][0]["unit_id"] = []
        with self.assertRaises(ValueError):
            self.parse(data)


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.key = enter(self, patch.object(
            unit_judge, "load_key", return_value="offline-key"
        ))
        self.request = enter(self, patch.object(
            unit_judge, "post_json", return_value=response()
        ))

    def judge(self):
        return unit_judge.judge_run(
            task(), run_record(), annotation(), "offline-judge", TEMPLATE
        )

    def test_request_body_headers_and_timeout(self):
        self.assertIsInstance(self.judge(), UnitJudgment)
        self.key.assert_called_once_with("REFINX_LLM_API_KEY", "key.md")
        self.request.assert_called_once_with(
            unit_judge.LLM_ENDPOINT, {"Authorization": "Bearer offline-key"},
            {
                "model": "offline-judge", "temperature": 0, "max_tokens": 32768,
                "messages": [{
                    "role": "user",
                    "content": unit_judge.build_prompt(
                        task(), run_record(), annotation(), TEMPLATE
                    ),
                }],
            },
            timeout=unit_judge.TIMEOUT,
        )

    def test_null_content_and_malformed_envelopes_retry(self):
        null_content = {"choices": [{"message": {"content": None}}]}
        for invalid in (null_content, {"choices": []}, {"choices": [{}]}):
            with self.subTest(response=invalid):
                self.request.reset_mock()
                self.request.side_effect = [invalid, response()]
                self.assertIsInstance(self.judge(), UnitJudgment)
                self.assertEqual(self.request.call_count, 2)

    def test_invalid_computations_retry_once(self):
        invalid = valid_judgment()
        invalid["computations"][0]["stated_result"] = "65.4%"
        self.request.side_effect = [response(json.dumps(invalid)), response()]
        self.assertIsInstance(self.judge().computations[0], Computation)
        self.assertEqual(self.request.call_count, 2)


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(enter(self, tempfile.TemporaryDirectory()))
        self.output = self.root / "runs" / "offline"
        self.output.mkdir(parents=True)
        (self.root / "annotations").mkdir()
        self.data_path = self.root / "paper/public-data/ReFInx_data/refinx_data.json"
        self.data_path.parent.mkdir(parents=True)
        prompt_path = self.root / "refinx/prompts/unit_judge.md"
        prompt_path.parent.mkdir(parents=True)
        prompt_path.write_text(TEMPLATE, encoding="utf-8")
        self.runs = [run_record()]
        self.annotations = [annotation()]
        self.tasks = [task()]
        enter(self, patch.object(unit_judge, "ROOT", self.root))
        self.key = enter(self, patch.object(
            unit_judge, "load_key", return_value="offline-key"
        ))
        self.request = enter(self, patch.object(
            unit_judge, "post_json", return_value=response()
        ))
        self.stdout = enter(self, redirect_stdout(io.StringIO()))

    def run_main(self, *arguments):
        write_jsonl(self.output / "runs.jsonl", [asdict(run) for run in self.runs])
        write_jsonl(
            self.root / "annotations" / "refined.jsonl",
            [asdict(ann) for ann in self.annotations],
        )
        self.data_path.write_text(json.dumps(self.tasks), encoding="utf-8")
        unit_judge.main([
            "--run", "offline", "--annotations", "refined",
            "--workers", "2", *arguments,
        ])

    def read_records(self, suffix=".jsonl"):
        return [
            json.loads(line)
            for line in (self.output / ("unit_judgments" + suffix)).read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    def test_invalid_then_valid_writes_one_judgment(self):
        invalid = valid_judgment()
        invalid["facts"].pop()
        self.request.side_effect = [response(json.dumps(invalid)), response()]
        self.run_main()
        self.assertEqual(self.read_records(), [{
            "case_id": "case-a", "judge_model": "deepseek-v4.1-flash",
            **valid_judgment(),
        }])
        self.assertEqual(self.read_records(".errors.jsonl"), [])
        self.assertEqual(self.request.call_count, 2)
        self.key.assert_called_once_with("REFINX_LLM_API_KEY", "key.md")
        self.assertEqual(
            self.stdout.getvalue().strip(),
            "judged=1 skipped_error=0 no_annotation=0 failed=0",
        )

    def test_two_invalid_responses_write_only_an_error(self):
        self.request.side_effect = [response("{"), response("[]")]
        self.run_main()
        self.assertEqual(self.read_records(), [])
        errors = self.read_records(".errors.jsonl")
        self.assertEqual(len(errors), 1)
        self.assertEqual(set(errors[0]), {"case_id", "error"})
        self.assertEqual(errors[0]["case_id"], "case-a")
        self.assertTrue(errors[0]["error"])
        self.assertEqual(self.request.call_count, 2)
        self.assertIn("judged=0 skipped_error=0 no_annotation=0 failed=1",
                      self.stdout.getvalue())

    def test_request_failure_is_recorded_without_validation_retry(self):
        self.request.side_effect = TimeoutError("offline")
        self.run_main()
        self.assertEqual(self.read_records(), [])
        self.assertIn("TimeoutError", self.read_records(".errors.jsonl")[0]["error"])
        self.request.assert_called_once()

    def test_error_and_unannotated_runs_are_skipped(self):
        self.runs.extend([
            run_record("case-error", "error"), run_record("case-unannotated"),
        ])
        self.annotations.append(annotation("case-error"))
        self.run_main()
        self.assertEqual([row["case_id"] for row in self.read_records()], ["case-a"])
        self.assertEqual(self.read_records(".errors.jsonl"), [])
        self.request.assert_called_once()
        self.assertEqual(
            self.stdout.getvalue().strip(),
            "judged=1 skipped_error=1 no_annotation=1 failed=0",
        )

    def test_only_the_latest_row_per_case_is_judged(self):

        self.runs.append(run_record("case-a", "error"))
        self.run_main()
        self.assertEqual(self.read_records(), [])
        self.request.assert_not_called()
        self.assertEqual(
            self.stdout.getvalue().strip(),
            "judged=0 skipped_error=1 no_annotation=0 failed=0",
        )

    def test_budget_run_with_blank_answer_is_still_judged(self):
        self.runs[0].stop_reason = "budget"
        self.runs[0].final_answer = " \n "
        self.run_main()
        self.assertEqual(len(self.read_records()), 1)
        body = self.request.call_args.args[2]
        self.assertIn("(no final response)", body["messages"][0]["content"])

    def test_resume_skips_successes_but_retries_historical_errors(self):
        previous = {
            "case_id": "case-a", "judge_model": "previous-judge", **valid_judgment(),
        }
        write_jsonl(self.output / "unit_judgments.jsonl", [previous])
        old_error = {"case_id": "case-b", "error": "earlier failure"}
        write_jsonl(self.output / "unit_judgments.errors.jsonl", [old_error])
        self.runs.append(run_record("case-b"))
        self.annotations.append(annotation("case-b"))
        self.tasks.append(task("case-b"))
        self.run_main("--resume", "--judge-model", "offline-judge")
        records = self.read_records()
        self.assertEqual([row["case_id"] for row in records], ["case-a", "case-b"])
        self.assertEqual(records[0], previous)
        self.assertEqual(records[1]["judge_model"], "offline-judge")
        self.assertEqual(self.read_records(".errors.jsonl"), [old_error])
        self.request.assert_called_once()
        self.request.reset_mock()
        self.key.reset_mock()
        self.run_main("--resume")
        self.request.assert_not_called()
        self.key.assert_not_called()
        self.assertEqual(self.read_records(), records)
        self.assertEqual(
            self.stdout.getvalue().splitlines()[-1],
            "judged=0 skipped_error=0 no_annotation=0 failed=0",
        )

    def test_without_resume_output_is_appended(self):
        previous = {
            "case_id": "case-a", "judge_model": "previous-judge", **valid_judgment(),
        }
        write_jsonl(self.output / "unit_judgments.jsonl", [previous])
        self.run_main()
        records = self.read_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0], previous)
        self.request.assert_called_once()

    def test_requests_use_workers_and_records_flush_on_the_main_thread(self):
        self.runs.append(run_record("case-b"))
        self.annotations.append(annotation("case-b"))
        self.tasks.append(task("case-b"))
        original_write = unit_judge._write_record

        def request(*args, **kwargs):
            self.assertIsNot(threading.current_thread(), threading.main_thread())
            return response()

        def write_record(handle, record):
            self.assertIs(threading.current_thread(), threading.main_thread())
            proxy = Mock(wraps=handle)
            original_write(proxy, record)
            self.assertEqual(
                [item[0] for item in proxy.mock_calls], ["write", "flush"]
            )

        self.request.side_effect = request
        with patch.object(unit_judge, "_write_record", side_effect=write_record):
            self.run_main()
        self.assertEqual(len(self.read_records()), 2)
        self.assertEqual(self.read_records(".errors.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
