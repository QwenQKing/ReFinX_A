from __future__ import annotations

import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import call, patch

from refinx.run import main
from refinx.schema import Judgment, Run, Task


def enter(test, cm):
    value = cm.__enter__()
    test.addCleanup(cm.__exit__, None, None, None)
    return value


class RunResumeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(enter(self, tempfile.TemporaryDirectory()))
        self.output = self.root / "runs" / "resume-test"
        self.tasks = [
            Task(
                case_id=case_id, query="What is 1 + 1?",
                as_of="2025-01-01T00:00:00Z", language="en",
                terminal_action="answer", gold_answer="2",
                accepted_alternatives=[], required_clarifications=[],
            )
            for case_id in ("a", "b", "c")
        ]
        self.args = [
            "--agent-model", "offline-model", "--judge-model", "offline-judge",
            "--name", "resume-test", "--workers", "1",
        ]
        enter(self, patch("refinx.run.ROOT", self.root))
        self.load_tasks = enter(
            self, patch("refinx.run.load_tasks", return_value=self.tasks)
        )
        self.evaluate = enter(
            self, patch("refinx.run.evaluate", side_effect=self.fake_evaluate)
        )
        self.stdout = enter(self, patch("sys.stdout", new_callable=io.StringIO))
        self.stderr = enter(self, patch("sys.stderr", new_callable=io.StringIO))

    def make_run(self, case_id, stop_reason="final"):
        return Run(
            case_id, "offline-model", "" if stop_reason == "error" else "2",
            stop_reason, [], [], 1, 0, 0, 0,
        )

    def fake_evaluate(self, task, model, judge_model, run=None,
                      harness="dsh", provider="default"):
        result = run if run is not None else self.make_run(task.case_id)
        if result.stop_reason == "error":
            return result, None
        return result, Judgment(task.case_id, True, "matches")

    def write_history(self, runs, judgments=()):
        self.output.mkdir(parents=True, exist_ok=True)
        contents = []
        for filename, records in (
            ("runs.jsonl", runs), ("judgments.jsonl", judgments),
        ):
            data = "".join(
                json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records
            ).encode("utf-8")
            (self.output / filename).write_bytes(data)
            contents.append(data)
        return tuple(contents)

    def read_records(self, filename):
        return [
            json.loads(line)
            for line in (self.output / filename).read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    def read_summary(self):
        return json.loads(
            (self.output / "summary.json").read_text(encoding="utf-8")
        )

    def test_resume_retries_errors_and_missing_cases(self):
        runs_before, judgments_before = self.write_history(
            [self.make_run("a"), self.make_run("b", "error")],
            [Judgment("a", True, "已有判断")],
        )

        main(self.args + ["--case-ids", "a,b,c", "--resume"])

        self.assertEqual(self.evaluate.call_count, 2)
        self.evaluate.assert_has_calls([
            call(task, "offline-model", "offline-judge",
                 harness="dsh", provider="default")
            for task in self.tasks[1:]
        ], any_order=True)
        self.assertTrue(
            (self.output / "runs.jsonl").read_bytes().startswith(runs_before)
        )
        self.assertTrue(
            (self.output / "judgments.jsonl").read_bytes().startswith(judgments_before)
        )
        runs = self.read_records("runs.jsonl")
        self.assertCountEqual([row["case_id"] for row in runs[2:]], ["b", "c"])
        self.assertTrue(all(row["stop_reason"] == "final" for row in runs[2:]))
        self.assertEqual(len(self.read_records("judgments.jsonl")), 3)
        self.assertEqual(self.read_summary(), {
            "model": "offline-model", "judge_model": "offline-judge",
            "applicable": 3, "unobservable": 0, "undecided": 0,
            "adjudicated": 3, "correct": 3, "ans_acc": 1.0, "coverage": 1.0,
        })
        self.assertEqual(
            self.stdout.getvalue().splitlines()[0], "resume: 1 done, 2 to run"
        )

    def test_resume_uses_final_after_error_and_last_judgment(self):
        runs_before, judgments_before = self.write_history(
            [self.make_run("a", "error"), self.make_run("a")],
            [Judgment("a", False, "old"), Judgment("a", True, "latest")],
        )

        main(self.args + ["--case-ids", "a", "--resume"])

        self.evaluate.assert_not_called()
        self.assertEqual((self.output / "runs.jsonl").read_bytes(), runs_before)
        self.assertEqual(
            (self.output / "judgments.jsonl").read_bytes(), judgments_before
        )
        summary = self.read_summary()
        self.assertEqual(summary["applicable"], 1)
        self.assertEqual(summary["unobservable"], 0)
        self.assertEqual(summary["adjudicated"], 1)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(
            self.stdout.getvalue().splitlines()[0], "resume: 1 done, 0 to run"
        )

    def test_resume_retries_last_error_and_excludes_stale_judgment(self):
        error = self.make_run("a", "error")
        self.write_history(
            [self.make_run("a"), error], [Judgment("a", True, "old success")]
        )
        self.evaluate.side_effect = None
        self.evaluate.return_value = error, None

        main(self.args + ["--case-ids", "a", "--resume"])

        self.evaluate.assert_called_once_with(
            self.tasks[0], "offline-model", "offline-judge",
            harness="dsh", provider="default",
        )
        self.assertEqual(len(self.read_records("runs.jsonl")), 3)
        summary = self.read_summary()
        self.assertEqual(summary["applicable"], 1)
        self.assertEqual(summary["unobservable"], 1)
        self.assertEqual(summary["adjudicated"], 0)
        self.assertEqual(summary["correct"], 0)
        self.assertEqual(summary["undecided"], 0)
        self.assertIsNone(summary["ans_acc"])
        self.assertEqual(summary["coverage"], 0.0)

    def test_resume_keeps_non_error_run_without_judgment(self):
        runs_before, _ = self.write_history([self.make_run("a", "budget")])
        (self.output / "judgments.jsonl").unlink()

        main(self.args + ["--case-ids", "a", "--resume"])

        self.evaluate.assert_not_called()
        self.assertEqual((self.output / "runs.jsonl").read_bytes(), runs_before)
        self.assertEqual((self.output / "judgments.jsonl").read_bytes(), b"")
        summary = self.read_summary()
        self.assertEqual(summary["applicable"], 1)
        self.assertEqual(summary["unobservable"], 0)
        self.assertEqual(summary["adjudicated"], 0)
        self.assertEqual(summary["coverage"], 0.0)

    def test_existing_nonempty_runs_require_resume_without_opening_files(self):
        runs_before, judgments_before = self.write_history(
            [self.make_run("a")], [Judgment("a", True, "existing")]
        )

        with patch.object(
            Path, "open", side_effect=AssertionError("Files must not be opened")
        ) as opened:
            with self.assertRaises(SystemExit) as raised:
                main(self.args + ["--case-ids", "a"])
            opened.assert_not_called()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--resume", self.stderr.getvalue())
        self.evaluate.assert_not_called()
        self.assertEqual((self.output / "runs.jsonl").read_bytes(), runs_before)
        self.assertEqual(
            (self.output / "judgments.jsonl").read_bytes(), judgments_before
        )
        self.assertFalse((self.output / "summary.json").exists())

    def test_resume_requires_name(self):
        with self.assertRaises(SystemExit) as raised:
            main(["--agent-model", "offline-model", "--resume"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--name", self.stderr.getvalue())
        self.load_tasks.assert_not_called()
        self.evaluate.assert_not_called()

    def test_resume_rejects_from_run(self):
        with self.assertRaises(SystemExit) as raised:
            main(["--from-run", "previous", "--name", "resume-test", "--resume"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--from-run", self.stderr.getvalue())
        self.load_tasks.assert_not_called()
        self.evaluate.assert_not_called()

    def test_resume_creates_missing_output_directory(self):
        self.assertFalse(self.output.exists())

        main(self.args + [
            "--case-ids", "a,b,c", "--resume",
            "--harness", "dsh", "--provider", "api",
        ])

        self.assertEqual(self.evaluate.call_count, 3)
        self.evaluate.assert_has_calls([
            call(task, "offline-model", "offline-judge",
                 harness="dsh", provider="api")
            for task in self.tasks
        ], any_order=True)
        self.assertEqual(len(self.read_records("runs.jsonl")), 3)
        self.assertEqual(len(self.read_records("judgments.jsonl")), 3)
        summary = self.read_summary()
        self.assertEqual(summary["applicable"], 3)
        self.assertEqual(summary["correct"], 3)
        self.assertEqual(summary["unobservable"], 0)
        self.assertEqual(
            self.stdout.getvalue().splitlines()[0], "resume: 0 done, 3 to run"
        )

    def test_resume_keeps_sample_selection_and_excludes_other_cases(self):
        sample_args = self.args + ["--sample", "2", "--seed", "7"]
        main(sample_args)
        self.assertEqual(json.loads(self.stdout.getvalue()), self.read_summary())
        selected = {row["case_id"] for row in self.read_records("runs.jsonl")}
        self.assertEqual(len(selected), 2)
        extra_id = next(task.case_id for task in self.tasks if task.case_id not in selected)
        for filename, record in (
            ("runs.jsonl", self.make_run(extra_id, "error")),
            ("judgments.jsonl", Judgment(extra_id, True, "outside selection")),
        ):
            with (self.output / filename).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record)) + "\n")
        self.evaluate.reset_mock()
        self.stdout.seek(0)
        self.stdout.truncate(0)

        main(sample_args + ["--resume"])

        self.evaluate.assert_not_called()
        summary = self.read_summary()
        self.assertEqual(summary["applicable"], 2)
        self.assertEqual(summary["unobservable"], 0)
        self.assertEqual(summary["adjudicated"], 2)
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(
            self.stdout.getvalue().splitlines()[0], "resume: 2 done, 0 to run"
        )

    def test_empty_runs_file_allows_fresh_run(self):
        self.write_history([], [Judgment("a", False, "stale")])

        main(self.args + ["--case-ids", "a"])

        self.evaluate.assert_called_once()
        self.assertEqual(len(self.read_records("runs.jsonl")), 1)
        self.assertEqual(
            self.read_records("judgments.jsonl"),
            [asdict(Judgment("a", True, "matches"))],
        )
        self.assertEqual(self.read_summary()["applicable"], 1)
        self.assertEqual(self.read_summary()["correct"], 1)

    def test_from_run_judges_latest_row_and_overwrites_destination(self):
        source_runs = [self.make_run("a", "error"), self.make_run("a")]
        rows = [asdict(run) for run in source_runs]
        del rows[0]["trajectory"]
        source = self.root / "runs" / "previous" / "runs.jsonl"
        source.parent.mkdir(parents=True)
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.write_history(
            [self.make_run("b")], [Judgment("b", False, "old destination")]
        )

        main(self.args + ["--from-run", "previous"])


        self.evaluate.assert_called_once_with(
            self.tasks[0], "offline-model", "offline-judge", source_runs[1]
        )
        self.assertEqual((self.output / "runs.jsonl").read_bytes(), source.read_bytes())
        self.assertEqual(
            self.read_records("judgments.jsonl"),
            [asdict(Judgment("a", True, "matches"))],
        )
        summary = self.read_summary()
        self.assertEqual(summary["applicable"], 1)
        self.assertEqual(summary["unobservable"], 0)
        self.assertEqual(summary["adjudicated"], 1)
        self.assertEqual(summary["correct"], 1)

    def skip_aware_evaluate(self, task, model, judge_model, run=None,
                            harness="dsh", provider="default", skip_judge=False):
        result = self.make_run(task.case_id)
        return result, None if skip_judge else Judgment(task.case_id, True, "matches")

    def test_skip_judge_writes_runs_only(self):
        self.evaluate.side_effect = self.skip_aware_evaluate
        main(self.args + ["--case-ids", "a,b,c", "--skip-judge"])
        self.evaluate.assert_has_calls([
            call(task, "offline-model", "offline-judge",
                 harness="dsh", provider="default", skip_judge=True)
            for task in self.tasks
        ], any_order=True)
        self.assertEqual(len(self.read_records("runs.jsonl")), 3)
        self.assertEqual(self.read_records("judgments.jsonl"), [])
        summary = self.read_summary()
        self.assertEqual(
            (summary["applicable"], summary["adjudicated"], summary["ans_acc"]),
            (3, 0, None),
        )

    def test_skip_judge_resumes_and_keeps_earlier_judgments(self):
        self.write_history([self.make_run("a")], [Judgment("a", True, "inline")])
        self.evaluate.side_effect = self.skip_aware_evaluate
        main(self.args + ["--case-ids", "a,b,c", "--resume", "--skip-judge"])
        self.assertEqual(self.evaluate.call_count, 2)
        self.assertEqual(len(self.read_judgments_lines()), 1)
        summary = self.read_summary()
        self.assertEqual(
            (summary["applicable"], summary["adjudicated"], summary["correct"]),
            (3, 1, 1),
        )

    def read_judgments_lines(self):
        return self.read_records("judgments.jsonl")

    def test_skip_judge_rejects_from_run(self):
        with self.assertRaises(SystemExit):
            main(self.args + ["--from-run", "previous", "--skip-judge"])
        self.evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
