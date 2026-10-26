from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from shutil import copyfile

from . import dsh_run
from .data import load_records
from .agent import run_agent
from .judge import judge_answer
from .net import ROOT
from .schema import AnsAccSummary, Judgment, Run, Task, ToolCall


def load_tasks():
    tasks = []
    for row in load_records(ROOT):
        evaluation = row["eval"]
        tasks.append(Task(
            case_id=row["case_id"], query=evaluation["query"],
            as_of=evaluation["as_of"], language=row["classification"]["language"],
            terminal_action=evaluation["terminal_action"],
            gold_answer=evaluation["gold_answer"],
            accepted_alternatives=row["diagnostic_contract"]["correctness_contract"][
                "accepted_alternatives"
            ],
            required_clarifications=evaluation["required_clarifications"],
        ))
    return tasks


def evaluate(task, model, judge_model, run=None, harness="dsh", provider="default",
             skip_judge=False):
    if run is None:
        if harness == "dsh":
            run = dsh_run.run_task(task, model, provider)
        else:
            run = run_agent(task, model)
    if run.stop_reason == "error" or skip_judge:
        return run, None
    if not run.final_answer.strip():
        return run, Judgment(task.case_id, False, "no final answer")
    return run, judge_answer(task, run.final_answer, judge_model)


def summarize(model, judge_model, runs, judgments):
    applicable = len(runs)
    adjudicated = sum(item.correct is not None for item in judgments)
    correct = sum(item.correct is True for item in judgments)
    return AnsAccSummary(
        model=model, judge_model=judge_model, applicable=applicable,
        unobservable=sum(run.stop_reason == "error" for run in runs),
        undecided=sum(item.correct is None for item in judgments),
        adjudicated=adjudicated, correct=correct,
        ans_acc=correct / adjudicated if adjudicated else None,
        coverage=adjudicated / applicable if applicable else 0.0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-model")
    parser.add_argument("--harness", choices=("agent", "dsh"), default="dsh")
    parser.add_argument("--provider", choices=("default", "api"), default="default")
    parser.add_argument("--judge-model", default="deepseek-v4.1-flash")
    parser.add_argument("--from-run")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-ids")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name")
    parser.add_argument("--resume", action="store_true")


    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.from_run:
        parser.error("--resume cannot be used with --from-run")
    if args.skip_judge and args.from_run:
        parser.error("--skip-judge cannot be used with --from-run")
    if args.resume and not args.name:
        parser.error("--resume requires --name")
    if not args.from_run and args.agent_model is None:
        parser.error("--agent-model is required unless --from-run is provided")
    tasks = load_tasks()
    if args.from_run:
        by_id = {task.case_id: task for task in tasks}
        latest = {}
        source = ROOT / "runs" / args.from_run / "runs.jsonl"
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)

                row.setdefault("trajectory", [])
                row["tool_calls"] = [ToolCall(**call) for call in row["tool_calls"]]


                latest[row["case_id"]] = Run(**row)
        previous_runs = list(latest.values())
        model = previous_runs[0].model if previous_runs else ""
        name = args.name or f"{args.from_run}-rejudge-{args.judge_model}"
    else:
        if args.case_ids:
            by_id = {task.case_id: task for task in tasks}
            tasks = [by_id[value.strip()] for value in args.case_ids.split(",")]
        else:
            tasks = random.Random(args.seed).sample(tasks, args.sample)
        model = args.agent_model
        name = args.name or f"{args.agent_model}-{datetime.now():%Y%m%d-%H%M%S}"
    output = ROOT / "runs" / name
    runs_path = output / "runs.jsonl"
    judgments_path = output / "judgments.jsonl"
    if (
        not args.from_run and not args.resume
        and runs_path.exists() and runs_path.stat().st_size > 0
    ):
        parser.error(
            f"{runs_path} already exists and is not empty; use --resume to continue"
        )
    output.mkdir(parents=True, exist_ok=True)
    if args.from_run:
        copyfile(source, runs_path)
    runs, judgments = [], []
    pending_tasks = tasks
    if args.resume:
        latest_runs, latest_judgments = {}, {}
        if runs_path.exists():
            with runs_path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    row.setdefault("trajectory", [])
                    row["tool_calls"] = [ToolCall(**call) for call in row["tool_calls"]]
                    latest_runs[row["case_id"]] = Run(**row)
        if judgments_path.exists():
            with judgments_path.open(encoding="utf-8") as handle:
                for line in handle:
                    judgment = Judgment(**json.loads(line))
                    latest_judgments[judgment.case_id] = judgment
        pending_tasks = [
            task for task in tasks
            if task.case_id not in latest_runs
            or latest_runs[task.case_id].stop_reason == "error"
        ]
        print(
            f"resume: {len(tasks) - len(pending_tasks)} done, "
            f"{len(pending_tasks)} to run"
        )
    mode = "a" if args.resume else "w"
    with (
        (nullcontext() if args.from_run else
         runs_path.open(mode, encoding="utf-8")) as runs_handle,
        judgments_path.open(mode, encoding="utf-8") as judgments_handle,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        if args.from_run:
            futures = [
                pool.submit(
                    evaluate, by_id[run.case_id], run.model, args.judge_model, run
                )
                for run in previous_runs
            ]
        else:
            options = {"skip_judge": True} if args.skip_judge else {}
            futures = [
                pool.submit(
                    evaluate, task, args.agent_model, args.judge_model,
                    harness=args.harness, provider=args.provider, **options,
                )
                for task in pending_tasks
            ]
        results = (future.result() for future in as_completed(futures))
        for run, judgment in results:
            runs.append(run)
            if runs_handle is not None:
                runs_handle.write(json.dumps(asdict(run), ensure_ascii=False) + "\n")
                runs_handle.flush()
            if judgment is not None:
                judgments.append(judgment)
                judgments_handle.write(json.dumps(asdict(judgment), ensure_ascii=False) + "\n")
                judgments_handle.flush()
    if args.resume:
        latest_runs.update((run.case_id, run) for run in runs)
        latest_judgments.update((judgment.case_id, judgment) for judgment in judgments)
        selected_ids = {task.case_id for task in tasks}
        runs = [
            run for case_id, run in latest_runs.items()
            if case_id in selected_ids
        ]
        judgments = [
            latest_judgments[run.case_id]
            for run in runs
            if run.stop_reason != "error" and run.case_id in latest_judgments
        ]
    summary = summarize(model, args.judge_model, runs, judgments)
    text = json.dumps(asdict(summary), ensure_ascii=False, indent=2)
    (output / "summary.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
