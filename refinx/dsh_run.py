from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from .net import ROOT, load_key
from .schema import Run, Task, ToolCall


def run_task(task: Task, model: str, provider: str = "default") -> Run:
    with tempfile.TemporaryDirectory() as directory:
        tool_log = Path(directory) / "tools.jsonl"
        trajectory_log = Path(directory) / "trajectory.jsonl"
        env = {
            **os.environ,
            "REFINX_AGENT_MODEL": model,
            "REFINX_AS_OF": task.as_of,
            "REFINX_LANGUAGE": task.language,
            "REFINX_TOOL_LOG": str(tool_log),
            "REFINX_TRAJECTORY_LOG": str(trajectory_log),

            "REFINX_PYTHON": sys.executable,
            "REFINX_ROOT": str(ROOT),
        }
        command = [
            "dsh.cmd" if os.name == "nt" else "dsh", "--profile", "headless", "--patch",
            str(ROOT / "dsh" / "refinx.patch.yml"),
        ]
        if provider == "api":

            command.extend(["--patch", str(ROOT / "dsh" / "api.patch.yml")])
        try:
            env["REFINX_LLM_API_KEY"] = load_key("REFINX_LLM_API_KEY", "key.md")
        except (OSError, ValueError):
            env["REFINX_LLM_API_KEY"] = "EMPTY"   
        host = urlsplit(env.get("REFINX_LLM_BASE_URL", "")).hostname
        if host:
            for name in ("NO_PROXY", "no_proxy"):
                env[name] = ",".join(filter(None, (env.get(name, ""), host)))
        command.extend(["--json", "-"])
        stdout = stderr = ""
        returncode = None
        timed_out = False
        started = time.monotonic()
        try:
            with subprocess.Popen(
                command, cwd=directory, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8",

                start_new_session=os.name != "nt",
            ) as process:
                try:
                    stdout, stderr = process.communicate(task.query, timeout=255)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if os.name == "nt":

                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            check=False,
                        )
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                returncode = process.returncode
        except OSError as exc:
            stderr = f"dsh: {type(exc).__name__}"
        latency_s = time.monotonic() - started

        answer = ""
        turns = prompt_tokens = completion_tokens = 0
        kind = None
        failed = False
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "status":
                phase = event.get("phase")
                if phase == "step_start":
                    turns += 1
                elif phase == "step_end":
                    usage = event.get("usage") or {}
                    prompt_tokens += sum(
                        usage.get(name, 0) or 0
                        for name in ("inputTokens", "cacheReadTokens", "cacheWriteTokens")
                    )
                    completion_tokens += usage.get("outputTokens", 0) or 0
                elif phase == "turn_end":
                    kind = (event.get("reason") or {}).get("kind")
                    failed = failed or kind not in ("completed", "blocked", "max-tokens")
            elif event.get("type") == "final":
                answer = event.get("text") or ""
            elif event.get("type") == "error":
                failed = True

        if timed_out:
            answer, stop_reason = "", "budget"
        elif (
            failed or kind not in ("completed", "blocked", "max-tokens")
            or (kind == "completed" and returncode != 0)
        ):
            stop_reason = "error"
            diagnostic = " | ".join(
                line for line in stderr.splitlines() if line.startswith("dsh:")
            )
            for name in ("REFINX_LLM_API_KEY",):
                key = env.get(name)
                if key:
                    diagnostic = diagnostic.replace(key, "[redacted]")
            print(
                f"{task.case_id} exit={returncode} reason={kind or 'none'} {diagnostic}",
                file=sys.stderr,
            )
        elif kind in ("blocked", "max-tokens") or any(
            line.startswith("refinx-limits: final answer forced")
            for line in stderr.splitlines()
        ):
            stop_reason = "budget"
        else:
            stop_reason = "final"

        trajectory = []
        if trajectory_log.exists():
            with trajectory_log.open(encoding="utf-8") as handle:
                trajectory = [json.loads(line) for line in handle]
        results = {
            entry["call_id"]: entry
            for entry in trajectory if entry["type"] == "tool_result"
        }
        calls = []
        for entry in trajectory:
            if entry["type"] != "tool_call" or entry["call_id"] not in results:
                continue
            result = results[entry["call_id"]]
            arguments = entry["arguments"]
            calls.append(ToolCall(
                entry["name"], arguments if isinstance(arguments, dict) else {},
                result["result"], not result["is_error"],
            ))
        return Run(
            case_id=task.case_id, model=model, final_answer=answer,
            stop_reason=stop_reason, tool_calls=calls, trajectory=trajectory,
            model_turns=turns,
            latency_s=latency_s, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
