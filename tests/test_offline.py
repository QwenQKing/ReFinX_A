from __future__ import annotations

import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from unittest.mock import call, patch
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

from refinx import dsh_run
from refinx.agent import BUDGET_PROMPT, run_agent
from refinx.judge import TIMEOUT as JUDGE_TIMEOUT, judge_answer
from refinx.mcp_server import call_tool
from refinx.net import LLM_ENDPOINT, get_json, load_key, post_json
from refinx.run import evaluate, main
from refinx.schema import Judgment, Run, Task, ToolCall
from refinx.tools import (
    calculate, get_financial_statements, get_price_history, search_web,
)


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.task = Task(
            case_id="offline", query="What is 1 + 1?",
            as_of="2025-01-01T00:00:00Z", language="en",
            terminal_action="answer", gold_answer="2",
            accepted_alternatives=[], required_clarifications=[],
        )
        network = patch("refinx.net.urlopen", side_effect=AssertionError("Network is disabled"))
        network.start()
        self.addCleanup(network.stop)
        endpoints = patch.dict(os.environ, {
            "REFINX_SEARCH_URL": "https://search.example/search",
            "REFINX_SEARCH_HEADERS": '{"X-Auth": "offline"}',
            "REFINX_SEARCH_BODY": '{"app_id": "offline-app"}',
            "REFINX_MARKET_DATA_URL": "https://data.example/api/a-share",
        })
        endpoints.start()
        self.addCleanup(endpoints.stop)

    def test_calculate(self):
        self.assertEqual(float(calculate("(120-100)/100*100")), 20)
        self.assertEqual(float(calculate("sqrt(16)")), 4)
        for expression in ("__import__('os')", "(1).__class__", "unknown_name"):
            with self.subTest(expression=expression):
                self.assertTrue(calculate(expression).startswith("Error:"))
        self.assertIn("** (power, not ^)", calculate("1.1^25"))
        self.assertIn("division by zero", calculate("1/0"))

    def test_search_cutoff(self):
        as_of = "2026-06-18T05:00:00Z"
        cutoff = int(datetime(2026, 6, 18, 5, tzinfo=timezone.utc).timestamp())
        hit = {"title": "before", "url": "https://example.com/a", "summary": "s",
               "publish_time": cutoff}
        items = [
            {**hit, "title": "after", "publish_time": cutoff + 1},
            {**hit, "title": "undated", "publish_time": 0},
            hit,
        ]
        with patch("refinx.tools.post_json", return_value={
            "status_code": 0, "status_msg": "OK", "data": items,
        }) as request:
            result = json.loads(search_web("茅台", language="zh", as_of=as_of))
        url, headers, body = request.call_args.args
        self.assertEqual(url, "https://search.example/search")
        self.assertEqual(headers, {"X-Auth": "offline"})
        self.assertEqual(body["app_id"], "offline-app")
        self.assertEqual(body["query"], "茅台")
        self.assertEqual(body["channels"], ["web", "news", "announcement", "report"])
        self.assertEqual(result, [{
            "title": "before", "url": "https://example.com/a", "snippet": "s",
            "date": "2026-06-18",
        }])

    def test_search_brave_fallback(self):
        with patch.dict(os.environ, {"REFINX_SEARCH": "brave"}):
            with patch("refinx.tools.load_key", return_value="offline-key"):
                with patch("refinx.tools.get_json", return_value={"web": {"results": []}}) as brave:
                    with patch("refinx.tools.post_json") as super_link:
                        result = search_web("x", language="en", as_of="2025-01-01T00:00:00Z")
        self.assertEqual(result, "[]")
        self.assertTrue(brave.call_args.args[0].startswith("https://api.search.brave.com/"))
        super_link.assert_not_called()

    def test_price_history_cutoff(self):
        as_of = "2026-06-18T05:00:00Z"
        beijing = timezone(timedelta(hours=8))
        cutoff_ms = int(datetime(2026, 6, 18, 5, tzinfo=timezone.utc).timestamp() * 1000)
        items = [
            {
                "date_ms": int(datetime(2026, 6, day, tzinfo=beijing).timestamp() * 1000),
                "open_price": 1250, "high_price": 1260, "low_price": 1240,
                "close_price": 1255.67, "volume": 100, "turnover": 125567,
            }
            for day in (17, 16)
        ]
        with patch("refinx.tools.load_key", return_value="offline-key") as key:
            with patch("refinx.tools.get_json", return_value={
                "code": 0, "data": {"item": items},
            }) as request:
                result = json.loads(get_price_history(
                    "600519.SH", "2026-06-10", "2026-06-30", as_of=as_of
                ))
        url, headers = request.call_args.args
        self.assertEqual(
            url.split("?", 1)[0],
            "https://data.example/api/a-share/prices/historical",
        )
        self.assertEqual(parse_qs(urlsplit(url).query), {
            "symbol": ["600519.SH"],
            "start": [str(int(datetime(2026, 6, 10, tzinfo=beijing).timestamp() * 1000))],
            "end": [str(cutoff_ms - 15 * 60 * 60 * 1000)],
            "interval": ["1d"], "adjust": ["forward"],
        })
        self.assertEqual(headers["X-api-key"], "offline-key")
        key.assert_called_once_with("REFINX_MARKET_DATA_KEY", "key_market_data.md")
        self.assertEqual(result, [
            {
                "date": f"2026-06-{day}", "open": 1250, "high": 1260, "low": 1240,
                "close": 1255.67, "volume": 100, "turnover": 125567,
            }
            for day in (16, 17)
        ])

    def test_price_history_empty_cutoff(self):
        with patch("refinx.tools.load_key", return_value="offline-key") as key:
            with patch("refinx.tools.get_json") as request:
                result = get_price_history(
                    "600519.SH", "2026-06-18", "2026-06-30",
                    as_of="2026-06-18T05:00:00Z",
                )
        self.assertEqual(
            result,
            "Error: no trading days completed before the information cutoff in this range",
        )
        request.assert_not_called()
        key.assert_not_called()

    def test_financial_statements_cutoff(self):
        as_of = "2026-06-18T05:00:00Z"
        beijing = timezone(timedelta(hours=8))
        published = {
            "symbol": "600519.SH", "ticker": "600519", "period": "quarterly",
            "fiscal_year": 2025, "fiscal_period": "Q1",
            "period_end_ms": int(datetime(2025, 3, 31, tzinfo=beijing).timestamp() * 1000),
            "report_date_ms": int(datetime(2026, 4, 30, tzinfo=beijing).timestamp() * 1000),
            "currency": "CNY", "total_revenue": 1000.5, "net_income": 200,
        }
        future = {
            **published, "fiscal_period": "Q2",
            "period_end_ms": int(datetime(2025, 6, 30, tzinfo=beijing).timestamp() * 1000),
            "report_date_ms": int(datetime(2026, 8, 15, tzinfo=beijing).timestamp() * 1000),
            "net_income": 250,
        }
        for statement, path in (
            ("income", "income-statements"), ("balance", "balance-sheets"),
            ("cash_flow", "cash-flow-statements"),
        ):
            with self.subTest(statement=statement):
                with patch("refinx.tools.load_key", return_value="offline-key") as key:
                    with patch("refinx.tools.get_json", return_value={
                        "code": 0, "data": {"item": [future, published]},
                    }) as request:
                        result = json.loads(get_financial_statements(
                            "600519.SH", statement, "quarterly", limit=1, as_of=as_of
                        ))
                url, headers = request.call_args.args
                self.assertEqual(
                    url.split("?", 1)[0],
                    "https://data.example/api/a-share/financials/" + path,
                )
                self.assertEqual(parse_qs(urlsplit(url).query), {
                    "symbol": ["600519.SH"], "period": ["quarterly"],
                    "start": [str(int(
                        datetime(2016, 6, 18, 5, tzinfo=timezone.utc).timestamp() * 1000
                    ))],
                    "end": [str(int(
                        datetime(2026, 6, 18, 5, tzinfo=timezone.utc).timestamp() * 1000
                    ))],
                })
                self.assertEqual(headers["X-api-key"], "offline-key")
                key.assert_called_once_with("REFINX_MARKET_DATA_KEY", "key_market_data.md")
                self.assertEqual(result, [{
                    "fiscal_year": 2025, "fiscal_period": "Q1",
                    "period_end": "2025-03-31", "report_date": "2026-04-30",
                    "currency": "CNY", "total_revenue": 1000.5, "net_income": 200,
                }])
        with patch("refinx.tools.get_json") as request:
            self.assertEqual(
                get_financial_statements("600519.SH", "quarterly", "quarterly", as_of=as_of),
                "Error: statement must be one of income, balance, cash_flow",
            )
        request.assert_not_called()

    def test_market_data_business_error(self):
        for function, arguments in (
            (get_price_history, ("600519", "2026-06-10", "2026-06-30")),
            (get_financial_statements, ("600519", "income", "quarterly")),
        ):
            with self.subTest(tool=function.__name__):
                with patch("refinx.tools.load_key", return_value="offline-key"):
                    with patch("refinx.tools.get_json", return_value={
                        "code": 1002, "message": "Unknown symbol: 600519",
                    }):
                        result = function(*arguments, as_of="2026-06-18T05:00:00Z")
                self.assertEqual(result, "Error: Unknown symbol: 600519")

    def test_agent_market_data_cutoff(self):
        requests = []

        def fake_request(body, timeout):
            requests.append(deepcopy(body))
            if len(requests) == 1:
                return {"choices": [{"message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                        for index, (name, arguments) in enumerate((
                            ("get_price_history", {
                                "symbol": "600519.SH",
                                "start_date": "2024-12-01", "end_date": "2025-01-10",
                            }),
                            ("get_financial_statements", {
                                "symbol": "600519.SH", "statement": "balance",
                                "period": "annual",
                            }),
                        ))
                    ],
                }}]}
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

        with patch("refinx.tools.load_key", return_value="offline-key"):
            with patch("refinx.tools.get_json", return_value={
                "code": 0, "data": {"item": []},
            }) as request:
                run = run_agent(self.task, "offline-model", request=fake_request)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(len(run.tool_calls), 2)
        self.assertTrue(all(call.ok for call in run.tool_calls))
        self.assertEqual(run.stop_reason, "final")
        self.assertEqual(run.final_answer, "done")
        cutoff_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        for call, expected_end in zip(
            request.call_args_list, (cutoff_ms - 15 * 60 * 60 * 1000, cutoff_ms)
        ):
            self.assertEqual(parse_qs(urlsplit(call.args[0]).query)["end"], [str(expected_end)])
        for tool in requests[0]["tools"]:
            self.assertNotIn("as_of", tool["function"]["parameters"]["properties"])

    def test_agent_trajectory(self):
        for raw_arguments, arguments, result, ok in (
            ('{"expression": "1 + 1"}', {"expression": "1 + 1"}, "2", True),
            ("[]", "[]", "Error: ValueError", False),
            ("{bad", "{bad", "Error: JSONDecodeError", False),
        ):
            with self.subTest(arguments=raw_arguments):
                responses = iter([
                    {"choices": [{"message": {
                        "role": "assistant", "reasoning_content": "I should calculate.",
                        "content": "Let me check.",
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "calculate", "arguments": raw_arguments},
                        }],
                    }}]},
                    {"choices": [{"message": {
                        "role": "assistant", "reasoning_content": "I can now answer.",
                        "content": "2",
                    }}]},
                ])

                def fake_request(body, timeout):
                    return next(responses)

                run = run_agent(self.task, "offline-model", request=fake_request)
                self.assertEqual(run.trajectory, [
                    {"step": 1, "type": "thinking", "text": "I should calculate."},
                    {"step": 1, "type": "text", "text": "Let me check."},
                    {
                        "step": 1, "type": "tool_call", "call_id": "call_1",
                        "name": "calculate", "arguments": arguments,
                    },
                    {
                        "step": 1, "type": "tool_result", "call_id": "call_1",
                        "result": result, "is_error": not ok,
                    },
                    {"step": 2, "type": "thinking", "text": "I can now answer."},
                    {"step": 2, "type": "text", "text": "2"},
                ])
                self.assertEqual(run.tool_calls, [ToolCall(
                    "calculate", arguments if isinstance(arguments, dict) else {},
                    result, ok,
                )])

    def test_agent_budget(self):
        requests = []

        def fake_request(body, timeout):
            requests.append(deepcopy(body))
            return {
                "choices": [{"message": {
                    "role": "assistant", "content": "2",
                    "reasoning_content": "This must not enter the history.",
                    "tool_calls": [{
                        "id": f"call_{len(requests)}", "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": json.dumps({"expression": "1 + 1"}),
                        },
                    }],
                }}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }

        run = run_agent(self.task, "offline-model", request=fake_request)
        self.assertLessEqual(len(run.tool_calls), 12)
        self.assertLessEqual(run.model_turns, 10)
        self.assertEqual(run.model_turns, len(requests))
        self.assertEqual(run.model_turns, 10)
        self.assertEqual(sum(call.ok for call in run.tool_calls), 8)
        self.assertFalse(run.tool_calls[-1].ok)
        self.assertEqual(
            [entry for entry in run.trajectory if entry["type"] == "tool_result"][-1],
            {
                "step": 9, "type": "tool_result", "call_id": "call_9",
                "result": "Error: per-tool call limit reached", "is_error": True,
            },
        )
        self.assertNotIn("tools", requests[-1])
        self.assertTrue(all("tools" in request for request in requests[:-1]))
        self.assertEqual(requests[-1]["messages"][-1]["content"], BUDGET_PROMPT)
        self.assertEqual(run.stop_reason, "budget")
        self.assertEqual(run.final_answer, "2")
        self.assertEqual(run.prompt_tokens, 3 * len(requests))
        self.assertEqual(run.completion_tokens, 2 * len(requests))
        for message in requests[-1]["messages"]:
            if message["role"] == "assistant":
                self.assertEqual(set(message), {"role", "content", "tool_calls"})

    def test_agent_timeout_forces_final(self):
        requests = []

        def fake_request(body, timeout):
            requests.append((deepcopy(body), timeout))
            if len(requests) == 1:
                raise socket.timeout("offline")
            return {
                "choices": [{"message": {"role": "assistant", "content": "final"}}],
            }

        with patch("refinx.agent.time.monotonic", side_effect=[0, 190, 190, 190]):
            run = run_agent(self.task, "offline-model", request=fake_request)
        self.assertEqual(run.model_turns, 2)
        self.assertEqual(run.stop_reason, "budget")
        self.assertEqual(run.final_answer, "final")
        self.assertEqual(run.trajectory, [{"step": 2, "type": "text", "text": "final"}])
        self.assertIn("tools", requests[0][0])
        self.assertNotIn("tools", requests[1][0])
        self.assertEqual([timeout for _, timeout in requests], [5, 50])
        self.assertEqual(requests[1][0]["messages"][-1]["content"], BUDGET_PROMPT)

    def test_batch_tool_limit(self):
        requests = []

        def fake_request(body, timeout):
            requests.append(deepcopy(body))
            tool_calls = []
            for index in range(14):
                name = "calculate" if index % 2 == 0 else "search_web"
                arguments = {"expression": "1 + 1"} if index % 2 == 0 else {"query": "example"}
                tool_calls.append({
                    "id": f"call_{len(requests)}_{index}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                })
            return {
                "choices": [{"message": {
                    "role": "assistant", "content": "2", "tool_calls": tool_calls,
                }}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        with patch("refinx.agent.search_web", return_value="[]") as search:
            with patch("refinx.agent.calculate", wraps=calculate) as arithmetic:
                run = run_agent(self.task, "offline-model", request=fake_request)
        self.assertEqual(search.call_count + arithmetic.call_count, 12)
        self.assertLessEqual(search.call_count, 8)
        self.assertLessEqual(arithmetic.call_count, 8)
        self.assertEqual(sum(call.ok for call in run.tool_calls), 12)
        self.assertEqual(sum(not call.ok for call in run.tool_calls), 2)
        self.assertEqual(run.model_turns, 2)
        self.assertNotIn("tools", requests[-1])
        self.assertEqual(run.stop_reason, "budget")
        search.assert_called_with(
            query="example", language=self.task.language, as_of=self.task.as_of
        )

    def test_mcp_context_errors_and_log(self):
        arguments = {"query": "茅台", "count": 3}
        result = "完整结果\n" + "x" * 9000
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "tools.jsonl"
            with patch.dict(os.environ, {
                "REFINX_LANGUAGE": "zh", "REFINX_AS_OF": self.task.as_of,
                "REFINX_TOOL_LOG": str(log),
            }):
                with patch("refinx.mcp_server.search_web", return_value=result) as search:
                    self.assertEqual(call_tool("search_web", arguments), result)
                search.assert_called_once_with(
                    **arguments, language="zh", as_of=self.task.as_of
                )
                rows = log.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(rows), 1)
                self.assertEqual(json.loads(rows[0]), {
                    "name": "search_web", "arguments": arguments,
                    "result": result, "ok": True,
                })
                with patch(
                    "refinx.mcp_server.calculate", side_effect=ValueError("private detail")
                ) as arithmetic:
                    self.assertEqual(
                        call_tool("calculate", {"expression": "bad"}), "Error: ValueError"
                    )
                arithmetic.assert_called_once_with(expression="bad")
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1], {
            "name": "calculate", "arguments": {"expression": "bad"},
            "result": "Error: ValueError", "ok": False,
        })

    def test_mcp_market_data_context(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "tools.jsonl"
            with patch.dict(os.environ, {
                "REFINX_AS_OF": self.task.as_of, "REFINX_TOOL_LOG": str(log),
            }):
                for name, arguments, result in (
                    ("get_price_history", {
                        "symbol": "600519.SH", "start_date": "2024-12-01",
                        "end_date": "2025-01-10",
                    }, "[]"),
                    ("get_financial_statements", {
                        "symbol": "600519.SH", "statement": "balance", "period": "annual",
                    }, "Error: unavailable"),
                ):
                    with self.subTest(tool=name):
                        with patch(
                            "refinx.mcp_server." + name, return_value=result
                        ) as function:
                            self.assertEqual(call_tool(name, arguments), result)
                        function.assert_called_once_with(**arguments, as_of=self.task.as_of)
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["ok"] for row in rows], [True, False])
        self.assertTrue(all("as_of" not in row["arguments"] for row in rows))

    def test_dsh_events_and_trajectory(self):
        self.task.query = "1 + 1 是多少？"
        answer = "\n  答案是 2。\n"
        events = [
            {"type": "session", "id": "offline-session"},
            {"type": "status", "phase": "step_start"},
            {"type": "status", "phase": "step_end", "usage": {
                "inputTokens": 10, "outputTokens": 3, "totalTokens": 13,
            }},
            {"type": "status", "phase": "step_start"},
            {"type": "status", "phase": "step_end", "usage": {
                "inputTokens": 20, "outputTokens": 7, "totalTokens": 27,
            }},
            {"type": "status", "phase": "turn_end", "reason": {"kind": "completed"}},
            {"type": "final", "text": answer},
        ]
        stdout = "not json\nnull\n" + "\n".join(json.dumps(event) for event in events)
        result = "完整结果\n" + "x" * 9000
        trajectory = [
            {"step": 1, "type": "thinking", "text": "I should search."},
            {"step": 1, "type": "text", "text": "Checking the evidence."},
            {
                "step": 1, "type": "tool_call", "call_id": "call_1",
                "name": "search_web", "arguments": {"query": "example"},
            },
            {
                "step": 1, "type": "tool_call", "call_id": "call_2",
                "name": "search_web", "arguments": "[]",
            },
            {
                "step": 1, "type": "tool_result", "call_id": "call_2",
                "result": "Error: per-tool call limit reached", "is_error": True,
            },
            {
                "step": 1, "type": "tool_result", "call_id": "call_1",
                "result": result, "is_error": False,
            },
            {"step": 2, "type": "text", "text": answer},
        ]
        with (
            patch("refinx.dsh_run.load_key", return_value="offline-key") as key,
            patch("refinx.dsh_run.subprocess.Popen") as popen,
            patch("refinx.dsh_run.time.monotonic", side_effect=[10, 12.5]),
        ):
            process = popen.return_value.__enter__.return_value
            process.returncode = 0

            def communicate(query, timeout):
                self.assertEqual(query, self.task.query)
                self.assertEqual(timeout, 255)
                options = popen.call_args.kwargs
                cwd = Path(options["cwd"])
                self.assertEqual(list(cwd.iterdir()), [])
                self.assertNotEqual(cwd, dsh_run.ROOT)
                self.assertEqual(options["encoding"], "utf-8")
                self.assertIs(options["text"], True)
                env = options["env"]
                self.assertEqual(env["REFINX_AGENT_MODEL"], "offline-model")
                self.assertEqual(env["REFINX_AS_OF"], self.task.as_of)
                self.assertEqual(env["REFINX_LANGUAGE"], self.task.language)
                self.assertEqual(env["REFINX_LLM_API_KEY"], "offline-key")
                self.assertEqual(env.get("PATH"), os.environ.get("PATH"))
                self.assertEqual(env["REFINX_PYTHON"], sys.executable)
                self.assertEqual(env["REFINX_ROOT"], str(dsh_run.ROOT))
                self.assertEqual(Path(env["REFINX_TOOL_LOG"]).parent, cwd)
                log = Path(env["REFINX_TRAJECTORY_LOG"])
                self.assertEqual(log, cwd / "trajectory.jsonl")
                log.write_text(
                    "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in trajectory),
                    encoding="utf-8",
                )
                return stdout, "refinx-limits: final answer forced (turns)\n"

            process.communicate.side_effect = communicate
            run = dsh_run.run_task(self.task, "offline-model")
            directory = Path(popen.call_args.kwargs["cwd"])
        key.assert_called_once_with("REFINX_LLM_API_KEY", "key.md")
        self.assertEqual(popen.call_args.args[0], [
            "dsh.cmd" if os.name == "nt" else "dsh", "--profile", "headless", "--patch",
            str(dsh_run.ROOT / "dsh" / "refinx.patch.yml"), "--json", "-",
        ])
        self.assertFalse(directory.exists())
        self.assertEqual(run.case_id, self.task.case_id)
        self.assertEqual(run.model, "offline-model")
        self.assertEqual(run.model_turns, 2)
        self.assertEqual((run.prompt_tokens, run.completion_tokens), (30, 10))
        self.assertEqual(run.latency_s, 2.5)
        self.assertEqual(run.stop_reason, "budget")
        self.assertEqual(run.final_answer, answer)
        self.assertEqual(run.trajectory, trajectory)
        self.assertEqual(run.tool_calls, [
            ToolCall("search_web", {"query": "example"}, result, True),
            ToolCall("search_web", {}, "Error: per-tool call limit reached", False),
        ])
        self.assertEqual([call.ok for call in run.tool_calls], [True, False])
        self.assertGreater(len(run.tool_calls[0].result), 9000)

    def test_dsh_cached_tokens(self):
        events = [
            {"type": "status", "phase": "step_start"},
            {"type": "status", "phase": "step_end", "usage": {
                "inputTokens": 301, "outputTokens": 4362, "totalTokens": 12599,
                "cacheReadTokens": 7936,
            }},
            {"type": "status", "phase": "step_start"},
            {"type": "status", "phase": "step_end", "usage": {
                "inputTokens": 20, "outputTokens": 7,
                "cacheReadTokens": None, "cacheWriteTokens": 50,
            }},
            {"type": "status", "phase": "turn_end", "reason": {"kind": "completed"}},
            {"type": "final", "text": "2"},
        ]
        with (
            patch("refinx.dsh_run.load_key", return_value="offline-key"),
            patch("refinx.dsh_run.subprocess.Popen") as popen,
        ):
            process = popen.return_value.__enter__.return_value
            process.returncode = 0
            process.communicate.return_value = (
                "\n".join(json.dumps(event) for event in events), "",
            )
            run = dsh_run.run_task(self.task, "offline-model")
        self.assertEqual(run.stop_reason, "final")
        self.assertEqual((run.prompt_tokens, run.completion_tokens), (8307, 4369))

    def test_dsh_api(self):
        host = "api.example"
        for previous, expected_upper, expected_lower in (
            ({}, host, host),
            ({"NO_PROXY": "", "no_proxy": ""}, host, host),
            (
                {"NO_PROXY": "localhost,127.0.0.1", "no_proxy": "internal.example"},
                f"localhost,127.0.0.1,{host}", f"internal.example,{host}",
            ),
        ):
            with self.subTest(previous=previous):

                with (
                    patch("refinx.dsh_run.os.environ", {
                        "HTTP_PROXY": "http://127.0.0.1:17891",
                        "HTTPS_PROXY": "http://127.0.0.1:17891",
                        "REFINX_LLM_BASE_URL": f"https://{host}/v1",
                        **previous,
                    }),
                    patch("refinx.dsh_run.load_key", return_value="offline-api-key") as key,
                    patch("refinx.dsh_run.subprocess.Popen") as popen,
                    patch("refinx.dsh_run.sys.stderr", new_callable=io.StringIO) as diagnostic,
                ):
                    process = popen.return_value.__enter__.return_value
                    process.returncode = 1
                    process.communicate.return_value = ("", "dsh: failure offline-api-key\n")
                    run = dsh_run.run_task(self.task, "offline-api-model", provider="api")
                key.assert_called_once_with("REFINX_LLM_API_KEY", "key.md")
                self.assertEqual(popen.call_args.args[0], [
                    "dsh.cmd" if os.name == "nt" else "dsh", "--profile", "headless", "--patch",
                    str(dsh_run.ROOT / "dsh" / "refinx.patch.yml"),
                    "--patch", str(dsh_run.ROOT / "dsh" / "api.patch.yml"),
                    "--json", "-",
                ])
                env = popen.call_args.kwargs["env"]
                self.assertEqual(env["REFINX_LLM_API_KEY"], "offline-api-key")
                self.assertEqual(env["NO_PROXY"], expected_upper)
                self.assertEqual(env["no_proxy"], expected_lower)
                self.assertEqual(env["HTTP_PROXY"], "http://127.0.0.1:17891")
                self.assertEqual(env["HTTPS_PROXY"], "http://127.0.0.1:17891")
                self.assertEqual(run.stop_reason, "error")
                self.assertNotIn("offline-api-key", diagnostic.getvalue())
                self.assertEqual(diagnostic.getvalue(), (
                    f"{self.task.case_id} exit=1 reason=none dsh: failure [redacted]\n"
                ))

    def test_dsh_default_route_without_key(self):

        with (
            patch("refinx.dsh_run.os.environ", {
                "NO_PROXY": "internal.example", "REFINX_LLM_BASE_URL": "http://127.0.0.1:3800/v1",
            }),
            patch("refinx.dsh_run.load_key", side_effect=ValueError("no key")),
            patch("refinx.dsh_run.subprocess.Popen") as popen,
            patch("refinx.dsh_run.sys.stderr", new_callable=io.StringIO),
        ):
            process = popen.return_value.__enter__.return_value
            process.returncode = 1
            process.communicate.return_value = ("", "")
            dsh_run.run_task(self.task, "offline-model")
        self.assertEqual(popen.call_args.args[0][4:], [
            str(dsh_run.ROOT / "dsh" / "refinx.patch.yml"), "--json", "-",
        ])
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["REFINX_LLM_API_KEY"], "EMPTY")
        self.assertEqual(env["NO_PROXY"], "internal.example,127.0.0.1")
        self.assertEqual(env["no_proxy"], "127.0.0.1")

    def test_dsh_timeout(self):
        partial = "\n".join(json.dumps(event) for event in [
            {"type": "status", "phase": "step_start"},
            {"type": "status", "phase": "step_end", "usage": {
                "inputTokens": 4, "outputTokens": 2, "totalTokens": 6,
            }},
            {"type": "final", "text": "discard this on timeout"},
        ])
        with (
            patch("refinx.dsh_run.load_key", return_value="offline-key"),
            patch("refinx.dsh_run.subprocess.Popen") as popen,
            patch("refinx.dsh_run.subprocess.run") as terminate,
            patch("refinx.dsh_run.os.killpg", create=True) as kill_group,
        ):
            process = popen.return_value.__enter__.return_value
            process.pid = 1234
            process.returncode = -9
            process.communicate.side_effect = [
                subprocess.TimeoutExpired("dsh.cmd", 255), (partial, ""),
            ]
            run = dsh_run.run_task(self.task, "offline-model")
            directory = Path(popen.call_args.kwargs["cwd"])
        self.assertEqual(process.communicate.call_args_list[0].args, (self.task.query,))
        self.assertEqual(process.communicate.call_args_list[0].kwargs, {"timeout": 255})
        self.assertEqual(process.communicate.call_count, 2)
        if os.name == "nt":
            terminate.assert_called_once_with(
                ["taskkill", "/PID", "1234", "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            kill_group.assert_not_called()
        else:
            kill_group.assert_called_once_with(1234, signal.SIGKILL)
            terminate.assert_not_called()
        self.assertFalse(directory.exists())
        self.assertEqual(run.stop_reason, "budget")
        self.assertEqual(run.final_answer, "")
        self.assertEqual(run.tool_calls, [])
        self.assertEqual(run.model_turns, 1)
        self.assertEqual((run.prompt_tokens, run.completion_tokens), (4, 2))

    def test_dsh_stop_reasons_and_diagnostics(self):
        for returncode, kind, error_event, forced, expected in (
            (0, "completed", False, False, "final"),
            (0, "completed", False, True, "budget"),
            (1, "blocked", False, False, "budget"),
            (1, "max-tokens", False, False, "budget"),
            (1, "completed", False, True, "error"),
            (0, "completed", True, True, "error"),
            (1, "blocked", True, True, "error"),
            (1, "max-tokens", True, True, "error"),
            (0, "aborted", False, True, "error"),
            (1, "error", False, False, "error"),
            (0, None, False, False, "error"),
        ):
            with self.subTest(returncode=returncode, kind=kind, error=error_event):
                events = [
                    {"type": "status", "phase": "turn_end", "reason": {"kind": kind}},
                    {"type": "final", "text": "2"},
                ]
                if kind is None:
                    events = events[1:]
                if error_event:
                    events.append({"type": "error", "message": "offline"})
                stderr = "unrelated output\ndsh: failure offline-key\n"
                if forced:
                    stderr += "refinx-limits: final answer forced (turns)\n"
                with (
                    patch("refinx.dsh_run.load_key", return_value="offline-key"),
                    patch("refinx.dsh_run.subprocess.Popen") as popen,
                    patch("refinx.dsh_run.sys.stderr", new_callable=io.StringIO) as diagnostic,
                ):
                    process = popen.return_value.__enter__.return_value
                    process.returncode = returncode
                    process.communicate.return_value = (
                        "\n".join(json.dumps(event) for event in events), stderr,
                    )
                    run = dsh_run.run_task(self.task, "offline-model")
                self.assertEqual(run.stop_reason, expected)
                self.assertEqual(run.final_answer, "2")
                self.assertEqual(run.tool_calls, [])
                self.assertNotIn("offline-key", diagnostic.getvalue())
                if expected == "error":
                    self.assertEqual(diagnostic.getvalue(), (
                        f"{self.task.case_id} exit={returncode} reason={kind or 'none'} "
                        "dsh: failure [redacted]\n"
                    ))
                else:
                    self.assertEqual(diagnostic.getvalue(), "")

    def test_evaluate_harness_selection(self):
        run = Run(self.task.case_id, "offline-model", "2", "final", [], [], 1, 0, 0, 0)
        with (
            patch("refinx.run.run_agent", return_value=run) as agent,
            patch("refinx.dsh_run.run_task", return_value=run) as dsh,
            patch("refinx.run.judge_answer") as judge,
        ):
            self.assertEqual(evaluate(self.task, "offline-model", "judge")[0], run)
            agent.assert_not_called()
            self.assertEqual(
                evaluate(self.task, "offline-model", "judge", harness="agent")[0], run
            )
            evaluate(self.task, "offline-model", "judge", run, harness="agent")
        agent.assert_called_once_with(self.task, "offline-model")
        dsh.assert_called_once_with(self.task, "offline-model", "default")
        self.assertEqual(judge.call_count, 3)

    def test_pinned_host_uses_pinned_opener(self):
        for pins, pinned in ((None, False), ("data.example=192.0.2.10", True)):
            with self.subTest(pins=pins):
                environ = {} if pins is None else {"REFINX_PINNED_HOSTS": pins}
                with (
                    patch.dict("refinx.net.os.environ", environ, clear=True),
                    patch("refinx.net._pinned_open") as pinned_open,
                    patch("refinx.net.urlopen") as direct,
                ):
                    opener = pinned_open if pinned else direct
                    opener.return_value.__enter__.return_value = io.BytesIO(b'{"code": 0}')
                    get_json("https://data.example/api/x", {}, 5)
                    self.assertEqual(pinned_open.called, pinned)
                    self.assertEqual(direct.called, not pinned)

    def test_key_dir_overrides_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "key_test.md").write_text("```\nfrom-key-dir\n```\n", encoding="utf-8")
            with patch.dict("refinx.net.os.environ", {"REFINX_KEY_DIR": directory}, clear=True):
                self.assertEqual(load_key("REFINX_TEST_KEY", "key_test.md"), "from-key-dir")

    def test_main_provider_selection(self):
        run = Run(self.task.case_id, "offline-model", "2", "final", [], [], 1, 0, 0, 0)
        for harness, provider_args, provider in (
            ("dsh", [], "default"),
            ("dsh", ["--provider", "api"], "api"),
            ("agent", ["--provider", "api"], "api"),
        ):
            with self.subTest(harness=harness, provider=provider):
                with tempfile.TemporaryDirectory() as directory:
                    with (
                        patch("refinx.run.ROOT", Path(directory)),
                        patch("refinx.run.load_tasks", return_value=[self.task]),
                        patch("refinx.run.run_agent", return_value=run) as agent,
                        patch("refinx.dsh_run.run_task", return_value=run) as dsh,
                        patch(
                            "refinx.run.judge_answer",
                            return_value=Judgment(self.task.case_id, True, "matches"),
                        ),
                        patch("sys.stdout", new_callable=io.StringIO),
                    ):
                        main([
                            "--agent-model", "offline-model", "--harness", harness,
                            "--sample", "1", "--workers", "1", "--name", "offline",
                            *provider_args,
                        ])
                if harness == "dsh":
                    dsh.assert_called_once_with(self.task, "offline-model", provider)
                    agent.assert_not_called()
                else:
                    agent.assert_called_once_with(self.task, "offline-model")
                    dsh.assert_not_called()

    def test_from_run_parallel_judging(self):
        tasks = [self.task, replace(self.task, case_id="offline-2")]
        runs = [
            Run(task.case_id, "offline-model", "2", "final", [], [], 1, 0, 0, 0)
            for task in tasks
        ]
        rows = [asdict(run) for run in runs]
        del rows[0]["trajectory"]
        barrier = Barrier(2, timeout=5)

        def fake_judge(task, final_answer, model):
            barrier.wait()
            return Judgment(task.case_id, True, "matches")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "runs" / "previous" / "runs.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with (
                patch("refinx.run.ROOT", root),
                patch("refinx.run.load_tasks", return_value=tasks),
                patch("refinx.run.judge_answer", side_effect=fake_judge) as judge,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                main([
                    "--from-run", "previous", "--name", "rejudged",
                    "--judge-model", "offline-judge", "--workers", "2",
                ])
            self.assertEqual(judge.call_count, 2)
            judge.assert_has_calls(
                [call(task, "2", "offline-judge") for task in tasks], any_order=True
            )
            output = root / "runs" / "rejudged"
            judgments = [
                json.loads(line)
                for line in (output / "judgments.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertCountEqual(
                judgments, [asdict(Judgment(task.case_id, True, "matches")) for task in tasks]
            )
            self.assertEqual((output / "runs.jsonl").read_bytes(), source.read_bytes())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["applicable"], 2)
            self.assertEqual(summary["adjudicated"], 2)
            self.assertEqual(summary["correct"], 2)

    def test_judge_unescaped_quotes(self):
        text = '{"correct": false, "reason": "称"基本跑通"有误"}'
        response = {"choices": [{"message": {"content": text}}]}
        with patch("refinx.judge.load_key", return_value="offline-key"):
            with patch("refinx.judge.post_json", return_value=response) as request:
                judgment = judge_answer(self.task, "2", "offline-judge")
        self.assertIs(judgment.correct, False)
        self.assertEqual(judgment.reason, text)
        self.assertEqual(request.call_count, 1)

    def test_judge_request(self):
        text = '{"correct": true, "reason": "matches"}'
        response = {"choices": [{"message": {"content": text}}]}
        with patch("refinx.judge.load_key", return_value="offline-key") as key:
            with patch("refinx.judge.post_json", return_value=response) as request:
                judge_answer(self.task, "2", "offline-judge")
        url, headers, body = request.call_args.args
        self.assertEqual(url, LLM_ENDPOINT)
        self.assertEqual(headers, {"Authorization": "Bearer offline-key"})
        self.assertEqual(body["model"], "offline-judge")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 16384)
        self.assertEqual(
            [message["role"] for message in body["messages"]], ["system", "user"]
        )
        self.assertEqual(request.call_args.kwargs["timeout"], JUDGE_TIMEOUT)
        key.assert_called_once_with("REFINX_LLM_API_KEY", "key.md")

    def test_judge_none_content(self):
        response = {"choices": [{"message": {"content": None}}]}
        with patch("refinx.judge.load_key", return_value="offline-key"):
            with patch("refinx.judge.post_json", return_value=response) as request:
                judgment = judge_answer(self.task, "2", "offline-judge")
        self.assertIsNone(judgment.correct)
        self.assertEqual(request.call_count, 2)

    def test_judge_network_failure(self):
        with patch("refinx.judge.load_key", return_value="offline-key"):
            with patch("refinx.judge.post_json", side_effect=URLError("offline")) as request:
                judgment = judge_answer(self.task, "2", "offline-judge")
        self.assertIsNone(judgment.correct)
        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
