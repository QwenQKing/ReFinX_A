from __future__ import annotations

import json
import sys
import time
from collections import Counter

from .net import LLM_ENDPOINT, is_timeout, load_key, post_json
from .schema import Run, Task, ToolCall
from .tools import (
    TOOLS, calculate, get_financial_statements, get_price_history, search_web,
)

WALL_CLOCK_BUDGET = 240
FINAL_ANSWER_RESERVE = 45
MAX_MODEL_TURNS = 10
MAX_TOOL_CALLS = 12
MAX_CALLS_PER_TOOL = 8
MAX_TOTAL_TOKENS = 250_000

SYSTEM_PROMPT = """You are a financial research agent. The information cutoff is {as_of} (UTC): treat anything after it as unknown and do not rely on knowledge of later events.
Use the tools when you need facts, market data or calculations:
- search_web: web search limited to pages dated on or before the cutoff. Snippets are short summaries, not full documents.
- get_price_history: forward-adjusted (前复权) daily prices of Shanghai/Shenzhen A-shares, only for trading days completed before the cutoff.
- get_financial_statements: A-share income, balance sheet and cash flow statements from filings published on or before the cutoff. Quarterly figures are year-to-date cumulative.
- calculate: evaluate an arithmetic expression exactly.
If the question lacks information needed to answer it reliably, ask the user for the specific missing information instead of guessing; you may add conditional answers for the plausible cases.
When you are done, reply with your final answer in the same language as the user's question, without calling tools. State the evidence and calculations your conclusions rely on."""
BUDGET_PROMPT = (
    "Budget reached. Give your final answer now based on what you have, "
    "without calling tools."
)


def request_model(body, timeout):
    return post_json(
        LLM_ENDPOINT,
        {"Authorization": "Bearer " + load_key("REFINX_LLM_API_KEY", "key.md")},
        body, timeout=timeout,
    )


def run_agent(task: Task, model: str, request=request_model) -> Run:
    started = time.monotonic()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(as_of=task.as_of)},
        {"role": "user", "content": task.query},
    ]
    calls = []
    trajectory = []
    counts = Counter()
    turns = prompt_tokens = completion_tokens = 0
    answer, stop_reason = "", "error"
    force_final = False
    while turns < MAX_MODEL_TURNS:
        remaining = WALL_CLOCK_BUDGET - (time.monotonic() - started)
        forced = (
            force_final or turns + 1 >= MAX_MODEL_TURNS or len(calls) >= MAX_TOOL_CALLS
            or remaining <= FINAL_ANSWER_RESERVE
            or prompt_tokens + completion_tokens >= MAX_TOTAL_TOKENS
        )
        if forced:
            messages.append({"role": "user", "content": BUDGET_PROMPT})
        body = {
            "model": model, "messages": messages,
            "temperature": 0, "max_tokens": 8192,
        }
        if not forced:
            body["tools"] = TOOLS
        timeout = max(
            1, min(90, remaining if forced else remaining - FINAL_ANSWER_RESERVE)
        )
        turns += 1
        try:
            response = request(body, timeout)
            usage = response.get("usage") or {}
            prompt_tokens += usage.get("prompt_tokens", 0) or 0
            completion_tokens += usage.get("completion_tokens", 0) or 0
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            thinking = message.get("reasoning_content")
            if thinking:
                trajectory.append({"step": turns, "type": "thinking", "text": thinking})
            if message.get("content"):
                trajectory.append({
                    "step": turns, "type": "text", "text": message["content"],
                })
            for call in tool_calls:
                arguments = call["function"]["arguments"]
                try:
                    parsed = json.loads(arguments)
                    if isinstance(parsed, dict):
                        arguments = parsed
                except json.JSONDecodeError:
                    pass
                trajectory.append({
                    "step": turns, "type": "tool_call", "call_id": call["id"],
                    "name": call["function"]["name"].removeprefix("mcp__refinx__"),
                    "arguments": arguments,
                })
            if forced or not tool_calls:
                answer = message.get("content") or ""
                stop_reason = "budget" if forced else "final"
                break
        except Exception as exc:
            if is_timeout(exc):
                if forced:
                    answer, stop_reason = "", "budget"
                    break
                force_final = True
                continue
            status = getattr(exc, "code", None)
            detail = f" HTTP {status}" if status is not None else ""
            print(f"{task.case_id} {type(exc).__name__}{detail}", file=sys.stderr)
            break
        messages.append({
            "role": "assistant", "content": message.get("content"),
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            name = call["function"]["name"]
            arguments, ok = {}, False
            try:
                parsed = json.loads(call["function"]["arguments"])
                if not isinstance(parsed, dict):
                    raise ValueError("Arguments must be a JSON object")
                arguments = parsed
                if name not in (
                    "search_web", "get_price_history", "get_financial_statements", "calculate"
                ):
                    result = "Error: unknown tool"
                elif len(calls) >= MAX_TOOL_CALLS:
                    result = "Error: total tool call limit reached"
                elif counts[name] >= MAX_CALLS_PER_TOOL:
                    result = "Error: per-tool call limit reached"
                else:
                    counts[name] += 1
                    if name == "search_web":
                        result = search_web(
                            **arguments, language=task.language, as_of=task.as_of
                        )
                    elif name == "get_price_history":
                        result = get_price_history(**arguments, as_of=task.as_of)
                    elif name == "get_financial_statements":
                        result = get_financial_statements(**arguments, as_of=task.as_of)
                    else:
                        result = calculate(**arguments)
                    ok = not result.startswith("Error:")
            except Exception as exc:
                result = f"Error: {type(exc).__name__}"

            calls.append(ToolCall(name, arguments, result, ok))
            trajectory.append({
                "step": turns, "type": "tool_result", "call_id": call["id"],
                "result": result, "is_error": not ok,
            })
            messages.append({
                "role": "tool", "tool_call_id": call["id"], "content": result,
            })
    return Run(
        case_id=task.case_id, model=model, final_answer=answer,
        stop_reason=stop_reason, tool_calls=calls, trajectory=trajectory,
        model_turns=turns,
        latency_s=time.monotonic() - started,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )
