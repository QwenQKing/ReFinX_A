from __future__ import annotations

import json
import re

from .net import LLM_ENDPOINT, load_key, post_json
from .schema import Judgment, Task

SYSTEM_PROMPT = "你是金融问答评分员。只输出一行 JSON，不要输出其他内容。"

TIMEOUT = 600
USER_PROMPT = """Judge whether the response to the financial question is correct. You see only the question and the final response, not how it was produced.

Question (information cutoff {as_of} UTC):
{query}

Expected behaviour: {expected}
Reference answer:
{gold_answer}

Also acceptable:
{accepted_alternatives}

Required clarifications:
{required_clarifications}

Response to judge:
{final_answer}

The response is correct if its substantive conclusions agree with the reference answer or an acceptable alternative (equivalent wording and rounding are fine) and it asserts no material claim that contradicts them. For a clarify task it must also ask for the key missing information.
Output only: {{"correct": true or false, "reason": "<one sentence>"}}"""


def _bullets(items):
    return "\n".join("- " + item for item in items) if items else "(none)"


def judge_answer(task: Task, final_answer: str, model: str) -> Judgment:
    expected = (
        "answer the question" if task.terminal_action == "answer" else
        "ask for the missing information listed below, optionally with conditional answers"
    )
    prompt = USER_PROMPT.format(
        as_of=task.as_of, query=task.query, expected=expected,
        gold_answer=task.gold_answer, final_answer=final_answer,
        accepted_alternatives=_bullets(task.accepted_alternatives),
        required_clarifications=_bullets(task.required_clarifications),
    )
    headers = {
        "Authorization": "Bearer " + load_key("REFINX_LLM_API_KEY", "key.md"),
    }
    body = {
        "model": model, "temperature": 0, "max_tokens": 16384,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    for _ in range(2):
        try:
            response = post_json(LLM_ENDPOINT, headers, body, timeout=TIMEOUT)
        except (OSError, ValueError):
            continue
        raw_text = response["choices"][0]["message"]["content"] or ""
        text = raw_text.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
        if fence:
            text = fence.group(1)
        try:
            verdict = json.loads(text)
            if type(verdict["correct"]) is bool and isinstance(verdict["reason"], str):
                return Judgment(task.case_id, verdict["correct"], verdict["reason"])
        except json.JSONDecodeError:
            match = re.search(r'"correct"\s*:\s*(true|false)', text)
            if match:
                return Judgment(task.case_id, match.group(1) == "true", raw_text)
        except (ValueError, KeyError, TypeError):
            pass
    return Judgment(task.case_id, None, "no verdict")
