from __future__ import annotations

import ast
import json
import math
import operator
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from .net import get_json, load_key, post_json

SEARCH_CHANNELS = ["web", "news", "announcement", "report"]


def _env_json(name):
    return json.loads(os.environ.get(name) or "{}")


def _market_data_url(path, params):
    base = os.environ.get("REFINX_MARKET_DATA_URL", "").rstrip("/")
    if not base:
        raise ValueError("REFINX_MARKET_DATA_URL is not set")
    return f"{base}/{path}?{urlencode(params)}"

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
}


def _evaluate(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        return _FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args))
    raise ValueError(
        "unsupported expression; use numbers, + - * / % ** (power, not ^), "
        "parentheses and " + ", ".join(_FUNCTIONS)
    )


def evaluate_expression(expression):
    return _evaluate(ast.parse(expression, mode="eval").body)


def calculate(expression):
    try:
        return str(evaluate_expression(expression))
    except (ArithmeticError, SyntaxError, TypeError, ValueError, RecursionError) as exc:

        return f"Error: {type(exc).__name__}: {exc}"


def search_web(query, count=5, *, language, as_of):
    if os.environ.get("REFINX_SEARCH") == "brave":
        return _search_brave(query, count, language=language, as_of=as_of)
    beijing = timezone(timedelta(hours=8))
    cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00")).timestamp()
    url = os.environ.get("REFINX_SEARCH_URL")
    if not url:
        raise ValueError("REFINX_SEARCH_URL is not set")


    response = post_json(
        url, _env_json("REFINX_SEARCH_HEADERS"),
        {"query": query, "channels": SEARCH_CHANNELS, "size": 20, "output": "block",
         **_env_json("REFINX_SEARCH_BODY")},
        timeout=90,
    )
    if response.get("status_code") != 0:
        return f"Error: {response.get('status_msg')}"

    items = [
        item for item in response.get("data") or []
        if 0 < (item.get("publish_time") or 0) <= cutoff
    ][:min(count, 10)]
    return json.dumps([
        {"title": item.get("title", ""), "url": item.get("url", ""),
         "snippet": item.get("summary") or item.get("content", ""),
         "date": datetime.fromtimestamp(item["publish_time"], beijing).strftime("%Y-%m-%d")}
        for item in items
    ], ensure_ascii=False)


def _search_brave(query, count, *, language, as_of):
    params = {
        "q": query, "count": min(count, 10),
        "search_lang": {"zh": "zh-hans", "en": "en"}[language],
        "result_filter": "web", "text_decorations": "false",
        "freshness": f"2000-01-01to{as_of[:10]}",
    }
    response = get_json(
        "https://api.search.brave.com/res/v1/web/search?" + urlencode(params),
        {"X-Subscription-Token": load_key("BRAVE_API_KEY", "key_search.md"),
         "Accept": "application/json"},
        timeout=90,
    )
    results = response.get("web", {}).get("results", [])
    return json.dumps([
        {"title": item["title"], "url": item["url"],
         "snippet": item.get("description", ""), "date": item.get("page_age")}
        for item in results
    ], ensure_ascii=False)


def get_price_history(symbol, start_date, end_date, *, as_of):
    beijing = timezone(timedelta(hours=8))
    cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    start = int(
        datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=beijing).timestamp() * 1000
    )
    end = min(
        int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=beijing).timestamp() * 1000),
        int(cutoff.timestamp() * 1000) - 15 * 60 * 60 * 1000,
    )
    if start > end:
        return "Error: no trading days completed before the information cutoff in this range"
    params = {
        "symbol": symbol, "start": start, "end": end,
        "interval": "1d", "adjust": "forward",
    }
    response = get_json(
        _market_data_url("prices/historical", params),
        {"X-api-key": load_key("REFINX_MARKET_DATA_KEY", "key_market_data.md"),
         "Accept": "application/json"},
        timeout=90,
    )
    if response["code"] != 0:
        return f"Error: {response['message']}"
    items = sorted(response["data"]["item"], key=lambda item: item["date_ms"])[-250:]
    return json.dumps([
        {
            "date": datetime.fromtimestamp(item["date_ms"] / 1000, beijing).strftime("%Y-%m-%d"),
            "open": item["open_price"], "high": item["high_price"],
            "low": item["low_price"], "close": item["close_price"],
            "volume": item["volume"], "turnover": item["turnover"],
        }
        for item in items
    ], ensure_ascii=False)


def get_financial_statements(symbol, statement, period, limit=4, *, as_of):
    beijing = timezone(timedelta(hours=8))
    cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    try:
        start = cutoff.replace(year=cutoff.year - 10)
    except ValueError:
        start = cutoff.replace(year=cutoff.year - 10, day=28)
    end = int(cutoff.timestamp() * 1000)
    path = {
        "income": "income-statements",
        "balance": "balance-sheets",
        "cash_flow": "cash-flow-statements",
    }.get(statement)
    if path is None:
        return "Error: statement must be one of income, balance, cash_flow"
    params = {
        "symbol": symbol, "period": period,
        "start": int(start.timestamp() * 1000), "end": end,
    }
    response = get_json(
        _market_data_url("financials/" + path, params),
        {"X-api-key": load_key("REFINX_MARKET_DATA_KEY", "key_market_data.md"),
         "Accept": "application/json"},
        timeout=90,
    )
    if response["code"] != 0:
        return f"Error: {response['message']}"
    items = sorted(
        (item for item in response["data"]["item"] if item["report_date_ms"] <= end),
        key=lambda item: item["period_end_ms"], reverse=True,
    )[:min(limit, 12)]
    return json.dumps([
        {
            **{
                key: value for key, value in item.items()
                if key not in ("symbol", "ticker", "period", "period_end_ms", "report_date_ms")
            },
            "period_end": datetime.fromtimestamp(
                item["period_end_ms"] / 1000, beijing
            ).strftime("%Y-%m-%d"),
            "report_date": datetime.fromtimestamp(
                item["report_date_ms"] / 1000, beijing
            ).strftime("%Y-%m-%d"),
        }
        for item in items
    ], ensure_ascii=False)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web", "description": "Search the web for facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer", "default": 5, "maximum": 10},
                },
                "required": ["query"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": (
                "Forward-adjusted (前复权) daily prices for a Shanghai (.SH) or Shenzhen (.SZ) A-share, "
                "only for trading days completed before the information cutoff. "
                "At most 250 most recent days per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "e.g. 600519.SH or 002386.SZ"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["symbol", "start_date", "end_date"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_statements",
            "description": (
                "Income, balance sheet or cash flow statements for a Shanghai (.SH) or "
                "Shenzhen (.SZ) A-share, only from filings published on or before the "
                "information cutoff. Quarterly figures are year-to-date cumulative as "
                "reported. report_date is the latest filing that contains the figures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "e.g. 600519.SH or 002386.SZ"},
                    "statement": {"type": "string", "enum": ["income", "balance", "cash_flow"]},
                    "period": {"type": "string", "enum": ["annual", "quarterly"]},
                    "limit": {"type": "integer", "default": 4, "maximum": 12},
                },
                "required": ["symbol", "statement", "period"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": ("Evaluate arithmetic with +, -, *, /, **, %, "
                            "abs, round, min, max, sqrt, log, exp."),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"], "additionalProperties": False,
            },
        },
    },
]
