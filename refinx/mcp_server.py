from __future__ import annotations

import json
import os
from dataclasses import asdict

from .schema import ToolCall
from .tools import (
    TOOLS, calculate, get_financial_statements, get_price_history, search_web,
)


def call_tool(name, arguments):
    ok = False
    try:
        if name == "search_web":
            result = search_web(
                **arguments, language=os.environ["REFINX_LANGUAGE"],
                as_of=os.environ["REFINX_AS_OF"],
            )
        elif name == "get_price_history":
            result = get_price_history(**arguments, as_of=os.environ["REFINX_AS_OF"])
        elif name == "get_financial_statements":
            result = get_financial_statements(
                **arguments, as_of=os.environ["REFINX_AS_OF"]
            )
        elif name == "calculate":
            result = calculate(**arguments)
        else:
            result = "Error: unknown tool"
        ok = not result.startswith("Error:")
    except Exception as exc:
        result = f"Error: {type(exc).__name__}"
    record = ToolCall(name, arguments, result, ok)
    with open(os.environ["REFINX_TOOL_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        handle.flush()
    return result


def main():
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.tools import Tool

    def search_web(query: str, count: int = 5) -> str:
        return call_tool("search_web", {"query": query, "count": count})

    def get_price_history(symbol: str, start_date: str, end_date: str) -> str:
        return call_tool("get_price_history", {
            "symbol": symbol, "start_date": start_date, "end_date": end_date,
        })

    def get_financial_statements(
        symbol: str, statement: str, period: str, limit: int = 4
    ) -> str:
        return call_tool("get_financial_statements", {
            "symbol": symbol, "statement": statement, "period": period, "limit": limit,
        })

    def calculate(expression: str) -> str:
        return call_tool("calculate", {"expression": expression})

    handlers = {
        function.__name__: function for function in (
            search_web, get_price_history, get_financial_statements, calculate,
        )
    }
    registered = []
    for entry in TOOLS:
        definition = entry["function"]
        tool = Tool.from_function(
            handlers[definition["name"]], name=definition["name"],
            description=definition["description"], structured_output=False,
        )

        tool.parameters = definition["parameters"]
        registered.append(tool)
    server = FastMCP("refinx", tools=registered)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
