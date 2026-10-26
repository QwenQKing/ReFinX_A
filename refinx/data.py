from __future__ import annotations

import json

TASKS = "data/tasks.json"
ORIGINAL = "paper/public-data/ReFInx_data/refinx_data.json"


def _nested(row):
    return {
        "case_id": row["id"],
        "classification": {"language": row["language"]},
        "eval": {
            name: row[name]
            for name in ("query", "as_of", "terminal_action", "gold_answer", "required_clarifications")
        },
        "diagnostic_contract": {
            "domains": row["domains"],
            "task_attributes": {
                "primary_deliverable": row["task_type"],
                "primary_subtype": row["task_subtype"],
                "workflow_attributes": row["workflow_attributes"],
            },
            "correctness_contract": {"accepted_alternatives": row["accepted_alternatives"]},
        },
    }


def load_records(root):

    tasks = root / TASKS
    if tasks.exists():
        with tasks.open(encoding="utf-8") as handle:
            return [_nested(row) for row in json.load(handle)]
    with (root / ORIGINAL).open(encoding="utf-8") as handle:
        return json.load(handle)
