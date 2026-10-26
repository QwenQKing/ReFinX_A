import json
import sys
from pathlib import Path

LIMIT = 131072 - 8192


def main():
    run_dir = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    runs_path = run_dir / "runs.jsonl"
    latest = {}
    with runs_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[row["case_id"]] = row
    marks_path = run_dir / "context_exceeded_131072.json"
    marks = json.loads(marks_path.read_text(encoding="utf-8")) if marks_path.exists() else {"cases": {}}
    without_usage, new = 0, {}
    for case_id, row in latest.items():
        usage = [entry for entry in row.get("trajectory", []) if entry.get("type") == "usage"]
        if not usage:
            without_usage += 1
            continue
        peak = max((entry.get("input") or 0) + (entry.get("cache_read") or 0) for entry in usage)
        if peak > LIMIT and row["stop_reason"] != "error":
            new[case_id] = (row, peak)
    print(f"{run_dir.name}: {len(latest)} cases, {without_usage} without usage entries, "
          f"{len(new)} new over {LIMIT}: {sorted((peak, cid) for cid, (_, peak) in new.items())}")
    if dry_run or not new:
        return
    with runs_path.open("a", encoding="utf-8") as handle:
        for row, _ in new.values():
            handle.write(json.dumps(dict(row, stop_reason="error", final_answer="", tool_calls=[], trajectory=[]),
                                    ensure_ascii=False) + "\n")
    marks["reason"] = (f"Largest request exceeded {LIMIT} prompt tokens (input + cache_read of the model's own "
                       "usage): with 8192 output tokens it would not fit the 131072-token context of the locally "
                       "served models. Marked unobservable by appending an error row; the original rows are kept "
                       "above.")
    marks["cases"].update({case_id: peak for case_id, (_, peak) in new.items()})
    marks_path.write_text(json.dumps(marks, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
