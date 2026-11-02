"""Summarize server R0-R4 results without inferring missing measurements."""

import argparse
import json
from pathlib import Path


def summarize(root):
    rows = []
    for path in sorted(root.glob("*/*/logs/R*.result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("exit_code") != 0:
            continue
        rows.append(record)
    groups = {}
    for record in rows:
        key = (record["workload"], record["system"])
        groups.setdefault(key, []).append(record)
    summary = []
    for (workload, system), records in sorted(groups.items()):
        records.sort(key=lambda item: item["round"])
        hits = [item["stage_cache_hits"] for item in records if item.get("stage_cache_hits") is not None]
        misses = [item["stage_cache_misses"] for item in records if item.get("stage_cache_misses") is not None]
        hit_rate = sum(hits) / (sum(hits) + sum(misses)) if hits and sum(hits) + sum(misses) else None
        summary.append({
            "workload": workload,
            "system": system,
            "completed_rounds": [item["round"] for item in records],
            "cumulative_seconds": sum(item["elapsed_seconds"] for item in records),
            "stage_cache_hit_rate": hit_rate,
            "rounds": {item["round"]: {"seconds": item["elapsed_seconds"],
                                        "hits": item.get("stage_cache_hits"),
                                        "misses": item.get("stage_cache_misses"),
                                        "export_sha256": item.get("export_sha256")} for item in records},
        })
    comparisons = []
    for workload in sorted({item["workload"] for item in rows}):
        for round_name in ("R0", "R1", "R2", "R3", "R4"):
            selected = [item for item in rows if item["workload"] == workload and item["round"] == round_name]
            hashes = {item["system"]: item.get("export_sha256") for item in selected}
            comparisons.append({"workload": workload, "round": round_name,
                                "systems_with_results": sorted(hashes),
                                "all_available_outputs_equal": len(set(hashes.values())) == 1 and None not in hashes.values()
                                if len(hashes) >= 2 else None})
    return {"kind": "iteraflow_server_result_summary", "completed_runs": len(rows),
            "systems": summary, "cross_system_output_checks": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {report['completed_runs']} completed runs")


if __name__ == "__main__":
    main()
