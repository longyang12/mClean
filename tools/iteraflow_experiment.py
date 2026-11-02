"""Small, reproducible IteraFlow cache and rewrite acceptance experiment."""

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import datasets
from datasets import Dataset
from loguru import logger

from data_juicer.core.data import NestedDataset
from data_juicer.core.data.iteraflow_cache import StageSnapshotCache
from data_juicer.ops.base_op import OP
from data_juicer.ops.filter.special_characters_filter import SpecialCharactersFilter
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.filter.words_num_filter import WordsNumFilter
from data_juicer.ops.filter_reorder import rewrite_filter_ops_feature_first


FIXTURE = [
    {"id": 1, "text": "one two three", "body": "one two three"},
    {"id": 2, "text": "four five", "body": "four five six"},
    {"id": 3, "text": "seven eight nine ten", "body": "seven eight"},
    {"id": 4, "text": "short!", "body": "short"},
    {"id": 5, "text": "eleven twelve thirteen", "body": "eleven twelve thirteen"},
]

ROUNDS = (
    ("R0", {}),
    ("R1", {"min_words": 3}),
    ("R2", {"min_words": 3, "min_length": 14}),
    ("R3", {"min_words": 3, "min_length": 14, "max_special": 0.12}),
    ("R4_smoke", {"min_words": 3, "min_length": 14, "max_special": 0.12, "words_source": "body"}),
)


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def operators(changes):
    return [
        WordsNumFilter(min_num=changes.get("min_words", 1), text_key=changes.get("words_source", "text"), num_proc=1),
        TextLengthFilter(min_len=changes.get("min_length", 1), num_proc=1),
        SpecialCharactersFilter(max_ratio=changes.get("max_special", 1.0), num_proc=1),
    ]


def run(source, changes, cache_root=None, rewrite=False):
    ops = operators(changes)
    if rewrite:
        ops = rewrite_filter_ops_feature_first(ops)
    cache = StageSnapshotCache(cache_root, source) if cache_root is not None else None
    start = perf_counter()
    with patch.object(OP, "runtime_np", return_value=None):
        result = source.process(ops, stage_cache=cache, open_monitor=False)
    elapsed = perf_counter() - start
    return {
        "seconds": round(elapsed, 6),
        "rows": len(result),
        "result_sha256": digest(result.to_list()),
        "hits": cache.hits if cache else None,
        "misses": cache.misses if cache else None,
        "operators": [op._name for op in ops],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, help="Optional local JSONL with a text field")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_rows < 1:
        parser.error("--max-rows must be positive")
    logger.remove()
    datasets.disable_progress_bars()
    if args.input_jsonl:
        with args.input_jsonl.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for _, line in zip(range(args.max_rows), stream)]
        if any("text" not in row for row in rows):
            parser.error("input rows require a text field")
        body_fallback = any("body" not in row for row in rows)
        rows = [{**row, "body": row.get("body", row["text"])} for row in rows]
    else:
        rows = FIXTURE
        body_fallback = False
    if not rows:
        parser.error("input has no rows")
    source = NestedDataset(Dataset.from_list(rows))
    report = {
        "kind": "functional_smoke_not_paper_benchmark",
        "python": platform.python_version(),
        "datasets": datasets.__version__,
        "input": str(args.input_jsonl) if args.input_jsonl else "built-in fixture",
        "input_rows": len(source),
        "input_sha256": digest(rows),
        "body_fallback_to_text": body_fallback,
        "rounds": [],
    }
    with tempfile.TemporaryDirectory(prefix="iteraflow-experiment-") as directory:
        for name, changes in ROUNDS:
            original = run(source, changes)
            compiled = run(source, changes, rewrite=True)
            cached = run(source, changes, cache_root=directory, rewrite=True)
            equal = original["result_sha256"] == compiled["result_sha256"] == cached["result_sha256"]
            report["rounds"].append({
                "name": name,
                "changes": changes,
                "original": original,
                "compiled": compiled,
                "cached": cached,
                "output_equal": equal,
            })
            if not equal:
                raise AssertionError(f"{name}: original, compiled, and cached outputs differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(report['rounds'])} rounds, all output_equal")


if __name__ == "__main__":
    main()
