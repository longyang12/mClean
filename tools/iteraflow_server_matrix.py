"""Generate and run the seven-workload R0-R4 server experiment matrix."""

import argparse
import copy
import glob
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "data-recipes"
WORKLOADS = {
    "arxiv": "redpajama-arxiv-refine-cpu-np1-cache.yaml",
    "stackex": "redpajama-stackexchange-refine-cpu-np1-cache.yaml",
    "wiki": "redpajama-wiki-refine-cpu-np1-cache.yaml",
    "github": "redpajama-code-refine-cpu-np1-cache.yaml",
    "c4": "redpajama-c4-refine-cpu-np1-cache.yaml",
    "llava": "llava-pretrain-refine-gpu-np1-cache.yaml",
    "msrvtt": "msr-vtt-refine-gpu-np1-cache.yaml",
}
SYSTEMS = {
    "no_cache": ("normal", False),
    "monolithic_cache": ("normal", True),
    "iteraflow": ("feature_first", True),
    "without_fclh": ("feature_first", True),
    "without_odtr": ("normal", True),
}
THRESHOLD_KEYS = (
    "min_ratio", "max_ratio", "min_len", "max_len", "min_num", "max_num",
    "min_score", "max_score", "min_ppl", "max_ppl", "max_width", "max_height",
    "prob_threshold",
)
R4_TARGET = {
    "arxiv": "flagged_words_filter",
    "stackex": "flagged_words_filter",
    "wiki": "flagged_words_filter",
    "github": "alphanumeric_filter",
    "c4": "perplexity_filter",
    "llava": "image_text_similarity_filter",
    "msrvtt": "video_frames_text_similarity_filter",
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_filters(config):
    return [(idx, name, params) for idx, entry in enumerate(config["process"])
            for name, params in entry.items() if name.endswith("_filter") and isinstance(params, dict)]


def modify_threshold(config, round_name):
    candidates = [(idx, name, params, next(key for key in THRESHOLD_KEYS
                                            if key in params and isinstance(params[key], (int, float))
                                            and not isinstance(params[key], bool)))
                  for idx, name, params in process_filters(config)
                  if any(key in params and isinstance(params[key], (int, float))
                         and not isinstance(params[key], bool) for key in THRESHOLD_KEYS)]
    if not candidates:
        raise ValueError("recipe has no numeric filter thresholds")
    position = {"R1": 0, "R2": len(candidates) // 2, "R3": -1}[round_name]
    idx, name, params, key = candidates[position]
    before = params[key]
    delta = max(abs(before) * 0.05, 0.01)
    after = before + delta if key.startswith("min_") else before - delta
    if isinstance(before, int):
        after = int(after) if after != before else before + (1 if key.startswith("min_") else -1)
    if key.endswith("ratio") or key.endswith("score") or key == "prob_threshold":
        after = min(1.0, max(0.0, after))
    if after == before:
        raise ValueError(f"cannot modify {name}.{key} from {before}")
    params[key] = after
    return {"operator_index": idx, "operator": name, "parameter": key, "before": before, "after": after}


def apply_r4(config, workload, overrides):
    target = R4_TARGET[workload]
    matched = [entry for entry in config["process"] if target in entry]
    if not matched:
        raise ValueError(f"{workload}: R4 target {target} absent")
    if workload in {"llava", "msrvtt"}:
        model = "openai/clip-vit-large-patch14"
        for entry in matched:
            entry[target]["hf_clip"] = model
        return {"operator": target, "upgrade": "CLIP-B/32 to CLIP-L/14", "model": model}
    replacement = overrides.get(workload)
    if not isinstance(replacement, dict) or not isinstance(replacement.get("operator"), str):
        raise ValueError(f"{workload}: R4 requires an override with operator and parameters")
    new_name = replacement["operator"]
    if new_name == target:
        raise ValueError(f"{workload}: R4 replacement must change the operator")
    if workload in {"arxiv", "stackex", "wiki"} and new_name != "toxicity_filter":
        raise ValueError(f"{workload}: paper R4 requires toxicity_filter")
    if workload == "github" and new_name != "ast_syntax_filter":
        raise ValueError("github: paper R4 requires ast_syntax_filter")
    if workload == "c4" and new_name != "perplexity_filter_5gram":
        raise ValueError("c4: paper R4 requires perplexity_filter_5gram")
    parameters = replacement.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"{workload}: parameters must be a mapping")
    matched[0].clear()
    matched[0][new_name] = copy.deepcopy(parameters)
    return {"operator": target, "upgrade": new_name, "parameters": parameters}


def generate(args):
    with args.inputs.open(encoding="utf-8") as stream:
        inputs = json.load(stream)
    overrides = {}
    if args.r4_overrides:
        with args.r4_overrides.open(encoding="utf-8") as stream:
            overrides = json.load(stream)
    workloads = list(WORKLOADS) if args.workload == "all" else [args.workload]
    systems = list(SYSTEMS) if args.system == "all" else [args.system]
    rounds = ["R0", "R1", "R2", "R3", "R4"] if args.round == "all" else [args.round]
    planned = []
    for workload in workloads:
        if workload not in inputs:
            raise ValueError(f"missing input path for {workload}")
        source = inputs[workload]
        if isinstance(source, list):
            source = " ".join(source)
        if not isinstance(source, str) or not source:
            raise ValueError(f"invalid input path for {workload}")
        resolved = []
        for item in source.split():
            matches = sorted(glob.glob(item)) if glob.has_magic(item) else [item]
            if not matches:
                if args.run:
                    raise FileNotFoundError(f"{workload}: source pattern has no files: {item}")
                matches = [item]
            resolved.extend(matches)
        source = " ".join(resolved)
        if args.run and any(not Path(item).exists() for item in resolved):
            raise FileNotFoundError(f"{workload}: source data not found: {source}")
        with (RECIPES / WORKLOADS[workload]).open(encoding="utf-8") as stream:
            base = yaml.safe_load(stream)
        for system in systems:
            for round_name in rounds:
                config = copy.deepcopy(base)
                config["dataset_path"] = source
                config["filter_execution_mode"], config["use_cache"] = SYSTEMS[system]
                config["use_checkpoint"] = False
                config["op_fusion"] = False
                config["open_tracer"] = False
                config["open_monitor"] = False
                config["skip_op_error"] = False
                config["column_aware_cache"] = system not in {"without_fclh", "without_odtr"}
                if args.custom_operator_path:
                    config["custom_operator_paths"] = [str(path) for path in args.custom_operator_path]
                changes = []
                depth = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 3}[round_name]
                for prior in ("R1", "R2", "R3")[:depth]:
                        changes.append(modify_threshold(config, prior))
                if round_name == "R4":
                    changes.append(apply_r4(config, workload, overrides))
                common = args.output_root / workload / system
                config["work_dir"] = str(common / "work")
                config["ds_cache_dir"] = str(common / "hf_cache")
                config["export_path"] = str(common / "export" / f"{round_name}.jsonl")
                path = common / "configs" / f"{round_name}.yaml"
                planned.append((path, config, changes, workload, system, round_name))
    for path, config, changes, workload, system, round_name in planned:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        metadata = {"workload": workload, "system": system, "round": round_name,
                    "config": str(path), "config_sha256": canonical_hash(config), "changes": changes}
        (path.with_suffix(".json")).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if args.run:
            log_path = path.parents[1] / "logs" / f"{round_name}.log"
            result_path = path.parents[1] / "logs" / f"{round_name}.result.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if result_path.exists() and not args.rerun:
                raise FileExistsError(f"result exists: {result_path}; use --rerun to replace")
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run([sys.executable, str(ROOT / "tools/process_data.py"), "--config", str(path)],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"IteraFlow stage cache: (\d+) hits, (\d+) misses", log_text)
            compiled = re.findall(
                r"IteraFlow compiled plan: (\d+) feature stages, (\d+) standalone reducers, "
                r"(\d+) fused reducers, (\d+) total stages",
                log_text,
            )
            if system in {"iteraflow", "without_fclh"} and completed.returncode == 0:
                if not compiled or int(compiled[-1][0]) == 0:
                    completed = subprocess.CompletedProcess(completed.args, 2)
            exported = Path(config["export_path"])
            result_path.write_text(json.dumps({
                **metadata,
                "elapsed_seconds": time.time() - started,
                "exit_code": completed.returncode,
                "log": str(log_path),
                "stage_cache_hits": int(matches[-1][0]) if matches else None,
                "stage_cache_misses": int(matches[-1][1]) if matches else None,
                "compiled_plan": {
                    "feature_stages": int(compiled[-1][0]),
                    "standalone_reducers": int(compiled[-1][1]),
                    "fused_reducers": int(compiled[-1][2]),
                    "total_stages": int(compiled[-1][3]),
                } if compiled else None,
                "export_sha256": file_sha256(exported) if completed.returncode == 0 and exported.is_file() else None,
            }, indent=2) + "\n")
            if completed.returncode:
                raise RuntimeError(f"{workload}/{system}/{round_name} failed; see {log_path}")
    print(f"Generated {len(planned)} configs under {args.output_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="JSON mapping workload names to server data paths")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--r4-overrides", type=Path, help="JSON mapping text/code workloads to R4 replacement operators")
    parser.add_argument(
        "--custom-operator-path", type=Path, action="append", default=[],
        help="Custom operator module or directory required by R4; repeat for multiple paths",
    )
    parser.add_argument("--workload", choices=["all", *WORKLOADS], default="all")
    parser.add_argument("--system", choices=["all", *SYSTEMS], default="all")
    parser.add_argument("--round", choices=["all", "R0", "R1", "R2", "R3", "R4"], default="all")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
