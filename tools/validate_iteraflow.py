"""Audit IteraFlow recipes and the local runtime for reviewer reproduction."""

import argparse
import glob
import importlib.metadata
import json
import platform
from pathlib import Path

import yaml


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", default="data-recipes/*-cache.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.recipes))
    recipes = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        dataset = str(config.get("dataset_path", ""))
        recipes.append({
            "path": str(path),
            "use_cache": config.get("use_cache"),
            "filter_execution_mode": config.get("filter_execution_mode", "normal"),
            "dataset_path": dataset,
            "dataset_exists": all(Path(item).exists() for item in dataset.split()) if dataset else False,
            "feature_first_ready": config.get("use_cache") is True and config.get("filter_execution_mode") == "feature_first",
        })
    report = {
        "kind": "iteraflow_recipe_runtime_audit",
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in ("datasets", "pyarrow", "numpy", "dill", "xxhash")},
        "recipe_count": len(recipes),
        "feature_first_ready_count": sum(item["feature_first_ready"] for item in recipes),
        "available_dataset_count": sum(item["dataset_exists"] for item in recipes),
        "recipes": recipes,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if recipes and report["feature_first_ready_count"] != len(recipes):
        raise SystemExit("one or more cache recipes are not feature_first ready")


if __name__ == "__main__":
    main()
