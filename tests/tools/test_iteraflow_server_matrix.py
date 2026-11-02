import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.iteraflow_server_matrix import generate, modify_threshold
from tools.summarize_iteraflow_server import summarize


class ServerMatrixTest(unittest.TestCase):
    def test_summary_uses_only_completed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for system, code, digest in (("iteraflow", 0, "same"), ("no_cache", 0, "same"),
                                         ("monolithic_cache", 1, "different")):
                path = root / "arxiv" / system / "logs" / "R0.result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"workload": "arxiv", "system": system, "round": "R0",
                                            "elapsed_seconds": 2.0, "exit_code": code,
                                            "stage_cache_hits": 1 if system == "iteraflow" else None,
                                            "stage_cache_misses": 1 if system == "iteraflow" else None,
                                            "export_sha256": digest}))
            report = summarize(root)
            self.assertEqual(report["completed_runs"], 2)
            self.assertTrue(report["cross_system_output_checks"][0]["all_available_outputs_equal"])
            self.assertEqual(report["systems"][0]["stage_cache_hit_rate"], 0.5)

    def test_rounds_accumulate_and_ablations_keep_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.json"
            inputs.write_text('{"arxiv": "/server/arxiv.jsonl"}')
            overrides = root / "r4.json"
            overrides.write_text('{"arxiv": {"operator": "toxicity_filter", "parameters": {"model": "server-model"}}}')
            args = SimpleNamespace(inputs=inputs, r4_overrides=overrides, workload="arxiv",
                                   system="all", round="all", output_root=root / "matrix",
                                   run=False, rerun=False, custom_operator_path=[])
            generate(args)
            configs = {}
            for round_name in ("R0", "R1", "R2", "R3", "R4"):
                path = args.output_root / "arxiv" / "iteraflow" / "configs" / f"{round_name}.yaml"
                configs[round_name] = yaml.safe_load(path.read_text())
            self.assertEqual(len(configs["R0"]["process"]), len(configs["R1"]["process"]))
            self.assertEqual(configs["R1"]["process"][6], configs["R4"]["process"][6])
            self.assertEqual(configs["R2"]["process"][11], configs["R4"]["process"][11])
            self.assertEqual(configs["R3"]["process"][15], configs["R4"]["process"][15])
            self.assertIn("toxicity_filter", str(configs["R4"]["process"]))
            ablation = yaml.safe_load((args.output_root / "arxiv" / "without_fclh" / "configs" / "R1.yaml").read_text())
            self.assertTrue(ablation["use_cache"])
            self.assertEqual(ablation["filter_execution_mode"], "feature_first")
            self.assertFalse(ablation["column_aware_cache"])
            self.assertFalse(ablation["skip_op_error"])
            without_odtr = yaml.safe_load(
                (args.output_root / "arxiv" / "without_odtr" / "configs" / "R1.yaml").read_text()
            )
            self.assertEqual(without_odtr["filter_execution_mode"], "normal")
            self.assertFalse(without_odtr["column_aware_cache"])
            self.assertEqual(len(list(args.output_root.rglob("*.yaml"))), 25)

    def test_threshold_round_changes_one_filter(self):
        config = {"process": [{"a_filter": {"min_score": 0.2}},
                              {"b_filter": {"max_score": 0.8}},
                              {"c_filter": {"min_len": 10}}]}
        original = copy.deepcopy(config)
        change = modify_threshold(config, "R2")
        self.assertEqual(change["operator"], "b_filter")
        self.assertEqual(config["process"][0], original["process"][0])
        self.assertEqual(config["process"][2], original["process"][2])


if __name__ == "__main__":
    unittest.main()
