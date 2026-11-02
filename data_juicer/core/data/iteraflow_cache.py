"""Persistent stage snapshots with conservative column-aware reuse."""

import json
import os
import shutil
import uuid

import xxhash
from datasets import load_from_disk

from data_juicer.core.data.dj_dataset import NestedDataset
from data_juicer.utils.constant import Fields
from data_juicer.utils.fingerprint_utils import (
    extract_owner_semantic_state,
    normalize_fingerprint_value,
    operator_code_signature,
)


class DeferredSnapshot:
    """Snapshot metadata; Arrow files are opened only at materialization."""

    def __init__(self, path, manifest):
        self.path = path
        self.column_names = manifest["output_columns"]
        self.num_rows = manifest["output_rows"]
        self._fingerprint = manifest["output_fingerprint"]

    def __len__(self):
        return self.num_rows

    def materialize(self):
        return NestedDataset(load_from_disk(self.path))


class StageSnapshotCache:
    def __init__(self, root, dataset, *, column_aware=True):
        self.root = os.path.join(root, "iteraflow_snapshots")
        os.makedirs(self.root, exist_ok=True)
        self.source = getattr(dataset, "_fingerprint", None)
        self.column_aware = column_aware
        self.row_lineage = self.source
        self.columns = {name: self._column_key(self.source, name) for name in dataset.column_names}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _column_key(key, name):
        return xxhash.xxh64_hexdigest(json.dumps([key, name]).encode("utf-8"))

    @staticmethod
    def materialize(dataset):
        return dataset.materialize() if isinstance(dataset, DeferredSnapshot) else dataset

    def _identity(self, op, contract):
        if op._name.endswith(".feature"):
            signature = op.get_feature_signature()
        elif op._name.endswith(".reduce"):
            signature = op.get_threshold_signature()
        elif op._name == "fused_filter_reduce":
            signature = op._op_cfg
        else:
            signature = getattr(op, "_op_cfg", None) or extract_owner_semantic_state(op)
        reads = sorted(contract.read_columns)
        input_columns = {
            name: self.columns.get(name)
            for name in reads
        }
        return {
            "operator": op._name,
            "signature": normalize_fingerprint_value(signature),
            "code": operator_code_signature(op, "stage"),
            "row": self.row_lineage,
            "reads": input_columns,
        }

    def _key(self, op, contract, dataset):
        identity = self._identity(op, contract)
        if hasattr(op, "filter_op"):
            identity["operator_code"] = operator_code_signature(op.filter_op, op._name.rsplit(".", 1)[-1])
        elif hasattr(op, "filter_ops"):
            identity["operator_code"] = [operator_code_signature(inner, "reduce") for inner in op.filter_ops]
        if not self.column_aware or contract.opaque or not contract.deterministic:
            identity["full_input"] = getattr(dataset, "_fingerprint", None)
            identity["column_aware"] = self.column_aware
        raw = json.dumps(identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return xxhash.xxh64_hexdigest(raw)

    def _path(self, key):
        return os.path.join(self.root, key)

    @staticmethod
    def _merge_writes(dataset, snapshot, writes):
        if len(dataset) != len(snapshot):
            return None
        result = NestedDataset(dataset)
        original_order = list(dataset.column_names)
        stats_keys = {name[6:] for name in writes if name.startswith("stats:")}
        for name in sorted(writes - {f"stats:{key}" for key in stats_keys}):
            if name not in snapshot.column_names:
                return None
            if name in result.column_names:
                result = NestedDataset(result.remove_columns(name))
            result = NestedDataset(result.add_column(name, snapshot[name]))
        if stats_keys:
            if Fields.stats not in snapshot.column_names:
                return None
            old_stats = result[Fields.stats] if Fields.stats in result.column_names else [{} for _ in range(len(result))]
            cached_stats = snapshot[Fields.stats]
            merged = []
            for old, cached in zip(old_stats, cached_stats):
                value = dict(old or {})
                for key in stats_keys:
                    if key not in cached:
                        return None
                    value[key] = cached[key]
                merged.append(value)
            if Fields.stats in result.column_names:
                result = NestedDataset(result.remove_columns(Fields.stats))
            result = NestedDataset(result.add_column(Fields.stats, merged))
        return NestedDataset(result.select_columns(original_order + [name for name in result.column_names if name not in original_order]))

    def before(self, op, dataset, *, allow_hit=True):
        contract = op.get_contract() if hasattr(op, "get_contract") else None
        if contract is None:
            from data_juicer.ops.operator_contract import OPAQUE_CONTRACT

            contract = OPAQUE_CONTRACT
        key = self._key(op, contract, dataset)
        path = self._path(key)
        manifest_path = os.path.join(path, "iteraflow.json")
        if allow_hit and not contract.opaque and contract.deterministic and os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
            if manifest.get("input_columns") == self.columns and "output_columns" in manifest:
                result = DeferredSnapshot(path, manifest)
            elif contract.row_effect == "preserve":
                snapshot = NestedDataset(load_from_disk(path))
                result = self._merge_writes(self.materialize(dataset), snapshot, contract.write_columns)
            else:
                result = None
            if result is not None:
                self.hits += 1
                self._advance(contract, key, result)
                return result, (key, contract, True)
        self.misses += 1
        return self._clear_stale_stats(self.materialize(dataset), contract), (key, contract, False)

    @staticmethod
    def _clear_stale_stats(dataset, contract):
        keys = {name[6:] for name in contract.write_columns if name.startswith("stats:")}
        if not keys or Fields.stats not in dataset.column_names:
            return dataset
        values = [{key: value for key, value in (row or {}).items() if key not in keys} for row in dataset[Fields.stats]]
        cleaned = NestedDataset(dataset.remove_columns(Fields.stats))
        return NestedDataset(cleaned.add_column(Fields.stats, values))

    def after(self, context, input_dataset, output_dataset):
        key, contract, hit = context
        if hit:
            return
        path = self._path(key)
        if not os.path.exists(path):
            temporary = f"{path}.{uuid.uuid4().hex}.tmp"
            try:
                output_dataset.save_to_disk(temporary)
                with open(os.path.join(temporary, "iteraflow.json"), "w", encoding="utf-8") as stream:
                    json.dump({
                        "input_columns": self.columns,
                        "output_columns": output_dataset.column_names,
                        "output_rows": len(output_dataset),
                        "output_fingerprint": output_dataset._fingerprint,
                    }, stream)
                os.rename(temporary, path)
            finally:
                if os.path.exists(temporary):
                    shutil.rmtree(temporary)
        self._advance(contract, key, output_dataset)

    def _advance(self, contract, key, dataset):
        if contract.opaque or not contract.deterministic:
            execution_key = uuid.uuid4().hex
            self.row_lineage = execution_key
            self.columns = {name: execution_key for name in dataset.column_names}
            return
        if contract.row_effect != "preserve":
            self.row_lineage = key
        for name in contract.write_columns:
            self.columns[name] = self._column_key(key, name)
        for name in dataset.column_names:
            self.columns.setdefault(name, self.source)
        self.columns = {name: fp for name, fp in self.columns.items() if name.startswith("stats:") or name in dataset.column_names}
