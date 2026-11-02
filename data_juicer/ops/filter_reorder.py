import numpy as np

from data_juicer.ops.base_op import Filter, Mapper, NON_STATS_FILTERS
from data_juicer.ops.operator_contract import OperatorContract
from data_juicer.utils.fingerprint_utils import generate_stage_fingerprint, operator_code_signature


class FilterFeatureOp:
    def __init__(self, filter_op):
        object.__setattr__(self, "filter_op", filter_op)
        object.__setattr__(self, "_name", f"{filter_op._name}.feature")
        object.__setattr__(
            self,
            "_op_cfg",
            {
                self._name: {
                    "base_op": filter_op._name,
                    "stage": "feature",
                    "signature": filter_op.get_feature_signature(),
                }
            },
        )

    def __getattr__(self, item):
        return getattr(self.filter_op, item)

    def __setattr__(self, key, value):
        if key in {"filter_op", "_name", "_op_cfg"}:
            object.__setattr__(self, key, value)
        else:
            setattr(self.filter_op, key, value)

    def run(self, dataset, *, exporter=None, tracer=None):
        return self.filter_op.run_feature_stage(dataset, exporter=exporter)

    def get_contract(self):
        return self.filter_op.get_contract("feature")


class FilterReduceOp:
    def __init__(self, filter_op):
        object.__setattr__(self, "filter_op", filter_op)
        object.__setattr__(self, "_name", f"{filter_op._name}.reduce")
        object.__setattr__(
            self,
            "_op_cfg",
            {
                self._name: {
                    "base_op": filter_op._name,
                    "stage": "reduce",
                    "signature": filter_op.get_threshold_signature(),
                }
            },
        )

    def __getattr__(self, item):
        return getattr(self.filter_op, item)

    def __setattr__(self, key, value):
        if key in {"filter_op", "_name", "_op_cfg"}:
            object.__setattr__(self, key, value)
        else:
            setattr(self.filter_op, key, value)

    def run(self, dataset, *, exporter=None, tracer=None):
        return self.filter_op.run_reduce_stage(dataset, tracer=tracer)

    def get_contract(self):
        return self.filter_op.get_contract("reduce")


class FusedReduceOp:
    """Evaluate pending predicates on one row space and prune once."""

    _name = "fused_filter_reduce"

    def __init__(self, filter_ops):
        self.filter_ops = tuple(filter_ops)
        self.batch_size = min(op.batch_size for op in self.filter_ops)
        self._op_cfg = {
            self._name: [
                {"name": op._name, "signature": op.get_threshold_signature()}
                for op in self.filter_ops
            ]
        }

    def use_cuda(self):
        return False

    def is_batched_op(self):
        return True

    def get_contract(self):
        reads = frozenset().union(*(op.get_contract("reduce").read_columns for op in self.filter_ops))
        return OperatorContract(reads, frozenset(), "filter")

    def _process_batch(self, samples):
        expected = len(next(iter(samples.values())))
        packed = np.full((expected + 7) // 8, 255, dtype=np.uint8)
        for op in self.filter_ops:
            mask = np.asarray(list(op.process(samples)), dtype=np.bool_)
            if mask.ndim != 1 or len(mask) != expected:
                raise ValueError("Fused reducer produced a mask with the wrong length")
            np.bitwise_and(packed, np.packbits(mask), out=packed)
        return np.unpackbits(packed, count=expected).astype(np.bool_).tolist()

    def run(self, dataset, *, exporter=None, tracer=None):
        signature = [
            {
                "name": op._name,
                "threshold": op.get_threshold_signature(),
                "code": operator_code_signature(op, "reduce"),
            }
            for op in self.filter_ops
        ]
        signature.append({"fused_code": operator_code_signature(self, "reduce")})
        fingerprint = generate_stage_fingerprint(
            getattr(dataset, "_fingerprint", None), self._name, "reduce", signature
        )
        result = dataset.filter(
            self._process_batch,
            batched=True,
            batch_size=self.batch_size,
            desc=self._name,
            new_fingerprint=fingerprint,
        )
        if tracer:
            for op in self.filter_ops:
                if op._name in tracer.op_list_to_trace:
                    individual = dataset.filter(op.process, batched=True)
                    tracer.trace_filter(op._name, dataset, individual)
        return result


def _is_reorderable_filter(op):
    return (
        isinstance(op, Filter)
        and op._name not in NON_STATS_FILTERS.modules
        and not op.get_contract("feature").opaque
        and not op.get_contract("reduce").opaque
    )


def rewrite_filter_ops_feature_first(ops):
    """Hoist feature extraction across audited row-preserving mappers.

    A row-changing or opaque operator closes the current reduction group.  This
    preserves representative selection for deduplicators and other barriers.
    """
    rewritten_ops = []
    pending = []
    pending_stats = set()

    def flush():
        if pending:
            rewritten_ops.append(FusedReduceOp(pending) if len(pending) > 1 else FilterReduceOp(pending[0]))
            pending.clear()
            pending_stats.clear()

    for op in ops:
        if _is_reorderable_filter(op):
            written = op.get_contract("feature").write_columns
            if pending_stats.intersection(written):
                flush()
            rewritten_ops.append(FilterFeatureOp(op))
            pending.append(op)
            pending_stats.update(written)
            continue

        contract = op.get_contract() if hasattr(op, "get_contract") else None
        mapper_barrier = (
            isinstance(op, Mapper)
            and contract is not None
            and not contract.opaque
            and contract.deterministic
            and contract.row_effect == "preserve"
            and not pending_stats.intersection(contract.read_columns | contract.write_columns)
        )
        if not mapper_barrier:
            flush()
        rewritten_ops.append(op)

    flush()
    return rewritten_ops
