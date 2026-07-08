from data_juicer.ops.base_op import Filter, NON_STATS_FILTERS


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


def _is_reorderable_filter(op):
    return isinstance(op, Filter) and op._name not in NON_STATS_FILTERS.modules


def _rewrite_filter_group(group):
    feature_ops = [FilterFeatureOp(op) for op in group]
    reduce_ops = [FilterReduceOp(op) for op in group]
    return feature_ops + reduce_ops


def rewrite_filter_ops_feature_first(ops):
    """
    Rewrite consecutive Filter segments into feature stages followed by reduce
    stages. Non-filter operators and NON_STATS_FILTERS stay in place.
    """

    rewritten_ops = []
    idx = 0
    while idx < len(ops):
        op = ops[idx]
        if not _is_reorderable_filter(op):
            rewritten_ops.append(op)
            idx += 1
            continue

        filter_group = [op]
        idx += 1
        while idx < len(ops) and _is_reorderable_filter(ops[idx]):
            filter_group.append(ops[idx])
            idx += 1
        rewritten_ops.extend(_rewrite_filter_group(filter_group))

    return rewritten_ops
