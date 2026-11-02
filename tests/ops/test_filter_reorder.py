import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset

from data_juicer.ops.base_op import OP
from data_juicer.core.data import NestedDataset
from data_juicer.core.data.iteraflow_cache import StageSnapshotCache
from data_juicer.core.data.iteraflow_cache import DeferredSnapshot
from data_juicer.ops.deduplicator.document_simhash_deduplicator import DocumentSimhashDeduplicator
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.filter.words_num_filter import WordsNumFilter
from data_juicer.ops.filter_reorder import FusedReduceOp, rewrite_filter_ops_feature_first
from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper
from data_juicer.ops.operator_contract import OPAQUE_CONTRACT, OperatorContract, contract_for, infer_contract
from data_juicer.utils.constant import Fields
from data_juicer.utils.fingerprint_utils import generate_column_fingerprint


class DynamicSchemaOperator:
    _name = "custom_dynamic"

    def run(self, dataset):
        return {self.output_key: dataset[self.input_key]}


class FilterReorderTest(unittest.TestCase):
    def test_fclh_ablation_uses_full_input_fingerprint(self):
        source = NestedDataset(Dataset.from_list([{"text": "one two"}]))
        feature = rewrite_filter_ops_feature_first([WordsNumFilter(min_num=1, num_proc=1)])[0]
        with tempfile.TemporaryDirectory() as directory:
            aware = StageSnapshotCache(directory, source, column_aware=True)
            coarse = StageSnapshotCache(directory, source, column_aware=False)
            contract = feature.get_contract()
            self.assertNotEqual(aware._key(feature, contract, source), coarse._key(feature, contract, source))
            changed = NestedDataset(source.add_column("label", ["new"]))
            aware.columns["label"] = aware._column_key("other", "label")
            coarse.columns["label"] = coarse._column_key("other", "label")
            self.assertEqual(aware._key(feature, contract, source), aware._key(feature, contract, changed))
            self.assertNotEqual(coarse._key(feature, contract, source), coarse._key(feature, contract, changed))

    def test_algorithm2_accepts_complete_schema_and_rejects_dynamic_schema(self):
        class Explicit:
            _name = "custom_explicit"
            read_columns = {"text"}
            write_columns = {"score"}
            row_effect = "preserve"

        self.assertEqual(infer_contract(Explicit), OperatorContract(frozenset({"text"}), frozenset({"score"})))
        self.assertEqual(infer_contract(DynamicSchemaOperator()), OPAQUE_CONTRACT)

    def test_packed_fused_mask_handles_byte_boundary_and_empty_batch(self):
        first = WordsNumFilter(min_num=2, num_proc=1)
        second = TextLengthFilter(min_len=9, num_proc=1)
        fused = FusedReduceOp([first, second])
        rows = [{"text": "word " * (idx + 1), "id": idx} for idx in range(11)]
        source = NestedDataset(Dataset.from_list(rows))
        with patch.object(OP, "runtime_np", return_value=None):
            for op in (first, second):
                source = op.run_feature_stage(source)
            expected = second.run_reduce_stage(first.run_reduce_stage(source))
            actual = fused.run(source)
        self.assertEqual(actual.to_list(), expected.to_list())
        self.assertEqual(fused._process_batch({Fields.stats: []}), [])

    def test_consecutive_cache_hits_defer_snapshot_read(self):
        source = NestedDataset(Dataset.from_list([{"text": "one two three"}]))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(OP, "runtime_np", return_value=None):
                source.process(
                    rewrite_filter_ops_feature_first([WordsNumFilter(min_num=1, num_proc=1)]),
                    stage_cache=StageSnapshotCache(directory, source), open_monitor=False,
                )
            cache = StageSnapshotCache(directory, source)
            stages = rewrite_filter_ops_feature_first([WordsNumFilter(min_num=1, num_proc=1)])
            with patch("data_juicer.core.data.iteraflow_cache.load_from_disk") as loader:
                current = source
                for stage in stages:
                    current, context = cache.before(stage, current)
                    self.assertTrue(context[2])
                    self.assertIsInstance(current, DeferredSnapshot)
                    cache.after(context, source, current)
                loader.assert_not_called()
            self.assertEqual(cache.materialize(current).to_list()[0]["text"], "one two three")

    def test_stage_cache_warm_hit_and_nondeterministic_descendant(self):
        source = NestedDataset(Dataset.from_list([{"text": "one two"}, {"text": "one two three"}]))

        class ChangingMapper:
            _name = "changing_mapper"
            _op_cfg = {"changing_mapper": {}}

            def __init__(self):
                self.calls = 0

            def get_contract(self):
                return OperatorContract(frozenset(), frozenset({"label"}), deterministic=False)

            def use_cuda(self):
                return False

            def run(self, dataset, **kwargs):
                self.calls += 1
                return NestedDataset(dataset.add_column("label", [str(self.calls)] * len(dataset)))

        with tempfile.TemporaryDirectory() as directory:
            filter_ops = lambda: rewrite_filter_ops_feature_first([WordsNumFilter(min_num=3, num_proc=1)])
            with patch.object(OP, "runtime_np", return_value=None):
                cold = StageSnapshotCache(directory, source)
                expected = source.process(filter_ops(), stage_cache=cold, open_monitor=False)
                warm = StageSnapshotCache(directory, source)
                actual = source.process(filter_ops(), stage_cache=warm, open_monitor=False)
            self.assertEqual((cold.hits, cold.misses), (0, 2))
            self.assertEqual((warm.hits, warm.misses), (2, 0))
            self.assertEqual(expected.to_list(), actual.to_list())

            mapper = ChangingMapper()
            for _ in range(2):
                cache = StageSnapshotCache(directory, source)
                prepared, context = cache.before(mapper, source)
                self.assertFalse(context[2])
                changed = mapper.run(prepared)
                cache.after(context, source, changed)
                descendant = filter_ops()[0]
                _, downstream_context = cache.before(descendant, changed)
                self.assertFalse(downstream_context[2])
            self.assertEqual(mapper.calls, 2)

    def test_stage_snapshots_reuse_features_without_rolling_back_other_columns(self):
        rows = [{"text": "one two", "label": "old"}, {"text": "one two three", "label": "old"}]

        class LabelMapper:
            _name = "test_label_mapper"

            def __init__(self, label):
                self.label = label
                self._op_cfg = {self._name: {"label": label}}

            def get_contract(self):
                return OperatorContract(frozenset(), frozenset({"label"}))

            def use_cuda(self):
                return False

            def run(self, dataset, **kwargs):
                return dataset.map(lambda sample: {**sample, "label": self.label})

        with tempfile.TemporaryDirectory() as directory:
            source = NestedDataset(Dataset.from_list(rows))
            first = WordsNumFilter(min_num=1, num_proc=1)
            first_cache = StageSnapshotCache(directory, source)
            with patch.object(OP, "runtime_np", return_value=None):
                first_result = source.process(
                    [LabelMapper("old"), *rewrite_filter_ops_feature_first([first])],
                    stage_cache=first_cache,
                    open_monitor=False,
                )
            self.assertEqual((first_cache.hits, first_cache.misses), (0, 3))
            self.assertEqual(len(first_result), 2)

            changed = WordsNumFilter(min_num=3, num_proc=1)
            second_cache = StageSnapshotCache(directory, source)
            with patch.object(OP, "runtime_np", return_value=None):
                result = source.process(
                    [LabelMapper("new"), *rewrite_filter_ops_feature_first([changed])],
                    stage_cache=second_cache,
                    open_monitor=False,
                )
            self.assertEqual((second_cache.hits, second_cache.misses), (1, 2))
            self.assertEqual([row["text"] for row in result], ["one two three"])
            self.assertEqual(result["label"], ["new"])
            self.assertEqual(result[Fields.stats][0]["num_words"], 3)

    def test_stage_cache_clears_stale_statistics_on_miss(self):
        source = NestedDataset(Dataset.from_list([{"text": "one two", Fields.stats: {"num_words": 99}}]))
        feature = rewrite_filter_ops_feature_first([WordsNumFilter(num_proc=1)])[0]
        with tempfile.TemporaryDirectory() as directory:
            cache = StageSnapshotCache(directory, source)
            cleaned, context = cache.before(feature, source)
            self.assertFalse(context[2])
            self.assertNotIn("num_words", cleaned[Fields.stats][0])
            with patch.object(OP, "runtime_np", return_value=None):
                result = feature.run(cleaned)
            self.assertEqual(result[Fields.stats][0]["num_words"], 2)

    def test_direct_filter_fingerprints_separate_feature_and_threshold(self):
        dataset = Dataset.from_list([{"text": "one two three"}])
        first = WordsNumFilter(min_num=1, max_num=10, num_proc=1)
        changed_threshold = WordsNumFilter(min_num=2, max_num=10, num_proc=1)
        changed_feature = WordsNumFilter(text_key="body", min_num=1, max_num=10, num_proc=1)
        changed_revision = WordsNumFilter(min_num=1, max_num=10, cache_revision="v2", num_proc=1)

        self.assertEqual(first.get_feature_fingerprint(dataset), changed_threshold.get_feature_fingerprint(dataset))
        self.assertNotEqual(first.get_threshold_fingerprint(dataset), changed_threshold.get_threshold_fingerprint(dataset))
        self.assertNotEqual(first.get_feature_fingerprint(dataset), changed_feature.get_feature_fingerprint(dataset))
        self.assertNotEqual(first.get_feature_fingerprint(dataset), changed_revision.get_feature_fingerprint(dataset))
        original_fingerprint = first.get_feature_fingerprint(dataset)
        with patch("data_juicer.ops.base_op.operator_code_signature", return_value="changed-code"):
            changed_code_fingerprint = first.get_feature_fingerprint(dataset)
        self.assertNotEqual(original_fingerprint, changed_code_fingerprint)

    def test_feature_stage_reuses_statistics_for_threshold_change(self):
        dataset = Dataset.from_list([{"text": "one two"}, {"text": "one two three four"}])
        first = WordsNumFilter(min_num=1, max_num=10, num_proc=1)
        with patch.object(WordsNumFilter, "runtime_np", return_value=None):
            feature_dataset = first.run_feature_stage(dataset)
        self.assertEqual(feature_dataset[Fields.stats][0]["num_words"], 2)
        self.assertEqual(feature_dataset[Fields.stats][1]["num_words"], 4)
        stricter = WordsNumFilter(min_num=3, max_num=10, num_proc=1)
        with patch.object(WordsNumFilter, "runtime_np", return_value=None):
            self.assertEqual(len(stricter.run_reduce_stage(feature_dataset)), 1)
        self.assertEqual(stricter.get_feature_fingerprint(dataset), first.get_feature_fingerprint(dataset))

    def test_contracts_are_conservative_and_use_actual_stat_key(self):
        op = SimpleNamespace(_name="alphanumeric_filter", tokenization=False, text_key="text", skip_op_error=False)
        self.assertEqual(contract_for(op, "feature").write_columns, frozenset({"stats:alnum_ratio"}))
        op.tokenization = True
        self.assertEqual(contract_for(op, "feature").write_columns, frozenset({"stats:alpha_token_ratio"}))
        op.skip_op_error = True
        self.assertEqual(contract_for(op, "feature"), OPAQUE_CONTRACT)
        self.assertEqual(contract_for(SimpleNamespace(_name="unknown")), OPAQUE_CONTRACT)

    def test_feature_inputs_are_not_mistaken_for_thresholds(self):
        motion = WordsNumFilter(num_proc=1)
        motion._name = "video_motion_score_filter"
        self.assertFalse(motion._is_threshold_signature_key("max_size"))
        self.assertTrue(motion._is_threshold_signature_key("max_score"))
        motion._name = "language_id_score_filter"
        self.assertTrue(motion._is_threshold_signature_key("lang"))
        self.assertTrue(motion._is_threshold_signature_key("any_or_all"))
        motion._name = "unreviewed_filter"
        self.assertFalse(motion._is_threshold_signature_key("max_size"))

    def test_column_fingerprint_is_stable_and_column_specific(self):
        kwargs = dict(
            dataset_fingerprint="dataset-fp",
            op_name="words_num_filter",
            stage_name="feature",
            column_signature={"text_key": "text"},
        )
        first = generate_column_fingerprint(column_name="num_words", **kwargs)
        second = generate_column_fingerprint(column_name="num_words", **kwargs)
        other = generate_column_fingerprint(column_name="num_chars", **kwargs)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_rewrite_fuses_reducers_across_row_preserving_mapper(self):
        ops = [
            WordsNumFilter(min_num=2, num_proc=1),
            WhitespaceNormalizationMapper(num_proc=1),
            TextLengthFilter(min_len=5, num_proc=1),
        ]
        rewritten = rewrite_filter_ops_feature_first(ops)
        self.assertEqual([op._name for op in rewritten], [
            "words_num_filter.feature", "whitespace_normalization_mapper",
            "text_length_filter.feature", "fused_filter_reduce",
        ])
        self.assertIsInstance(rewritten[-1], FusedReduceOp)

        rows = [
            {"text": "one two", "id": 1},
            {"text": " one ", "id": 2},
            {"text": "one two three", "id": 3},
        ]

        def run(operations):
            dataset = Dataset.from_list(rows)
            with patch.object(OP, "runtime_np", return_value=None):
                for op in operations:
                    dataset = op.run(dataset)
            return [(row["id"], row["text"]) for row in dataset]

        self.assertEqual(run(ops), run(rewritten))

    def test_rewrite_flushes_before_row_changing_or_unknown_operator(self):
        first = WordsNumFilter(num_proc=1)
        other = TextLengthFilter(num_proc=1)
        barrier = SimpleNamespace(_name="unknown", run=lambda dataset: dataset)
        rewritten = rewrite_filter_ops_feature_first([first, barrier, other])
        self.assertEqual([op._name for op in rewritten], [
            "words_num_filter.feature", "words_num_filter.reduce", "unknown",
            "text_length_filter.feature", "text_length_filter.reduce",
        ])
        repeated = rewrite_filter_ops_feature_first([first, WordsNumFilter(num_proc=1)])
        self.assertEqual([op._name for op in repeated], [
            "words_num_filter.feature", "words_num_filter.reduce",
            "words_num_filter.feature", "words_num_filter.reduce",
        ])
        dedup = DocumentSimhashDeduplicator(num_proc=1)
        with_dedup = rewrite_filter_ops_feature_first([first, dedup, other])
        self.assertEqual([op._name for op in with_dedup], [
            "words_num_filter.feature", "words_num_filter.reduce",
            "document_simhash_deduplicator", "text_length_filter.feature",
            "text_length_filter.reduce",
        ])


if __name__ == "__main__":
    unittest.main()
