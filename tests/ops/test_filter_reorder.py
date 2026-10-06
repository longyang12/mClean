import unittest

from data_juicer.utils.fingerprint_utils import generate_column_fingerprint


class FilterReorderTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
