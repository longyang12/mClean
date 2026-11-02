"""Conservative operator semantics used by IteraFlow's compiler.

An unknown operator is opaque.  In particular, a missing declaration must
never be interpreted as an empty read or write set.
"""

from dataclasses import dataclass
from typing import FrozenSet

from data_juicer.utils.constant import HashKeys


@dataclass(frozen=True)
class OperatorContract:
    read_columns: FrozenSet[str]
    write_columns: FrozenSet[str]
    row_effect: str = "preserve"
    deterministic: bool = True
    opaque: bool = False

    def __post_init__(self):
        if self.row_effect not in {"preserve", "filter", "reorder"}:
            raise ValueError(f"Unsupported row effect: {self.row_effect}")
        if self.opaque and (self.read_columns or self.write_columns):
            raise ValueError("Opaque contracts must not claim known dependencies")


OPAQUE_CONTRACT = OperatorContract(frozenset(), frozenset(), "reorder", False, True)


def _declared_contract(op):
    """Infer a contract from explicit schema attributes on a custom operator.

    This is intentionally strict: both sets must be supplied and every value
    must be a string. Partial declarations are rejected instead of guessed.
    """
    reads = getattr(op, "read_columns", None)
    writes = getattr(op, "write_columns", None)
    if reads is None or writes is None:
        return None
    if not all(isinstance(value, str) for value in (*reads, *writes)):
        return OPAQUE_CONTRACT
    return OperatorContract(
        frozenset(reads),
        frozenset(writes),
        getattr(op, "row_effect", "preserve"),
        bool(getattr(op, "deterministic", True)),
    )


def infer_contract(op):
    """Resolve a custom operator schema without guessing partial dependencies."""
    return _declared_contract(op) or OPAQUE_CONTRACT


# Values are logical names.  `stats:<key>` addresses one entry in the
# physical __dj__stats__ dictionary, rather than the whole Arrow column.
FILTER_STATS = {
    "alphanumeric_filter": ("alnum_ratio", "alpha_token_ratio"),
    "average_line_length_filter": ("avg_line_length",),
    "character_repetition_filter": ("char_rep_ratio",),
    "flagged_words_filter": ("flagged_words_ratio",),
    "image_aspect_ratio_filter": ("aspect_ratios",),
    "image_shape_filter": ("image_width", "image_height"),
    "image_size_filter": ("image_sizes",),
    "image_text_matching_filter": ("image_text_matching_score",),
    "image_text_similarity_filter": ("image_text_similarity",),
    "language_id_score_filter": ("lang", "lang_score"),
    "maximum_line_length_filter": ("max_line_length",),
    "perplexity_filter": ("perplexity",),
    "special_characters_filter": ("special_char_ratio",),
    "text_length_filter": ("text_len",),
    "video_aesthetics_filter": ("video_frames_aesthetics_score",),
    "video_frames_text_similarity_filter": ("video_frames_text_similarity",),
    "video_motion_score_filter": ("video_motion_score",),
    "video_nsfw_filter": ("video_nsfw_score",),
    "video_watermark_filter": ("video_watermark_prob",),
    "word_repetition_filter": ("word_rep_ratio",),
    "words_num_filter": ("num_words",),
}

IMAGE_FILTERS = {
    "image_aspect_ratio_filter", "image_shape_filter", "image_size_filter",
    "image_text_matching_filter", "image_text_similarity_filter",
}
VIDEO_FILTERS = {
    "video_aesthetics_filter", "video_frames_text_similarity_filter",
    "video_motion_score_filter", "video_nsfw_filter", "video_watermark_filter",
}
TEXT_AND_IMAGE_FILTERS = {"image_text_matching_filter", "image_text_similarity_filter"}
TEXT_AND_VIDEO_FILTERS = {"video_frames_text_similarity_filter"}
TEXT_MAPPERS = {
    "clean_copyright_mapper", "clean_email_mapper", "clean_links_mapper",
    "fix_unicode_mapper", "punctuation_normalization_mapper",
    "whitespace_normalization_mapper",
}


def contract_for(op, stage=None):
    """Return a declared contract for supported paper-workload operators.

    Other operators remain opaque until audited.  This deliberately avoids
    guessing dependencies from the presence of `text_key` on the base class.
    """
    name = op._name
    if getattr(op, "skip_op_error", False):
        return OPAQUE_CONTRACT
    if name in FILTER_STATS:
        stats = FILTER_STATS[name]
        if name == "alphanumeric_filter":
            stats = ("alpha_token_ratio",) if op.tokenization else ("alnum_ratio",)
        output = frozenset(f"stats:{key}" for key in stats)
        if name in TEXT_AND_IMAGE_FILTERS:
            source = {op.text_key, op.image_key}
        elif name in TEXT_AND_VIDEO_FILTERS:
            source = {op.text_key, op.video_key}
        elif name in IMAGE_FILTERS:
            source = {op.image_key, op.image_bytes_key}
        elif name in VIDEO_FILTERS:
            source = {op.video_key, op.video_bytes_key}
        else:
            source = {op.text_key}
        if name in TEXT_AND_IMAGE_FILTERS:
            source.add(op.image_bytes_key)
        if name in TEXT_AND_VIDEO_FILTERS:
            source.add(op.video_bytes_key)
        if stage == "feature":
            return OperatorContract(frozenset(source), output)
        if stage == "reduce":
            return OperatorContract(output, frozenset(), "filter")
        return OperatorContract(frozenset(source), output, "filter")
    if stage is not None:
        return OPAQUE_CONTRACT
    if name in TEXT_MAPPERS:
        column = frozenset({op.text_key})
        return OperatorContract(column, column)
    if name == "document_simhash_deduplicator":
        return OperatorContract(frozenset({op.text_key}), frozenset({HashKeys.simhash}), "filter")
    return infer_contract(op)
