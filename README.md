# DecoupDAG: Eliminating Compute Redundancy in Iterative Data Curation

DecoupDAG is the open-source research prototype accompanying our SIGMOD
submission. It is implemented on top of
[Data-Juicer](https://github.com/datajuicer/data-juicer) and targets redundant
computation in iterative data-cleaning pipelines, especially when users tune
filter thresholds or update data-processing operators.

DecoupDAG preserves Data-Juicer's YAML recipes, operator registry, dataset
abstraction, and text/multimodal processing support. It adds two execution
optimizations:

1. **Operator Decoupling and Topological Reordering (ODTR)** separates expensive
   feature extraction from threshold-based filtering.
2. **Fine-Grained Column-Aware Lineage Hashing (FCLH)** refines cache identity
   from table-level stages toward logical feature columns.


## Design Overview

Iterative curation commonly changes thresholds while keeping expensive feature
computations unchanged. In the original execution pattern:

```text
F1 -> F2 -> ... -> Fn
```

ODTR rewrites a safe Filter group as:

```text
F1.feature -> F2.feature -> ... -> Fn.feature
    -> F1.reduce -> F2.reduce -> ... -> Fn.reduce
```

The feature stage computes row-preserving statistics, while the reduce stage
uses those statistics to apply filtering policies. The intended full design
also treats data-writing operators as dependency barriers and combines
compatible reducer masks with:

```text
B_final = B1 AND B2 AND ... AND Bn
```

FCLH associates lineage metadata with generated feature columns. An operator's
cache identity is intended to depend on its declared read columns and its
normalized code/configuration, rather than on unrelated columns appended to
the same table.

## Changes to Data-Juicer

| Component | Main files | Purpose |
| --- | --- | --- |
| Filter decomposition | `data_juicer/ops/base_op.py` | Adds feature and reduce execution stages |
| Feature-first rewriting | `data_juicer/ops/filter_reorder.py` | Wraps Filter stages and rewrites consecutive Filter groups |
| Executor integration | `data_juicer/core/executor/default_executor.py` | Enables the optimized execution mode |
| Stage fingerprints | `data_juicer/utils/fingerprint_utils.py` | Separates feature and threshold signatures |
| Dataset cache tracing | `data_juicer/core/data/dj_dataset.py` | Records map/filter fingerprint and cache information |
| Configuration | `data_juicer/config/config.py` and `data_juicer/config/config_all.yaml` | Adds `filter_execution_mode` |

The current implementation keeps the physical Hugging Face Datasets cache at
the stage/table level. Logical statistic-column fingerprints are attached to
feature-stage datasets for inspection and future lineage-aware cache
management.

## Repository Layout

```text
data_juicer/       Data-Juicer core, operators, executor, and fingerprints
data-recipes/      Recipes used by the paper workloads
demos/             End-to-end examples
tests/             Unit and integration tests
docs/              Data-Juicer documentation
run_cpu.sh         CPU experiment commands
run_gpu.sh         GPU experiment commands
```

## Installation

The project requires Python 3.10 or newer. The current `uv.lock` resolves the
main environment with:

```text
datasets 3.6.0
pyarrow 19.0.1
numpy 1.26.4
dill 0.3.8
xxhash 3.5.0
ray 2.52.0 (optional)
```

Install with `uv`:

```bash
git clone https://github.com/longyang12/DecoupDAG.git
cd DecoupDAG
uv sync
```

Alternatively, install the editable package with optional dependencies:

```bash
python -m pip install -e ".[nlp,dev]"
```

For distributed execution, install:

```bash
python -m pip install -e ".[distributed]"
```

Vision, audio, and model-backed operators may require additional optional
dependencies and system libraries.

## Quick Start

The normal Data-Juicer execution path remains available:

```bash
python tools/process_data.py \
  --config data-recipes/redpajama-c4-refine-cpu-np1-nocache.yaml
```

Enable the current DecoupDAG feature-first path in a YAML recipe:

```yaml
filter_execution_mode: feature_first
use_checkpoint: false
op_fusion: false
```

Then run the recipe with the same command. The optimized path rewrites
consecutive statistics-based Filters while leaving non-statistical Filters and
other operators in their original positions.

To compare against the baseline, use the same input data, operator
configuration, output directory, and cache environment, and set:

```yaml
filter_execution_mode: normal
```

## Citation

If you use DecoupDAG, please cite:

```bibtex
@inproceedings{decoupdag-sigmod,
  title     = {DecoupDAG: Eliminating Compute Redundancy in Iterative Data Curation via Operator Decoupling},
  author    = {<authors>},
  booktitle = {Proceedings of the ACM SIGMOD International Conference on Management of Data},
  year      = {<year>}
}
```

Because DecoupDAG is derived from Data-Juicer, please also cite the original
Data-Juicer work.

## License

DecoupDAG follows the Apache License 2.0 terms inherited from Data-Juicer.
See [LICENSE](LICENSE) for the complete license and third-party notices.
For the Chinese overview, see [README_ZH.md](README_ZH.md).
