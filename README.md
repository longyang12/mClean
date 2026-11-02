# IteraFlow

**Eliminating Compute Redundancy in Iterative Data Curation via Operator Decoupling**

IteraFlow is the research prototype accompanying our ICDE submission. It extends
[Data-Juicer](https://github.com/datajuicer/data-juicer) with an execution engine
for iterative data curation, where filter thresholds and feature extractors are
repeatedly updated over the same raw corpus.

IteraFlow addresses two sources of redundant computation:

- **Operator Decoupling and Topological Reordering (ODTR)** splits a statistical
  filter into a row-preserving feature extractor and a threshold reducer. Feature
  stages are reordered within Mapper and Deduplicator barriers, and compatible
  reducers are fused into one packed-bitmask filtering pass.
- **Fine-Grained Column-Aware Lineage Hashing (FCLH)** derives cache keys from an
  operator's code, configuration, and referenced column lineages. Cached table
  snapshots are reused without invalidating operators whose input columns are
  unchanged.

## Implementation

The optimized execution path is enabled by two YAML options:

```yaml
filter_execution_mode: feature_first
use_cache: true
```

| Component | Location |
| --- | --- |
| Filter feature/reducer decomposition | `data_juicer/ops/base_op.py` |
| Barrier-aware ODTR compiler and bitmask fusion | `data_juicer/ops/filter_reorder.py` |
| Operator read/write schemas | `data_juicer/ops/operator_contract.py` |
| Column lineage and table snapshot cache | `data_juicer/core/data/iteraflow_cache.py` |
| Runtime integration | `data_juicer/core/executor/default_executor.py` |
| Stable code/config fingerprints | `data_juicer/utils/fingerprint_utils.py` |

Unknown operators are treated as write barriers unless they expose complete
`read_columns` and `write_columns` schemas. `feature_first` disables per-sample
error skipping because feature extraction must preserve cardinality before the
fused reduction stage.

## Installation

IteraFlow requires Python 3.10 or later. The paper environment uses Datasets
3.6.0, PyArrow 19.0.1, NumPy 1.26.4, dill 0.3.8, and xxhash 3.5.0.

```bash
git clone https://github.com/longyang12/IteraFlow.git
cd IteraFlow
uv sync
```

An editable installation is also supported:

```bash
python -m pip install -e ".[nlp,dev]"
```

Image and video workloads require the corresponding Data-Juicer optional
dependencies and access to the models specified in their recipes.

## Quick Start

Run an IteraFlow recipe with the standard Data-Juicer entry point:

```bash
python tools/process_data.py \
  --config data-recipes/redpajama-arxiv-refine-cpu-np1-cache.yaml
```

Each run logs the compiled plan and cache statistics:

```text
IteraFlow compiled plan: ... feature stages, ... fused reducers, ... total stages
IteraFlow stage cache: ... hits, ... misses
```

Snapshots are stored in `<work_dir>/iteraflow_snapshots`. Reuse the same
`work_dir` across iterative rounds and use a separate directory for each system
variant.

## Reproducing the ICDE Experiments

The paper evaluates seven workloads over five rounds:

| Workload | Modality | Samples | Size | Pipeline |
| --- | --- | ---: | ---: | --- |
| ArXiv | Text | 1,558,306 | 87.35 GB | 5M + 10F + 1D |
| StackExchange | Text | 29,825,086 | 74.55 GB | 5M + 11F + 1D |
| Wikipedia | Text | 29,834,171 | 111.89 GB | 5M + 11F + 1D |
| GitHub | Text | 28,793,312 | 212.18 GB | 6M + 8F + 1D |
| C4 | Text | 364,599,496 | 806.77 GB | 5M + 9F + 1D |
| LLaVA | Image-text | 595,375 | 73.14 GB | 2M + 11F |
| MSRVTT | Video-text | 180,000 | 27.04 GB | 7F |

The trace consists of R0 cold execution; R1 head-filter threshold tuning; R2
middle-filter threshold tuning; R3 tail-filter threshold tuning; and R4 operator
or model upgrading. R4 replaces flagged-word filtering with toxicity
classification for knowledge text, KenLM 3-gram with 5-gram for C4,
alphanumeric filtering with AST syntax validation for GitHub, and CLIP-B with
CLIP-L for multimodal workloads.

Create a JSON file mapping workload names to server paths:

```json
{
  "arxiv": "/data/arxiv/*.jsonl",
  "stackex": "/data/stackexchange/*.jsonl",
  "wiki": "/data/wikipedia/*.jsonl",
  "github": "/data/github/*.jsonl",
  "c4": "/data/c4/*.jsonl",
  "llava": "/data/llava/chat-dj.jsonl",
  "msrvtt": "/data/msrvtt/msrvtt_train_dj.jsonl"
}
```

For R4, provide server-registered text and code operators. Their modules can be
loaded with Data-Juicer's `custom_operator_paths` option.

```json
{
  "arxiv": {"operator": "toxicity_filter", "parameters": {}},
  "stackex": {"operator": "toxicity_filter", "parameters": {}},
  "wiki": {"operator": "toxicity_filter", "parameters": {}},
  "github": {"operator": "ast_syntax_filter", "parameters": {}},
  "c4": {"operator": "perplexity_filter_5gram", "parameters": {}}
}
```

Generate the complete experiment matrix:

```bash
python tools/iteraflow_server_matrix.py \
  --inputs /path/to/server-inputs.json \
  --r4-overrides /path/to/r4-overrides.json \
  --custom-operator-path /path/to/r4_operators \
  --output-root /scratch/iteraflow-icde
```

Add `--run` to execute the generated configurations. A single workload, system,
or round can be selected with `--workload`, `--system`, and `--round`.

| System | Configuration |
| --- | --- |
| No-Cache | `normal`, cache disabled |
| Monolithic-Cache | `normal`, Data-Juicer table cache enabled |
| IteraFlow | `feature_first`, FCLH enabled |
| w/o FCLH | `feature_first`, table-level cache keys |
| w/o ODTR | monolithic filters and table-level cache keys |

Each run writes its YAML, configuration hash, raw log, wall-clock time, stage
cache counts, and output SHA-256 under the experiment root. Summarize completed
runs with:

```bash
python tools/summarize_iteraflow_server.py \
  --root /scratch/iteraflow-icde \
  --output /scratch/iteraflow-icde/summary.json
```

## Tests

```bash
export NUMBA_CACHE_DIR=/tmp/iteraflow-numba-cache
export MPLCONFIGDIR=/tmp/iteraflow-mpl-cache

python -m unittest discover -s tests/ops -p test_filter_reorder.py -q
python -m unittest discover -s tests/tools -p test_iteraflow_server_matrix.py -q
python -m unittest discover -s tests/utils -p test_fingerprint_utils.py -q
```

The tests cover filter decomposition, barrier placement, fused packed masks,
column-aware invalidation, lazy snapshot reuse, schema fallbacks, experiment
round generation, ablation settings, and result aggregation.

## Citation

```bibtex
@inproceedings{yang2026iteraflow,
  title     = {IteraFlow: Eliminating Compute Redundancy in Iterative Data Curation via Operator Decoupling},
  author    = {Yang, Long and Wang, Jing and Li, Yaqi and Shao, Yuchen and Liu, Xuan and Wan, Chengcheng and Shi, Liang},
  booktitle = {Proceedings of the IEEE International Conference on Data Engineering},
  year      = {2026}
}
```

IteraFlow is built on Data-Juicer. Please cite Data-Juicer when using the
underlying operator library or recipes.

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE).
