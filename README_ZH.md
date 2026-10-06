# DecoupDAG: Eliminating Compute Redundancy in Iterative Data Curation via Operator Decoupling

**DecoupDAG** 是我们准备投稿 SIGMOD 的论文工作的开源研究项目，用于构建可复用、可缓存的数据清洗流水线。本项目是在 [Data-Juicer](https://github.com/datajuicer/data-juicer) 基础上的**源码级二次开发**：保留 Data-Juicer 的配置驱动算子体系、Hugging Face Datasets 执行模型，以及文本和多模态数据支持，并针对论文实验扩展算子执行和指纹计算机制。

大规模数据清洗中，一个 Filter 通常先为每个样本计算昂贵的统计特征（例如困惑度、重复率），再根据阈值删除样本。传统执行方式把这两个步骤绑定在一起，修改阈值或调整算子顺序时会重复计算已经有效的特征。DecoupDAG 将二者拆成独立的可缓存阶段，并对兼容的连续 Filter 组采用“先计算特征、后执行过滤”的顺序。

[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-000000.svg)](LICENSE)

> 本仓库不包含论文实验使用的数据集。运行 recipe 前请先阅读[数据集与路径](#数据集与路径)。

英文版：[`README.md`](README.md)

## 项目改动概览

DecoupDAG 在 Data-Juicer 上增加了三个相互配合的部分：

1. **算子解耦。** 将产生统计量的 `Filter` 拆成特征阶段（`compute_stats`）和归约阶段（`process`）。特征阶段把统计结果写入 Data-Juicer 的 `__dj__stats__` 列，归约阶段根据配置的保留条件决定哪些行继续保留。
2. **算子先特征后过滤的重排。** 对一段连续且可重排的 Filter，DecoupDAG 将 `f1, f2, ..., fn` 改写为 `f1.feature, f2.feature, ..., fn.feature, f1.reduce, f2.reduce, ..., fn.reduce`。这样可以在删除行之前一次性计算昂贵特征。Mapper、Deduplicator 以及注册在 `NON_STATS_FILTERS` 中的非统计 Filter 保持原位置，并作为重排边界。
3. **细粒度的阶段/列感知哈希。** 为特征阶段和归约阶段生成独立的 Hugging Face Dataset fingerprint。特征 fingerprint 排除只影响阈值的参数，归约 fingerprint 保留阈值和区间参数；只改阈值时可以复用统计列缓存，只重新执行轻量的归约阶段。

实现仍然采用配置驱动方式，可处理文本、图像、音频和视频 recipe。论文实验对应的配置位于 [`data-recipes/`](data-recipes/)。

## 系统流程

```mermaid
flowchart LR
    A[输入数据集] --> B[Data-Juicer 加载器]
    B --> C[Mapper / Deduplicator]
    C --> D[连续统计型 Filter]
    D --> E[特征阶段\nDataset.map(compute_stats)]
    E --> F[归约阶段\nDataset.filter(process)]
    F --> G[导出结果]
    E -. 缓存 .-> H[HF Datasets cache]
    F -. 缓存 .-> H
```

`normal` 模式下，每个 Filter 会连续执行自己的特征和归约阶段。`feature_first` 模式下，`DefaultExecutor` 只重写连续的统计型 `Filter`；特征块内部和归约块内部的顺序保持不变，不会跨越 Mapper 或非统计 Filter 移动算子。

## 核心实现

### 1. Filter 解耦

Data-Juicer 原始的 `Filter.run` 会把计算统计量的 map 和消费统计量的 filter 连在一起。DecoupDAG 保留原有公共接口，并增加两个显式阶段：

- `Filter.run_feature_stage(dataset, exporter=None)` 调用 `Dataset.map(self.compute_stats, ...)`，并附加特征 fingerprint。
- `Filter.run_reduce_stage(dataset, tracer=None)` 调用 `Dataset.filter(self.process, ...)`，并附加归约 fingerprint。
- `Filter.run(...)` 仍然保留，用于普通的连续执行路径。

[`data_juicer/ops/filter_reorder.py`](data_juicer/ops/filter_reorder.py) 中的 `FilterFeatureOp` 和 `FilterReduceOp` 是对同一个底层 Filter 的轻量包装。它们转发算子属性，并为阶段提供独立名称和配置，便于执行日志和 checkpoint 记录。

对一个可重排的 Filter 组，执行逻辑可以表示为：

```text
D0 = 输入数据集
D1 = map(compute_stats_1, D0)  # f1.feature
D2 = map(compute_stats_2, D1)  # f2.feature
...
Dk = filter(process_1, Dk-1)   # f1.reduce
Dk+1 = filter(process_2, Dk)   # f2.reduce
...
```

特征阶段写入或扩展 `__dj__stats__`，归约阶段只判断行是否保留。这样，当只修改阈值时，已经计算好的统计量可以直接复用。

### 2. `feature_first` 重排

通过下面的配置项启用：

```yaml
filter_execution_mode: feature_first
```

入口位于 [`data_juicer/core/executor/default_executor.py`](data_juicer/core/executor/default_executor.py)，重排逻辑位于 [`data_juicer/ops/filter_reorder.py`](data_juicer/ops/filter_reorder.py) 的 `rewrite_filter_ops_feature_first`：

1. 按 recipe 原始顺序加载算子。
2. 找到由统计型 `Filter` 组成的最长连续片段。
3. 将每个片段替换为“全部 feature wrapper + 全部 reduce wrapper”。
4. 其他算子保持原位置。

[`data_juicer/config/config.py`](data_juicer/config/config.py) 会校验执行模式。当前版本中，`feature_first` 与 checkpoint、operator fusion 不兼容，因此配置初始化时会发出警告并自动关闭这两项。

### 3. 细粒度阶段/列级别 fingerprint

每个阶段的 fingerprint 由输入数据集 fingerprint 和语义阶段签名确定。 [`data_juicer/utils/fingerprint_utils.py`](data_juicer/utils/fingerprint_utils.py) 中的 `generate_stage_fingerprint` 可以概括为：

```text
stage_fp = xxhash64(
    input_dataset_fp,
    {
        "type": "decoupled_filter_stage",
        "op_name": operator_name,
        "stage": "feature" | "reduce"
    },
    normalize(stage_signature)
)
```

[`data_juicer/ops/base_op.py`](data_juicer/ops/base_op.py) 负责拆分签名：

- `get_feature_signature()` 删除阈值类参数：`min_*`、`max_*`、`*_threshold`、`threshold`，以及 `min_closed_interval`、`max_closed_interval`、`reversed_range`。
- `get_threshold_signature()` 保留上述阈值和区间语义；如果算子没有可识别的阈值参数，则使用完整算子配置作为归约签名。
- 两类 fingerprint 作为 `new_fingerprint` 传给 `Dataset.map` 和 `Dataset.filter`，使 Hugging Face Datasets 可以独立寻址两个阶段的缓存。

在阶段 fingerprint 之外，`generate_column_fingerprint` 和
`generate_column_fingerprints` 会为每个逻辑统计列单独生成 fingerprint。每个列的
fingerprint 由输入数据集 fingerprint、算子名称、执行阶段、列名和归一化后的特征签名共同决定。
`Filter.run_feature_stage` 会从 `__dj__stats__` 中收集本次产生的统计 key，并在可访问的
feature-stage 数据集上保存 `_dj_column_fingerprints`。因此，分析器可以观察到每个统计列的
失效范围；物理 Arrow cache 仍然使用 map/filter 的 stage fingerprint。

统计列（`Fields.stats == "__dj__stats__"`）是特征阶段的物化边界。也就是说，DecoupDAG 在统计列这个语义边界上实现列级别复用；Hugging Face Datasets 在物理层面可能把完整 Arrow 表记录放在一个 cache 文件中，但缓存失效由统计阶段的签名驱动，而不是把整条 pipeline 当作不可分割的黑盒。只修改阈值会改变归约 key，不会使特征 key 失效；修改特征参数则会按预期重新计算特征阶段。

哈希前会在 [`data_juicer/utils/fingerprint_utils.py`](data_juicer/utils/fingerprint_utils.py) 中进行稳定归一化：

- 字典按 key 排序，嵌套值递归归一化；
- 路径、bytes、NumPy 标量、set、list、tuple 使用稳定表示；
- callable 会记录 module、限定名、wrapper 深度、代码位置和绑定算子的语义状态；
- 进程数、batch size、内存限制和 Ray runtime environment 等运行时参数不会进入算子语义状态。

[`data_juicer/core/data/dj_dataset.py`](data_juicer/core/data/dj_dataset.py) 会记录每次 `map`/`filter` 的输入 fingerprint、请求 fingerprint、输出 fingerprint、行数、cache 文件数量和 callable identity，便于检查命中与失效原因。

最小实现检查可以运行：

```bash
pytest -q tests/utils/test_fingerprint_utils.py tests/ops/test_filter_reorder.py
```

## 安装

DecoupDAG 要求 Python 3.10 或更高版本，锁文件记录了仓库使用的环境。

```bash
git clone https://github.com/longyang12/DecoupDAG.git
cd DecoupDAG

# 推荐：使用 uv 按锁文件安装。
uv sync

# 或使用 pip 安装可编辑包。
python -m pip install -e .
```

根据使用的 recipe 安装可选依赖：

```bash
# 使用 SimHash 和 NLP 工具的文本 recipe。
python -m pip install -e ".[nlp]"

# 多模态 recipe。
python -m pip install -e ".[nlp,vision]"

# Ray 执行（可选）。
python -m pip install -e ".[distributed]"
```

`all` extra 会安装全部可选依赖，体积较大。GPU 算子还需要匹配的 PyTorch/CUDA 环境。

## 快速开始

### 运行论文 recipe

仓库中的 recipe 含有特定机器上的数据、输出和缓存路径。运行前请复制配置并修改 `dataset_path`、`export_path`、`ds_cache_dir`：

```bash
cp data-recipes/redpajama-arxiv-refine-cpu-np1-cache.yaml /tmp/decoupdag-arxiv.yaml
# 编辑 /tmp/decoupdag-arxiv.yaml，填写本机有效的输入、输出和缓存路径。
python tools/process_data.py --config /tmp/decoupdag-arxiv.yaml
```

启用缓存的 arXiv 配置已经包含：

```yaml
use_cache: true
filter_execution_mode: feature_first
```

要运行未优化的 baseline，在相同数据和算子参数下将 `filter_execution_mode` 改为 `normal` 或删除该配置项。对比实验时请保持数据、缓存目录和软件环境一致。

### 最小配置

下面的配置除输入文件外不依赖仓库中的固定路径：

```yaml
project_name: decoupdag-demo
dataset_path: ./data/input.jsonl
export_path: ./outputs/clean.jsonl
ds_cache_dir: ./outputs/cache
use_cache: true
np: 4
open_tracer: true
filter_execution_mode: feature_first

process:
  - clean_email_mapper:
  - alphanumeric_filter:
      tokenization: false
      min_ratio: 0.10
      max_ratio: 0.95
  - words_num_filter:
      lang: en
      tokenization: true
      min_num: 5
      max_num: 100000
```

运行：

```bash
python tools/process_data.py --config path/to/decoupdag-demo.yaml
```

再次运行相同配置，可以观察 `map`/`filter` 阶段缓存的复用。`NestedDataset` 输出的日志会显示每个阶段使用的 fingerprint。

## 复现实验

### Recipe

仓库提供主要工作负载的 CPU/GPU、cache/no-cache 配置：

| 工作负载 | Recipe 家族 | 主要算子 |
| --- | --- | --- |
| RedPajama arXiv | `redpajama-arxiv-refine-cpu-np1-{cache,nocache}.yaml` | 文本规范化、质量过滤、SimHash 去重 |
| RedPajama C4 | `redpajama-c4-refine-cpu-np1-{cache,nocache}.yaml` | 文本规范化、质量过滤、SimHash 去重 |
| RedPajama code | `redpajama-code-refine-cpu-np1-{cache,nocache}.yaml` | 面向代码的长度/重复过滤、SimHash 去重 |
| StackExchange | `redpajama-stackexchange-refine-cpu-np1-{cache,nocache}.yaml` | 语言/质量过滤、SimHash 去重 |
| Wikipedia | `redpajama-wiki-refine-cpu-np1-{cache,nocache}.yaml` | 语言/质量过滤、SimHash 去重 |
| LLaVA 预训练 | `llava-pretrain-refine-{cpu,gpu}-np1-{cache,nocache}.yaml` | 多模态图文过滤 |
| MSR-VTT | `msr-vtt-refine-{cpu,gpu}-np1-{cache,nocache}.yaml` | 视频质量、运动、NSFW、水印和图文/视频过滤 |

文件名中的 `cache` 表示启用缓存，`nocache` 用于测量重复计算成本。`np1` 配置使用一个本地 worker，便于复现论文测量；只有在有意改变实验设置时才调整 `np`。

### 运行脚本

- [`run_base_pipeline.sh`](run_base_pipeline.sh)：包含主要 recipe 的 cache/no-cache 执行顺序；使用前请修改绝对路径。
- [`run_cpu.sh`](run_cpu.sh) 和 [`run_gpu.sh`](run_gpu.sh)：CPU/GPU 实验命令列表。
- [`run_test.sh`](run_test.sh)：本地小规模运行脚本，其中也包含需要修改的绝对路径。

所有输入数据、模型 checkpoint 和生成结果都被 Git 忽略。请按照数据集提供方的官方说明下载，并遵守相应的许可证和访问条款。

## 数据集与路径

仓库不分发论文实验数据。运行 recipe 前需要：

1. 从对应数据集的官方来源获取数据，并确认使用权限。
2. 将 recipe 中的 `dataset_path` 改为本机路径。
3. 将 `export_path` 和 `ds_cache_dir` 指向有足够空间的目录。
4. 对需要模型的多模态算子安装可选依赖并准备模型权重。

缓存可能包含完整的中间 Arrow 数据，规模会接近甚至超过输入数据规模；请为 `ds_cache_dir` 预留空间。

## 代码结构

```text
DecoupDAG/
├── data_juicer/
│   ├── core/data/dj_dataset.py           # Dataset 包装、fingerprint 日志
│   ├── core/executor/default_executor.py # feature_first 接入
│   ├── ops/base_op.py                    # Filter 阶段拆分和签名
│   ├── ops/filter_reorder.py             # feature/reduce 包装和重排
│   └── utils/fingerprint_utils.py        # 稳定归一化和阶段哈希
├── data-recipes/                         # 论文实验 YAML
├── data/                                 # 本地数据（被 Git 忽略）
├── tools/process_data.py                 # 命令行入口脚本
├── run_*.sh                              # 实验命令列表
├── tests/                                # 上游和扩展测试
├── pyproject.toml                        # 依赖和 extras
└── uv.lock                               # 锁定环境
```

## 扩展配置项

| 配置项 | 可选值 | 作用 |
| --- | --- | --- |
| `filter_execution_mode` | `normal`, `feature_first` | 选择原始执行顺序或解耦的先特征后过滤顺序。 |
| `use_cache` | `true`, `false` | 开启或关闭 Hugging Face Datasets 缓存管理。 |
| `ds_cache_dir` | 目录路径 | 指定可复现实验的中间缓存位置。 |
| `open_tracer` | `true`, `false` | 记录每个 Filter 前后的样本变化，便于调试但会增加开销。 |
| `op_fusion` | `true`, `false` | `feature_first` 下保持关闭；配置层会自动关闭。 |
| `use_checkpoint` | `true`, `false` | `feature_first` 下保持关闭；配置层会自动关闭。 |

## 限制与复现注意事项

- `feature_first` 只处理连续的统计型 `Filter` 片段，不是任意自定义算子的依赖分析器。
- Mapper、Deduplicator 或 `NON_STATS_FILTERS` 成员会打断可重排片段，并保持 recipe 中的原始位置。
- 当 `use_cache: true`、`ds_cache_dir` 持久存在，且输入数据、recipe 和软件环境一致时，缓存收益最明显。
- 修改特征参数会使特征缓存失效；只修改阈值或区间语义时，会使归约缓存失效，同时保留特征结果。
- 当前项目中，`feature_first` 会按设计关闭 checkpoint 和 operator fusion。
- 多模态及模型算子需要额外依赖和模型权重，仓库只提供 recipe，不提供这些外部资产。

## 与 Data-Juicer 的关系

Data-Juicer 提供基础数据格式、算子注册、配置系统、导出器、tracer 和分布式执行能力。DecoupDAG 修改 Filter 的执行路径和 fingerprint 构造，同时保留上游算子接口和 recipe 风格。上游代码和第三方组件的许可信息见 [`LICENSE`](LICENSE)。

如果使用基础系统或基于本项目开发，请在引用 DecoupDAG 论文的同时引用 Data-Juicer。该论文计划投稿 SIGMOD：

```bibtex
@inproceedings{datajuicer,
  title     = {Data-Juicer: A One-Stop Data Processing System for Large Language Models},
  author    = {Chen, Daoyuan and Huang, Yilun and Ma, Zhijian and Chen, Hesen and Pan, Xuchen and Ge, Ce and Gao, Dawei and Xie, Yuexiang and Liu, Zhaoyang and Gao, Jinyang and Li, Yaliang and Ding, Bolin and Zhou, Jingren},
  booktitle = {International Conference on Management of Data},
  year      = {2024}
}
```

DecoupDAG 论文的 BibTeX 请以论文最终发布页面中的版本为准。

## 开源协议

DecoupDAG 采用 Apache License 2.0，完整协议文本和第三方组件声明见 [`LICENSE`](LICENSE)。
