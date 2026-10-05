# Representation Similarity

This tool compares saved KDFP students with their teacher on 1,000 examples
from the GSM8K training set. It computes:

- Linear centered kernel alignment (CKA).
- Representational similarity analysis (RSA), using cosine similarity between
  examples and Spearman correlation between teacher and student similarity
  matrices.

The comparison uses the final non-padding token from sequences truncated and
padded to exactly 1,024 tokens. Teacher and student tokenizers are applied
independently, but the same examples and deterministic ordering are used.

## Running

Install the repository environment and package first if necessary:

```bash
conda env create -f environment.yaml
conda activate kdfp
pip install -e .
```

Copy `config.example.yaml` to `config.yaml`, update `inputs`, and run:

```bash
rep-sim -c config.yaml
```

The `rep-sim` command is installed by `pip install -e .`. If the config path
is relative, the tool first checks under `representation_similarity/`, then
falls back to the supplied path. Therefore, from the repository root,
`rep-sim -c config.yaml` prefers `representation_similarity/config.yaml` and
then falls back to `config.yaml` in the current directory. Omitting `-c`
uses the same lookup order for `config.yaml`.

Each input can be either an experiment directory:

```text
experiment/
  seed3/
    pl_config.json
    student.bin
  seed33/
    pl_config.json
    student.bin
```

or a single seed directory containing `pl_config.json` and `student.bin`.
The teacher configuration must be identical across all seeds under one input.
The analysis uses three model categories: the unmodified teacher, the
unmodified base student loaded from its configured model name, and each trained
student loaded from its TransformerLens `student.bin`. All models are created
through `HookedTransformer.from_pretrained_no_processing`, so activation
extraction follows the same TransformerLens path. The base student is compared
to the teacher separately and is not included in trained-seed aggregates.

## Configuration

All analysis parameters belong in the YAML configuration. The exact input
configuration is copied to the output directory as `config.yaml`.

| Parameter | Description |
| --- | --- |
| `inputs` | List of experiment or seed directories. |
| `output_dir` | Directory for figures and result files. |
| `samples` | Number of GSM8K train examples, default 1,000. |
| `analysis_seeds` | List of dataset ordering and analysis seeds. Defaults to `[67]`. |
| `seq_len` | Must be `1024`. |
| `batch_size` | Inference batch size. |
| `device` | PyTorch device, usually `cuda` or `cpu`. |
| `dtype` | PyTorch dtype, for example `bfloat16` or `float32`. |
| `logging.level` | CLI log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `logging.progress` | Whether to show input, activation, and metric progress bars. |
| `dataset.name` | Hugging Face dataset name. |
| `dataset.subset` | Dataset configuration, `main` for GSM8K. |
| `dataset.split` | Dataset split, normally `train`. |
| `dataset.format` | `question` or `question_and_answer`. |
| `activation_component` | `resid`, `attn_out`, or `mlp`. |
| `metrics` | Any combination of `cka` and `rsa`. |
| `pooling.teacher_bins` | Must be `4`. |
| `pooling.student_bins` | Must be `4`. |

Inputs may provide a concise label for each independent student experiment:

```yaml
inputs:
  - label: student_small
    path: path/to/experiment
```

The label is used exactly as written in plot titles, logs, summaries, metadata,
and cross-student rankings. A filesystem-safe version is used for output
directories and filenames. Legacy string entries such as `- path/to/experiment`
remain supported; the input directory basename is used as the label. Labels
must be unique after filesystem-safe sanitization.

## Layer pooling

The script first computes the complete teacher-layer by student-layer
similarity matrix. It then splits each model's selected layers into four
contiguous groups and averages the values in each teacher/student group pair.
Consequently, every saved heatmap and CSV is exactly 4 by 4. Axis labels show
the first and last layer represented by each pool.

Pooling is applied to similarity values, not activations. This preserves the
standard CKA and RSA calculations for individual layer pairs.

## Similarity Metrics

### CKA

CKA measures linear similarity between two activation representations after
centering examples. For each teacher/student layer pair, this implementation
compares the cross-covariance structure with each representation's
within-covariance structure using Frobenius norms. CKA is relatively invariant
to feature rotation and isotropic scaling, so it measures shared linear
structure rather than matching individual neuron coordinates. Higher CKA means
the representations are more similar.

### RSA

RSA measures similarity between representational geometry. For each layer, the
tool computes an example-by-example cosine-similarity matrix, takes its upper
triangle, ranks the entries, and computes the Spearman correlation between the
teacher and student rankings. RSA therefore asks whether the two models arrange
the examples similarly, even when their feature coordinates differ. RSA ranges
from `-1` to `1`; higher RSA means more similar geometry.

CKA and RSA measure different properties and are reported separately. They are
not averaged into a composite score.

## Cross-Student Rankings

The top-level `summary/` directory ranks each independent student by its
similarity to the teacher. It includes `base_student` and every trained student
as separate records. It never computes similarity between two students and does
not average different students together.

For each metric, a ranking CSV and JSON entry are generated for every scalar
summary below:

- `global_mean`, `global_median`, `global_p10`, `global_p25`,
  `global_minimum`, `global_maximum`, and `global_range`: statistics over all
  teacher-layer/student-layer pairs.
- `global_fraction_at_or_above_0_25`, `global_fraction_at_or_above_0_5`,
  and `global_fraction_at_or_above_0_75`: fractions of layer pairs meeting each
  similarity threshold.
- `student_best_mean`, `student_best_median`, `student_best_p10`,
  `student_best_p25`, `student_best_minimum`, `student_best_maximum`,
  `student_best_range`, and `student_best_std`: for each student layer, the
  best matching teacher layer, summarized across student layers.
- `teacher_best_mean`, `teacher_best_median`, `teacher_best_p10`,
  `teacher_best_p25`, `teacher_best_minimum`, `teacher_best_maximum`,
  `teacher_best_range`, and `teacher_best_std`: the corresponding teacher-layer
  perspective.
- `mapped_mean`, `mapped_median`, `mapped_p10`, `mapped_p25`,
  `mapped_minimum`, `mapped_maximum`, `mapped_range`, and `mapped_std`:
  similarities for the teacher/student hook pairs at the same positions in the
  `pl_config.json` hook lists. These are the layer pairings used during
  distillation.

Similarity statistics are ranked higher-is-better. Standard deviation and range
are ranked lower-is-better and should be interpreted as consistency diagnostics,
not as primary similarity scores. Every ranking records both an all-student rank
and a trained-student-only rank.

## Outputs

Each input model gets its own directory. Aggregate results for that model are
under `summary/`, while individual trained seeds are under `seeds/`:

```text
<output_dir>/
  base_student/
    summary/
      cka/
      rsa/
      summary.json
  <input_model_name>/
    summary/
      cka/
      rsa/
      summary.json
    seeds/
      seed3/
        cka/
        rsa/
        summary.json
      seed33/
        cka/
        rsa/
        summary.json
  summary/
      summary.json
      student_metrics.csv
      data/
        cka_<scalar>.csv
        rsa_<scalar>.csv
```

Each metric directory contains `matrix.csv`, `full_matrix.csv`,
`heatmap.png`, and `summary.json`. Aggregate directories additionally contain
`std.csv` and `full_std.csv`. Model summaries include key statistics, seed
aggregates, base-student values, and trained-minus-base deltas. The top-level
`summary` directory ranks each independent student by its similarity to the
teacher. It includes the base student and trained students as separate records,
and does not compare or average activations between different students.
`summary.json` contains the detailed representation and all scalar rankings.
`student_metrics.csv` is a wide one-row-per-student table containing only
`global_mean`, `global_std`, and `global_p25`, plus `mapped_mean`, `mapped_std`,
and `mapped_p25`, for both CKA and RSA. Each value has a corresponding rank
column; there is no kind or average-rank column. The full ranking CSV data files
are under `summary/data/`.

The output root also contains `metadata.json` and `config.yaml`.

Heatmaps use low-to-high layer-pool ordering on the y-axis: the lowest teacher
layer pool is at the bottom and the highest teacher layer pool is at the top.

The teacher and student layer lists come from each seed's `pl_config.json`, so
the analysis respects the layer selection/mapping recorded during distillation.

## Monitoring

Progress bars and logging are configured in YAML:

```yaml
logging:
  level: INFO
  progress: true
```

Set `logging.level` to `DEBUG` for more detailed stage messages, or set
`logging.progress` to `false` for non-interactive environments.

The analysis can be repeated over multiple dataset/order seeds:

```yaml
analysis_seeds:
  - 67
  - 123
  - 456
```

The results are averaged elementwise across these analysis seeds. Individual
analysis-seed directories are not written; trained checkpoint seed directories
remain part of the per-model outputs.

## Reproducibility

The analysis seeds Python, NumPy, and PyTorch with the configured `seed`. CUDA
runs enable PyTorch deterministic algorithms, deterministic cuDNN behavior,
disable cuDNN benchmarking and TF32, and configure deterministic cuBLAS
workspace behavior. Runs are expected to be reproducible on the same hardware
and software stack; changing GPU architecture, CUDA, PyTorch, or model/data
versions can still change numerical results. The effective reproducibility
settings and runtime versions are recorded in `metadata.json`.
