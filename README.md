# RAPID: Increasing Efficiency of Internal Distillation via Randomized Methods

This repository contains the experiment harness and analysis code for the paper
[RAPID: Increasing Efficiency of Internal Distillation via Randomized Methods](https://arxiv.org/abs/2611.04081)
by Ryan Swift and Konstantinos Psounis. See the [paper PDF](https://arxiv.org/pdf/2611.04081).

## Citation

If you use this code or build on this work, please cite:

```bibtex
@misc{swift2026rapid,
  title         = {RAPID: Increasing Efficiency of Internal Distillation via Randomized Methods},
  author        = {Swift, Ryan and Psounis, Konstantinos},
  year          = {2026},
  eprint        = {2611.04081},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2611.04081}
}
```

## Requirements and setup

The experiments use CUDA and BF16. The paper's experiments were run on one
server with two NVIDIA RTX A6000 48 GB GPUs. The checked-in environment pins
Python and the experiment dependencies; `pip install -e .` installs the local
command-line tools.

Create and activate the environment from the repository root:

```bash
conda env create -f environment.yaml --name rapid
conda activate rapid
pip install -e .
```

The experiment configs download models and datasets from Hugging Face. Before
running Gemma experiments, accept the model terms on Hugging Face and log in:

```bash
hf auth login
```

The experiment configs also initialize Weights & Biases runs. Log in if you
want to track runs online:

```bash
wandb login
```

## Reproducing the main results

From the repository root, run the Gemma 3 and Qwen 2.5 SmolTalk experiment
wrappers:

```bash
./scripts/run_smoltalk_gemma3.sh
./scripts/run_smoltalk_qwen.sh
```

Each wrapper runs its four configs sequentially and stops if a config fails.
Together, the configs cover randomized and learned online projectors for the
residual-only and all-activation objectives. The configs use seeds `3`, `33`,
and `333`, train on 40M tokens with sequence length 1024, and evaluate ARC
Challenge, BBH, GPQA, and MMLU. The first config in each wrapper also runs the
configured fine-tuning and traditional KD baselines. The YAML files contain the
full model, data, training, and evaluation settings:

- Gemma 3: `rapid/configs/gemma3/g3_smoltalk_*.yaml`
- Qwen 2.5: `rapid/configs/qwen25/qwen25_smoltalk_*.yaml`

Wrapper progress is appended to `logs/smoltalk_experiments.log`.

## Running the experiment harness directly

The `rapid` command runs one experiment suite from a YAML config. Use the small
debug suite to smoke-test the harness and environment:

```bash
rapid run -c rapid/configs/debug.yaml
```

`debug.yaml` is a short smoke test, not a paper reproduction run: it uses one
seed, 32 training tokens, sequence length 32, and a one-sample evaluation
limit. To run another suite directly, pass its config path in the same way:

```bash
rapid run -c rapid/configs/<suite>.yaml
```

## Results and plots

Each run writes timestamped result tables in CSV, Markdown, and Parquet formats
to the suite's score directory:

```text
<directories.root>/<suite_title>/scores/results_df_<timestamp>.{csv,md,parquet}
```

The experiment checkpoints are saved under the configured models directory.
For the main SmolTalk configs, this is under
`iclr_fix/<suite_title>/models/students/`.

To average a result table across seeds and generate plots, pass its Parquet file
to `plot-gen`:

```bash
plot-gen iclr_fix/<suite_title>/scores/results_df_<timestamp>.parquet
```

`plot-gen` writes to a `plots/` directory beside the Parquet file. It saves the
averaged table under `plots/Averaged_Data/` in Parquet, CSV, and Markdown
formats, along with per-task score plots and an average-score plot. If the input
contains perplexity results, it also creates perplexity plots.

## Representation-similarity analysis

After the main SmolTalk runs have saved their checkpoints, reproduce the CKA
and RSA analysis with:

```bash
rep-sim -c representation_similarity/g3_smoltalk.yaml
rep-sim -c representation_similarity/q25_smoltalk.yaml
```

These configs compare each trained student with its teacher on 500 examples
from the GSM8K training split. They use sequence length 1024, analysis seeds
`123`, `45`, and `67`, and compute both CKA and RSA on residual-stream
activations. The Gemma analysis writes to `iclr_rep_sim/g3/`; the Qwen analysis
writes to `iclr_rep_sim/q25/`.

The input paths in the configs point to checkpoints produced by the SmolTalk
wrappers. For details on the analysis config, metrics, and output files, see
[`representation_similarity/README.md`](representation_similarity/README.md).

## Chain-of-Thought evaluation (work in progress)

The following CoT evaluation variants are a work in progress:

```bash
./scripts/cot/run_smoltalk_gemma3.sh
./scripts/cot/run_smoltalk_qwen.sh
```

Each wrapper runs its four configs sequentially. Progress is appended to
`logs/smoltalk_cot_experiments.log`. These runs are separate from the main
results above.

## Configuration reference

The harness validates experiment YAML files against the schemas in
`rapid/schemas/`; unknown config fields are rejected. Main experiment configs
have these top-level sections:

| Section | Purpose and key fields |
| --- | --- |
| `suite_title` | Names the suite and its output directory. |
| `models` | Teacher, pretrained student, and shallow-student model names, tokenizer names, and optional layer reduction. |
| `directories` | Output roots for caches, models, Ray, and Lightning. |
| `experiment_space` | Seeds, student types, objective variants, projection direction and training method, offline-alignment values, and include/exclude filters. |
| `training_params` | Dataset and tokenization, sequence length, batch sizes, token budget, epochs, trials, device and dtype, baselines, resource limits, offline-projector settings, and Weights & Biases project/group. |
| `hook_types` | Activation hooks to capture once (`single`) and for each transformer layer (`per_layer`). |
| `hparams` | Fixed hyperparameters and/or Ray Tune search spaces for learning rate, weight decay, temperature, and gradient clipping. |
| `tasks` | Evaluation task names and optional sample limit or forced rerun. |

`experiment_space.objective.weight_variants` selects which loss components are
active. For example, `with_resid` selects output distillation, ground-truth
labels, and residual activations; `balanced` selects the output, label,
embedding, attention-output, MLP-output, and residual objectives. The dynamic
loss weighting implementation determines the per-step weights for active
components.

Representation-similarity YAML files use a separate schema. Their key fields
are `inputs` and `output_dir`; sample count and `analysis_seeds`; `seq_len`,
`batch_size`, `device`, and `dtype`; dataset name, subset, split, and text
format; activation component; CKA/RSA metrics; four-bin pooling settings; and
logging level/progress options. The required analysis sequence length is 1024,
and both pooling bin counts must be 4. The checked-in examples are
`representation_similarity/g3_smoltalk.yaml` and
`representation_similarity/q25_smoltalk.yaml`.
