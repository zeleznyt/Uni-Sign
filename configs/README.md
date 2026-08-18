# Data-Config Guide

Every training or evaluation run requires `--data_config PATH`. The JSON file defines the active datasets and the run-wide pose/model settings that must remain compatible across them.

For the underlying annotation, keypoint, loader, and landmark formats, see [Datasets, Loaders, and Pose Layouts](../docs/DATASET.md).

## Start from an Example

- [`ytasl.json`](./ytasl.json) shows a single pose-only YTASL dataset with ordinary sampling.
- [`mixed_json_example.json`](./mixed_json_example.json) shows two YTASL-compatible training sources with explicit dataset sampling ratios.

Copy the closest example, change its paths, and run from the repository root. Relative paths inside a config are resolved from the process's current working directory, not from the config file's directory.

```bash
cp configs/ytasl.json configs/my_experiment.json
# Edit configs/my_experiment.json, then:
deepspeed --include localhost:0 fine_tuning.py \
  --data_config configs/my_experiment.json \
  --task SLT \
  --batch-size 8 \
  --epochs 20 \
  --output_dir out/my_experiment
```

## Complete Shape

```json
{
  "name": "mixed_example",
  "layout": "pruned",
  "graph": "ytasl",
  "target_language": "English",
  "normalization": "none",
  "metric_text_transform": null,
  "train": [
    {
      "name": "dataset_a_train",
      "loader": "ytasl_json",
      "annotation_path": "data/dataset_a/annotations.train.json",
      "pose_roots": [
        "data/dataset_a/raw_keypoints"
      ],
      "rgb": null,
      "weight": 2.0
    },
    {
      "name": "dataset_b_train",
      "loader": "ytasl_json",
      "annotation_path": "data/dataset_b/annotations.train.json",
      "pose_roots": [
        "data/dataset_b/raw_keypoints"
      ],
      "rgb": null,
      "weight": 1.0
    }
  ],
  "dev": {
    "name": "dataset_a_dev",
    "loader": "ytasl_json",
    "annotation_path": "data/dataset_a/annotations.dev.json",
    "pose_roots": [
      "data/dataset_a/raw_keypoints"
    ],
    "rgb": null
  },
  "test": null
}
```

JSON does not support comments. The field reference below is therefore kept outside the runnable examples.

## Top-level Fields

| Field | Required | Values and behavior |
| --- | --- | --- |
| `name` | No | Human-readable run/data name. If omitted, the config filename stem is used. |
| `layout` | Yes | `default`, `pruned`, or `isharah`. Selects the retained local-JSON landmarks and contributes to graph selection. |
| `graph` | Yes | `ytasl`, `default`, or `original`. Selects the ST-GCN adjacency family. |
| `target_language` | Yes | Non-empty language name used by the model's task prefix, for example `English`. |
| `normalization` | No | `none` by default, or `signspace` for local JSON keypoints. |
| `metric_text_transform` | No | `null` by default, or `csl_daily_char`. The latter applies CSL-Daily character formatting only for SLT with `--original_metric_implementation`. |
| `train` | Run-dependent | One dataset object, a list of objects, or `null`. Required for training; its datasets and paths are ignored in evaluation-only mode. |
| `dev` | Run-dependent | One dataset object, a list of objects, or `null`. Required for training checkpoint selection. |
| `test` | No | One dataset object, a list of objects, or `null`. |

The layout and graph are run-wide. Consult the [compatibility table](../docs/DATASET.md#layouts-and-graphs) before combining formats.

## Dataset-spec Fields

Each object inside `train`, `dev`, or `test` accepts:

| Field | Required | Values and behavior |
| --- | --- | --- |
| `name` | No | Label shown in setup and sampling reports. Falls back to the loader name. |
| `loader` | Yes | `ytasl_json`, `isharah_json`, `original_pickle`, or `csl_news`. |
| `annotation_path` | Yes | Path to the annotation JSON or compressed pickle expected by the loader. |
| `pose_roots` | Yes | Ordered list of directories containing pose files. A single string is also accepted. |
| `rgb` | No | `null` for pose-only, a root-path string, or an object containing `root`. See [RGB](#rgb). |
| `weight` | No | Positive relative dataset sampling ratio. It is meaningful for train datasets and activates weighted sampling when present on any train spec. |

Compatibility aliases are also accepted:

- `pose_root` for one pose directory when `pose_roots` is absent.
- `rgb_root` when `rgb` is absent.

Prefer the plural `pose_roots` and explicit `rgb` forms in new configs.

## Split Composition

A split can be a single object:

```json
"dev": {
  "name": "how2sign_dev",
  "loader": "ytasl_json",
  "annotation_path": "data/how2sign/annotations.dev.json",
  "pose_roots": ["data/how2sign/raw_keypoints"],
  "rgb": null
}
```

It can be a list, in which case the usable samples are combined into one configured split:

```json
"train": [
  {
    "name": "how2sign_train",
    "loader": "ytasl_json",
    "annotation_path": "data/how2sign/annotations.train.json",
    "pose_roots": ["data/how2sign/raw_keypoints"],
    "rgb": null
  },
  {
    "name": "phoenix_train",
    "loader": "ytasl_json",
    "annotation_path": "data/phoenix/annotations.train.json",
    "pose_roots": ["data/phoenix/raw_keypoints"],
    "rgb": null
  }
]
```

Or it can be explicitly disabled:

```json
"test": null
```

A missing split field behaves like `null`.

### Training requirements

A normal training run requires non-empty `train` and `dev` sections. `dev` is used for checkpoint selection. `test` may be `null`; this avoids evaluating the test set during training.

### Evaluation-only requirements

With `--eval`, the train dataset is not constructed and its paths are not checked by preflight. The JSON split value must still be an object, list, or `null`. At least one of `dev` or `test` must be non-null. Every configured dev/test split is evaluated.

## Paths and Preflight Validation

Run from the repository root when using paths such as `data/YTASL/...`.

Before constructing the model, config preflight checks:

- required top-level values and supported enum values;
- loader names and annotation-file existence;
- pose-root existence and whether a root contains direct `*.json` or `*.pkl` files as appropriate;
- positive weights;
- RGB compatibility and root existence.

The local JSON loaders then match annotation clip names to pose filename stems. Missing clips are filtered and counted. Multiple pose roots are searched/indexed in order; the first matching file wins.

## Sampling and Weights

There are two distinct training modes. The startup report states which one is active.

### Default: concatenation

If no train dataset object contains `weight`, all usable datasets are concatenated and shuffled without replacement:

```text
mode: concatenation
replacement: False
```

Dataset exposure follows usable dataset size. If dataset A has 1,000 samples and dataset B has 100, the nominal epoch contains 1,100 positions: 1,000 from A and 100 from B, in shuffled order.

This is the behavior demonstrated by [`ytasl.json`](./ytasl.json). An omitted weight does **not** silently turn on equal-dataset sampling.

### Dataset-weighted sampling

If any train dataset object explicitly contains `weight`, weighted mode is activated for the entire train split. A missing weight on another train object then defaults to `1.0`.

Weights are relative dataset-level ratios and do not need to sum to one. For each dataset, every sample receives:

```text
per_sample_weight = configured_dataset_weight / usable_dataset_size
```

This makes the total probability mass of each dataset proportional to its configured ratio, independent of its size. Ratios `2.0 : 1.0` and `1.0 : 0.5` both normalize to expected shares of `66.67% : 33.33%`.

The nominal epoch length remains the total number of usable samples. For datasets of 1,000 and 100 samples with weights `2.0` and `1.0`:

- nominal epoch draws: `1,100`;
- expected draws from A: approximately `733`;
- expected draws from B: approximately `367`.

The epoch does not become 1,050 samples, and the configured ratio is not an exact per-epoch quota.

### What `replacement: True` means

`replacement` is not a JSON field and should not be added to the config. It is a derived line in the startup report.

Weighted mode draws each sample index independently from the configured probability distribution. After an index is drawn, it remains eligible to be drawn again. Therefore:

- one source sample can appear multiple times in the same epoch;
- another source sample can be absent from that epoch;
- actual per-epoch dataset counts fluctuate around the reported expected shares.

This is sampling with replacement. It affects selection only; it does not duplicate files on disk.

Omit all train weights when you want ordinary concatenation. Avoid adding a weight to a single-dataset train section unless repeated/omitted sample selection is intentional.

### Distributed training

Dataset-weighted mode performs one deterministic global weighted draw for each epoch and shards it across ranks. Each rank receives the same number of positions. If the nominal dataset length is not divisible by the world size, the sampler rounds the global draw count up to the nearest divisible size. The setup report's shares and draw counts remain nominal/expected values.

Unweighted distributed sampling can likewise pad indices to equalize rank lengths.

## RGB

Pose-only:

```json
"rgb": null
```

RGB root as a string:

```json
"rgb": "data/csl_daily/videos"
```

Equivalent explicit object:

```json
"rgb": {
  "root": "data/csl_daily/videos"
}
```

`ytasl_json` and `isharah_json` are pose-only. `original_pickle` and `csl_news` retain RGB support.

RGB is enabled only when at least one active spec requests it and every active spec can provide it. If one active dataset is pose-only or lacks an RGB root, the setup report warns and forces the entire run to pose-only. There is no RGB CLI switch and no silent RGB activation.

For training, active specs include train, dev, and test. In evaluation-only mode, they include dev and test; the train section is ignored.

## Reading the Setup Report

A successful weighted configuration prints information in this form:

```text
Data config setup:
  layout: pruned
  graph: ytasl
  target_language: English
  normalization: none
  rgb_requested: False
  rgb_enabled: False
  [train] dataset_a_train (ytasl_json)
    annotation_samples: 1000
    usable_samples: 1000
    filtered_missing_pose_samples: 0
  [train] dataset_b_train (ytasl_json)
    annotation_samples: 100
    usable_samples: 100
    filtered_missing_pose_samples: 0
  [train sampling]
    mode: dataset_weighted
    replacement: True
    nominal_epoch_draws: 1100
    dataset_a_train: source_samples=1000 expected_share=66.67% expected_draws=733.3 configured_weight=2.0
    dataset_b_train: source_samples=100 expected_share=33.33% expected_draws=366.7 configured_weight=1.0
```

Check this report before a long run. In particular, verify `usable_samples`, filtered missing-pose counts, RGB fallback, sampling mode, expected shares, and expected draws.

## Training and Evaluation Commands

Train from scratch:

```bash
deepspeed --include localhost:0,1 fine_tuning.py \
  --data_config configs/my_experiment.json \
  --task SLT \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --lr 3e-4 \
  --output_dir out/my_experiment
```

Start a new run from model weights while resetting optimizer and scheduler state:

```bash
deepspeed --include localhost:0,1 fine_tuning.py \
  --data_config configs/my_experiment.json \
  --task SLT \
  --finetune out/previous/best_checkpoint.pth \
  --output_dir out/my_experiment_finetuned
```

Resume full model, optimizer, scheduler, epoch, and RNG state from a DeepSpeed checkpoint directory:

```bash
deepspeed --include localhost:0,1 fine_tuning.py \
  --data_config configs/my_experiment.json \
  --task SLT \
  --resume out/my_experiment \
  --epochs 20 \
  --output_dir out/my_experiment
```

Evaluate a `.pth` checkpoint on every configured non-null dev/test split:

```bash
deepspeed --include localhost:0 fine_tuning.py \
  --data_config configs/my_experiment.json \
  --task SLT \
  --eval \
  --finetune out/my_experiment/best_checkpoint.pth \
  --output_dir out/my_experiment_eval
```

Use `--resume` instead of `--finetune` to evaluate a DeepSpeed checkpoint with its stored client state. Do not supply both options.

## Common Errors

- **Annotation or pose root does not exist:** run from the expected working directory or use absolute paths.
- **Pose root reports zero files:** files must be directly inside the root and use the extension expected by the loader.
- **Train split has zero usable samples:** annotation clip names do not match pose filename stems, or all corresponding files are missing.
- **Graph/layout tensor-size failure:** use a compatible loader, layout, and graph combination from the [dataset guide](../docs/DATASET.md#layouts-and-graphs).
- **RGB is forced off:** at least one active dataset is pose-only or lacks a compatible RGB root.
- **Unexpected 50/50 dataset exposure:** explicit equal weights activate dataset-weighted sampling; remove every train weight to restore natural size-proportional exposure.
