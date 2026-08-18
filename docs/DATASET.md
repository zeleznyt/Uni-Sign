# Datasets, Loaders, and Pose Layouts

Uni-Sign reads every configured split through one `ConfiguredDataset`. A split may contain one dataset specification or a list of specifications. Each specification selects a format adapter with its `loader` field.

The loader name describes the on-disk format, not necessarily the dataset identity. For example, YTASL, How2Sign, Phoenix, and a converted WLASL export can all use `ytasl_json` when their annotations and per-clip keypoints follow the schema below.

For instructions on composing and running JSON configs, see the [data-config guide](../configs/README.md).

## Supported Loaders

| Loader | Annotation | Pose files | RGB | Typical use |
| --- | --- | --- | --- | --- |
| `ytasl_json` | Nested JSON by video and clip | One MediaPipe-style JSON per clip | Pose-only | YTASL and datasets converted to the same schema |
| `isharah_json` | Same nested JSON structure | One Isharah-style JSON per clip | Pose-only | Isharah/MSLR data |
| `original_pickle` | Gzip-compressed pickle | Original Uni-Sign pickle files | Optional | Original CSL-Daily/WLASL-style prepared data |
| `csl_news` | JSON list | CSL-News pickle files | Optional | Original CSL-News format |

The `ytasl_json` path has been exercised locally and in mixed-dataset cluster training with YTASL and How2Sign. The original pickle and CSL-News adapters were migrated from the upstream implementation but could not be runtime-tested during this transition because the corresponding prepared data was unavailable.

## YTASL Keypoint Dataset

The published keypoint release is available as the [YouTube-ASL Clip Keypoint Dataset](http://hdl.handle.net/11234/1-5898). It is also referenced by the [T5_for_SLT project](https://github.com/zeleznyt/T5_for_SLT).

Use the release's citation and licence information when publishing work based on it. This repository consumes the extracted clips as annotation JSON plus one keypoint JSON per clip; it does not use the H5 data workflow documented by T5_for_SLT.

The paths can be arranged freely. A common layout is:

```text
data/YTASL/
├── YT.annotations.train.json
├── YT.annotations.dev.json
├── YT.annotations.test.json
└── raw_keypoints/
    ├── clip_000001.json
    ├── clip_000002.json
    └── ...
```

Point a config at these files with the `ytasl_json` loader. See [`configs/ytasl.json`](../configs/ytasl.json) for a complete example.

## Local JSON Format

`ytasl_json` and `isharah_json` share the same annotation structure, filename matching, root indexing, and missing-file handling. Their expected landmark counts and keypoint conversion differ.

### Annotation JSON

The annotation root is an object keyed by source video ID. Each video contains `clip_order`, and every listed clip contains a `translation`:

```json
{
  "video_001": {
    "clip_order": [
      "clip_000001",
      "clip_000002"
    ],
    "clip_000001": {
      "translation": "first example translation"
    },
    "clip_000002": {
      "translation": "second example translation"
    }
  }
}
```

The order in `clip_order` is the sample order within that video. A clip's keypoint filename must be `<clip_name>.json`; the parent video ID is not part of pose-file lookup.

### Per-clip keypoint JSON

Each pose file contains a `cropped_keypoints` list with one object per frame:

```json
{
  "cropped_keypoints": [
    {
      "pose_landmarks": [[0.31, 0.12], [0.32, 0.15]],
      "right_hand_landmarks": [[0.60, 0.41], [0.61, 0.40]],
      "left_hand_landmarks": [[0.20, 0.43], [0.21, 0.42]],
      "face_landmarks": [[0.45, 0.15], [0.46, 0.15]]
    }
  ]
}
```

The abbreviated arrays above only illustrate the nesting. Real frames must use these sizes:

| Loader | Body | Right hand | Left hand | Face |
| --- | ---: | ---: | ---: | ---: |
| `ytasl_json` | 33 | 21 | 21 | 478 |
| `isharah_json` | 25 | 21 | 21 | 19 |

Each landmark must provide two coordinates, `[x, y]`. The loader adds a third confidence channel. A present, correctly sized group receives confidence `1`. An empty group is zero-filled and receives confidence `0`. A non-empty group with an unexpected number of landmarks raises an error. For `ytasl_json`, all four group keys must be present even when a group is an empty list.

### Pose roots and matching

`pose_roots` may contain multiple directories. Only `.json` files directly inside each root are indexed; discovery is not recursive. Roots are processed in the configured order, and the first file for a repeated clip name wins.

Annotation clips without a matching pose filename are removed before training or evaluation. The startup report shows the annotation count, usable count, number filtered for missing pose, and several missing-name examples. An empty required train split is an error; empty dev/test splits are treated as unavailable with a warning.

## Isharah

Use `isharah_json` for the Isharah/MSLR landmark format and pair it with:

```json
{
  "layout": "isharah",
  "graph": "ytasl"
}
```

The relevant competition pages are:

- [MSLR CSLR Task 1](https://www.codabench.org/competitions/13266/)
- [MSLR CSLR Task 2](https://www.codabench.org/competitions/13267/)

The configured annotation still needs the common nested JSON schema described above. Isharah pose frames use 25 body, 21 landmarks per hand, and 19 face landmarks. The loader selects nine upper-body landmarks, retains both full hands, and retains all 19 face landmarks.

## Original Pickle Format

`original_pickle` preserves the upstream Uni-Sign prepared-data format.

The annotation file is a gzip-compressed pickled mapping. Each sample is expected to contain:

```text
name:       sample identifier
text:       translation or class label
video_path: relative video path ending in .mp4
gloss:      optional string or token list
```

The loader replaces the `.mp4` suffix in `video_path` with `.pkl` and searches the configured `pose_roots` in order. A pose pickle contains `keypoints` and `scores`, plus optional `start` and `end` offsets. The expected original pose representation has 133 landmarks and is converted to 9 body, 21 left-hand, 21 right-hand, and 18 face points.

Use `layout: "default"` with `graph: "default"` or `graph: "original"` for this loader.

## CSL-News Format

`csl_news` preserves the upstream CSL-News behavior. Its annotation file is a JSON list whose samples contain:

```json
{
  "video": "relative/video.mp4",
  "pose": "relative/pose.pkl",
  "text": "target text"
}
```

Pose pickles contain `keypoints` and `scores` in the original 133-landmark representation. The loader uses the first 99% of the annotation list for `train` and the final 1% for either non-training phase. It retries another random item when a sample cannot be loaded.

Use `layout: "default"` with `graph: "default"` or `graph: "original"`.

## Layouts and Graphs

`layout` controls which landmarks the local JSON loaders retain. `graph` controls the ST-GCN adjacency definition built for those retained points. These settings apply to the entire run, so all datasets combined in a config must produce compatible shapes.

Recommended combinations are:

| Input format | `layout` | `graph` | Body | Each hand | Face |
| --- | --- | --- | ---: | ---: | ---: |
| YTASL-style JSON, full selection | `default` | `ytasl` | 25 | 21 | 37 |
| YTASL-style JSON, compact selection | `pruned` | `ytasl` | 9 | 21 | 18 |
| YTASL-style JSON, Isharah-compatible selection | `isharah` | `ytasl` | 9 | 21 | 19 |
| Isharah JSON | `isharah` | `ytasl` | 9 | 21 | 19 |
| Original pickle or CSL-News | `default` | `default` or `original` | 9 | 21 | 18 |

For `ytasl_json`, the selections are:

- `default`: body landmarks `0..24`; 37 selected face landmarks; all 21 points from each hand.
- `pruned`: body landmarks `[0, 7, 8, 11, 12, 13, 14, 15, 16]`; face landmarks `[4, 13, 14, 61, 81, 93, 152, 159, 172, 178, 291, 311, 323, 386, 397, 402, 472, 477]`; both full hands.
- `isharah`: the same nine body landmarks as `pruned`; face landmarks `[0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185, 267, 269, 270, 291, 314, 321, 375, 405]`; both full hands.

`graph: "original"` currently uses the same adjacency-name path as `graph: "default"`; it remains a supported compatibility alias. Do not mix incompatible graph/layout pairs merely because each value is accepted independently by config validation—the resulting adjacency sizes must match the loader output.

## Normalization

The top-level `normalization` field accepts:

- `none`: keep the local JSON coordinates as stored and append the generated confidence channel.
- `signspace`: globally normalize the body and locally normalize hands and face using the sign-space normalization implementation.

This switch applies to `ytasl_json` and `isharah_json`. The original pickle loaders retain the upstream confidence-thresholding and crop/scale normalization behavior.

The optional CLI flag `--normalize_text` is separate from pose normalization. It normalizes target text only for `ytasl_json` train and dev samples; test text and other loaders remain unchanged.

## Frame Selection

`--max_length` defaults to 256 frames. Shorter samples keep all frames. Longer training samples use a sorted random subset, while dev and test use deterministic near-uniform sampling across the full sequence.

## RGB Behavior

The two local JSON loaders are pose-only. Set `rgb` to `null` for them.

`original_pickle` and `csl_news` can use RGB when `rgb` specifies a root directory. RGB is a run-wide mode: if any active dataset requests RGB but another active dataset cannot supply it, Uni-Sign prints warnings and forces the entire run to pose-only. RGB is never enabled implicitly.

See [RGB configuration](../configs/README.md#rgb) for accepted JSON forms and active-split behavior.

## Targets and Tasks

- `ytasl_json` and `isharah_json` read `translation` and return no gloss sequence.
- `csl_news` reads `text` and returns no gloss sequence.
- `original_pickle` reads `text` and optionally `gloss`.

SLT uses the translation/text target. ISLR uses the text target as the prediction label. CSLR replaces the sentence target with the returned gloss sequence, so a loader/config intended for CSLR must provide glosses; the current local JSON and CSL-News loaders do not.
