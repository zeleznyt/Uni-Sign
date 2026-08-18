import json
from pathlib import Path


SPLITS = ("train", "dev", "test")
SUPPORTED_LOADERS = {
    "ytasl_json",
    "isharah_json",
    "original_pickle",
    "csl_news",
}
LOCAL_JSON_LOADERS = {"ytasl_json", "isharah_json"}
PICKLE_POSE_LOADERS = {"original_pickle", "csl_news"}
SUPPORTED_METRIC_TEXT_TRANSFORMS = {None, "csl_daily_char"}


def load_data_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Data config root must be a JSON object.")
    for field in ("layout", "graph", "target_language"):
        if not config.get(field):
            raise ValueError(f"Data config must define non-empty top-level '{field}'.")

    if config["layout"] not in ("default", "pruned", "isharah"):
        raise ValueError("Data config 'layout' must be 'default', 'pruned', or 'isharah'.")
    if config["graph"] not in ("ytasl", "default", "original"):
        raise ValueError("Data config 'graph' must be 'ytasl', 'default', or 'original'.")
    if not isinstance(config["target_language"], str):
        raise ValueError("Data config 'target_language' must be a string.")

    normalization = config.get("normalization", "none")
    if normalization not in ("none", "signspace"):
        raise ValueError("Data config 'normalization' must be 'none' or 'signspace'.")

    metric_transform = config.get("metric_text_transform")
    if metric_transform not in SUPPORTED_METRIC_TEXT_TRANSFORMS:
        raise ValueError(
            "Data config 'metric_text_transform' must be null or 'csl_daily_char'."
        )

    for split in SPLITS:
        normalize_split_specs(config, split)
    return config


def normalize_split_specs(config, split):
    value = config.get(split)
    if value is None:
        return []
    if isinstance(value, dict):
        specs = [value]
    elif isinstance(value, list):
        specs = value
    else:
        raise ValueError(f"Data config split '{split}' must be an object, list, or null.")

    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(
                f"Data config split '{split}' item {index} must be a JSON object."
            )
    return specs


def iter_split_specs(config, splits=SPLITS):
    for split in splits:
        for spec in normalize_split_specs(config, split):
            yield split, spec


def get_required_split_specs(config, split):
    specs = normalize_split_specs(config, split)
    if len(specs) == 0:
        raise ValueError(f"Data config split '{split}' is required for this run.")
    return specs


def spec_name(spec):
    return spec.get("name") or spec.get("loader") or "unnamed"


def spec_pose_roots(spec):
    roots = spec.get("pose_roots")
    if roots is None:
        root = spec.get("pose_root")
        roots = [] if root is None else [root]
    if isinstance(roots, str):
        roots = [roots]
    if not isinstance(roots, list) or not all(isinstance(root, str) for root in roots):
        raise ValueError(
            f"Dataset spec '{spec_name(spec)}' must define 'pose_roots' as a list of paths."
        )
    return roots


def spec_rgb_config(spec):
    if "rgb" in spec:
        return spec["rgb"]
    return spec.get("rgb_root")


def spec_rgb_root(spec):
    rgb = spec_rgb_config(spec)
    if rgb is None:
        return None
    if isinstance(rgb, str):
        return rgb
    if isinstance(rgb, dict):
        root = rgb.get("root")
        if root is None or isinstance(root, str):
            return root
    raise ValueError(
        f"Dataset spec '{spec_name(spec)}' must define RGB as null, a root path, or an object with 'root'."
    )


def preflight_data_config(config, active_splits=SPLITS):
    active_splits = tuple(active_splits)
    active_specs = list(iter_split_specs(config, active_splits))
    rgb_requested = any(
        spec_rgb_config(spec) is not None for _, spec in active_specs
    )
    rgb_compatible = rgb_requested

    report = {
        "layout": config.get("layout"),
        "graph": config.get("graph"),
        "target_language": config.get("target_language"),
        "normalization": config.get("normalization", "none"),
        "metric_text_transform": config.get("metric_text_transform"),
        "rgb_requested": rgb_requested,
        "rgb_enabled": False,
        "splits": [],
        "warnings": [],
        "errors": [],
    }

    for split, spec in active_specs:
        name = spec_name(spec)
        loader = spec.get("loader")
        annotation_path = spec.get("annotation_path")
        rgb_config = spec_rgb_config(spec)
        weight = spec.get("weight")

        split_report = {
            "split": split,
            "name": name,
            "loader": loader,
            "annotation_path": annotation_path,
            "annotation_exists": False,
            "pose_roots": [],
            "rgb": rgb_config,
            "rgb_root": None,
            "weight": weight,
        }

        if loader not in SUPPORTED_LOADERS:
            supported = ", ".join(sorted(SUPPORTED_LOADERS))
            report["errors"].append(
                f"{split}/{name}: unsupported loader '{loader}'. Supported loaders: {supported}."
            )

        if not annotation_path:
            report["errors"].append(f"{split}/{name}: missing required 'annotation_path'.")
        else:
            annotation_exists = Path(annotation_path).is_file()
            split_report["annotation_exists"] = annotation_exists
            if not annotation_exists:
                report["errors"].append(
                    f"{split}/{name}: annotation file does not exist: {annotation_path}"
                )

        if weight is not None:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
                report["errors"].append(
                    f"{split}/{name}: weight must be a positive number when provided."
                )

        try:
            roots = spec_pose_roots(spec)
        except ValueError as exc:
            report["errors"].append(str(exc))
            roots = []

        if len(roots) == 0:
            report["errors"].append(f"{split}/{name}: missing required 'pose_roots'.")

        pose_pattern = "*.pkl" if loader in PICKLE_POSE_LOADERS else "*.json"
        for root in roots:
            root_path = Path(root)
            exists = root_path.is_dir()
            file_count = len(list(root_path.glob(pose_pattern))) if exists else 0
            root_report = {
                "path": root,
                "exists": exists,
                "file_pattern": pose_pattern,
                "file_count": file_count,
            }
            split_report["pose_roots"].append(root_report)
            if not exists:
                report["errors"].append(f"{split}/{name}: pose root does not exist: {root}")
            elif file_count == 0:
                report["warnings"].append(
                    f"{split}/{name}: pose root contains no {pose_pattern} files: {root}"
                )

        try:
            rgb_root = spec_rgb_root(spec)
        except ValueError as exc:
            report["errors"].append(str(exc))
            rgb_root = None
        split_report["rgb_root"] = rgb_root

        if rgb_requested:
            if loader in LOCAL_JSON_LOADERS:
                rgb_compatible = False
                report["warnings"].append(
                    f"{split}/{name}: loader '{loader}' is pose-only and cannot provide RGB."
                )
            elif rgb_root is None:
                rgb_compatible = False
                report["warnings"].append(
                    f"{split}/{name}: RGB was requested but this dataset has no RGB root."
                )
            else:
                rgb_path = Path(rgb_root)
                if not rgb_path.is_dir():
                    rgb_compatible = False
                    report["errors"].append(
                        f"{split}/{name}: RGB root does not exist: {rgb_root}"
                    )

        report["splits"].append(split_report)

    report["rgb_enabled"] = bool(rgb_requested and rgb_compatible)
    if rgb_requested and not report["rgb_enabled"]:
        report["warnings"].append(
            "RGB is not available consistently across all active datasets; forcing the entire run to pose-only."
        )
    return report


def format_data_setup_report(report, dataset_summaries=None, train_sampling=None):
    lines = [
        "Data config setup:",
        f"  layout: {report.get('layout')}",
        f"  graph: {report.get('graph')}",
        f"  target_language: {report.get('target_language')}",
        f"  normalization: {report.get('normalization')}",
        f"  metric_text_transform: {report.get('metric_text_transform')}",
        f"  rgb_requested: {report.get('rgb_requested')}",
        f"  rgb_enabled: {report.get('rgb_enabled')}",
    ]

    summaries = dataset_summaries or {}
    summary_by_key = {}
    for split, split_summaries in summaries.items():
        for summary in split_summaries:
            summary_by_key[(split, summary.get("name"), summary.get("loader"))] = summary

    for item in report.get("splits", []):
        lines.append(f"  [{item['split']}] {item['name']} ({item.get('loader')})")
        lines.append(
            f"    annotation: {item.get('annotation_path')} exists={item.get('annotation_exists')}"
        )
        for root in item.get("pose_roots", []):
            lines.append(
                f"    pose_root: {root['path']} exists={root['exists']} "
                f"{root['file_pattern']}_count={root['file_count']}"
            )
        lines.append(f"    rgb: {item.get('rgb')}")
        lines.append(f"    weight: {item.get('weight')}")

        summary = summary_by_key.get((item["split"], item["name"], item.get("loader")))
        if summary is not None:
            lines.append(f"    annotation_samples: {summary.get('annotation_samples')}")
            lines.append(f"    usable_samples: {summary.get('usable_samples')}")
            missing_count = summary.get("missing_pose_samples")
            if missing_count is not None:
                lines.append(f"    filtered_missing_pose_samples: {missing_count}")
            missing_examples = summary.get("missing_pose_examples") or []
            if missing_examples:
                lines.append(f"    missing_pose_examples: {', '.join(missing_examples)}")

    if train_sampling is not None:
        lines.append("  [train sampling]")
        lines.append(f"    mode: {train_sampling['mode']}")
        lines.append(f"    replacement: {train_sampling['replacement']}")
        lines.append(
            f"    nominal_epoch_draws: {train_sampling['nominal_epoch_draws']}"
        )
        for dataset in train_sampling["datasets"]:
            details = (
                f"source_samples={dataset['source_samples']} "
                f"expected_share={dataset['expected_share']:.2%} "
                f"expected_draws={dataset['expected_draws']:.1f}"
            )
            if dataset["configured_weight"] is not None:
                details += f" configured_weight={dataset['configured_weight']}"
            lines.append(f"    {dataset['name']}: {details}")

    for warning in report.get("warnings", []):
        lines.append(f"  WARNING: {warning}")
    for error in report.get("errors", []):
        lines.append(f"  ERROR: {error}")
    return "\n".join(lines)
