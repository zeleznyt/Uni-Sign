import json
from pathlib import Path


SPLITS = ("train", "dev", "test")


def load_data_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Data config root must be a JSON object.")
    if "layout" not in config:
        raise ValueError("Data config must define top-level 'layout'.")
    if "target_language" not in config:
        raise ValueError("Data config must define top-level 'target_language'.")

    return config


def normalize_split_specs(config, split):
    value = config.get(split)
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError(f"Data config split '{split}' must be an object, list, or null.")


def iter_split_specs(config):
    for split in SPLITS:
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
    if not isinstance(roots, list):
        raise ValueError(f"Dataset spec '{spec_name(spec)}' must define 'pose_roots' as a list.")
    return roots


def spec_rgb_config(spec):
    return spec.get("rgb", None)


def preflight_data_config(config):
    report = {
        "layout": config.get("layout"),
        "target_language": config.get("target_language"),
        "splits": [],
        "warnings": [],
        "errors": [],
    }

    for split, spec in iter_split_specs(config):
        name = spec_name(spec)
        loader = spec.get("loader")
        annotation_path = spec.get("annotation_path")
        rgb_config = spec_rgb_config(spec)

        split_report = {
            "split": split,
            "name": name,
            "loader": loader,
            "annotation_path": annotation_path,
            "annotation_exists": False,
            "pose_roots": [],
            "rgb": rgb_config,
            "weight": spec.get("weight", 1.0),
        }

        if not loader:
            report["errors"].append(f"{split}/{name}: missing required 'loader'.")
        if not annotation_path:
            report["errors"].append(f"{split}/{name}: missing required 'annotation_path'.")
        else:
            annotation_exists = Path(annotation_path).is_file()
            split_report["annotation_exists"] = annotation_exists
            if not annotation_exists:
                report["errors"].append(f"{split}/{name}: annotation file does not exist: {annotation_path}")

        try:
            roots = spec_pose_roots(spec)
        except ValueError as exc:
            report["errors"].append(str(exc))
            roots = []

        if len(roots) == 0:
            report["errors"].append(f"{split}/{name}: missing required 'pose_roots'.")

        for root in roots:
            root_path = Path(root)
            exists = root_path.is_dir()
            json_count = len(list(root_path.glob("*.json"))) if exists else 0
            root_report = {
                "path": root,
                "exists": exists,
                "json_count": json_count,
            }
            split_report["pose_roots"].append(root_report)
            if not exists:
                report["errors"].append(f"{split}/{name}: pose root does not exist: {root}")
            elif json_count == 0:
                report["warnings"].append(f"{split}/{name}: pose root contains no JSON files: {root}")

        if rgb_config is not None:
            report["warnings"].append(
                f"{split}/{name}: RGB config is set but local RGB loading is not implemented; continuing pose-only."
            )

        report["splits"].append(split_report)

    return report


def format_data_setup_report(report, dataset_summaries=None):
    lines = [
        "Data config setup:",
        f"  layout: {report.get('layout')}",
        f"  target_language: {report.get('target_language')}",
    ]

    summaries = dataset_summaries or {}
    summary_by_key = {}
    for split, split_summaries in summaries.items():
        for summary in split_summaries:
            summary_by_key[(split, summary.get("name"), summary.get("loader"))] = summary

    for item in report.get("splits", []):
        lines.append(f"  [{item['split']}] {item['name']} ({item.get('loader')})")
        lines.append(f"    annotation: {item.get('annotation_path')} exists={item.get('annotation_exists')}")
        for root in item.get("pose_roots", []):
            lines.append(
                f"    pose_root: {root['path']} exists={root['exists']} json_count={root['json_count']}"
            )
        lines.append(f"    rgb: {item.get('rgb')}")
        lines.append(f"    weight: {item.get('weight')}")

        summary = summary_by_key.get((item["split"], item["name"], item.get("loader")))
        if summary is not None:
            lines.append(f"    annotation_clips: {summary.get('annotation_clips')}")
            lines.append(f"    usable_clips: {summary.get('usable_clips')}")
            lines.append(f"    filtered_missing_pose_clips: {summary.get('missing_pose_clips')}")
            missing_examples = summary.get("missing_pose_examples") or []
            if missing_examples:
                lines.append(f"    missing_pose_examples: {', '.join(missing_examples)}")

    for warning in report.get("warnings", []):
        lines.append(f"  WARNING: {warning}")
    for error in report.get("errors", []):
        lines.append(f"  ERROR: {error}")
    return "\n".join(lines)
