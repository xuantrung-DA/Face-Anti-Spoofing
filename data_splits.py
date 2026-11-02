"""Explicit subject/session-disjoint splits; never infer identities from filenames."""

import csv
from copy import copy
from pathlib import Path


def manifest_datasets(dataset, root_dir, manifest_path, train_transform, eval_transform, *, real_label):
    """Load CSV columns path,label,group,split; preserve each model's label polarity."""
    root = Path(root_dir).resolve()
    samples = {name: [] for name in ("train", "val", "test")}
    groups, seen = {}, set()
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "group", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Split CSV requires path,label,group,split columns.")
        for row in reader:
            split = row["split"].strip()
            label, group = row["label"].strip(), row["group"].strip()
            if split not in samples or label not in {"real", "fake"} or not group:
                raise ValueError("Each row needs train/val/test, real/fake, and a nonempty group.")
            relative = Path(row["path"])
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Manifest image must exist inside the dataset root.")
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Manifest paths must identify supported image files.")
            if path in seen:
                raise ValueError("An image occurs more than once in the split manifest.")
            if group in groups and groups[group] != split:
                raise ValueError(f"Group {group!r} crosses split boundaries.")
            seen.add(path)
            groups[group] = split
            numeric_label = real_label if label == "real" else 1 - real_label
            samples[split].append((str(path), numeric_label))
    result = []
    for split, rows in samples.items():
        if {label for _, label in rows} != {0, 1}:
            raise ValueError(f"Split {split!r} must contain both real and fake images.")
        view = copy(dataset)
        view.samples = rows
        view.transform = train_transform if split == "train" else eval_transform
        result.append(view)
    return tuple(result)
