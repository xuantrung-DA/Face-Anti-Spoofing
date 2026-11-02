import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import cdcnCBAM as cbam
from data_splits import manifest_datasets


def write_manifest(tmp_path, mutate=None):
    rows = []
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            path = tmp_path / f"{split}-{label}.jpg"
            path.touch()
            rows.append(dict(path=path.name, label=label, group=split, split=split))
    if mutate:
        mutate(rows)
    manifest = tmp_path / "splits.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_independent_transforms_and_legacy_label_polarities(tmp_path):
    manifest = write_manifest(tmp_path)
    source = SimpleNamespace(samples=[], transform=None)
    train_tf, eval_tf = object(), object()
    for real_label in (0, 1):
        train, val, test = manifest_datasets(
            source, tmp_path, manifest, train_tf, eval_tf, real_label=real_label
        )
        assert train.transform is train_tf
        assert val.transform is test.transform is eval_tf
        assert source.transform is None
        assert train.samples[0][1] == real_label
        val.transform = None
        assert train.transform is train_tf and test.transform is eval_tf


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r[2].update(group="train"), "crosses"),
    (lambda r: r[2].update(path=r[0]["path"]), "more than once"),
    (lambda r: r[2].update(path="../outside.jpg"), "inside"),
    (lambda r: r[0].update(label="fake"), "both"),
])
def test_manifest_rejects_invalid_splits(tmp_path, mutation, match):
    manifest = write_manifest(tmp_path, mutation)
    with pytest.raises(ValueError, match=match):
        manifest_datasets(SimpleNamespace(), tmp_path, manifest, None, None, real_label=0)


def test_pad_error_names_follow_spoof_positive_contract():
    labels = np.array([0, 0, 1, 1])
    metrics = cbam.compute_binary_metrics(labels, np.array([0.1, 0.2, 0.1, 0.9]))
    assert metrics["apcer"] == 0.5
    assert metrics["bpcer"] == 0.0
    assert metrics["acer"] == 0.25


def test_fixed_threshold_evaluation_never_selects_from_test_labels(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Threshold search must not run on test data")
    monkeypatch.setattr(cbam, "best_thr_by_acer", forbidden)
    loader = DataLoader(TensorDataset(torch.tensor([[-2.0], [2.0]]), torch.tensor([0, 1])), batch_size=2)
    metrics, _, _ = cbam.evaluate(torch.nn.Flatten(start_dim=0), loader, "cpu", threshold=0.8)
    assert metrics["thr"] == 0.8
    with pytest.raises(ValueError):
        cbam.evaluate(torch.nn.Identity(), loader, "cpu")


def test_training_restores_best_weights_and_validation_threshold(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path)
    class TinyDataset:
        def __init__(self, *args, **kwargs):
            self.samples = []
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            label = self.samples[idx][1]
            return torch.tensor([float(label)]), torch.tensor(float(label))
    class TinyModel(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
        def forward(self, x):
            return super().forward(x).squeeze(-1)
    validation_weights, tested = [], []
    def evaluate(model, loader, device, *, threshold=None, select_threshold=False):
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if select_threshold:
            validation_weights.append(weights)
            acer = 0.1 if len(validation_weights) == 1 else 0.2
        else:
            tested.append((weights, threshold))
            acer = 0.1
        return dict(acc=0.9, apcer=acer, bpcer=acer, acer=acer, auc=0.9, f1=0.9, thr=0.7), None, None
    monkeypatch.setattr(cbam, "FASDataset", TinyDataset)
    monkeypatch.setattr(cbam, "CDCNpp_CBAM", TinyModel)
    monkeypatch.setattr(cbam, "evaluate", evaluate)
    monkeypatch.setattr(cbam, "DEVICE", "cpu")
    monkeypatch.setattr(cbam, "USE_CUDA", False)
    monkeypatch.setattr(torch.onnx, "export", lambda *a, **kw: None)
    cbam.train_loop(tmp_path, epochs=2, batch_size=2, split_manifest=manifest, output_dir=tmp_path / "run")
    assert tested[0][1] == 0.7
    assert any(not torch.equal(validation_weights[0][k], validation_weights[1][k]) for k in validation_weights[0])
    assert all(torch.equal(tested[0][0][k], validation_weights[0][k]) for k in validation_weights[0])
