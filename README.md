# Face Anti-Spoofing with CDCN and Attention

PyTorch implementations of spatial-attention (SA) and CBAM variants of a
CDCN-based face presentation-attack detector, developed as a module for a
Secure Login System. This repository contains training code, a webcam capture
utility, and legacy PyTorch/ONNX artifacts; it is not a complete login service.

[![CI](https://github.com/xuantrung-DA/Face-Anti-Spoofing/actions/workflows/ci.yml/badge.svg)](https://github.com/xuantrung-DA/Face-Anti-Spoofing/actions/workflows/ci.yml)

## Evidence and current status

The former 96.9% accuracy / 0.032 ACER headline is withdrawn as a verified
benchmark. The legacy source used image-level random splits, and CBAM selected
its operating threshold again on test labels. The public artifacts do not include
the dataset manifest, identity split, or run provenance needed to establish which
weights produced those numbers.

Corrected training now requires an explicit group-disjoint manifest, keeps
train/evaluation transforms separate, restores an independent best-weight
snapshot, and uses a validation-selected threshold for CBAM test evaluation.
These fixes do not retroactively validate old metrics or weights. No new
accuracy, generalization, or latency result is claimed until a new experiment
is completed and published.

Original method: [Searching Central Difference Convolutional Networks for Face
Anti-Spoofing (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.html).
This code implements attention variants, not the paper's official benchmark.

## Install and verify without private data

Python 3.12 is the maintenance-test environment. Run from the repository root:

```bash
git clone https://github.com/xuantrung-DA/Face-Anti-Spoofing.git
cd Face-Anti-Spoofing
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python cdcnCBAM.py --help
```

Tests use synthetic samples and do not measure FAS accuracy. For CUDA, install
compatible PyTorch/torchvision wheels for the intended device. Requirements pin
direct dependencies; they are not a full transitive lock.

## Dataset and split contract

Supply images you are permitted to use and a local CSV:

```csv
path,label,group,split
real/subject01.jpg,real,subject01,train
fake/subject01.jpg,fake,subject01,train
real/subject02.jpg,real,subject02,val
fake/subject02.jpg,fake,subject02,val
real/subject03.jpg,real,subject03,test
fake/subject03.jpg,fake,subject03,test
```

Paths are relative to `--data-dir`. Use the true identity/leakage unit as `group`:
related frames, sessions, live images and attacks from one identity must remain
in one split. The loader rejects duplicate paths, missing files, cross-split
groups, and splits without both classes. It cannot detect an identity assigned
two different group IDs; audit source metadata before training. There is no
automatic image-level random-split fallback.

```bash
python cdcnCBAM.py --data-dir data --split-manifest splits.csv --epochs 5 --output-dir checkpoints/new-cbam
python cdcnSA.py --data-dir data --split-manifest splits.csv --epochs 5
```

CBAM selects its checkpoint and threshold on validation ACER, then applies that
fixed threshold to test. SA evaluates its final epoch at a fixed 0.5 threshold.
Use a matched selection policy before comparing variants. Default seed: 2025;
cross-device bitwise reproducibility is not guaranteed.

## Label and artifact contracts

| Variant | Live label | Spoof label | Sigmoid score means |
| --- | ---: | ---: | --- |
| SA | 1 | 0 | Live probability |
| CBAM | 0 | 1 | Spoof probability |

Legacy polarities are preserved for compatibility. APCER is spoof accepted as
live; BPCER is live rejected as spoof; ACER is their mean. CBAM checkpoints
record the label mapping and selected validation threshold.

`cdcnpp80k.pt` and `cdcnpp_cbam.onnx` are legacy artifacts with unverified run
provenance. They have not been retrained or replaced by this update. Only load
trusted PyTorch files. New training outputs are ignored by Git.

## Webcam capture

[collect.py](collect.py) uses OpenCV, Pillow and MTCNN. Install `mtcnn` and a
compatible TensorFlow runtime separately; optional capture dependencies are not
installed in test CI. Capture is local and interactive, not an evaluated
inference endpoint. Obtain consent before collecting or sharing face images.

## Next reproducible experiment

Publish dataset attribution, identity counts, split manifest hash, dependency
versions, seed, checkpoint hash, validation threshold, and per-run test metrics.
Use multiple seeds and subject-disjoint or cross-dataset evaluation before making
generalization claims. Keep raw faces and private manifests outside Git.
