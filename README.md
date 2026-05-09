# ct-mm

Source code for multimodal tri-class classification of lung adenocarcinoma spectrum nodules using 3D CT patches and structured radiological semantic features.

This repository accompanies a multicenter retrospective study on preoperative classification of lung adenocarcinoma spectrum nodules into atypical adenomatous hyperplasia/adenocarcinoma in situ (AAH/AIS), minimally invasive adenocarcinoma (MIA), and invasive adenocarcinoma (IAC).

The framework supports CT-only, semantic-only, and multimodal models.

## Code structure

```text
code/
  model.py              Model definitions
  utils.py              Dataset, feature processing, augmentation, and evaluation utilities
  train_stable.py       Model training
  infer.py              Model inference
  calibrate_ld_sr.py    Post-hoc size-aware calibration
  cam.py                Grad-CAM visualization
  test_process.py       CT and mask preprocessing to 1.0-mm isotropic space
  real_gen.py           3D CT patch generation
```

## Environment

The code was tested under the following environment:

- OS: Linux 6.8.0-52-generic x86_64
- Python: 3.9.21
- PyTorch: 2.6.0+cu124
- TorchVision: 0.21.0+cu124
- CUDA: 12.4
- cuDNN: 9.1.0
- GPU: NVIDIA RTX 5880 Ada Generation

Core Python packages:

| Package | Version |
|---|---:|
| numpy | 1.26.4 |
| pandas | 2.3.2 |
| SimpleITK | 2.5.2 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.9.4 |

Other compatible Python, PyTorch, CUDA, and TorchVision configurations may also work, but were not formally tested.

## Task definition

The classification task includes three pathological categories:

| Index | Class |
|---:|---|
| 0 | AAH/AIS |
| 1 | MIA |
| 2 | IAC |

The code supports three model modes:

| Mode | Description |
|---|---|
| `img_only` | 3D CT patch only |
| `txt_only` | Structured radiological semantic features only |
| `mm` | Multimodal fusion of 3D CT patches and structured semantic features |

## Input format

The public code assumes that all patient identifiers have been de-identified and that the input tables have been standardized.

The training CSV should contain at least the following columns:

```text
path
subject
nod_id
label or label_idx
density
spiculation
lobulation
smooth_sharp
bronchus_sign
pleural_indentation
cord_sign
irregular
round_like
vascular_convergence
```

Optional size-related fields used for post-hoc calibration include:

```text
long_diameter
solid_ratio
```

The expected class labels are:

```text
AAH/AIS
MIA
IAC
```

The expected attenuation categories are:

```text
GGO
PartSolid
Solid
```

## Basic usage

### Train a CT-only model

```bash
python code/train_stable.py \
  --samples_csv /path/to/samples.csv \
  --out_dir /path/to/output/img_only \
  --mode img_only \
  --img_backbone r2plus1d_18
```

### Train a semantic-only model

```bash
python code/train_stable.py \
  --samples_csv /path/to/samples.csv \
  --out_dir /path/to/output/txt_only \
  --mode txt_only
```

### Train a multimodal model

```bash
python code/train_stable.py \
  --samples_csv /path/to/samples.csv \
  --out_dir /path/to/output/mm \
  --mode mm \
  --img_backbone r2plus1d_18 \
  --use_gate
```

### Run inference

```bash
python code/infer.py \
  --ckpt_path /path/to/best.pt \
  --samples_csv /path/to/samples.csv \
  --out_csv /path/to/predictions.csv
```

### Post-hoc calibration

```bash
python code/calibrate_ld_sr.py \
  --preds_csv /path/to/predictions.csv \
  --out_dir /path/to/calibration_output
```

### Grad-CAM visualization

```bash
python code/cam.py \
  --ckpt_path /path/to/best.pt \
  --samples_csv /path/to/samples.csv \
  --out_dir /path/to/cam_output \
  --cam_layer layer3 \
  --score_source image
```

## Data availability

This repository contains source code only.

Patient CT images, lesion masks, structured radiological feature tables, pathology labels, and trained model weights are not included because they contain sensitive clinical information and are subject to institutional data-use restrictions.

De-identified data availability should follow the data availability statement in the associated manuscript.

## Notes

The preprocessing scripts are provided for standardized CT and lesion-mask inputs. Institution-specific data parsing, private directory conventions, and internal data-cleaning rules are not included.

Users should verify CT-mask alignment and inspect generated overlay images before model training.

## Citation

If you use this code, please cite the associated manuscript. Citation information will be updated after publication.

```text
Citation to be added after publication.
```

## License

No open-source license has been applied to this repository at this stage. All rights are reserved by the authors unless otherwise stated.
