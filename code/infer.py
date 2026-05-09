#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference script for tri-class lung nodule classification.

This script loads a trained checkpoint and exports prediction probabilities.

Expected samples.csv columns:
    path
    subject
    nod_id
    label or label_idx
    density
    structured semantic feature columns used during training

Outputs:
    y_true
    y_pred
    p_AAH_AIS
    p_MIA
    p_IAC

If --calibrator_path is provided, the script also outputs:
    y_pred_calib
    p_calib_AAH_AIS
    p_calib_MIA
    p_calib_IAC
"""

from __future__ import annotations

import argparse
import os.path as osp
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from model import ImgOnlyModel, TabOnlyModel, UnifiedModel
from utils import LungDataset, read_samples_for_task, resolve_device, seed_all


BASE_PROB_COLS = ["p_AAH_AIS", "p_MIA", "p_IAC"]
CALIB_PROB_COLS = ["p_calib_AAH_AIS", "p_calib_MIA", "p_calib_IAC"]


class LogisticCalibrator(torch.nn.Module):
    def __init__(self, in_dim: int = 5, num_classes: int = 3) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def to_abs_path(path: str, data_root: str | None = None) -> str:
    path = str(path)

    if osp.isabs(path) or not data_root:
        return path

    return osp.join(data_root, path)


def infer_in_channels_from_first_sample(df: pd.DataFrame) -> int:
    arr = np.load(df.iloc[0]["path"])

    if arr.ndim == 4:
        return int(arr.shape[0])

    if arr.ndim == 3:
        return 1

    raise RuntimeError(f"Unexpected patch shape: {arr.shape}")


def load_checkpoint(path: str, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(
            "Unsupported checkpoint format. Expected a checkpoint saved by train_stable.py."
        )

    required = ["args", "stats", "den_vocab"]
    missing = [k for k in required if k not in ckpt]

    if missing:
        raise RuntimeError(f"Checkpoint is missing required fields: {missing}")

    return ckpt


def build_model_from_checkpoint(
    ckpt: Dict,
    device: torch.device,
    in_channels_fallback: int,
):
    args = ckpt["args"]
    stats = ckpt["stats"]
    den_vocab = ckpt["den_vocab"]

    mode = args.get("mode", "mm")
    img_backbone = args.get("img_backbone", "r2plus1d_18")
    img_trans_layers = int(args.get("img_trans_layers", 0))
    d_img = int(args.get("d_img", 256))
    d_txt = int(args.get("d_txt", 256))
    drop = float(args.get("drop", 0.2))
    use_gate = bool(args.get("use_gate", False))
    mix_rule = args.get("mix_rule", "prob_anchor")
    alpha_cap = float(args.get("alpha_cap", 0.30))

    in_channels = args.get("in_channels", None)
    if in_channels is None:
        in_channels = args.get("in_ch", None)
    if in_channels is None:
        in_channels = in_channels_fallback
    in_channels = int(in_channels)

    tab_columns = stats.get("tab_columns", None)
    if not tab_columns:
        raise RuntimeError("Checkpoint does not contain stats['tab_columns'].")

    den_num_classes = len(den_vocab) if den_vocab else 4

    if mode == "img_only":
        model = ImgOnlyModel(
            img_backbone=img_backbone,
            in_ch=in_channels,
            d_img=d_img,
            drop=drop,
            img_trans_layers=img_trans_layers,
            out_classes=3,
        )

    elif mode == "txt_only":
        model = TabOnlyModel(
            d_txt=d_txt,
            drop=drop,
            den_num_classes=den_num_classes,
            out_classes=3,
            num_names=tab_columns,
        )

    elif mode == "mm":
        model = UnifiedModel(
            mode="mm",
            task_out_classes=3,
            img_backbone=img_backbone,
            in_ch=in_channels,
            img_trans_layers=img_trans_layers,
            d_img=d_img,
            d_txt=d_txt,
            den_num_classes=den_num_classes,
            drop=drop,
            use_gate=use_gate,
            mix_rule=mix_rule,
            alpha_cap=alpha_cap,
            num_names=tab_columns,
        )

    else:
        raise RuntimeError(f"Unsupported model mode: {mode}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    return model, mode, in_channels, stats, den_vocab


def prepare_samples(args) -> pd.DataFrame:
    ns = argparse.Namespace(task="tri", samples_csv=args.samples_csv)
    df = read_samples_for_task(ns).copy()

    if args.data_root:
        df["path"] = df["path"].map(lambda p: to_abs_path(p, args.data_root))

    missing = ~df["path"].map(lambda p: osp.isfile(str(p)))
    if missing.any():
        examples = df.loc[missing, "path"].head(5).tolist()
        raise RuntimeError(f"Some patch files are missing. Examples: {examples}")

    return df.reset_index(drop=True)


@torch.no_grad()
def run_inference(
    model,
    loader: DataLoader,
    mode: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    del mode

    all_true = []
    all_pred = []
    all_prob = []

    model.eval()

    for x_img, x_tab, y, ld_z in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_num = x_tab[0].to(device, non_blocking=True)
        x_den = x_tab[1].to(device, non_blocking=True)
        ld_z = ld_z.to(device, non_blocking=True)

        logits, _ = model(x_img, (x_num, x_den), ld_z, x_den)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(all_true).astype(int)
    y_pred = np.concatenate(all_pred).astype(int)
    prob = np.concatenate(all_prob).astype(np.float32)

    return y_true, y_pred, prob


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> None:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(
        f"[{prefix}] "
        f"acc={acc:.4f}  "
        f"macro_f1={f1m:.4f}  "
        f"macro_precision={prec:.4f}  "
        f"macro_recall={rec:.4f}"
    )


def load_size_calibrator(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict):
        raise RuntimeError("Invalid calibrator checkpoint.")

    in_dim = int(ckpt.get("in_dim", 5))
    out_classes = int(ckpt.get("out_classes", 3))

    model = LogisticCalibrator(in_dim=in_dim, num_classes=out_classes).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    stats = ckpt.get("stats", {})
    meta = ckpt.get("meta", {})

    return model, stats, meta


def build_calibrator_features(
    df: pd.DataFrame,
    prob: np.ndarray,
    stats: Dict,
    eps: float = 1e-6,
) -> np.ndarray:
    if "long_diameter" not in df.columns:
        raise RuntimeError("Calibrator requires 'long_diameter' in samples.csv.")

    if "solid_ratio" not in df.columns:
        raise RuntimeError("Calibrator requires 'solid_ratio' in samples.csv.")

    p = np.clip(prob.astype(np.float32), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    log_p = np.log(p)

    ld = pd.to_numeric(df["long_diameter"], errors="coerce").to_numpy(dtype=np.float32)
    sr = pd.to_numeric(df["solid_ratio"], errors="coerce").to_numpy(dtype=np.float32)

    if np.isnan(ld).any() or np.isnan(sr).any():
        raise RuntimeError(
            "Calibrator cannot be applied because long_diameter or solid_ratio contains missing values."
        )

    mean_ld = float(stats["mean_long_diameter"])
    std_ld = float(stats["std_long_diameter"])
    mean_sr = float(stats["mean_solid_ratio"])
    std_sr = float(stats["std_solid_ratio"])

    if std_ld < 1e-6:
        std_ld = 1.0
    if std_sr < 1e-6:
        std_sr = 1.0

    ld_z = (ld - mean_ld) / std_ld
    sr_z = (sr - mean_sr) / std_sr

    x = np.concatenate(
        [
            log_p,
            ld_z[:, None].astype(np.float32),
            sr_z[:, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return x


@torch.no_grad()
def apply_calibrator(
    calibrator,
    df: pd.DataFrame,
    prob: np.ndarray,
    stats: Dict,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    x = build_calibrator_features(df, prob, stats)
    xt = torch.tensor(x, dtype=torch.float32, device=device)

    logits = calibrator(xt)
    prob_calib = torch.softmax(logits, dim=1).detach().cpu().numpy()
    pred_calib = prob_calib.argmax(axis=1).astype(int)

    return pred_calib, prob_calib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--samples_csv", required=True)
    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--data_root", default="")
    parser.add_argument("--calibrator_path", default="")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed_all(args.seed)

    device = resolve_device(args.gpu)
    print(f"[DEVICE] {device}")

    df = prepare_samples(args)

    ckpt = load_checkpoint(args.ckpt_path, device)
    in_channels_fallback = infer_in_channels_from_first_sample(df)

    model, mode, in_channels, stats, den_vocab = build_model_from_checkpoint(
        ckpt=ckpt,
        device=device,
        in_channels_fallback=in_channels_fallback,
    )

    need_image = mode != "txt_only"

    dataset = LungDataset(
        df=df,
        stats=stats,
        den_vocab=den_vocab,
        in_ch=in_channels,
        task="tri",
        augment=None,
        train=False,
        load_image=need_image,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"[DATA] samples={len(dataset)}  mode={mode}  in_channels={in_channels}")

    y_true, y_pred, prob = run_inference(
        model=model,
        loader=loader,
        mode=mode,
        device=device,
    )

    print_metrics(y_true, y_pred, prefix="BASE")

    out = df.copy().reset_index(drop=True)
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out[BASE_PROB_COLS[0]] = prob[:, 0]
    out[BASE_PROB_COLS[1]] = prob[:, 1]
    out[BASE_PROB_COLS[2]] = prob[:, 2]

    if args.calibrator_path:
        calibrator, calib_stats, _ = load_size_calibrator(args.calibrator_path, device)

        y_pred_calib, prob_calib = apply_calibrator(
            calibrator=calibrator,
            df=out,
            prob=prob,
            stats=calib_stats,
            device=device,
        )

        out["y_pred_calib"] = y_pred_calib
        out[CALIB_PROB_COLS[0]] = prob_calib[:, 0]
        out[CALIB_PROB_COLS[1]] = prob_calib[:, 1]
        out[CALIB_PROB_COLS[2]] = prob_calib[:, 2]

        print_metrics(y_true, y_pred_calib, prefix="CALIB")

    out_dir = osp.dirname(osp.abspath(args.out_csv))
    if out_dir:
        import os

        os.makedirs(out_dir, exist_ok=True)

    out.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv}")


if __name__ == "__main__":
    main()