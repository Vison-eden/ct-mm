#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc size-aware logistic calibrator for tri-class prediction.

The calibrator is trained after the image/semantic/multimodal model has been
frozen. It fits a small multiclass logistic regression model using:

    [log(p_AAH_AIS), log(p_MIA), log(p_IAC), long_diameter_z, solid_ratio_z]

where long_diameter_z and solid_ratio_z are standardized using the calibration
set statistics.

Expected input CSV columns:
    y_true
    p_AAH_AIS
    p_MIA
    p_IAC
    long_diameter
    solid_ratio

Outputs:
    size_calibrator_ld_sr.pt
    size_calibrator_ld_sr.json
    preds_calib_ld_sr.csv
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit


PROB_COLS = ["p_AAH_AIS", "p_MIA", "p_IAC"]
CALIB_PROB_COLS = ["p_calib_AAH_AIS", "p_calib_MIA", "p_calib_IAC"]
REQUIRED_COLS = ["y_true", *PROB_COLS, "long_diameter", "solid_ratio"]


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "") for c in out.columns]
    return out


class LogisticCalibrator(nn.Module):
    def __init__(self, in_dim: int = 5, num_classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def build_features_from_predictions(
    df: pd.DataFrame,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict, pd.DataFrame]:
    df = normalize_columns(df)
    check_required_columns(df)

    df = df.dropna(subset=["y_true"]).reset_index(drop=True)

    prob = df[PROB_COLS].to_numpy(dtype=np.float32)
    prob = np.clip(prob, eps, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)

    log_prob = np.log(prob).astype(np.float32)

    long_diameter = pd.to_numeric(df["long_diameter"], errors="coerce").to_numpy(dtype=np.float32)
    solid_ratio = pd.to_numeric(df["solid_ratio"], errors="coerce").to_numpy(dtype=np.float32)
    y = pd.to_numeric(df["y_true"], errors="raise").astype(int).to_numpy(dtype=np.int64)

    valid = ~(np.isnan(long_diameter) | np.isnan(solid_ratio))

    if int(valid.sum()) == 0:
        raise RuntimeError("No usable rows: long_diameter and solid_ratio are entirely missing.")

    df_used = df.loc[valid].reset_index(drop=True)
    y = y[valid]
    log_prob = log_prob[valid]
    long_diameter = long_diameter[valid]
    solid_ratio = solid_ratio[valid]

    mean_ld = float(long_diameter.mean())
    std_ld = float(long_diameter.std())
    if std_ld < 1e-6:
        std_ld = 1.0

    mean_sr = float(solid_ratio.mean())
    std_sr = float(solid_ratio.std())
    if std_sr < 1e-6:
        std_sr = 1.0

    ld_z = (long_diameter - mean_ld) / std_ld
    sr_z = (solid_ratio - mean_sr) / std_sr

    x = np.concatenate(
        [
            log_prob,
            ld_z[:, None].astype(np.float32),
            sr_z[:, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    stats = {
        "mean_long_diameter": mean_ld,
        "std_long_diameter": std_ld,
        "mean_solid_ratio": mean_sr,
        "std_solid_ratio": std_sr,
    }

    meta = {
        "feature_order": [
            "log_p_AAH_AIS",
            "log_p_MIA",
            "log_p_IAC",
            "long_diameter_z",
            "solid_ratio_z",
        ],
        "probability_columns": list(PROB_COLS),
        "raw_size_columns": ["long_diameter", "solid_ratio"],
        "eps": float(eps),
    }

    return x, y, stats, meta, df_used


def stratified_split(
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(val_ratio),
        random_state=int(seed),
    )
    index = np.arange(len(y))
    train_idx, val_idx = next(splitter.split(index, y))
    return train_idx, val_idx


def train_calibrator(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 2000,
    val_ratio: float = 0.1,
    patience: int = 200,
    seed: int = 42,
) -> Tuple[LogisticCalibrator, Dict]:
    train_idx, val_idx = stratified_split(y, val_ratio, seed)

    x_train = torch.tensor(x[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_idx], dtype=torch.long, device=device)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[val_idx], dtype=torch.long, device=device)

    model = LogisticCalibrator(in_dim=x.shape[1], num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best = {
        "best_epoch": -1,
        "best_val_nll": float("inf"),
        "best_val_acc": None,
        "best_val_macro_f1": None,
    }
    bad_epochs = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(x_val)
            val_nll = float(F.cross_entropy(logits_val, y_val).item())
            pred_val = logits_val.argmax(dim=1).detach().cpu().numpy()

        val_acc = float(accuracy_score(y[val_idx], pred_val))
        val_f1 = float(f1_score(y[val_idx], pred_val, average="macro"))

        if val_nll < best["best_val_nll"] - 1e-6:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best.update(
                best_epoch=int(epoch),
                best_val_nll=val_nll,
                best_val_acc=val_acc,
                best_val_macro_f1=val_f1,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"[E{epoch:04d}] "
                f"train_nll={float(loss.item()):.4f}  "
                f"val_nll={val_nll:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"val_macro_f1={val_f1:.4f}"
            )

        if bad_epochs >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    report = {
        **best,
        "stopped_epoch": int(epoch),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "val_ratio": float(val_ratio),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
    }

    return model, report


@torch.no_grad()
def apply_calibrator(
    model: LogisticCalibrator,
    x: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    logits = model(xt)
    prob = F.softmax(logits, dim=1).detach().cpu().numpy()
    pred = prob.argmax(axis=1).astype(int)
    return prob, pred


def resolve_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=200)

    args = parser.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.gpu)
    print(f"[DEVICE] {device}")

    df = pd.read_csv(args.preds_csv)
    x, y, stats, meta, df_used = build_features_from_predictions(df)

    print(f"[DATA] usable_samples={len(y)}")

    if "y_pred" in df_used.columns:
        base_pred = pd.to_numeric(df_used["y_pred"], errors="coerce").fillna(-1).astype(int).values
        valid_pred = base_pred >= 0
        if valid_pred.any():
            print(
                f"[BASE] "
                f"acc={accuracy_score(y[valid_pred], base_pred[valid_pred]):.4f}  "
                f"macro_f1={f1_score(y[valid_pred], base_pred[valid_pred], average='macro'):.4f}"
            )

    model, report = train_calibrator(
        x=x,
        y=y,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
    )

    prob_calib, pred_calib = apply_calibrator(model, x, device)

    print(
        f"[CALIB] "
        f"acc={accuracy_score(y, pred_calib):.4f}  "
        f"macro_f1={f1_score(y, pred_calib, average='macro'):.4f}"
    )

    df_out = df_used.copy()
    df_out["y_pred_calib"] = pred_calib
    df_out[CALIB_PROB_COLS[0]] = prob_calib[:, 0]
    df_out[CALIB_PROB_COLS[1]] = prob_calib[:, 1]
    df_out[CALIB_PROB_COLS[2]] = prob_calib[:, 2]

    out_csv = osp.join(args.out_dir, "preds_calib_ld_sr.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv}")

    checkpoint = {
        "model_type": "multiclass_logistic_ld_sr",
        "in_dim": int(x.shape[1]),
        "out_classes": 3,
        "stats": stats,
        "meta": meta,
        "train_report": report,
        "model": model.state_dict(),
    }

    ckpt_path = osp.join(args.out_dir, "size_calibrator_ld_sr.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"[SAVE] {ckpt_path}")

    json_path = osp.join(args.out_dir, "size_calibrator_ld_sr.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stats": stats,
                "meta": meta,
                "train_report": report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[WRITE] {json_path}")


if __name__ == "__main__":
    main()