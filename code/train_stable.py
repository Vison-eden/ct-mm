#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ImgOnlyModel, TabOnlyModel, UnifiedModel
from utils import (
    Aug3D,
    FEATS_NUM,
    LungDataset,
    evaluate,
    read_samples_for_task,
    resolve_device,
    seed_all,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument(
        "--mode",
        choices=["mm", "img_only", "txt_only"],
        default="mm",
    )

    parser.add_argument("--img_backbone", default="r2plus1d_18")
    parser.add_argument("--in_channels", type=int, default=1)

    parser.add_argument("--d_img", type=int, default=256)
    parser.add_argument("--d_txt", type=int, default=256)
    parser.add_argument("--drop", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument(
        "--aug",
        choices=["none", "light"],
        default="light",
    )

    return parser


def make_split(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy().reset_index(drop=True)

    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n_val = max(1, int(round(len(indices) * float(val_ratio))))

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df.iloc[val_idx].copy().reset_index(drop=True)

    print(f"[SPLIT] train={len(df_train)}  val={len(df_val)}")

    return df_train, df_val


def build_density_vocab(df: pd.DataFrame) -> Dict[str, int]:
    if "density" not in df.columns:
        return {"UNK": 0}

    values = sorted(set(df["density"].astype(str)))
    values = ["UNK"] + [x for x in values if x != "UNK"]

    return {name: i for i, name in enumerate(values)}


def build_feature_stats(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:

    df_train = df_train.copy()
    df_val = df_val.copy()

    tab_columns = [c for c in FEATS_NUM if c in df_train.columns]

    if len(tab_columns) == 0:
        ignore = {
            "path",
            "label",
            "label_idx",
            "subject",
            "nod_id",
            "density",
        }
        candidates = []
        for col in df_train.columns:
            if col in ignore:
                continue
            if pd.api.types.is_numeric_dtype(df_train[col]):
                candidates.append(col)
        tab_columns = candidates

    stats = {
        "mean": {},
        "std": {},
        "tab_columns": list(tab_columns),
    }

    for col in tab_columns:
        values = pd.to_numeric(df_train[col], errors="coerce").astype(float)
        mean = values.mean()
        std = values.std()

        if np.isnan(mean):
            mean = 0.0
        if np.isnan(std) or std < 1e-6:
            std = 1.0

        stats["mean"][col] = float(mean)
        stats["std"][col] = float(std)

    if "long_diameter" in df_train.columns:
        ld = pd.to_numeric(df_train["long_diameter"], errors="coerce").astype(float)
        ld_mean = ld.mean()
        ld_std = ld.std()

        stats["mean"]["long_diameter"] = float(
            0.0 if np.isnan(ld_mean) else ld_mean
        )
        stats["std"]["long_diameter"] = float(
            1.0 if np.isnan(ld_std) or ld_std < 1e-6 else ld_std
        )
    else:
        stats["mean"]["long_diameter"] = 0.0
        stats["std"]["long_diameter"] = 1.0

    for frame in (df_train, df_val):
        for col in tab_columns:
            frame[col] = (
                pd.to_numeric(frame[col], errors="coerce")
                .astype(float)
                .fillna(stats["mean"][col])
            )

    return df_train, df_val, stats


def build_model(
    args,
    den_vocab: Dict[str, int],
    tab_columns,
    device: torch.device,
):

    out_classes = 3

    if args.mode == "txt_only":
        model = TabOnlyModel(
            d_txt=args.d_txt,
            drop=args.drop,
            den_num_classes=len(den_vocab),
            out_classes=out_classes,
            num_names=tab_columns,
        )

    elif args.mode == "img_only":
        model = ImgOnlyModel(
            img_backbone=args.img_backbone,
            in_ch=args.in_channels,
            d_img=args.d_img,
            drop=args.drop,
            out_classes=out_classes,
        )

    else:
        model = UnifiedModel(
            mode="mm",
            task_out_classes=out_classes,
            img_backbone=args.img_backbone,
            in_ch=args.in_channels,
            d_img=args.d_img,
            d_txt=args.d_txt,
            den_num_classes=len(den_vocab),
            drop=args.drop,
            num_names=tab_columns,
        )

    return model.to(device)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
) -> float:

    model.train()

    total_loss = 0.0
    total_n = 0

    for x_img, x_tab, y, ld_z in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_num = x_tab[0].to(device, non_blocking=True)
        x_den = x_tab[1].to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ld_z = ld_z.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(
            x_img,
            (x_num, x_den),
            ld_z,
            x_den,
        )

        loss = F.cross_entropy(logits, y.long())

        if not torch.isfinite(loss):
            continue

        loss.backward()
        optimizer.step()

        batch_size = int(x_img.size(0))
        total_loss += float(loss.item()) * batch_size
        total_n += batch_size

    return total_loss / max(total_n, 1)


def save_checkpoint(
    path: str,
    model,
    epoch: int,
    args,
    stats: Dict,
    den_vocab: Dict[str, int],
    best_f1: float,
) -> None:

    ckpt = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "best_macro_f1": float(best_f1),
        "args": {
            "mode": args.mode,
            "img_backbone": args.img_backbone,
            "in_channels": int(args.in_channels),
            "d_img": int(args.d_img),
            "d_txt": int(args.d_txt),
        },
        "stats": {
            "mean": dict(stats["mean"]),
            "std": dict(stats["std"]),
            "tab_columns": list(stats["tab_columns"]),
        },
        "den_vocab": dict(den_vocab),
    }

    torch.save(ckpt, path)


def main() -> None:
    args = build_argparser().parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.gpu)

    df = read_samples_for_task(args)

    if "nod_id" in df.columns:
        df = df.drop_duplicates("nod_id").reset_index(drop=True)

    if "label_idx" not in df.columns:
        raise RuntimeError("samples_csv must contain label_idx.")

    if df["label_idx"].isna().any():
        raise RuntimeError("label_idx contains missing values.")

    df_train, df_val = make_split(
        df=df,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    den_vocab = build_density_vocab(df)

    for frame in (df_train, df_val):
        if "density" not in frame.columns:
            frame["density"] = "UNK"
        frame["density"] = frame["density"].astype(str)
        frame["density"] = frame["density"].map(
            lambda x: x if x in den_vocab else "UNK"
        )

    df_train, df_val, stats = build_feature_stats(
        df_train=df_train,
        df_val=df_val,
    )

    need_image = args.mode != "txt_only"

    train_set = LungDataset(
        df_train,
        stats=stats,
        den_vocab=den_vocab,
        in_ch=args.in_channels,
        task="tri",
        augment=Aug3D(args.aug),
        train=True,
        load_image=need_image,
    )

    val_set = LungDataset(
        df_val,
        stats=stats,
        den_vocab=den_vocab,
        in_ch=args.in_channels,
        task="tri",
        augment=None,
        train=False,
        load_image=need_image,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"[DATA] train={len(train_set)}  val={len(val_set)}")
    print(f"[MODE] {args.mode}")

    model = build_model(
        args=args,
        den_vocab=den_vocab,
        tab_columns=stats["tab_columns"],
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    best_epoch = -1

    best_path = osp.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_acc, val_f1 = evaluate(
            model,
            val_loader,
            task="tri",
            device=device,
            name="VAL",
            verbose=True,
        )

        print(
            f"[E{epoch:03d}/{args.epochs}] "
            f"loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_epoch = int(epoch)

            save_checkpoint(
                path=best_path,
                model=model,
                epoch=epoch,
                args=args,
                stats=stats,
                den_vocab=den_vocab,
                best_f1=best_f1,
            )

            print(f"[SAVE] {best_path}")

    meta = {
        "mode": args.mode,
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_f1),
    }

    with open(osp.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] best_macro_f1={best_f1:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
