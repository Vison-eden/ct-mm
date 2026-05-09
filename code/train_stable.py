#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training script for tri-class multimodal lung nodule classification.

This public version supports:
  - image-only, semantic-only, and multimodal training
  - tri-class classification: AAH/AIS, MIA, IAC
  - optional gated probability fusion
  - optional contrastive alignment loss between CT and semantic branches

The input samples.csv should use de-identified paths and standardized English
column names. See utils.py for required columns.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from model import ImgOnlyModel, TabOnlyModel, UnifiedModel
from utils import (
    Aug3D,
    FEATS_ALL,
    FEATS_NUM,
    LungDataset,
    discover_cat_levels,
    dump_val_predictions_from_best,
    evaluate,
    expand_structured_features,
    norm_id_core,
    read_samples_for_task,
    resolve_device,
    seed_all,
)


def info_nce(
    z_img: torch.Tensor | None,
    z_tab: torch.Tensor | None,
    temperature: float = 0.07,
) -> torch.Tensor:
    if z_img is None or z_tab is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor(0.0, device=device)

    z_img = F.normalize(z_img.float(), dim=1)
    z_tab = F.normalize(z_tab.float(), dim=1)

    logits = z_img @ z_tab.t()
    logits = logits / max(float(temperature), 1e-6)

    target = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, target)
    loss_t2i = F.cross_entropy(logits.t(), target)

    return 0.5 * (loss_i2t + loss_t2i)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_ids", default="")

    parser.add_argument(
        "--mode",
        choices=["mm", "img_only", "txt_only"],
        default="mm",
    )
    parser.add_argument(
        "--img_backbone",
        choices=["r3d_18", "mc3_18", "r2plus1d_18"],
        default="r2plus1d_18",
    )
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--img_trans_layers", type=int, default=0)
    parser.add_argument("--d_img", type=int, default=256)
    parser.add_argument("--d_txt", type=int, default=256)
    parser.add_argument("--drop", type=float, default=0.2)

    parser.add_argument("--use_gate", action="store_true")
    parser.add_argument(
        "--mix_rule",
        choices=["prob_anchor", "logits"],
        default="prob_anchor",
    )
    parser.add_argument("--alpha_cap", type=float, default=0.30)

    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument(
        "--aug",
        choices=["none", "light", "strong"],
        default="light",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_class_weight", action="store_true")

    parser.add_argument("--lambda_align", type=float, default=0.05)
    parser.add_argument("--align_temperature", type=float, default=0.07)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--aux_img_w", type=float, default=0.0)
    parser.add_argument("--aux_txt_w", type=float, default=0.0)

    return parser


def make_split(
    df: pd.DataFrame,
    val_ids: str,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["__ID__"] = df["subject"].astype(str).map(norm_id_core)

    if val_ids and osp.isfile(val_ids):
        val_df = pd.read_csv(val_ids)
        if "nod_id" not in val_df.columns:
            raise RuntimeError("--val_ids must contain a 'nod_id' column.")

        val_core_set = set(val_df["nod_id"].astype(str).map(norm_id_core))

        df_val = df[df["__ID__"].isin(val_core_set)].copy()
        df_train = df[~df["__ID__"].isin(val_core_set)].copy()

        if len(df_val) == 0:
            raise RuntimeError("No validation samples were matched from --val_ids.")

        print(
            f"[SPLIT] train={len(df_train)} rows, "
            f"val={len(df_val)} rows, "
            f"val_ids={len(val_core_set)}"
        )

        return df_train.reset_index(drop=True), df_val.reset_index(drop=True)

    y = df["label_idx"].astype(int).values

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(val_ratio),
        random_state=int(seed),
    )
    idx_train, idx_val = next(splitter.split(np.zeros(len(df)), y))

    df_train = df.iloc[idx_train].copy()
    df_val = df.iloc[idx_val].copy()

    print(f"[SPLIT] train={len(df_train)} rows, val={len(df_val)} rows")

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def build_density_vocab(df: pd.DataFrame) -> Dict[str, int]:
    base_order = ["UNK", "GGO", "PartSolid", "Solid"]
    observed = sorted(set(df["density"].astype(str)))

    order = [x for x in base_order if x in observed or x == "UNK"]
    order += [x for x in observed if x not in order]

    return {name: i for i, name in enumerate(order)}


def build_feature_stats(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    cat_levels = discover_cat_levels(df_train)

    df_train, tab_columns = expand_structured_features(df_train, cat_levels)
    df_val, _ = expand_structured_features(df_val, cat_levels)

    stats = {
        "mean": {},
        "std": {},
        "tab_columns": list(tab_columns),
        "cat_levels": cat_levels,
    }

    for col in tab_columns:
        values = pd.to_numeric(df_train[col], errors="coerce").astype(float)
        mean = values.mean()
        std = values.std()

        stats["mean"][col] = float(0.0 if np.isnan(mean) else mean)
        stats["std"][col] = float(1.0 if np.isnan(std) or std < 1e-6 else std)

    ld = pd.to_numeric(df_train.get("long_diameter"), errors="coerce").astype(float)
    ld_mean = ld.mean()
    ld_std = ld.std()

    stats["mean"]["long_diameter"] = float(0.0 if np.isnan(ld_mean) else ld_mean)
    stats["std"]["long_diameter"] = float(
        1.0 if np.isnan(ld_std) or ld_std < 1e-6 else ld_std
    )

    for frame in (df_train, df_val):
        for col in tab_columns:
            frame[col] = (
                pd.to_numeric(frame[col], errors="coerce")
                .astype(float)
                .fillna(stats["mean"][col])
            )

    return df_train, df_val, stats


def infer_in_channels(df_train: pd.DataFrame, user_value: int | None) -> int:
    if user_value is not None:
        return int(user_value)

    arr = np.load(df_train.iloc[0]["path"])

    if arr.ndim == 4:
        return int(arr.shape[0])

    if arr.ndim == 3:
        return 1

    raise RuntimeError(f"Unexpected patch shape: {arr.shape}")


def build_model(
    args,
    in_channels: int,
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
            in_ch=in_channels,
            d_img=args.d_img,
            drop=args.drop,
            img_trans_layers=args.img_trans_layers,
            out_classes=out_classes,
        )

    else:
        model = UnifiedModel(
            mode="mm",
            task_out_classes=out_classes,
            img_backbone=args.img_backbone,
            in_ch=in_channels,
            img_trans_layers=args.img_trans_layers,
            d_img=args.d_img,
            d_txt=args.d_txt,
            den_num_classes=len(den_vocab),
            drop=args.drop,
            use_gate=args.use_gate,
            mix_rule=args.mix_rule,
            alpha_cap=args.alpha_cap,
            num_names=tab_columns,
        )

    return model.to(device)


def make_scheduler(optimizer, args):
    warmup = max(0, int(args.warmup_epochs))
    total = max(1, int(args.epochs) - warmup)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(max(1, warmup))

        progress = float(epoch - warmup) / float(total)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return max(float(args.min_lr) / float(args.lr), cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_class_weight(df_train: pd.DataFrame, device: torch.device, enabled: bool):
    if not enabled:
        return None

    counts = (
        df_train["label_idx"]
        .value_counts()
        .reindex([0, 1, 2])
        .fillna(0)
        .astype(float)
        .values
    )

    inv = 1.0 / np.clip(counts, 1.0, None)
    inv = inv * (len(inv) / inv.sum())

    weight = torch.tensor(inv, dtype=torch.float32, device=device)
    print(f"[CLASS_WEIGHT] {inv}")

    return weight


def compute_main_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight,
    label_smoothing: float,
    use_gate: bool,
    mix_rule: str,
    extra: Dict,
) -> torch.Tensor:
    if use_gate and mix_rule == "prob_anchor" and extra.get("p_gate") is not None:
        return F.nll_loss(logits, target.long())

    return F.cross_entropy(
        logits,
        target.long(),
        weight=class_weight,
        label_smoothing=float(label_smoothing),
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    args,
    class_weight,
    epoch: int,
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

        use_amp = bool(args.amp and device.type == "cuda")

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, extra = model(x_img, (x_num, x_den), ld_z, x_den)

            loss = compute_main_loss(
                logits=logits,
                target=y,
                class_weight=class_weight,
                label_smoothing=args.label_smoothing,
                use_gate=args.use_gate,
                mix_rule=args.mix_rule,
                extra=extra,
            )

            if args.mode == "mm":
                if args.aux_img_w > 0 and "logits_img" in extra:
                    loss = loss + float(args.aux_img_w) * F.cross_entropy(
                        extra["logits_img"],
                        y.long(),
                        weight=class_weight,
                        label_smoothing=float(args.label_smoothing),
                    )

                if args.aux_txt_w > 0 and "logits_txt" in extra:
                    loss = loss + float(args.aux_txt_w) * F.cross_entropy(
                        extra["logits_txt"],
                        y.long(),
                        weight=class_weight,
                        label_smoothing=float(args.label_smoothing),
                    )

                if args.lambda_align > 0 and extra.get("pi") is not None:
                    if args.warmup_epochs <= 0:
                        align_weight = float(args.lambda_align)
                    else:
                        align_weight = float(args.lambda_align) * min(
                            1.0,
                            float(epoch) / float(args.warmup_epochs),
                        )

                    loss = loss + align_weight * info_nce(
                        extra.get("pi"),
                        extra.get("pt"),
                        temperature=args.align_temperature,
                    )

        if not torch.isfinite(loss):
            print("[WARN] non-finite loss detected; batch skipped.")
            continue

        if scaler is not None:
            scaler.scale(loss).backward()

            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

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
        "args": vars(args),
        "stats": {
            "mean": {k: float(v) for k, v in stats["mean"].items()},
            "std": {k: float(v) for k, v in stats["std"].items()},
            "cat_levels": stats.get("cat_levels", {}),
            "tab_columns": list(stats.get("tab_columns", [])),
        },
        "den_vocab": dict(den_vocab),
        "feat_num": list(stats.get("tab_columns", FEATS_NUM)),
        "feat_all": FEATS_ALL,
    }

    torch.save(ckpt, path)


def build_model_ctor_for_dump(args, in_channels, den_vocab, tab_columns):
    if args.mode == "txt_only":
        return lambda **kwargs: TabOnlyModel(
            d_txt=args.d_txt,
            drop=args.drop,
            den_num_classes=len(den_vocab),
            out_classes=3,
            num_names=tab_columns,
        )

    if args.mode == "img_only":
        return lambda **kwargs: ImgOnlyModel(
            img_backbone=args.img_backbone,
            in_ch=in_channels,
            d_img=args.d_img,
            drop=args.drop,
            img_trans_layers=args.img_trans_layers,
            out_classes=3,
        )

    return lambda **kwargs: UnifiedModel(
        mode="mm",
        task_out_classes=3,
        img_backbone=args.img_backbone,
        in_ch=in_channels,
        img_trans_layers=args.img_trans_layers,
        d_img=args.d_img,
        d_txt=args.d_txt,
        den_num_classes=len(den_vocab),
        drop=args.drop,
        use_gate=args.use_gate,
        mix_rule=args.mix_rule,
        alpha_cap=args.alpha_cap,
        num_names=tab_columns,
    )


def main() -> None:
    args = build_argparser().parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.gpu)

    df = read_samples_for_task(args)
    df = df.drop_duplicates("nod_id").reset_index(drop=True)

    if df["label_idx"].isna().any():
        raise RuntimeError("label_idx contains missing values.")

    df_train, df_val = make_split(
        df=df,
        val_ids=args.val_ids,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    den_vocab = build_density_vocab(df)
    df_train, df_val, stats = build_feature_stats(df_train, df_val)

    for frame in (df_train, df_val):
        frame["density"] = frame["density"].astype(str).map(
            lambda x: x if x in den_vocab else "UNK"
        )

    need_image = args.mode != "txt_only"

    if need_image:
        in_channels = infer_in_channels(df_train, args.in_channels)
    else:
        in_channels = 1

    train_set = LungDataset(
        df_train,
        stats=stats,
        den_vocab=den_vocab,
        in_ch=in_channels,
        task="tri",
        augment=Aug3D(args.aug),
        train=True,
        load_image=need_image,
    )
    val_set = LungDataset(
        df_val,
        stats=stats,
        den_vocab=den_vocab,
        in_ch=in_channels,
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
        drop_last=True,
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
    print(f"[DENSITY] {den_vocab}")
    print(f"[FEATURES] {stats['tab_columns']}")

    model = build_model(
        args=args,
        in_channels=in_channels,
        den_vocab=den_vocab,
        tab_columns=stats["tab_columns"],
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args)

    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    class_weight = make_class_weight(
        df_train=df_train,
        device=device,
        enabled=args.use_class_weight,
    )

    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    best_path = osp.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            class_weight=class_weight,
            epoch=epoch,
        )

        val_acc, val_f1 = evaluate(
            model,
            val_loader,
            task="tri",
            device=device,
            name="VAL",
            verbose=True,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[E{epoch:03d}/{args.epochs}] "
            f"lr={lr_now:.2e}  "
            f"loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1 + 1e-6:
            best_f1 = float(val_f1)
            best_epoch = int(epoch)
            bad_epochs = 0

            save_checkpoint(
                path=best_path,
                model=model,
                epoch=epoch,
                args=args,
                stats=stats,
                den_vocab=den_vocab,
                best_f1=best_f1,
            )

            print(f"[SAVE] best checkpoint updated: {best_path}")

        else:
            bad_epochs += 1

            if bad_epochs >= args.patience:
                print(
                    f"[EARLY_STOP] best_macro_f1={best_f1:.4f} "
                    f"at epoch {best_epoch}"
                )
                break

    model_ctor = build_model_ctor_for_dump(
        args=args,
        in_channels=in_channels,
        den_vocab=den_vocab,
        tab_columns=stats["tab_columns"],
    )

    dump_val_predictions_from_best(
        out_dir=args.out_dir,
        model_ctor=model_ctor,
        model_kwargs={},
        val_loader=val_loader,
        task="tri",
        device=device,
        out_name="preds_val.csv",
    )

    meta = {
        "args": vars(args),
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_f1),
        "features": list(stats["tab_columns"]),
        "density_vocab": dict(den_vocab),
    }

    with open(osp.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[DONE] best_macro_f1={best_f1:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()