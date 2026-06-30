#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PIPELINE_VERSION = "public-check-v1"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)

    for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    return pd.read_csv(path)


def resolve_path(path: str, base_dir: str | None = None) -> str:
    path = str(path).strip()

    if osp.isabs(path) or not base_dir:
        return path

    return osp.join(base_dir, path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "") for c in out.columns]
    return out


def inspect_patch(path: str) -> Dict:
    arr = np.load(path, mmap_mode="r")

    if arr.ndim == 3:
        shape = (1,) + tuple(arr.shape)
    elif arr.ndim == 4:
        shape = tuple(arr.shape)
    else:
        raise RuntimeError(f"unsupported shape: {arr.shape}")

    arr_small = np.asarray(arr)
    finite = np.isfinite(arr_small)

    if finite.any():
        mean = float(np.nanmean(arr_small))
        std = float(np.nanstd(arr_small))
        vmin = float(np.nanmin(arr_small))
        vmax = float(np.nanmax(arr_small))
    else:
        mean = std = vmin = vmax = float("nan")

    return {
        "shape": list(shape),
        "channels": int(shape[0]),
        "depth": int(shape[1]),
        "height": int(shape[2]),
        "width": int(shape[3]),
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "finite_fraction": float(finite.mean()),
    }


def summarize_counts(df: pd.DataFrame, col: str) -> Dict:
    if col not in df.columns:
        return {}

    counts = df[col].astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def check_samples(args) -> Dict:
    df = normalize_columns(read_table(args.samples_csv))

    required = ["path", "label_idx"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"samples.csv missing required columns: {missing}")

    ensure_dir(args.out_dir)

    rows = []
    failures = []

    max_items = len(df) if args.max_items <= 0 else min(int(args.max_items), len(df))

    for idx, row in df.head(max_items).iterrows():
        path = resolve_path(row["path"], args.data_root)

        if not osp.isfile(path):
            failures.append(
                {
                    "row": int(idx),
                    "path": path,
                    "reason": "missing file",
                }
            )
            continue

        try:
            info = inspect_patch(path)
            rows.append(
                {
                    "row": int(idx),
                    "path": path,
                    "label_idx": row.get("label_idx", np.nan),
                    "subject": row.get("subject", ""),
                    "nod_id": row.get("nod_id", ""),
                    **info,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "row": int(idx),
                    "path": path,
                    "reason": str(exc),
                }
            )

    detail = pd.DataFrame(rows)
    fail_df = pd.DataFrame(failures)

    detail_path = osp.join(args.out_dir, "patch_check.csv")
    failure_path = osp.join(args.out_dir, "patch_failures.csv")
    report_path = osp.join(args.out_dir, "check_report.json")

    detail.to_csv(detail_path, index=False)

    if not fail_df.empty:
        fail_df.to_csv(failure_path
