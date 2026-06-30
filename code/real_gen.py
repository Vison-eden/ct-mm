#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


GENERIC_FEATURES = [f"feat_{i:02d}" for i in range(1, 10)]
PIPELINE_VERSION = "public-samples-v1"


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


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


def clean_id(x) -> str:
    return str(x).strip().replace(" ", "").replace("\t", "")


def resolve_path(path: str, base_dir: str | None = None) -> str:
    path = str(path).strip()

    if osp.isabs(path) or not base_dir:
        return path

    return osp.join(base_dir, path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "") for c in out.columns]
    return out


def canonical_label(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, np.integer)):
        if int(x) in {0, 1, 2}:
            return int(x)

    if isinstance(x, float) and np.isfinite(x):
        if int(x) in {0, 1, 2}:
            return int(x)

    s = str(x).strip()
    s2 = s.lower().replace(" ", "").replace("-", "_")

    if s in {"AAH/AIS", "MIA", "IAC"}:
        return {"AAH/AIS": 0, "MIA": 1, "IAC": 2}[s]

    aliases = {
        "aah": 0,
        "ais": 0,
        "aah/ais": 0,
        "aah_ais": 0,
        "preinvasive": 0,
        "pre_invasive": 0,
        "mia": 1,
        "iac": 2,
    }

    return aliases.get(s2, np.nan)


def canonical_density(x) -> str:
    if pd.isna(x):
        return "UNK"

    s = str(x).strip().lower().replace(" ", "").replace("_", "-")

    if s in {"ggo", "ggn", "groundglass", "ground-glass"}:
        return "GGO"

    if s in {"partsolid", "part-solid", "mixed"}:
        return "PartSolid"

    if s in {"solid"}:
        return "Solid"

    return "UNK"


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def inspect_npy(path: str) -> Dict:
    arr = np.load(path, mmap_mode="r")

    if arr.ndim == 3:
        shape = (1,) + tuple(arr.shape)
    elif arr.ndim == 4:
        shape = tuple(arr.shape)
    else:
        raise RuntimeError(f"Unsupported patch shape {arr.shape}: {path}")

    return {
        "shape": list(shape),
        "channels": int(shape[0]),
        "depth": int(shape[1]),
        "height": int(shape[2]),
        "width": int(shape[3]),
    }


def build_generic_features(df: pd.DataFrame, n_features: int) -> pd.DataFrame:
    out = df.copy()
    feature_names = [f"feat_{i:02d}" for i in range(1, int(n_features) + 1)]

    existing = [c for c in feature_names if c in out.columns]
    if existing:
        for c in feature_names:
            if c not in out.columns:
                out[c] = np.nan
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    ignore = {
        "path",
        "patch_path",
        "file",
        "filename",
        "label",
        "label_idx",
        "category",
        "tri_label",
        "subject",
        "patient",
        "patient_id",
        "nod_id",
        "nodule_id",
        "density",
        "center_z",
        "center_y",
        "center_x",
        "spacing_z",
        "spacing_y",
        "spacing_x",
        "size_mm",
        "size_pix",
        "channels",
    }

    candidates = []
    for col in out.columns:
        if col in ignore:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        if values.notna().sum() > 0:
            candidates.append(col)

    for i, feat in enumerate(feature_names):
        if i < len(candidates):
            out[feat] = pd.to_numeric(out[candidates[i]], errors="coerce")
        else:
            out[feat] = np.nan

    return out


def maybe_copy_patch(src: str, dst_dir: str, nod_id: str, copy_files: bool) -> str:
    if not copy_files:
        return osp.abspath(src)

    ensure_dir(dst_dir)

    ext = ".npy"
    dst = osp.abspath(osp.join(dst_dir, f"{nod_id}{ext}"))

    if osp.abspath(src) != dst:
        shutil.copy2(src, dst)

    return dst


def build_samples(args) -> pd.DataFrame:
    df = normalize_columns(read_table(args.input_table))

    path_col = args.path_col or pick_first_existing(
        df,
        ["path", "patch_path", "npy_path", "file", "filename"],
    )
    if path_col is None:
        raise RuntimeError("Input table must contain a patch path column.")

    label_col = args.label_col or pick_first_existing(
        df,
        ["label_idx", "label", "category", "tri_label"],
    )
    if label_col is None:
        raise RuntimeError("Input table must contain a label column.")

    subject_col = args.subject_col or pick_first_existing(
        df,
        ["subject", "patient", "patient_id"],
    )

    nod_col = args.nod_col or pick_first_existing(
        df,
        ["nod_id", "nodule_id", "case_id", "id"],
    )

    density_col = args.density_col or pick_first_existing(df, ["density"])

    df = build_generic_features(df, n_features=args.n_features)

    rows = []
    failures = []

    patch_dir = osp.join(args.out_dir, "patches")
    ensure_dir(args.out_dir)

    for idx, row in df.iterrows():
        raw_path = resolve_path(row[path_col], args.data_root)

        if not osp.isfile(raw_path):
            failures.append(
                {
                    "row": int(idx),
                    "reason": "missing patch file",
                    "path": raw_path,
                }
            )
            continue

        try:
            nod_id = clean_id(row[nod_col]) if nod_col else osp.splitext(osp.basename(raw_path))[0]
            subject = clean_id(row[subject_col]) if subject_col else nod_id.split("_")[0]

            label_idx = canonical_label(row[label_col])
            if pd.isna(label_idx):
                raise RuntimeError("unmapped label")

            patch_info = inspect_npy(raw_path)
            path_out = maybe_copy_patch(
                src=raw_path,
                dst_dir=patch_dir,
                nod_id=nod_id,
                copy_files=bool(args.copy_patches),
            )

            out_row = {
                "subject": subject,
                "nod_id": nod_id,
                "path": path_out,
                "label_idx": int(label_idx),
                "density": canonical_density(row[density_col]) if density_col else "UNK",
                "channels": int(patch_info["channels"]),
            }

            for feat in [f"feat_{i:02d}" for i in range(1, int(args.n_features) + 1)]:
                out_row[feat] = row.get(feat, np.nan)

            rows.append(out_row)

        except Exception as exc:
            failures.append(
                {
                    "row": int(idx),
                    "reason": str(exc),
                    "path": raw_path,
                }
            )

    samples = pd.DataFrame(rows)

    base_cols = ["subject", "nod_id", "path", "label_idx", "density", "channels"]
    feat_cols = [f"feat_{i:02d}" for i in range(1, int(args.n_features) + 1)]
    ordered_cols = base_cols + feat_cols

    if samples.empty:
        samples = pd.DataFrame(columns=ordered_cols)
    else:
        for col in ordered_cols:
            if col not in samples.columns:
                samples[col] = np.nan
        samples = samples[ordered_cols]

    samples_path = osp.join(args.out_dir, "samples.csv")
    samples.to_csv(samples_path, index=False)

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "input_table": osp.abspath(args.input_table),
        "output_samples": osp.abspath(samples_path),
        "n_input_rows": int(len(df)),
        "n_success": int(len(samples)),
        "n_failed": int(len(failures)),
        "feature_columns": feat_cols,
    }

    with open(osp.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if failures:
        pd.DataFrame(failures).to_csv(osp.join(args.out_dir, "failures.csv"), index=False)

    print(f"[DONE] samples={len(samples)} failures={len(failures)} out={args.out_dir}")
    print(f"[WRITE] {samples_path}")

    return samples


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_table", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--data_root", default="")
    parser.add_argument("--path_col", default="")
    parser.add_argument("--label_col", default="")
    parser.add_argument("--subject_col", default="")
    parser.add_argument("--nod_col", default="")
    parser.add_argument("--density_col", default="")

    parser.add_argument("--n_features", type=int, default=9)
    parser.add_argument("--copy_patches", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    seed_all(args.seed)
    build_samples(args)


if __name__ == "__main__":
    main()
