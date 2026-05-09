#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 3D CT patches from 1.0-mm isotropic CT volumes.

Expected ct_index.csv columns:
    subject
    nod_id
    ct_iso
    center_z
    center_y
    center_x

Optional columns:
    mask_iso
    label
    density
    structured semantic features

The output samples.csv is compatible with the public training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk


WINDOW = (-1000.0, 400.0)
PAD_HU = -1000.0
PAD_NORM = 0.0
PIPELINE_VERSION = "public-patch3d-v1"

PUBLIC_SEMANTIC_COLS = [
    "density",
    "long_diameter",
    "solid_ratio",
    "spiculation",
    "lobulation",
    "smooth_sharp",
    "bronchus_sign",
    "pleural_indentation",
    "cord_sign",
    "irregular",
    "round_like",
    "vascular_convergence",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def parse_sizes_mm(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def window_normalize(arr_hu: np.ndarray) -> np.ndarray:
    lo, hi = WINDOW
    arr = np.clip(arr_hu, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return arr.astype(np.float32)


def read_itk_array(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing_xyz = img.GetSpacing()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return arr, spacing_zyx


def center_from_mask(mask_path: str) -> Tuple[int, int, int]:
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        raise RuntimeError(f"Empty mask: {mask_path}")

    center = coords.mean(axis=0)
    return tuple(int(round(float(v))) for v in center)


def get_center_zyx(row: pd.Series) -> Tuple[int, int, int]:
    has_center = all(c in row.index and pd.notna(row[c]) for c in ["center_z", "center_y", "center_x"])

    if has_center:
        return (
            int(round(float(row["center_z"]))),
            int(round(float(row["center_y"]))),
            int(round(float(row["center_x"]))),
        )

    if "mask_iso" in row.index and pd.notna(row["mask_iso"]):
        return center_from_mask(str(row["mask_iso"]))

    raise RuntimeError("Missing center_z/center_y/center_x and no usable mask_iso.")


def crop_centered_cube(
    arr: np.ndarray,
    center_zyx: Tuple[int, int, int],
    size: int,
    pad_value: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    zc, yc, xc = [int(v) for v in center_zyx]
    half = size // 2

    src_z0 = zc - half
    src_y0 = yc - half
    src_x0 = xc - half
    src_z1 = src_z0 + size
    src_y1 = src_y0 + size
    src_x1 = src_x0 + size

    in_z0 = max(0, src_z0)
    in_y0 = max(0, src_y0)
    in_x0 = max(0, src_x0)
    in_z1 = min(arr.shape[0], src_z1)
    in_y1 = min(arr.shape[1], src_y1)
    in_x1 = min(arr.shape[2], src_x1)

    out_z0 = in_z0 - src_z0
    out_y0 = in_y0 - src_y0
    out_x0 = in_x0 - src_x0

    out_z1 = out_z0 + max(0, in_z1 - in_z0)
    out_y1 = out_y0 + max(0, in_y1 - in_y0)
    out_x1 = out_x0 + max(0, in_x1 - in_x0)

    patch = np.full((size, size, size), pad_value, dtype=arr.dtype)

    if in_z1 > in_z0 and in_y1 > in_y0 and in_x1 > in_x0:
        patch[out_z0:out_z1, out_y0:out_y1, out_x0:out_x1] = arr[
            in_z0:in_z1,
            in_y0:in_y1,
            in_x0:in_x1,
        ]

    meta = {
        "src_z0": int(src_z0),
        "src_y0": int(src_y0),
        "src_x0": int(src_x0),
        "src_z1": int(src_z1),
        "src_y1": int(src_y1),
        "src_x1": int(src_x1),
        "in_z0": int(in_z0),
        "in_y0": int(in_y0),
        "in_x0": int(in_x0),
        "in_z1": int(in_z1),
        "in_y1": int(in_y1),
        "in_x1": int(in_x1),
        "out_z0": int(out_z0),
        "out_y0": int(out_y0),
        "out_x0": int(out_x0),
        "out_z1": int(out_z1),
        "out_y1": int(out_y1),
        "out_x1": int(out_x1),
        "center_z": int(zc),
        "center_y": int(yc),
        "center_x": int(xc),
    }

    return patch, meta


def patch_guard(patch01: np.ndarray, min_std: float = 1e-5, max_black_frac: float = 0.995) -> str:
    if not np.all(np.isfinite(patch01)):
        return "non-finite values"

    std = float(np.std(patch01))
    if std < min_std:
        return f"near-constant patch: std={std:.2e}"

    black_frac = float((patch01 <= 1e-6).mean())
    if black_frac >= max_black_frac:
        return f"near-empty patch: black_frac={black_frac:.4f}"

    return ""


def save_qc_figure(
    ct_volume01: np.ndarray,
    patch01: np.ndarray,
    crop_meta: Dict[str, int],
    out_png: str,
    title: str,
) -> None:
    ensure_dir(osp.dirname(out_png))

    z = int(np.clip(crop_meta["center_z"], 0, ct_volume01.shape[0] - 1))
    y0 = max(0, crop_meta["src_y0"])
    x0 = max(0, crop_meta["src_x0"])
    y1 = min(ct_volume01.shape[1], crop_meta["src_y1"])
    x1 = min(ct_volume01.shape[2], crop_meta["src_x1"])

    patch_z = patch01.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=160)

    axes[0].imshow(ct_volume01[z], cmap="gray", vmin=0.0, vmax=1.0)
    rect = patches.Rectangle(
        (x0, y0),
        max(1, x1 - x0),
        max(1, y1 - y0),
        fill=False,
        linewidth=1.2,
        edgecolor="red",
    )
    axes[0].add_patch(rect)
    axes[0].set_title("Full CT slice")
    axes[0].axis("off")

    axes[1].imshow(patch01[patch_z], cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Patch center slice")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def signature_for_patch(size_mm: float, size_pix: int) -> Dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "size_mm": float(size_mm),
        "size_pix": int(size_pix),
        "window": list(WINDOW),
        "pad_hu": float(PAD_HU),
        "pad_norm": float(PAD_NORM),
    }


def hash_dict(obj: Dict) -> str:
    text = json.dumps(obj, sort_keys=True)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, obj: Dict) -> None:
    ensure_dir(osp.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def should_skip(meta_path: str, npy_path: str, signature: Dict, cover: bool) -> bool:
    if cover:
        return False

    if not osp.isfile(npy_path) or osp.getsize(npy_path) <= 0:
        return False

    meta = read_json(meta_path)
    if not meta:
        return False

    return meta.get("signature_hash") == hash_dict(signature)


def merge_metadata(
    ct_index: pd.DataFrame,
    metadata_csv: str | None,
) -> pd.DataFrame:
    df = ct_index.copy()

    if not metadata_csv:
        return df

    meta = read_table(metadata_csv)

    if "nod_id" not in meta.columns:
        raise RuntimeError("metadata_csv must contain a 'nod_id' column.")

    meta["nod_id"] = meta["nod_id"].map(clean_id)
    df["nod_id"] = df["nod_id"].map(clean_id)

    extra_cols = [c for c in meta.columns if c != "nod_id"]
    df = df.merge(meta[["nod_id"] + extra_cols], on="nod_id", how="left")

    return df


def validate_ct_index(df: pd.DataFrame) -> None:
    required = ["subject", "nod_id", "ct_iso"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"ct_index.csv missing required columns: {missing}")

    has_center = all(c in df.columns for c in ["center_z", "center_y", "center_x"])
    has_mask = "mask_iso" in df.columns

    if not has_center and not has_mask:
        raise RuntimeError("ct_index.csv must contain center_z/y/x or mask_iso.")


def process_size(
    df: pd.DataFrame,
    out_root: str,
    size_mm: float,
    export_png: bool,
    cover: bool,
) -> None:
    out_dir = osp.join(out_root, f"3d-{int(size_mm)}mm-wide")
    meta_dir = osp.join(out_dir, "_meta")
    qc_dir = osp.join(out_dir, "img")

    ensure_dir(out_dir)
    ensure_dir(meta_dir)

    if export_png:
        ensure_dir(qc_dir)

    rows = []
    failures = []

    for _, row in df.iterrows():
        subject = clean_id(row["subject"])
        nod_id = clean_id(row["nod_id"])
        ct_path = str(row["ct_iso"])

        if not osp.isfile(ct_path):
            failures.append(
                {
                    "subject": subject,
                    "nod_id": nod_id,
                    "reason": f"ct_iso not found: {ct_path}",
                }
            )
            continue

        try:
            ct_hu, spacing_zyx = read_itk_array(ct_path)
            center = get_center_zyx(row)

            spacing_y = float(spacing_zyx[1])
            size_pix = int(max(8, round(float(size_mm) / max(spacing_y, 1e-6))))

            signature = signature_for_patch(size_mm=size_mm, size_pix=size_pix)

            npy_path = osp.abspath(osp.join(out_dir, f"{nod_id}.npy"))
            meta_path = osp.join(meta_dir, f"{nod_id}.json")

            if should_skip(meta_path, npy_path, signature, cover=cover):
                old_meta = read_json(meta_path) or {}

                out_row = {
                    "subject": subject,
                    "nod_id": nod_id,
                    "path": npy_path,
                    "size_mm": float(size_mm),
                    "size_pix": int(size_pix),
                    "channels": 1,
                    "spacing_z": float(spacing_zyx[0]),
                    "spacing_y": float(spacing_zyx[1]),
                    "spacing_x": float(spacing_zyx[2]),
                    "center_z": int(old_meta.get("center_z", center[0])),
                    "center_y": int(old_meta.get("center_y", center[1])),
                    "center_x": int(old_meta.get("center_x", center[2])),
                }

                for col in ["label"] + PUBLIC_SEMANTIC_COLS:
                    if col in row.index:
                        out_row[col] = row[col]

                rows.append(out_row)
                continue

            ct01 = window_normalize(ct_hu)
            patch01, crop_meta = crop_centered_cube(
                ct01,
                center_zyx=center,
                size=size_pix,
                pad_value=PAD_NORM,
            )

            guard_msg = patch_guard(patch01)
            if guard_msg:
                failures.append(
                    {
                        "subject": subject,
                        "nod_id": nod_id,
                        "reason": guard_msg,
                    }
                )
                continue

            patch = patch01[None, ...].astype(np.float32)
            np.save(npy_path, patch)

            if export_png:
                qc_path = osp.join(qc_dir, f"{nod_id}.png")
                save_qc_figure(
                    ct_volume01=ct01,
                    patch01=patch01,
                    crop_meta=crop_meta,
                    out_png=qc_path,
                    title=f"{nod_id} | {int(size_mm)} mm",
                )

            meta = {
                **signature,
                **crop_meta,
                "pipeline_version": PIPELINE_VERSION,
                "signature_hash": hash_dict(signature),
                "subject": subject,
                "nod_id": nod_id,
                "ct_iso": ct_path,
                "spacing_z": float(spacing_zyx[0]),
                "spacing_y": float(spacing_zyx[1]),
                "spacing_x": float(spacing_zyx[2]),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_json(meta_path, meta)

            out_row = {
                "subject": subject,
                "nod_id": nod_id,
                "path": npy_path,
                "size_mm": float(size_mm),
                "size_pix": int(size_pix),
                "channels": 1,
                "spacing_z": float(spacing_zyx[0]),
                "spacing_y": float(spacing_zyx[1]),
                "spacing_x": float(spacing_zyx[2]),
                "center_z": int(center[0]),
                "center_y": int(center[1]),
                "center_x": int(center[2]),
            }

            for col in ["label"] + PUBLIC_SEMANTIC_COLS:
                if col in row.index:
                    out_row[col] = row[col]

            rows.append(out_row)

        except Exception as exc:
            failures.append(
                {
                    "subject": subject,
                    "nod_id": nod_id,
                    "reason": str(exc),
                }
            )

    samples_path = osp.join(out_dir, "samples.csv")
    failures_path = osp.join(out_dir, "failures.csv")

    samples = pd.DataFrame(rows)

    base_cols = [
        "subject",
        "nod_id",
        "path",
        "size_mm",
        "size_pix",
        "channels",
        "spacing_z",
        "spacing_y",
        "spacing_x",
        "center_z",
        "center_y",
        "center_x",
    ]

    ordered_cols = base_cols + [c for c in ["label"] + PUBLIC_SEMANTIC_COLS if c in samples.columns]
    if samples.empty:
        samples = pd.DataFrame(columns=ordered_cols)
    else:
        for col in ordered_cols:
            if col not in samples.columns:
                samples[col] = np.nan
        samples = samples[ordered_cols]

    samples.to_csv(samples_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)

    print(
        f"[SIZE {int(size_mm)}mm] "
        f"success={len(samples)}  fail={len(failures)}  out={out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct_index_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes_mm", default="32")
    parser.add_argument("--metadata_csv", default="")
    parser.add_argument("--base_dir", default="")
    parser.add_argument("--sid", default="")
    parser.add_argument("--no_png", action="store_true")
    parser.add_argument("--cover", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed_all(args.seed)
    ensure_dir(args.out_dir)

    df = read_table(args.ct_index_csv)
    validate_ct_index(df)

    df["subject"] = df["subject"].map(clean_id)
    df["nod_id"] = df["nod_id"].map(clean_id)

    base_dir = args.base_dir or osp.dirname(osp.abspath(args.ct_index_csv))

    df["ct_iso"] = df["ct_iso"].map(lambda p: resolve_path(p, base_dir))

    if "mask_iso" in df.columns:
        df["mask_iso"] = df["mask_iso"].map(
            lambda p: resolve_path(p, base_dir) if pd.notna(p) and str(p).strip() else p
        )

    if args.sid:
        keep = {clean_id(x) for x in args.sid.split(",") if clean_id(x)}
        df = df[df["subject"].isin(keep)].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No cases to process.")

    df = merge_metadata(
        ct_index=df,
        metadata_csv=args.metadata_csv or None,
    )

    sizes_mm = parse_sizes_mm(args.sizes_mm)

    for size_mm in sizes_mm:
        process_size(
            df=df,
            out_root=args.out_dir,
            size_mm=size_mm,
            export_png=not args.no_png,
            cover=args.cover,
        )


if __name__ == "__main__":
    main()