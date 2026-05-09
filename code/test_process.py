#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocess CT and lesion masks to 1.0-mm isotropic NIfTI volumes.

Expected input table columns:
    subject
    ct_path
    mask_path or mask_dir
    nod_id optional

Notes:
    - ct_path can be a DICOM series directory or a NIfTI file.
    - mask_path can contain one or more paths separated by semicolons.
    - If mask_path is absent, mask_dir is scanned for .nii, .nii.gz, .nrrd, or .mha files.
    - Masks are assumed to be in the same physical coordinate system as the CT.
    - This public version intentionally keeps the preprocessing assumptions explicit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk


ISO_SPACING = (1.0, 1.0, 1.0)
CT_WINDOW = (-1000.0, 400.0)
PIPELINE_VERSION = "public-iso1mm-v1"


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


def read_ct_image(ct_path: str) -> sitk.Image:
    if osp.isdir(ct_path):
        reader = sitk.ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(ct_path)

        if not file_names:
            raise RuntimeError(f"No readable DICOM series found in: {ct_path}")

        reader.SetFileNames(file_names)
        image = reader.Execute()
        return sitk.Cast(image, sitk.sitkFloat32)

    if osp.isfile(ct_path):
        image = sitk.ReadImage(ct_path)
        return sitk.Cast(image, sitk.sitkFloat32)

    raise FileNotFoundError(f"CT path not found: {ct_path}")


def mask_to_binary(mask_img: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(mask_img)
    arr = np.nan_to_num(arr, nan=0.0)

    if arr.ndim == 4:
        arr = arr.max(axis=-1)

    arr = (arr > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)

    try:
        out.CopyInformation(mask_img)
    except Exception:
        pass

    return out


def resample_image(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float],
    is_mask: bool = False,
    default_value: float = 0.0,
) -> sitk.Image:
    old_size = np.array(image.GetSize(), dtype=np.int64)
    old_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    new_spacing = np.array(target_spacing, dtype=np.float64)

    new_size = np.ceil(old_size * old_spacing / new_spacing).astype(np.int64)
    new_size = [int(x) for x in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(x) for x in new_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    )
    resampler.SetOutputPixelType(sitk.sitkUInt8 if is_mask else sitk.sitkFloat32)

    return resampler.Execute(image)


def resample_mask_to_reference(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    mask_img = mask_to_binary(mask_img)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(sitk.sitkUInt8)

    out = resampler.Execute(mask_img)
    out = mask_to_binary(out)
    return out


def nonzero_center_zyx(mask_arr: np.ndarray) -> Tuple[int, int, int]:
    coords = np.argwhere(mask_arr > 0)

    if coords.size == 0:
        raise RuntimeError("Mask is empty after resampling.")

    center = coords.mean(axis=0)
    z, y, x = [int(round(float(v))) for v in center]

    z = int(np.clip(z, 0, mask_arr.shape[0] - 1))
    y = int(np.clip(y, 0, mask_arr.shape[1] - 1))
    x = int(np.clip(x, 0, mask_arr.shape[2] - 1))

    return z, y, x


def save_overlay(
    ct_img: sitk.Image,
    mask_img: sitk.Image,
    out_png: str,
) -> Tuple[int, int, int]:
    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask = sitk.GetArrayFromImage(mask_img).astype(np.uint8)

    z, y, x = nonzero_center_zyx(mask)

    ct2d = np.clip(ct[z], CT_WINDOW[0], CT_WINDOW[1])
    ct2d = (ct2d - CT_WINDOW[0]) / (CT_WINDOW[1] - CT_WINDOW[0] + 1e-8)

    ensure_dir(osp.dirname(out_png))

    fig = plt.figure(figsize=(5, 5), dpi=160)
    plt.imshow(ct2d, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imshow(mask[z], cmap="Reds", alpha=0.35)
    plt.scatter([x], [y], s=10)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return z, y, x


def parse_mask_paths(row: pd.Series, base_dir: str | None = None) -> List[str]:
    paths: List[str] = []

    if "mask_path" in row and pd.notna(row["mask_path"]) and str(row["mask_path"]).strip():
        raw = str(row["mask_path"]).replace(",", ";")
        paths = [p.strip() for p in raw.split(";") if p.strip()]
        return [resolve_path(p, base_dir) for p in paths]

    if "mask_dir" in row and pd.notna(row["mask_dir"]) and str(row["mask_dir"]).strip():
        mask_dir = resolve_path(str(row["mask_dir"]), base_dir)

        if not osp.isdir(mask_dir):
            raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

        exts = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")
        candidates = []

        for root, _, files in os.walk(mask_dir):
            for fn in files:
                if fn.lower().endswith(exts):
                    candidates.append(osp.join(root, fn))

        candidates.sort()

        if not candidates:
            raise FileNotFoundError(f"No mask files found in: {mask_dir}")

        return candidates

    raise RuntimeError("Each row must contain either mask_path or mask_dir.")


def make_nod_ids(subject: str, row: pd.Series, n_masks: int) -> List[str]:
    if "nod_id" in row and pd.notna(row["nod_id"]) and str(row["nod_id"]).strip():
        nod_id = clean_id(row["nod_id"])

        if n_masks == 1:
            return [nod_id]

        return [f"{nod_id}-{i + 1}" for i in range(n_masks)]

    if n_masks == 1:
        return [subject]

    return [f"{subject}-{i + 1}" for i in range(n_masks)]


def subject_is_done(out_dir: str, subject: str, cover: bool) -> bool:
    if cover:
        return False

    meta_path = osp.join(out_dir, "proc_meta.json")
    ct_path = osp.join(out_dir, f"{subject}_iso_ct.nii.gz")

    if not osp.isfile(meta_path) or not osp.isfile(ct_path):
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("pipeline_version") != PIPELINE_VERSION:
            return False

        if not meta.get("ct_index_rows"):
            return False

        return True

    except Exception:
        return False


def process_subject(
    row: pd.Series,
    out_root: str,
    base_dir: str | None,
    cover: bool = False,
) -> Tuple[List[dict], dict]:
    subject = clean_id(row["subject"])
    ct_path = resolve_path(row["ct_path"], base_dir)
    mask_paths = parse_mask_paths(row, base_dir)
    nod_ids = make_nod_ids(subject, row, len(mask_paths))

    subject_dir = osp.join(out_root, subject)
    ensure_dir(subject_dir)

    if subject_is_done(subject_dir, subject, cover=cover):
        meta_path = osp.join(subject_dir, "proc_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("ct_index_rows", []), {
            "subject": subject,
            "status": "skipped",
            "reason": "existing outputs reused",
        }

    ct_img = read_ct_image(ct_path)
    iso_ct = resample_image(
        ct_img,
        target_spacing=ISO_SPACING,
        is_mask=False,
        default_value=-1024.0,
    )

    iso_ct_path = osp.abspath(osp.join(subject_dir, f"{subject}_iso_ct.nii.gz"))
    sitk.WriteImage(iso_ct, iso_ct_path)

    ct_index_rows = []
    mask_records = []

    for mask_path, nod_id in zip(mask_paths, nod_ids):
        mask_img = sitk.ReadImage(mask_path)
        iso_mask = resample_mask_to_reference(mask_img, iso_ct)
        mask_arr = sitk.GetArrayFromImage(iso_mask).astype(np.uint8)

        voxels = int(np.count_nonzero(mask_arr))
        if voxels <= 0:
            raise RuntimeError(f"Mask became empty after resampling: {mask_path}")

        iso_mask_path = osp.abspath(osp.join(subject_dir, f"{nod_id}_iso_mask.nii.gz"))
        overlay_path = osp.abspath(osp.join(subject_dir, f"{nod_id}_overlay.png"))

        sitk.WriteImage(iso_mask, iso_mask_path)
        center_z, center_y, center_x = save_overlay(iso_ct, iso_mask, overlay_path)

        ct_index_rows.append(
            {
                "subject": subject,
                "nod_id": nod_id,
                "ct_iso": iso_ct_path,
                "mask_iso": iso_mask_path,
                "center_z": center_z,
                "center_y": center_y,
                "center_x": center_x,
            }
        )

        mask_records.append(
            {
                "nod_id": nod_id,
                "source_mask": osp.basename(mask_path),
                "iso_mask": iso_mask_path,
                "overlay": overlay_path,
                "voxels": voxels,
                "center_z": center_z,
                "center_y": center_y,
                "center_x": center_x,
            }
        )

    meta = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject": subject,
        "ct_path": ct_path,
        "iso_spacing": list(ISO_SPACING),
        "iso_ct": iso_ct_path,
        "masks": mask_records,
        "ct_index_rows": ct_index_rows,
    }

    with open(osp.join(subject_dir, "proc_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return ct_index_rows, {
        "subject": subject,
        "status": "ok",
        "n_masks": len(mask_records),
    }


def validate_input_table(df: pd.DataFrame) -> None:
    required = ["subject", "ct_path"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if "mask_path" not in df.columns and "mask_dir" not in df.columns:
        raise RuntimeError("Input table must contain either mask_path or mask_dir.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_dir", default="")
    parser.add_argument("--sid", default="")
    parser.add_argument("--cover", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    log_path = osp.join(args.out_dir, "preprocess.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    df = read_table(args.input_csv)
    validate_input_table(df)

    df["subject"] = df["subject"].map(clean_id)

    if args.sid:
        keep = {clean_id(x) for x in args.sid.split(",") if clean_id(x)}
        df = df[df["subject"].isin(keep)].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No cases to process.")

    all_rows = []
    status_rows = []

    for _, row in df.iterrows():
        subject = clean_id(row["subject"])

        try:
            rows, status = process_subject(
                row=row,
                out_root=args.out_dir,
                base_dir=args.base_dir or None,
                cover=args.cover,
            )
            all_rows.extend(rows)
            status_rows.append(status)
            logging.info("[OK] %s", subject)

        except Exception as exc:
            status_rows.append(
                {
                    "subject": subject,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
            logging.exception("[FAIL] %s: %s", subject, exc)

    ct_index_path = osp.join(args.out_dir, "ct_index.csv")
    status_path = osp.join(args.out_dir, "preprocess_status.csv")

    pd.DataFrame(all_rows).to_csv(ct_index_path, index=False)
    pd.DataFrame(status_rows).to_csv(status_path, index=False)

    print(f"[DONE] processed={len(status_rows)}")
    print(f"[WRITE] {ct_index_path}")
    print(f"[WRITE] {status_path}")


if __name__ == "__main__":
    main()