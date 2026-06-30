#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os.path as osp
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset


IDX2NAME = {0: "AAH/AIS", 1: "MIA", 2: "IAC"}
NAME2IDX = {v: k for k, v in IDX2NAME.items()}

FEATS_NUM = [f"feat_{i:02d}" for i in range(1, 10)]
FEATS_ALL = FEATS_NUM + ["density"]

DENSITY_ORDER = ["UNK", "GGO", "PartSolid", "Solid"]

LABEL_ALIASES = {
    "aah": "AAH/AIS",
    "ais": "AAH/AIS",
    "aah/ais": "AAH/AIS",
    "aah_ais": "AAH/AIS",
    "aah-ais": "AAH/AIS",
    "preinvasive": "AAH/AIS",
    "pre-invasive": "AAH/AIS",
    "mia": "MIA",
    "iac": "IAC",
}


def seed_all(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu)}")
    return torch.device("cpu")


def norm_id_core(x) -> str:
    s = str(x).strip().upper().replace(" ", "").replace("\t", "")
    return re.split(r"[-_]", s)[0]


def canonical_label(x):
    if pd.isna(x):
        return None

    if isinstance(x, (int, np.integer)):
        v = int(x)
        return v if v in IDX2NAME else None

    if isinstance(x, float) and np.isfinite(x):
        v = int(x)
        return v if v in IDX2NAME else None

    s = str(x).strip()

    if s in NAME2IDX:
        return NAME2IDX[s]

    key = s.lower().replace(" ", "").replace("-", "_")
    key = key.replace("preinvasivelesion", "preinvasive")
    key = LABEL_ALIASES.get(key, key)

    if key in NAME2IDX:
        return NAME2IDX[key]

    return None


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


def to_float_safe(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"yes", "y", "true", "present", "positive", "1"}:
            return 1.0
        if s in {"no", "n", "false", "absent", "negative", "0"}:
            return 0.0

    try:
        return float(x)
    except Exception:
        return np.nan


def discover_cat_levels(df_train: pd.DataFrame) -> Dict[str, List[str]]:
    del df_train
    return {}


def expand_structured_features(
    df: pd.DataFrame,
    _cat_levels=None,
) -> Tuple[pd.DataFrame, List[str]]:
    del _cat_levels

    out = df.copy()

    for col in FEATS_NUM:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = out[col].apply(to_float_safe)

    return out, list(FEATS_NUM)


def _choose_samples_csv(args) -> str:
    for attr in ("samples_csv", "samples_csv_tri"):
        path = getattr(args, attr, "")
        if path:
            return path

    raise SystemExit("Please provide --samples_csv.")


def _make_generic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    existing = [c for c in FEATS_NUM if c in out.columns]
    if existing:
        for c in FEATS_NUM:
            if c not in out.columns:
                out[c] = np.nan
            out[c] = out[c].apply(to_float_safe)
        return out

    ignore = {
        "path",
        "label",
        "label_idx",
        "label_group",
        "category",
        "tri_label",
        "subject",
        "nod_id",
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

    numeric_candidates = []
    for col in out.columns:
        if col in ignore:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        if values.notna().sum() > 0:
            numeric_candidates.append(col)

    for i, feat in enumerate(FEATS_NUM):
        if i < len(numeric_candidates):
            out[feat] = pd.to_numeric(out[numeric_candidates[i]], errors="coerce")
        else:
            out[feat] = np.nan

    return out


def read_samples_for_task(args) -> pd.DataFrame:
    task = getattr(args, "task", "tri")
    if task != "tri":
        raise ValueError("Only tri-class classification is supported in this release.")

    df = pd.read_csv(_choose_samples_csv(args))
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    if "path" not in df.columns:
        raise RuntimeError("samples.csv must contain a 'path' column.")

    if "nod_id" not in df.columns:
        df["nod_id"] = df["path"].map(lambda p: osp.splitext(osp.basename(str(p)))[0])

    if "subject" not in df.columns:
        df["subject"] = df["nod_id"].map(norm_id_core)

    if "density" not in df.columns:
        df["density"] = "UNK"

    df["density"] = df["density"].apply(canonical_density)

    if "label_idx" not in df.columns:
        label_col = None
        for cand in ("label", "label_group", "category", "tri_label"):
            if cand in df.columns:
                label_col = cand
                break

        if label_col is None:
            raise RuntimeError(
                "samples.csv must contain 'label_idx' or one of: "
                "label, label_group, category, tri_label."
            )

        df["label_idx"] = df[label_col].apply(canonical_label)
    else:
        df["label_idx"] = df["label_idx"].apply(canonical_label)

    if df["label_idx"].isna().any():
        bad = df.loc[df["label_idx"].isna(), ["nod_id"]].head(5).to_dict("records")
        raise RuntimeError(f"Unmapped labels were found. Examples: {bad}")

    df["label_idx"] = df["label_idx"].astype(int)

    if "long_diameter" not in df.columns:
        df["long_diameter"] = np.nan
    df["long_diameter"] = pd.to_numeric(df["long_diameter"], errors="coerce")

    df = _make_generic_features(df)
    df, _ = expand_structured_features(df)

    return df.reset_index(drop=True)


class Aug3D:
    def __init__(self, mode: str = "none") -> None:
        self.mode = str(mode).lower()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return x.astype(np.float32)

        if np.random.rand() < 0.5:
            x = np.flip(x, axis=-1).copy()

        if np.random.rand() < 0.5:
            x = np.flip(x, axis=-2).copy()

        if self.mode == "light" and np.random.rand() < 0.3:
            x = x + 0.01 * np.random.randn(*x.shape).astype(np.float32)

        return x.astype(np.float32)


class LungDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        stats: Dict,
        den_vocab: Dict[str, int],
        in_ch: int,
        task: str = "tri",
        augment: Aug3D | None = None,
        train: bool = False,
        load_image: bool = True,
    ) -> None:
        if task != "tri":
            raise ValueError("Only tri-class classification is supported in this release.")

        self.df = df.reset_index(drop=True).copy()
        self.stats = stats
        self.den_vocab = dict(den_vocab)
        self.in_ch = int(in_ch) if in_ch is not None else None
        self.augment = augment
        self.train = bool(train)
        self.load_image = bool(load_image)
        self.tab_columns = list(stats.get("tab_columns", FEATS_NUM))

        tab = []
        for col in self.tab_columns:
            if col not in self.df.columns:
                self.df[col] = np.nan
            tab.append(pd.to_numeric(self.df[col], errors="coerce").astype(float).values)

        tab = np.stack(tab, axis=1).astype(np.float32)

        mean = np.array(
            [float(stats.get("mean", {}).get(c, 0.0)) for c in self.tab_columns],
            dtype=np.float32,
        )
        std = np.array(
            [float(stats.get("std", {}).get(c, 1.0)) for c in self.tab_columns],
            dtype=np.float32,
        )
        std[std < 1e-6] = 1.0

        tab = np.where(np.isnan(tab), mean[None, :], tab)
        self.tab = (tab - mean[None, :]) / std[None, :]

        if "density" not in self.df.columns:
            self.df["density"] = "UNK"

        density = self.df["density"].astype(str).map(
            lambda s: s if s in self.den_vocab else "UNK"
        )
        self.den = density.map(lambda s: self.den_vocab.get(s, 0)).astype(int).values

        self.y = pd.to_numeric(self.df["label_idx"], errors="raise").astype(int).values

        self.ld_mu = float(stats.get("mean", {}).get("long_diameter", 0.0))
        self.ld_std = float(stats.get("std", {}).get("long_diameter", 1.0))
        if self.ld_std < 1e-6:
            self.ld_std = 1.0

    def __len__(self) -> int:
        return len(self.df)

    def _load_patch(self, path: str) -> np.ndarray:
        arr = np.load(path)

        if arr.ndim == 3:
            arr = arr[None, ...]

        if arr.ndim != 4:
            raise RuntimeError(
                f"Expected patch shape (C,D,H,W) or (D,H,W), got {arr.shape}."
            )

        if self.in_ch is not None and arr.shape[0] != self.in_ch:
            if arr.shape[0] > self.in_ch:
                arr = arr[: self.in_ch]
            else:
                pad = np.repeat(arr[-1:], self.in_ch - arr.shape[0], axis=0)
                arr = np.concatenate([arr, pad], axis=0)

        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        mu = float(arr.mean())
        sd = float(arr.std())
        if sd < 1e-6:
            sd = 1.0

        arr = ((arr - mu) / sd).astype(np.float32)
        arr = np.clip(arr, -5.0, 5.0)

        if self.train and self.augment is not None:
            arr = self.augment(arr)

        return arr.astype(np.float32)

    def __getitem__(self, idx: int):
        if self.load_image:
            x_img = torch.from_numpy(self._load_patch(str(self.df.iloc[idx]["path"])))
        else:
            x_img = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

        x_num = torch.from_numpy(self.tab[idx].astype(np.float32))
        x_den = torch.tensor(int(self.den[idx]), dtype=torch.long)

        ld_raw = self.df.iloc[idx].get("long_diameter", np.nan)
        ld_z = 0.0 if pd.isna(ld_raw) else (float(ld_raw) - self.ld_mu) / self.ld_std
        ld_z = torch.tensor(float(ld_z), dtype=torch.float32)

        y = torch.tensor(int(self.y[idx]), dtype=torch.long)

        return x_img, (x_num, x_den), y, ld_z


@torch.no_grad()
def evaluate(
    model,
    loader,
    task: str,
    device: torch.device,
    name: str = "VAL",
    verbose: bool = True,
    **kwargs,
):
    del kwargs

    if task != "tri":
        raise ValueError("Only tri-class classification is supported in this release.")

    model.eval()

    y_true, y_pred = [], []

    for x_img, x_tab, y, ld_z in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_num = x_tab[0].to(device, non_blocking=True)
        x_den = x_tab[1].to(device, non_blocking=True)
        ld_z = ld_z.to(device, non_blocking=True)

        logits, _ = model(x_img, (x_num, x_den), ld_z, x_den)
        pred = torch.softmax(logits, dim=1).argmax(dim=1)

        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    if verbose:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        print(
            f"[{name}] Acc={acc:.4f}  Macro-F1={f1m:.4f}  "
            f"Macro-P={prec:.4f}  Macro-R={rec:.4f}"
        )
        print(cm)

    return acc, f1m


@torch.no_grad()
def dump_val_predictions_from_best(
    out_dir: str,
    model_ctor,
    model_kwargs: Dict,
    val_loader,
    task: str,
    device: torch.device,
    out_name: str = "preds_val.csv",
) -> None:
    if task != "tri":
        raise ValueError("Only tri-class classification is supported in this release.")

    import os.path as osp

    ckpt_path = osp.join(out_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = model_ctor(**model_kwargs).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    all_true, all_pred, all_prob = [], [], []

    for x_img, x_tab, y, ld_z in val_loader:
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

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)

    df = val_loader.dataset.df.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["p_AAH_AIS"] = prob[:, 0]
    df["p_MIA"] = prob[:, 1]
    df["p_IAC"] = prob[:, 2]

    out_path = osp.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    print(f"[DUMP] wrote {out_path}")
