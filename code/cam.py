#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grad-CAM visualization for 3D CT classification models.

This script exports CT slice, Grad-CAM heatmap, and overlay images for a trained
image-only or multimodal model checkpoint.

Expected inputs:
    --ckpt_path   path to best.pt
    --samples_csv path to samples.csv
    --out_dir     output directory

The samples.csv should use the standardized public format described in utils.py.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import traceback
from types import SimpleNamespace
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import ImgOnlyModel, TabOnlyModel, UnifiedModel
from utils import IDX2NAME, LungDataset, read_samples_for_task, resolve_device


def seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_abs_path(path: str, base_dir: str | None = None) -> str:
    path = str(path)
    if osp.isabs(path) or not base_dir:
        return path
    return osp.join(base_dir, path)


def normalize01_percentile(
    x: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    eps: float = 1e-12,
) -> np.ndarray:
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))

    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def transform_cam_for_display(
    cam01: np.ndarray,
    mode: str = "pct_gamma",
    p_low: float = 1.0,
    p_high: float = 99.0,
    gamma: float = 0.7,
) -> np.ndarray:
    cam01 = np.clip(cam01.astype(np.float32), 0.0, 1.0)

    if mode == "strict":
        return cam01

    cam = normalize01_percentile(cam01, p_low, p_high)

    if gamma > 0:
        cam = np.power(cam, float(gamma)).astype(np.float32)

    return np.clip(cam, 0.0, 1.0).astype(np.float32)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict):
        return {
            "model": ckpt,
            "args": {},
            "stats": {},
            "den_vocab": {},
        }

    if "model" not in ckpt:
        ckpt = {"model": ckpt, "args": {}, "stats": {}, "den_vocab": {}}

    ckpt.setdefault("args", {})
    ckpt.setdefault("stats", {})
    ckpt.setdefault("den_vocab", {})

    return ckpt


def infer_in_channels_from_npy(path: str) -> int:
    arr = np.load(path)

    if arr.ndim == 4:
        return int(arr.shape[0])

    if arr.ndim == 3:
        return 1

    raise RuntimeError(f"Unexpected patch shape: {arr.shape}")


def build_model_from_checkpoint(
    ckpt: Dict,
    device: torch.device,
    in_ch_fallback: int,
):
    args = ckpt.get("args", {}) or {}
    stats = ckpt.get("stats", {}) or {}
    den_vocab = ckpt.get("den_vocab", {}) or {}

    mode = args.get("mode", "mm")
    img_backbone = args.get("img_backbone", "r2plus1d_18")
    img_trans_layers = int(args.get("img_trans_layers", 0))
    d_img = int(args.get("d_img", 256))
    d_txt = int(args.get("d_txt", 256))
    drop = float(args.get("drop", 0.2))
    use_gate = bool(args.get("use_gate", False))
    mix_rule = args.get("mix_rule", "prob_anchor")
    alpha_cap = float(args.get("alpha_cap", 0.30))

    in_ch = args.get("in_channels", None)
    if in_ch is None:
        in_ch = args.get("in_ch", None)
    if in_ch is None:
        in_ch = in_ch_fallback
    in_ch = int(in_ch)

    tab_columns = stats.get("tab_columns", None)
    if not tab_columns:
        raise RuntimeError(
            "The checkpoint does not contain stats['tab_columns']. "
            "Please use a checkpoint saved by train_stable.py."
        )

    den_num_classes = len(den_vocab) if den_vocab else 4

    if mode == "txt_only":
        model = TabOnlyModel(
            d_txt=d_txt,
            drop=drop,
            den_num_classes=den_num_classes,
            out_classes=3,
            num_names=tab_columns,
        )

    elif mode == "img_only":
        model = ImgOnlyModel(
            img_backbone=img_backbone,
            in_ch=in_ch,
            d_img=d_img,
            drop=drop,
            img_trans_layers=img_trans_layers,
            out_classes=3,
        )

    elif mode == "mm":
        model = UnifiedModel(
            mode="mm",
            task_out_classes=3,
            img_backbone=img_backbone,
            in_ch=in_ch,
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
        raise RuntimeError(f"Unsupported checkpoint mode: {mode}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    return model, mode, in_ch, stats, den_vocab


class ActivationGradientHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.activation = None
        self.handle = module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        if torch.is_tensor(output):
            self.activation = output
        elif isinstance(output, (tuple, list)):
            tensors = [x for x in output if torch.is_tensor(x)]
            self.activation = tensors[-1] if tensors else None
        else:
            self.activation = None

        if self.activation is not None and self.activation.requires_grad:
            self.activation.retain_grad()

    def close(self) -> None:
        self.handle.remove()


def get_cam_layer(model: torch.nn.Module, layer_name: str):
    if hasattr(model, "img") and hasattr(model.img, "backbone"):
        backbone = model.img.backbone

        if hasattr(backbone, "net"):
            net = backbone.net

            if layer_name == "layer2":
                return net.layer2
            if layer_name == "layer3":
                return net.layer3
            if layer_name == "layer4":
                return net.layer4

    raise RuntimeError(
        "Could not find a supported convolutional CAM layer. "
        "This script currently supports ResNet-style 3D backbones."
    )


def forward_logits(
    model: torch.nn.Module,
    mode: str,
    x_img: torch.Tensor,
    x_num: torch.Tensor,
    x_den: torch.Tensor,
    ld_z: torch.Tensor,
):
    if mode == "img_only":
        return model(x_img, None, None, None)

    if mode == "mm":
        return model(x_img, (x_num, x_den), ld_z, x_den)

    raise RuntimeError("Grad-CAM is not available for semantic-only models.")


def compute_gradcam_3d(
    model: torch.nn.Module,
    mode: str,
    x_img: torch.Tensor,
    x_num: torch.Tensor,
    x_den: torch.Tensor,
    ld_z: torch.Tensor,
    target_class: int,
    cam_layer: str = "layer3",
    score_source: str = "image",
) -> np.ndarray:
    layer = get_cam_layer(model, cam_layer)
    hook = ActivationGradientHook(layer)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        logits, extra = forward_logits(model, mode, x_img, x_num, x_den, ld_z)

        score_logits = logits

        if mode == "mm" and score_source == "image":
            if isinstance(extra, dict) and "logits_img" in extra:
                score_logits = extra["logits_img"]

        score = score_logits[:, int(target_class)].sum()
        score.backward(retain_graph=False)

        activation = hook.activation

        if activation is None or activation.grad is None:
            raise RuntimeError("Failed to capture activations or gradients.")

        gradients = activation.grad

        if activation.ndim != 5:
            raise RuntimeError(f"Expected 5D activation, got {tuple(activation.shape)}.")

        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        _, _, depth, height, width = x_img.shape
        cam = F.interpolate(
            cam,
            size=(depth, height, width),
            mode="trilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().float().cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max - cam_min < 1e-12:
            cam[:] = 0.0
        else:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-12)

        return np.clip(cam, 0.0, 1.0).astype(np.float32)

    finally:
        hook.close()
        model.zero_grad(set_to_none=True)


def save_ct_image(ct2d: np.ndarray, out_path: str) -> None:
    ensure_dir(osp.dirname(out_path))
    fig = plt.figure(figsize=(6, 6), dpi=200)
    plt.imshow(ct2d, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_cam_image(cam2d: np.ndarray, out_path: str) -> None:
    ensure_dir(osp.dirname(out_path))
    fig = plt.figure(figsize=(6, 6), dpi=200)
    im = plt.imshow(cam2d, cmap="jet", vmin=0.0, vmax=1.0)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def save_overlay_image(
    ct2d: np.ndarray,
    cam2d: np.ndarray,
    out_path: str,
    alpha: float = 0.45,
) -> None:
    ensure_dir(osp.dirname(out_path))
    fig = plt.figure(figsize=(6, 6), dpi=200)
    plt.imshow(ct2d, cmap="gray", vmin=0.0, vmax=1.0)
    im = plt.imshow(cam2d, cmap="jet", alpha=float(alpha), vmin=0.0, vmax=1.0)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def prepare_dataframe(samples_csv: str, data_root: str | None = None) -> pd.DataFrame:
    args = SimpleNamespace(task="tri", samples_csv=samples_csv)
    df = read_samples_for_task(args).copy()

    df["path"] = df["path"].map(lambda p: to_abs_path(p, data_root))

    missing = ~df["path"].map(lambda p: osp.isfile(str(p)))
    if missing.any():
        examples = df.loc[missing, "path"].head(5).tolist()
        raise RuntimeError(f"Some patch files are missing. Examples: {examples}")

    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--samples_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--data_root", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--cam_layer", choices=["layer2", "layer3", "layer4"], default="layer3")
    parser.add_argument("--score_source", choices=["image", "final"], default="image")
    parser.add_argument("--cam_class", choices=["pred", "gt"], default="pred")

    parser.add_argument("--slice_mode", choices=["center", "max_cam"], default="center")
    parser.add_argument("--alpha", type=float, default=0.45)

    parser.add_argument("--cam_vis", choices=["pct_gamma", "strict"], default="pct_gamma")
    parser.add_argument("--cam_p_low", type=float, default=1.0)
    parser.add_argument("--cam_p_high", type=float, default=99.0)
    parser.add_argument("--cam_gamma", type=float, default=0.7)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_pdf", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    seed_all(args.seed)
    ensure_dir(args.out_dir)

    device = resolve_device(args.gpu)
    print(f"[DEVICE] {device}")

    df = prepare_dataframe(args.samples_csv, args.data_root or None)

    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].reset_index(drop=True)

    in_ch_fallback = infer_in_channels_from_npy(str(df.iloc[0]["path"]))

    ckpt = load_checkpoint(args.ckpt_path, device)
    model, mode, in_ch, stats, den_vocab = build_model_from_checkpoint(
        ckpt=ckpt,
        device=device,
        in_ch_fallback=in_ch_fallback,
    )

    if mode == "txt_only":
        raise RuntimeError("Grad-CAM is not available for semantic-only checkpoints.")

    dataset = LungDataset(
        df=df,
        stats=stats,
        den_vocab=den_vocab,
        in_ch=in_ch,
        task="tri",
        augment=None,
        train=False,
        load_image=True,
    )

    png_dir = osp.join(args.out_dir, "png")
    pdf_dir = osp.join(args.out_dir, "pdf")
    ensure_dir(png_dir)

    if args.save_pdf:
        ensure_dir(pdf_dir)

    failures = []

    print(f"[DATA] samples={len(dataset)}  mode={mode}  in_channels={in_ch}")

    for i in range(len(dataset)):
        row = df.iloc[i]
        sample_id = str(row.get("nod_id", osp.splitext(osp.basename(str(row["path"])))[0]))
        sample_id = sample_id.replace("/", "_").replace("\\", "_")

        out_ct_png = osp.join(png_dir, f"{sample_id}_ct.png")
        out_cam_png = osp.join(png_dir, f"{sample_id}_cam.png")
        out_overlay_png = osp.join(png_dir, f"{sample_id}_overlay.png")

        if (
            not args.force
            and osp.isfile(out_ct_png)
            and osp.isfile(out_cam_png)
            and osp.isfile(out_overlay_png)
        ):
            continue

        try:
            x_img, x_tab, y, ld_z = dataset[i]
            x_num, x_den = x_tab

            x_img = x_img.unsqueeze(0).to(device, non_blocking=True)
            x_num = x_num.unsqueeze(0).to(device, non_blocking=True)
            x_den = x_den.unsqueeze(0).to(device, non_blocking=True)
            ld_z = ld_z.unsqueeze(0).to(device, non_blocking=True)

            if args.cam_class == "pred":
                with torch.no_grad():
                    logits, _ = forward_logits(model, mode, x_img, x_num, x_den, ld_z)
                    target_class = int(torch.argmax(logits, dim=1).item())
            else:
                target_class = int(y.item())

            cam3d = compute_gradcam_3d(
                model=model,
                mode=mode,
                x_img=x_img,
                x_num=x_num,
                x_den=x_den,
                ld_z=ld_z,
                target_class=target_class,
                cam_layer=args.cam_layer,
                score_source=args.score_source,
            )

            vol = x_img[0].detach().cpu().numpy()
            vol = vol[0] if vol.ndim == 4 else vol

            if args.slice_mode == "center":
                z = int(vol.shape[0] // 2)
            else:
                z = int(np.argmax(cam3d.reshape(cam3d.shape[0], -1).mean(axis=1)))

            ct2d = normalize01_percentile(vol[z], 1.0, 99.0)
            cam2d = np.clip(cam3d[z].astype(np.float32), 0.0, 1.0)
            cam2d_vis = transform_cam_for_display(
                cam2d,
                mode=args.cam_vis,
                p_low=args.cam_p_low,
                p_high=args.cam_p_high,
                gamma=args.cam_gamma,
            )

            save_ct_image(ct2d, out_ct_png)
            save_cam_image(cam2d_vis, out_cam_png)
            save_overlay_image(ct2d, cam2d_vis, out_overlay_png, alpha=args.alpha)

            if args.save_pdf:
                save_ct_image(ct2d, osp.join(pdf_dir, f"{sample_id}_ct.pdf"))
                save_cam_image(cam2d_vis, osp.join(pdf_dir, f"{sample_id}_cam.pdf"))
                save_overlay_image(
                    ct2d,
                    cam2d_vis,
                    osp.join(pdf_dir, f"{sample_id}_overlay.pdf"),
                    alpha=args.alpha,
                )

            label_name = IDX2NAME.get(int(y.item()), str(int(y.item())))
            pred_name = IDX2NAME.get(target_class, str(target_class))
            print(f"[OK] {sample_id}  gt={label_name}  cam_class={pred_name}  slice={z}")

        except Exception as exc:
            failures.append(
                {
                    "index": i,
                    "sample_id": sample_id,
                    "error": str(exc),
                    "trace": traceback.format_exc()[:4000],
                }
            )
            print(f"[ERROR] {sample_id}: {exc}")

    if failures:
        fail_path = osp.join(args.out_dir, "cam_failures.csv")
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"[WARN] failures saved to {fail_path}")
    else:
        print("[DONE] no failures")


if __name__ == "__main__":
    main()