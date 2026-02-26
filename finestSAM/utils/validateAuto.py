import os
import cv2
import torch
import math
import numpy as np
import lightning as L
from box import Box
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from finestSAM.model.model import FinestSAM
from finestSAM.data.dataset import load_test_dataset
from finestSAM.utils import seed_everything
from scipy.optimize import linear_sum_assignment
from monai.metrics import compute_dice, compute_hausdorff_distance
from typing import Dict, Any, Optional
import torch.nn.functional as F


def save_prediction_visual(
    *,
    out_dir: str,
    base_name: str,
    image: np.ndarray,
    gt_mask: Optional[torch.Tensor],
    pred_masks: torch.Tensor,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis("off")
    ax.imshow(image)

    def _iter_masks(mask_tensor: torch.Tensor):
        if mask_tensor.ndim == 2:
            yield mask_tensor
        elif mask_tensor.ndim == 3:
            for m in mask_tensor:
                yield m
        elif mask_tensor.ndim == 4:
            for m in mask_tensor.squeeze(1):
                yield m

    def _to_binary(mask: torch.Tensor) -> np.ndarray:
        mask_np = mask.detach().cpu().numpy()
        if mask_np.ndim > 2:
            mask_np = np.squeeze(mask_np)
        return (mask_np > 0.5).astype(np.uint8)

    legend_handles = []

    if gt_mask is not None:
        for gm in _iter_masks(gt_mask):
            gt_np = _to_binary(gm)
            if gt_np.any():
                ax.contour(gt_np, levels=[0.5], colors="lime", linewidths=2.0, linestyles="-")
        legend_handles.append(Line2D([0], [0], color="lime", lw=2, label="GT"))

    for pm in _iter_masks(pred_masks):
        pred_np = _to_binary(pm)
        if pred_np.any():
            ax.contour(pred_np, levels=[0.5], colors="red", linewidths=2.0, linestyles="-")
    legend_handles.append(Line2D([0], [0], color="red", lw=2, label="Prediction"))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title("GT (Green) vs Prediction (Red)")

    fig.tight_layout()
    output_path = os.path.join(out_dir, f"{base_name}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path



def compute_iou_matrix(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    """
    Computes the IoU matrix between two sets of masks.
    """
    n = masks1.shape[0]
    m = masks2.shape[0]
    
    if n == 0 or m == 0:
        return torch.zeros((n, m), device=masks1.device)
    
    masks1 = masks1.flatten(1).float()
    masks2 = masks2.flatten(1).float()
    
    intersection = torch.mm(masks1, masks2.t())
    
    area1 = masks1.sum(1).unsqueeze(1)
    area2 = masks2.sum(1).unsqueeze(0)
    union = area1 + area2 - intersection
    
    iou = intersection / (union + 1e-6)
    return iou


def match_masks(
    pred_masks: torch.Tensor, 
    gt_masks: torch.Tensor, 
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Matches predicted masks to ground truth masks using greedy assignment based on IoU.
    Vectorized computation of DSC and HD95 for matched pairs.

    Args:
        - pred_masks (torch.Tensor): Tensor of shape (N, H, W) or (N, 1, H, W) containing predicted binary masks.
        - gt_masks (torch.Tensor): Tensor of shape (M, H, W) or (M, 1, H, W) containing ground truth binary masks.
        - iou_threshold (float): Minimum IoU required to consider a match valid.

    Returns:
        - Dict[str, Any]: A dictionary containing TP, FP, FN counts, precision, recall, and lists of matched IoUs, DSCs, HD95s, and matched indices.    
    """
    if pred_masks.ndim == 4:
        pred_masks = pred_masks.squeeze(1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
        
    pred_masks = (pred_masks > 0).float()
    gt_masks = (gt_masks > 0).float()
    
    N = pred_masks.shape[0]
    M = gt_masks.shape[0]
    
    if N == 0 or M == 0:
        return {
            "tp": 0, "fp": N, "fn": M,
            "precision": 0.0, "recall": 0.0,
            "matched_ious": [], "matched_dscs": [], "matched_hd95s": [], "matches": []
        }

    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
    
    cost_matrix = -iou_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    valid_r = []
    valid_c = []
    matched_ious = []
    
    # 1. Filtering Matches Based on IoU Threshold
    for r, c in zip(row_ind, col_ind):
        iou_val = iou_matrix[r, c].item()
        if iou_val >= iou_threshold:
            valid_r.append(r)
            valid_c.append(c)
            matched_ious.append(iou_val)

    TP = len(valid_r)
    FP = N - TP
    FN = M - TP
    
    matched_dscs = []
    matched_hd95s = []
    
    # 2. Computing DSC and HD95 for Matched Pairs in Batch
    if TP > 0:
        # Formato per MONAI: (Batch, Channel, H, W)
        p_masks_batch = pred_masks[valid_r].unsqueeze(1)
        g_masks_batch = gt_masks[valid_c].unsqueeze(1)
        
        dsc_batch = compute_dice(y_pred=p_masks_batch, y=g_masks_batch, ignore_empty=False)
        hd95_batch = compute_hausdorff_distance(
            y_pred=p_masks_batch, y=g_masks_batch, include_background=False, percentile=95
        )
        
        for i in range(TP):
            matched_dscs.append(dsc_batch[i].item())
            
            hd95_val = hd95_batch[i].item()
            
            # Fixing NaN/Inf
            if math.isnan(hd95_val) or math.isinf(hd95_val):
                matched_hd95s.append(float('nan'))
            else:
                matched_hd95s.append(hd95_val)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    return {
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "precision": precision, 
        "recall": recall,
        "matched_ious": matched_ious,
        "matched_dscs": matched_dscs,
        "matched_hd95s": matched_hd95s,
        "matches": list(zip(valid_r, valid_c))
    }

def evaluate_amg(
    cfg: Box, 
    dataset_path: str, 
    checkpoint_path: str = None, 
    iou_threshold: float = 0.5,
    output_images: int = 0,
    out_dir: Optional[str] = None
):
    """
    Main function to evaluate the AMG predictor on a test dataset. 
    It computes both correlation metrics (IoU, DSC, HD95) for matched pairs and detection quality metrics (PPV, TPR, FPRatio) globally.
    
    Args:
        - cfg: Configuration object containing model and evaluation settings.
        - dataset_path: Path to the root of the test dataset.
        - checkpoint_path: Optional path to a model checkpoint to load.
        - iou_threshold: IoU threshold to determine true positive matches between predicted and ground truth masks.
        - output_images: Number of images to save visualizations for.
    Returns:
        - Prints out the evaluation results including mean IoU, DSC, HD95 for matched pairs, and global PPV, TPR, FPRatio.   
    """


    if checkpoint_path:
        cfg.model.checkpoint = checkpoint_path
        
    print(f"Loading checkpoint from: {cfg.model.checkpoint}")
    
    fabric = L.Fabric(
        accelerator=cfg.device,
        devices=cfg.num_devices,
        strategy="auto",
        precision=cfg.precision
    )
    
    seed_everything(fabric, cfg.seed_device, deterministic=True)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    
    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.eval()
        model.to(fabric.device)
        
    amg = model.get_automatic_predictor(
        pred_iou_thresh=0.8,
        stability_score_thresh=0.8,
        box_nms_thresh=0.7,
        min_mask_region_area=1
    )
    
    img_size = model.model.image_encoder.img_size
    dataloader = load_test_dataset(cfg, img_size, dataset_path, fabric=fabric)
    
    vis_dir = os.path.join(cfg.out_dir, "amg_validation_vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Starting AMG Evaluation on {len(dataloader)} images...")
    
    all_tps, all_fps, all_fns = [], [], []
    all_matched_ious, all_matched_dscs, all_matched_hd95s = [], [], []
    
    processed_images = 0
    saved_images = 0

    total_batches = len(dataloader)
    save_batch_indices = set()
    max_images = int(output_images or 0)
    if total_batches > 0:
        if max_images <= 0 or max_images >= total_batches:
            save_batch_indices = set(range(total_batches))
        else:
            lin_positions = np.linspace(0, total_batches - 1, num=max_images, dtype=int)
            save_batch_indices = set(int(pos) for pos in lin_positions.tolist())

    # Global segmentation accumulators (micro-average across the whole dataset)
    all_global_ious, all_global_dscs, all_global_hd95s = [], [], []
    total_intersection = 0.0
    total_union = 0.0
    total_pred_sum = 0.0
    total_gt_sum = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, batch in pbar:
            batch_results = []

            for data in batch:
                original_image = data["original_image"]
                gt_masks = data["gt_masks"]

                anns = amg.generate(original_image)

                if len(anns) == 0:
                    pred_masks = torch.empty((0, gt_masks.shape[1], gt_masks.shape[2]), device=fabric.device)
                else:
                    pred_masks_np = np.stack([ann['segmentation'] for ann in anns], axis=0)
                    pred_masks = torch.from_numpy(pred_masks_np).to(fabric.device)

                gt_masks = gt_masks.to(fabric.device)

                metrics = match_masks(pred_masks, gt_masks, iou_threshold=iou_threshold)

                # Global (per-image) segmentation metrics on the union of all masks
                h, w = gt_masks.shape[-2], gt_masks.shape[-1]
                if pred_masks.numel() == 0:
                    combined_pred = torch.zeros((h, w), device=fabric.device)
                else:
                    combined_pred = (pred_masks > 0).any(dim=0).float()

                if gt_masks.numel() == 0:
                    combined_gt = torch.zeros((h, w), device=fabric.device)
                else:
                    combined_gt = (gt_masks > 0).any(dim=0).float()

                intersection = (combined_pred * combined_gt).sum()
                pred_sum = combined_pred.sum()
                gt_sum = combined_gt.sum()
                union = pred_sum + gt_sum - intersection

                iou_global = (intersection / (union + 1e-6)).item() if union > 0 else 0.0
                dsc_global = (2 * intersection / (pred_sum + gt_sum + 1e-6)).item() if (pred_sum + gt_sum) > 0 else 0.0

                if pred_sum == 0 and gt_sum == 0:
                    hd95_global = float('nan')
                else:
                    hd95_val = compute_hausdorff_distance(
                        y_pred=combined_pred.unsqueeze(0).unsqueeze(0),
                        y=combined_gt.unsqueeze(0).unsqueeze(0),
                        include_background=False,
                        percentile=95
                    )[0].item()
                    hd95_global = float('nan') if math.isnan(hd95_val) or math.isinf(hd95_val) else hd95_val

                all_global_ious.append(iou_global)
                all_global_dscs.append(dsc_global)
                all_global_hd95s.append(hd95_global)

                total_intersection += intersection.item()
                total_union += union.item()
                total_pred_sum += pred_sum.item()
                total_gt_sum += gt_sum.item()

                all_tps.append(metrics["tp"])
                all_fps.append(metrics["fp"])
                all_fns.append(metrics["fn"])

                all_matched_ious.extend(metrics["matched_ious"])
                all_matched_dscs.extend(metrics["matched_dscs"])
                all_matched_hd95s.extend(metrics["matched_hd95s"])

                batch_results.append({
                    "data": data,
                    "metrics": metrics,
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                    "global_iou": iou_global,
                    "global_dsc": dsc_global,
                    "global_hd95": hd95_global,
                })

                processed_images += 1

            # Update progress bar with current batch metrics
            sum_tp_curr = sum(all_tps)
            sum_fp_curr = sum(all_fps)
            sum_fn_curr = sum(all_fns)

            curr_ppv = sum_tp_curr / (sum_tp_curr + sum_fp_curr) if (sum_tp_curr + sum_fp_curr) > 0 else 0.0
            curr_tpr = sum_tp_curr / (sum_tp_curr + sum_fn_curr) if (sum_tp_curr + sum_fn_curr) > 0 else 0.0

            pbar.set_postfix({
                "mIoU": f"{np.mean(all_matched_ious):.3f}" if all_matched_ious else "0.000",
                "mDSC": f"{np.mean(all_matched_dscs):.3f}" if all_matched_dscs else "0.000",
                "IoU(G)": f"{np.mean(all_global_ious):.3f}" if all_global_ious else "0.000",
                "DSC(G)": f"{np.mean(all_global_dscs):.3f}" if all_global_dscs else "0.000",
                "PPV(G)": f"{curr_ppv:.3f}",
                "TPR(G)": f"{curr_tpr:.3f}"
            })

            # Save every sample for the selected batches
            if i in save_batch_indices:
                for sample_idx, res in enumerate(batch_results):
                    data = res["data"]
                    pred_masks_batch = res["pred_masks"]

                    if data.get("original_image") is None or data.get("original_size") is None:
                        continue

                    gt_mask = data.get("gt_masks")
                    if gt_mask is not None:
                        gt_mask = gt_mask.float()
                        if gt_mask.ndim == 3:
                            gt_mask = gt_mask.any(dim=0).float()
                        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
                        gt_mask = F.interpolate(gt_mask, size=data["original_size"], mode="nearest").squeeze()

                    if pred_masks_batch.numel() == 0:
                        pred_to_save = torch.zeros((1, *data["original_size"]), device=fabric.device)
                    else:
                        pred_to_save = F.interpolate(
                            pred_masks_batch.unsqueeze(1).float(),
                            size=data["original_size"],
                            mode="nearest"
                        ).squeeze(1)

                    save_prediction_visual(
                        out_dir= out_dir or cfg.out_dir,
                        base_name=f"val_b{i:04d}_s{sample_idx:03d}",
                        image=np.array(data["original_image"]),
                        gt_mask=gt_mask,
                        pred_masks=pred_to_save.cpu(),
                    )
                    saved_images += 1

    # Global detection metrics
    sum_tp = sum(all_tps)
    sum_fp = sum(all_fps)
    sum_fn = sum(all_fns)
    
    global_ppv = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    global_tpr = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    global_fpratio = sum_fp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    
    mean_matched_iou = np.mean(all_matched_ious) if all_matched_ious else 0.0
    mean_matched_dsc = np.mean(all_matched_dscs) if all_matched_dscs else 0.0
    
    # Using np.nanmean to ignore NaN values in HD95 when computing the mean for matched pairs
    mean_matched_hd95 = np.nanmean(all_matched_hd95s) if all_matched_hd95s else 0.0

    # Global segmentation metrics (micro across dataset)
    global_iou_micro = (total_intersection / (total_union + 1e-6)) if total_union > 0 else 0.0
    global_dsc_micro = (2 * total_intersection / (total_pred_sum + total_gt_sum + 1e-6)) if (total_pred_sum + total_gt_sum) > 0 else 0.0
    mean_global_iou = np.mean(all_global_ious) if all_global_ious else 0.0
    mean_global_dsc = np.mean(all_global_dscs) if all_global_dscs else 0.0
    mean_global_hd95 = np.nanmean(all_global_hd95s) if all_global_hd95s else 0.0
    
    print("\n=== AMG Evaluation Results ===")
    print(f"Images Processed: {processed_images}")
    print(f"IoU Threshold for Match: {iou_threshold}")
    print(f"------------------------------")
    print(f"Correlation (Matched Pairs Only):")
    print(f"  Mean IoU:  {mean_matched_iou:.4f}")
    print(f"  Mean DSC:  {mean_matched_dsc:.4f}")
    print(f"  Mean HD95: {mean_matched_hd95:.4f} (NaNs ignored)")
    print(f"------------------------------")
    print(f"Global Segmentation (All Masks, OR-union per immagine):")
    print(f"  Mean IoU (macro):  {mean_global_iou:.4f}")
    print(f"  Mean DSC (macro):  {mean_global_dsc:.4f}")
    print(f"  Mean HD95 (macro): {mean_global_hd95:.4f} (NaNs ignored)")
    print(f"  IoU (micro):       {global_iou_micro:.4f}")
    print(f"  DSC (micro):       {global_dsc_micro:.4f}")
    print(f"------------------------------")
    print(f"Detection Quality (Global Micro-average):")
    print(f"  Global PPV (Precision):     {global_ppv:.4f}")
    print(f"  Global TPR (Recall):        {global_tpr:.4f}")
    print(f"  Global FPRatio (FP/Preds):  {global_fpratio:.4f}")
    print(f"  Total TP: {sum_tp} | Total FP: {sum_fp} | Total FN: {sum_fn}")
    print(f"==============================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkoint path")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for validation")
    parser.add_argument("--output_images", type=int, default=10, help="Number of output images to save")
    
    args = parser.parse_args()
    
    from finestSAM.config import cfg_training as cfg
    evaluate_amg(cfg, args.dataset_path, args.checkpoint, args.iou_thresh, output_images=args.output_images)


# Example usage:
# python validateV2.py --dataset_path /path/to/dataset --checkpoint /path/to/checkpoint.ckpt --iou_thresh 0.5 --output_images 10