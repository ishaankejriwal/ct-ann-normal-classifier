# Evaluation, thresholding, and Grad-CAM utilities.

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from torch.amp import autocast


def calculate_detailed_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float | None = None) -> dict:
    # Compute binary classification metrics and confusion-matrix details.

    # Replace invalid values before thresholding and metric computation.
    y_true = np.nan_to_num(y_true, nan=0).astype(int)
    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)

    if threshold is None:
        # Select threshold from a precision-recall tradeoff when not fixed.
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
            if len(thresholds) == 0:
                threshold = 0.5
            else:
                f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
                weighted_f1 = f1_scores * (precisions[:-1] ** 0.3)
                threshold = float(thresholds[int(np.argmax(weighted_f1))])
        except Exception:
            threshold = 0.5

    y_pred_binary = (y_pred_proba > threshold).astype(int)

    # Compute core classification metrics.
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    # Compute AUC with a safe fallback for degenerate label sets.
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5

    # Unpack confusion matrix and derive additional rates.
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "ppv": ppv,
        "npv": npv,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "raw_confusion_matrix": cm.tolist(),
    }


def find_optimal_threshold_conservative(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Find a conservative probability threshold using weighted F1.

    # Build precision-recall curve for candidate thresholds.
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_indices = thresholds >= 0.1

    if len(valid_indices.nonzero()[0]) == 0:
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        weighted_f1 = f1_scores * (precisions ** 0.3)
        best_idx = int(np.argmax(weighted_f1))
        return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    valid_thresholds = thresholds[valid_indices]
    # Score threshold candidates with precision-weighted F1.
    valid_precisions = precisions[:-1][valid_indices]
    valid_recalls = recalls[:-1][valid_indices]

    f1_scores = 2 * valid_precisions * valid_recalls / (valid_precisions + valid_recalls + 1e-8)
    weighted_f1 = f1_scores * (valid_precisions ** 0.3)
    return float(valid_thresholds[int(np.argmax(weighted_f1))])


def print_detailed_metrics(metrics: dict, fold: int | None = None, epoch: int | None = None) -> None:
    # Print formatted metrics for quick terminal inspection.

    # Build a compact optional prefix for fold and epoch context.
    prefix = f"[Fold {fold}] " if fold else ""
    prefix += f"[Epoch {epoch}] " if epoch else ""

    print(f"\n{prefix}DETAILED METRICS")
    print("=" * 60)
    print(f"Threshold:   {metrics['threshold']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"PPV:         {metrics['ppv']:.4f}")
    print(f"NPV:         {metrics['npv']:.4f}")


def evaluate_without_tta(model, val_loader, device: torch.device):
    # Run deterministic validation without test-time augmentation.

    # Switch to eval mode and disable gradient tracking.
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.float().to(device)

            with autocast("cuda" if device.type == "cuda" else "cpu"):
                # Execute one forward pass per batch without augmentation.
                outputs = model(videos)
                preds = torch.sigmoid(outputs)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    return np.array(all_preds), np.array(all_labels), val_loader.dataset.dataframe.reset_index(drop=True)


def generate_gradcam_for_tensor(model, video_tensor: torch.Tensor, device: torch.device) -> list[np.ndarray]:
    # Generate per-frame Grad-CAM heatmaps for a single video tensor.

    # Use eval mode to keep inference behavior deterministic.
    model.eval()

    target_layer = None
    # Select the last convolutional layer in the CNN backbone.
    for module in model.cnn.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:
        raise RuntimeError("No Conv2d found in model.cnn for Grad-CAM.")

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, outputs):
        activations.clear()
        activations.append(outputs)

        def save_grad(grad):
            gradients.clear()
            gradients.append(grad)

        outputs.register_hook(save_grad)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    # Iterate frames and compute Grad-CAM map per frame.
    heatmaps: list[np.ndarray] = []
    time_steps, _, height, width = video_tensor.shape

    with torch.backends.cudnn.flags(enabled=False):
        with torch.set_grad_enabled(True):
            for t in range(time_steps):
                one_step = video_tensor[t].unsqueeze(0).unsqueeze(0).to(device).float()

                model.zero_grad(set_to_none=True)
                logits = model(one_step)
                if logits.ndim > 1:
                    logits = logits.squeeze()
                score = logits
                score.backward(retain_graph=False)

                if not activations or not gradients:
                    heatmaps.append(np.zeros((height, width), dtype=np.float32))
                    continue

                activation = activations[0]
                gradient = gradients[0]
                # Compute channel weights from spatially pooled gradients.
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=False))

                if cam.shape[-2:] != (height, width):
                    cam = F.interpolate(cam.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)

                cam = cam.squeeze(0)
                # Normalize map to [0, 1] for visualization.
                cam = cam - cam.min()
                cam = cam / cam.max().clamp(min=1e-6)
                heatmaps.append(cam.detach().cpu().float().numpy())

    hook_handle.remove()
    return heatmaps


def save_gradcam_video(video_tensor: torch.Tensor, heatmaps: list[np.ndarray], output_path: str, fps: int = 5) -> None:
    # Overlay Grad-CAM heatmaps on frames and write an MP4 video.

    # Ensure destination directory exists before writing video.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    time_steps, _, height, width = video_tensor.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for t in range(time_steps):
        # Convert tensor frame to uint8 image.
        frame = video_tensor[t].permute(1, 2, 0).cpu().numpy()
        frame = np.clip((frame * 255), 0, 255).astype(np.uint8)

        heatmap = (heatmaps[t] * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if heatmap.ndim == 2:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        if frame.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # Blend original frame with heatmap overlay.
        blended = cv2.addWeighted(frame.astype(np.float32), 0.6, heatmap.astype(np.float32), 0.4, 0.0)
        writer.write(np.clip(blended, 0, 255).astype(np.uint8))

    writer.release()
