# Model and loss components for stable video classification.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class GradientStabilizer:
    # Clip gradients and detect unstable updates.

    def __init__(self, model: nn.Module, max_norm: float = 1.0, patience: int = 2) -> None:
        # Persist model reference and stability thresholds.
        self.model = model
        self.max_norm = max_norm
        self.patience = patience
        self.unstable_steps = 0
        self.grad_norm_log: list[float] = []

    def check_and_clip_gradients(self, logit_stats: dict[str, float] | None = None) -> tuple[bool, str | None]:
        # Clip gradient norms and return instability status.

        # Compute global L2 norm across all parameter gradients.
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        # Track recent norms for optional debugging and trend checks.
        self.grad_norm_log.append(total_norm)
        if len(self.grad_norm_log) > 10:
            self.grad_norm_log.pop(0)

        if logit_stats is not None:
            # Flag unstable logits before optimizer step.
            if abs(logit_stats["max"]) > 15.0 or abs(logit_stats["min"]) > 15.0:
                self.unstable_steps += 1
                if self.unstable_steps >= self.patience:
                    return True, "LOGIT_EXPLOSION"
            else:
                self.unstable_steps = 0

        if total_norm > self.max_norm:
            # Apply conservative clipping when gradients exceed target norm.
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.unstable_steps += 1
            if self.unstable_steps >= self.patience:
                return True, "GRAD_EXPLOSION"
        else:
            self.unstable_steps = 0

        return False, None


class StabilizedFocalLoss(nn.Module):
    # Compute binary focal loss with conservative logit stabilization.

    def __init__(
        self,
        alpha: float = 0.48,
        gamma: float = 1.0,
        label_smoothing: float = 0.02,
        temperature: float = 2.0,
        max_logit: float = 10.0,
    ) -> None:
        super().__init__()
        # Persist focal-loss and stabilization hyperparameters.
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        self.max_logit = max_logit
        self.loss_history: list[float] = []
        self.logit_stats: dict[str, list[float]] = {"min": [], "max": [], "mean": [], "std": []}

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Return stabilized focal loss on logits.

        # Cache logit statistics for stability monitoring.
        with torch.no_grad():
            self.logit_stats["min"].append(inputs.min().item())
            self.logit_stats["max"].append(inputs.max().item())
            self.logit_stats["mean"].append(inputs.mean().item())
            self.logit_stats["std"].append(inputs.std().item())
            if len(self.logit_stats["min"]) > 50:
                for key in self.logit_stats:
                    self.logit_stats[key].pop(0)

        # Apply temperature scaling and hard clipping before BCE logits.
        logits = torch.clamp(inputs / self.temperature, -self.max_logit, self.max_logit)
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute element-wise BCE and focal reweighting terms.
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_factor = (1 - p_t).pow(self.gamma)

        loss = (alpha_t * focal_factor * bce).mean()
        # Add a small regularization term to discourage extreme logits.
        regularization = 0.001 * torch.mean(logits.abs())
        total_loss = loss + regularization

        self.loss_history.append(float(loss.item()))
        # Keep a bounded window for stability checks.
        if len(self.loss_history) > 20:
            self.loss_history.pop(0)

        return total_loss

    def is_loss_stable(self) -> bool:
        # Return whether the recent loss window is stable.

        if len(self.loss_history) < 10:
            return True
        return float(np.std(self.loss_history[-10:])) < 0.5

    def get_logit_stats(self) -> dict[str, float] | None:
        # Return smoothed logit statistics from recent batches.

        if not self.logit_stats["min"]:
            return None
        return {
            "min": float(np.mean(self.logit_stats["min"][-10:])),
            "max": float(np.mean(self.logit_stats["max"][-10:])),
            "mean": float(np.mean(self.logit_stats["mean"][-10:])),
            "std": float(np.mean(self.logit_stats["std"][-10:])),
        }

    def update_epoch(self) -> None:
        # Provide epoch-hook compatibility for training loops.
        pass


class StabilizedCNNLSTM(nn.Module):
    # Combine EfficientNet, BiLSTM, and attention for video classification.

    def __init__(self, num_classes: int = 1, hidden_size: int = 256, num_layers: int = 2, dropout_rate: float = 0.3) -> None:
        super().__init__()

        # Build EfficientNet backbone and freeze early layers for stability.
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for index, (_, param) in enumerate(backbone.named_parameters()):
            if index < 25:
                param.requires_grad = False

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 1280
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.feature_projection = nn.Linear(self.feature_dim, hidden_size)
        self.feature_dropout = nn.Dropout(dropout_rate * 0.3)

        # Define sequence encoder and attention head.
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.lstm_layer_norm = nn.LayerNorm(hidden_size * 2)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=dropout_rate * 0.3,
            batch_first=True,
        )

        # Build classification head with normalization and dropout.
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(hidden_size // 2, num_classes),
        )

        self._init_weights_conservative()

    def _init_weights_conservative(self) -> None:
        # Initialize trainable layers with conservative gains.

        # Apply initializer rules per module type.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.7)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param, gain=0.7)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run forward pass on a [B, T, C, H, W] batch.

        # Flatten time into batch for frame-level CNN feature extraction.
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input to CNN contains NaN or Inf before EfficientNet")

        # Extract frame embeddings using the CNN backbone.
        features = self.cnn(x)
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("NaN/Inf in CNN features before normalization")

        # Clean and normalize features before sequence modeling.
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = self.feature_norm(features)
        features = self.feature_projection(features)
        features = torch.nan_to_num(features, nan=0.0)
        features = self.feature_dropout(features)

        # Encode temporal context and apply self-attention.
        features = features.view(batch_size, time_steps, -1)
        lstm_out, _ = self.lstm(features)
        lstm_out = torch.nan_to_num(lstm_out, nan=0.0)
        lstm_out = self.lstm_layer_norm(lstm_out)

        attended_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        attended_out = torch.nan_to_num(attended_out, nan=0.0)
        pooled = torch.mean(attended_out, dim=1)

        # Pool temporal tokens and produce a single logit per sample.
        output = self.classifier(pooled)
        output = torch.nan_to_num(output, nan=0.0)
        return output.squeeze(-1)
