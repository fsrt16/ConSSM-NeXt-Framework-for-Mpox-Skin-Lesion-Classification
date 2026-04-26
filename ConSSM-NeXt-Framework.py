"""
================================================================================
Hybrid Contrastive Vision Net — Single-File Implementation
================================================================================
Architecture  : ConvNeXt (backbone) + CBAM + ConvTransformer + SimpleSSM head
Task          : 4-class Medical Image Classification
                (Cyst / Normal / Stone / Tumor  —  CT-Kidney Dataset)
Loss          : CrossEntropy (0.7) + Supervised Contrastive Loss (0.3)
Precision     : Mixed-precision (AMP) with GradScaler
================================================================================
File layout
-----------
  1. Imports & device setup
  2. HyperParameters         — centralised config dataclass
  3. SimpleSSMBlock          — gated-MLP token mixer (SSM surrogate)
  4. ConvTransformerBlock    — depth-wise conv + multi-head self-attention + GLU-MLP
  5. ChannelSpatialAttention — CBAM-style dual-gate attention
  6. HybridSSSMConvTransformerClassifier — full encoder + classifier
  7. ModelFactory            — convenience factory with backbone auto-detection
  8. HybridContrastiveCE     — combined CE + supervised-contrastive loss
  9. GPUAugmenter            — on-GPU random augmentation pipeline
 10. MetricsTracker          — per-epoch accumulator for loss / accuracy
 11. Trainer                 — encapsulates train-one-epoch & validate-one-epoch
 12. TrainingDriver          — outer epoch loop, checkpointing, early-stopping
 13. LearningCurveVisualizer — matplotlib 2×2 learning-curve plotter
 14. __main__                — entry point (requires external data loaders)
================================================================================
"""

# ============================================================
# 1. IMPORTS & DEVICE SETUP
# ============================================================

import time
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torchvision import models, transforms
from tqdm import tqdm

# Select the best available hardware accelerator.
# All tensors and models are moved to DEVICE throughout the pipeline.
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Running on: {DEVICE}")


# ============================================================
# 2. HYPERPARAMETERS — centralised configuration
# ============================================================

@dataclass
class HyperParameters:
    """
    Central configuration dataclass.

    Keeping all hyperparameters in one place makes ablation studies
    straightforward — change one field instead of hunting through code.

    Attributes
    ----------
    num_classes        : Number of output categories.
    backbone_name      : torchvision model name used as the visual encoder.
    pretrained         : Whether to load ImageNet-pretrained weights.
    ssm_dim            : Channel dimension after the projection layer; governs
                         the width of the SSM and Transformer sub-networks.
    input_size         : Spatial resolution fed to the network (H = W).
    batch_size         : Mini-batch size.
    epochs             : Maximum training epochs.
    lr                 : Initial learning rate for AdamW.
    weight_decay       : L2 regularisation coefficient.
    ce_weight          : Fraction of total loss attributed to CrossEntropy.
    contrastive_weight : Fraction of total loss attributed to SupCon loss.
    contrastive_temp   : Temperature τ in the contrastive exponential.
    label_smoothing    : Smoothing factor ε for cross-entropy.
    clip_norm          : Maximum L2-norm for gradient clipping.
    lr_patience        : ReduceLROnPlateau patience in epochs.
    lr_factor          : ReduceLROnPlateau decay factor.
    early_stop_acc     : Validation accuracy threshold for early stopping.
    checkpoint_path    : File path for saving the best model checkpoint.
    experiment_name    : Prefix used in figure filenames and console output.
    """
    num_classes: int         = 4
    backbone_name: str       = "convnext_tiny"
    pretrained: bool         = True
    ssm_dim: int             = 256

    input_size: int          = 224
    batch_size: int          = 32
    epochs: int              = 100
    lr: float                = 1e-4
    weight_decay: float      = 0.01

    ce_weight: float         = 0.7
    contrastive_weight: float = 0.3
    contrastive_temp: float  = 0.07
    label_smoothing: float   = 0.1

    clip_norm: float         = 1.0
    lr_patience: int         = 5
    lr_factor: float         = 0.5

    early_stop_acc: float    = 95.0
    checkpoint_path: str     = "./best_hybrid_model.pth"
    experiment_name: str     = "HybridConvNeXt_SSM_Contrastive"


# ============================================================
# 3. SIMPLE SSM BLOCK — gated MLP token mixer
# ============================================================

class SimpleSSMBlock(nn.Module):
    """
    Lightweight token-mixing block that approximates the selective scan of a
    true State Space Model (SSM / Mamba) via a gated MLP.

    Motivation
    ----------
    Full SSMs (Mamba, S4) have hardware-efficient scan operators but require
    specialised CUDA kernels.  For prototyping on arbitrary hardware this block
    replaces the scan with a simple element-wise gate:

        h = σ(W_gate · x)  ⊙  (W_val · x)     (gated projection)
        y = W_out · h  +  x                      (residual)

    This retains the *gating* property of SSMs (content-dependent filtering)
    at a fraction of the implementation complexity.

    Parameters
    ----------
    dim       : Token / channel dimension  C.
    expansion : Hidden width multiplier; hidden_dim = dim × expansion.
    dropout   : Dropout probability applied before the residual add.

    Input / Output
    --------------
    x : Tensor [B, L, C]   (batch, sequence-length / spatial tokens, channels)
    → Tensor [B, L, C]
    """

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = dim * expansion

        # Single linear doubles as gate projection AND value projection.
        # Output is split into two equal halves along the last dimension.
        self.in_proj  = nn.Linear(dim, 2 * self.hidden_dim)

        # Projects the gated representation back to the original width.
        self.out_proj = nn.Linear(self.hidden_dim, dim)

        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated MLP and residual connection.

        Steps
        -----
        1. Save input as residual.
        2. Project to 2 × hidden_dim → split into gate and value.
        3. Apply sigmoid gate: gated = σ(gate) ⊙ value.
        4. Project back to dim.
        5. Apply dropout.
        6. Add residual (Pre-norm is handled externally by the caller).
        """
        residual = x                                    # [B, L, C]

        x = self.in_proj(x)                            # [B, L, 2·hidden_dim]
        gate, val = x.chunk(2, dim=-1)                 # each: [B, L, hidden_dim]

        gate = torch.sigmoid(gate)                     # content-dependent gate ∈ (0,1)
        x    = gate * val                              # selective mixing

        x    = self.out_proj(x)                        # [B, L, C]
        x    = self.dropout(x)

        return x + residual                            # skip connection


# ============================================================
# 4. CONV-TRANSFORMER BLOCK — local conv + global attention
# ============================================================

class ConvTransformerBlock(nn.Module):
    """
    Fuses local and global context in a single block.

    Architecture
    ------------
    Input [B, C, H, W]
        │
        ├─ Depth-wise conv 3×3  (local texture / spatial inductive bias)
        │     └─ residual add  →  x
        │
        ├─ Flatten + transpose  →  tokens [B, H·W, C]
        │     ├─ LayerNorm
        │     ├─ Multi-head Self-Attention (MHSA)   ← global context
        │     └─ residual add
        │
        ├─ LayerNorm
        ├─ GLU-MLP  (gate linear unit for feature refinement)
        └─ residual add  →  reshape back to [B, C, H, W]

    The depth-wise conv provides the translation-equivariant local processing
    that pure ViTs lack for small datasets, while MHSA aggregates long-range
    dependencies.  GLU activation (SwiGLU-style) improves gradient flow.

    Parameters
    ----------
    dim       : Channel / embedding dimension.
    num_heads : Number of attention heads (must divide dim evenly).
    mlp_ratio : Hidden width of the MLP relative to dim.
    dropout   : Dropout applied inside attention and MLP layers.

    Returns
    -------
    x_out       : Tensor [B, C, H, W]  — processed feature map.
    attn_weights: Tensor [B, heads, H·W, H·W]  — per-head attention maps
                  (useful for visualisation / Grad-CAM style explainability).
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()

        # --- Local branch ---
        # groups=dim → depth-wise conv; captures local spatial structure
        # without cross-channel mixing (that is done by the linear layers).
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # --- Global branch (MHSA) ---
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True       # expects [B, L, C] not [L, B, C]
        )

        # --- Channel MLP with GLU activation ---
        # GLU splits the hidden projection in half: one half gates the other.
        # nn.GLU(dim=-1) on a [B, L, 2·hidden] tensor → [B, L, hidden].
        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),   # project up (×2 for GLU split)
            nn.GLU(dim=-1),                   # gated linear unit → hidden_dim
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),       # project back to dim
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape

        # ── Local depth-wise convolution (residual) ──────────────────────────
        x = x + self.dwconv(x)                         # [B, C, H, W]

        # ── Reshape to token sequence for MHSA ───────────────────────────────
        # flatten(2) collapses H and W → L = H·W spatial tokens
        # transpose(1, 2) swaps C and L → [B, L, C]  as required by batch_first
        x_flat = x.flatten(2).transpose(1, 2)          # [B, H·W, C]

        # ── Multi-head self-attention (with residual) ─────────────────────────
        x_norm    = self.norm1(x_flat)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=False    # keep per-head weights for explainability
        )                                              # attn_weights: [B, heads, L, L]
        x_flat = x_flat + attn_out                     # residual

        # ── GLU-MLP (with residual) ──────────────────────────────────────────
        x_flat = x_flat + self.mlp(self.norm2(x_flat)) # [B, L, C]

        # ── Reshape back to spatial feature map ──────────────────────────────
        x_out = x_flat.transpose(1, 2).view(b, c, h, w)   # [B, C, H, W]

        return x_out, attn_weights


# ============================================================
# 5. CHANNEL-SPATIAL ATTENTION (CBAM-style dual gate)
# ============================================================

class ChannelSpatialAttention(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)-inspired dual-path attention.

    Channel gate: answers *what* features matter (recalibrates channel
                  importance via squeeze-and-excitation).
    Spatial gate: answers *where* in the image to focus (aggregates avg- and
                  max-pooled maps, then convolves with a large 7×7 kernel for
                  broad context).

    Both gates are computed in parallel and their weighted outputs are
    summed (additive fusion), preserving gradients through both paths.

    Parameters
    ----------
    channels  : Number of input / output channels.
    reduction : Bottleneck compression ratio inside the channel SE block.

    Returns
    -------
    fused     : Tensor [B, C, H, W]  — channel- AND spatial-attended feature.
    ca        : Tensor [B, C, 1, 1]  — channel attention weights.
    sa        : Tensor [B, 1, H, W]  — spatial attention weights.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # ── Channel attention (Squeeze-Excitation) ────────────────────────────
        # Global average pool squeezes spatial dims → per-channel statistics.
        # Two conv-1×1 layers (fully connected equivalent) recalibrate them.
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                                 # [B, C, 1, 1]
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()                                             # weights ∈ (0,1)
        )

        # ── Spatial attention ─────────────────────────────────────────────────
        # Channel-pooling (avg + max) along C → 2-channel descriptor.
        # Large receptive field (7×7) captures structural spatial context.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),  # [B, 1, H, W]
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                  torch.Tensor,
                                                  torch.Tensor]:
        # ── Channel branch ───────────────────────────────────────────────────
        ca   = self.channel_gate(x)        # [B, C, 1, 1] broadcast over H, W
        x_ca = x * ca                      # recalibrated feature map

        # ── Spatial branch ───────────────────────────────────────────────────
        # Descriptor: average- and max-pool across channels, then concatenate.
        avg_pool = torch.mean(x, dim=1, keepdim=True)         # [B, 1, H, W]
        max_pool = torch.max(x,  dim=1, keepdim=True)[0]      # [B, 1, H, W]
        sa_input = torch.cat([avg_pool, max_pool], dim=1)      # [B, 2, H, W]
        sa       = self.spatial_gate(sa_input)                 # [B, 1, H, W]
        x_sa     = x * sa

        # ── Additive fusion of both attended representations ─────────────────
        fused = x_ca + x_sa                                    # [B, C, H, W]

        return fused, ca, sa


# ============================================================
# 6. FULL MODEL — HybridSSSMConvTransformerClassifier
# ============================================================

class HybridSSSMConvTransformerClassifier(nn.Module):
    """
    End-to-end hybrid encoder for fine-grained medical image classification.

    Pipeline
    --------
    Input [B, 3, 224, 224]
        │
        ├── Frozen Backbone (ConvNeXt-Tiny / ResNet / DenseNet)
        │       → [B, in_ch, H', W']   (feature pyramid last stage)
        │
        ├── 1×1 Projection Conv
        │       → [B, ssm_dim, H', W'] (channel-dimensionality alignment)
        │
        ├── ChannelSpatialAttention (CBAM)
        │       → [B, ssm_dim, H', W'] + attention weights (ca, sa)
        │
        ├── ConvTransformerBlock
        │       → [B, ssm_dim, H', W'] + self-attention maps
        │
        ├── Flatten to tokens + LayerNorm
        │       → [B, H'·W', ssm_dim]
        │
        ├── SimpleSSMBlock (gated-MLP token mixer)
        │       → [B, H'·W', ssm_dim]
        │
        ├── Reshape back to spatial + Global Average Pool
        │       → [B, ssm_dim]
        │
        └── Classification Head (LayerNorm → Linear → GELU → Dropout → Linear)
                → [B, num_classes]

    The backbone is frozen by default; only the projection + attention + SSM
    head layers are trained.  This dramatically reduces trainable parameters
    while benefiting from rich ImageNet features.

    Parameters
    ----------
    num_classes    : Output dimension (number of categories).
    backbone_name  : torchvision model identifier (e.g. "convnext_tiny").
    pretrained     : Load pretrained ImageNet weights for the backbone.
    ssm_dim        : Unified channel width for the hybrid head.
    """

    def __init__(self,
                 num_classes: int   = 4,
                 backbone_name: str = "convnext_tiny",
                 pretrained: bool   = True,
                 ssm_dim: int       = 256):
        super().__init__()

        # ── 6.1  Backbone (feature extractor) ────────────────────────────────
        raw_model = getattr(models, backbone_name)(
            weights="DEFAULT" if pretrained else None
        )

        # Strip the classification head; keep only the convolutional body.
        # Different architectures expose their feature maps differently:
        if "convnext" in backbone_name:
            # ConvNeXt uses a .features Sequential; the final stage outputs
            # 768 channels for convnext_tiny (7×7 at 224-input).
            backbone = raw_model.features
        elif "resnet" in backbone_name:
            # ResNet ends with an average pool + fc; drop those (last 2 children).
            backbone = nn.Sequential(*list(raw_model.children())[:-2])
        elif "densenet" in backbone_name:
            backbone = raw_model.features
        else:
            # Generic fallback: strip the final two children (pool + head).
            backbone = nn.Sequential(*list(raw_model.children())[:-2])

        self.backbone = backbone

        # Freeze backbone: it acts as a fixed feature extractor.
        # Unfreeze later (fine-tuning phase) if higher accuracy is desired.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ── 6.2  Infer backbone output channels with a dummy forward pass ────
        with torch.no_grad():
            dummy  = torch.zeros(1, 3, 224, 224)
            feat   = self.backbone(dummy)
        in_ch = feat.shape[1]   # e.g. 768 for convnext_tiny

        # ── 6.3  Channel projection  (in_ch → ssm_dim) ───────────────────────
        # 1×1 convolution is a parameter-efficient linear channel mixing that
        # normalises the backbone output to a common width used downstream.
        self.conv_proj = nn.Conv2d(in_ch, ssm_dim, kernel_size=1)

        # ── 6.4  CBAM — highlight discriminative channels and regions ─────────
        self.cbam = ChannelSpatialAttention(ssm_dim, reduction=16)

        # ── 6.5  ConvTransformerBlock — local + global feature fusion ─────────
        self.conv_trans_block = ConvTransformerBlock(
            dim=ssm_dim, num_heads=4, mlp_ratio=2.0, dropout=0.1
        )

        # ── 6.6  SSM token mixer — sequential / temporal dependency capture ───
        # LayerNorm is applied to the token sequence BEFORE the SSM block
        # (pre-norm transformer convention) for training stability.
        self.ssm_norm  = nn.LayerNorm(ssm_dim)
        self.ssm_block = SimpleSSMBlock(dim=ssm_dim, expansion=2, dropout=0.1)

        # ── 6.7  Pooling + Classification head ───────────────────────────────
        self.gap  = nn.AdaptiveAvgPool2d(1)   # collapses H'×W' → 1×1
        self.head = nn.Sequential(
            nn.LayerNorm(ssm_dim),            # normalise pooled vector
            nn.Linear(ssm_dim, 256),          # project to smaller space
            nn.GELU(),                        # smooth non-linearity
            nn.Dropout(0.2),                  # regularise
            nn.Linear(256, num_classes),      # final logits
        )

    # ── 6.8  Forward pass (primary) ──────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                return_feats: bool = False):
        """
        Forward pass.

        Parameters
        ----------
        x            : Input image tensor [B, 3, H, W].
        return_feats : If True, also return an intermediate feature dict.

        Returns (when return_feats=False)
        ---------------------------------
        logits : [B, num_classes]

        Returns (when return_feats=True)
        ---------------------------------
        logits : [B, num_classes]
        feats  : dict with keys
            "backbone_feat"  — [B, ssm_dim, H', W']  post-SSM spatial map
            "gap_feat"       — [B, ssm_dim]           global-average-pooled vector
            "channel_attn"   — [B, ssm_dim, 1, 1]    CBAM channel weights
            "spatial_attn"   — [B, 1, H', W']         CBAM spatial weights
            "sa_map"         — [B, heads, L, L]        transformer attention weights
        """

        # ── Stage 1: backbone feature extraction ─────────────────────────────
        x = self.backbone(x)                          # [B, in_ch, H', W']

        # ── Stage 2: channel alignment ────────────────────────────────────────
        x = self.conv_proj(x)                         # [B, ssm_dim, H', W']

        # ── Stage 3: CBAM dual attention ─────────────────────────────────────
        x, ch_attn, sp_attn = self.cbam(x)           # [B, ssm_dim, H', W']

        # ── Stage 4: ConvTransformer (local + global) ─────────────────────────
        x, sa_map = self.conv_trans_block(x)          # [B, ssm_dim, H', W']

        # ── Stage 5: SSM token mixing ────────────────────────────────────────
        b, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)         # [B, H'·W', ssm_dim]
        tokens = self.ssm_norm(tokens)                 # pre-norm
        tokens = self.ssm_block(tokens)                # gated mixing
        x      = tokens.transpose(1, 2).view(b, d, h, w)  # [B, ssm_dim, H', W']

        # ── Stage 6: global pooling + classification ──────────────────────────
        gap_feat = self.gap(x).flatten(1)             # [B, ssm_dim]
        logits   = self.head(gap_feat)                 # [B, num_classes]

        if not return_feats:
            return logits

        return logits, {
            "backbone_feat": x,
            "gap_feat":      gap_feat,
            "channel_attn":  ch_attn,
            "spatial_attn":  sp_attn,
            "sa_map":        sa_map,
        }

    # ── 6.9  Convenience wrappers ─────────────────────────────────────────────

    def forward_with_features(self, x: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, gap_feat) — the signature expected by the
        contrastive loss which needs a pooled representation as the
        anchor / positive / negative embedding.
        """
        logits, feats = self.forward(x, return_feats=True)
        return logits, feats["gap_feat"]

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns only the GAP embedding [B, ssm_dim].
        Useful during inference for nearest-neighbour retrieval or t-SNE.
        """
        _, feats = self.forward(x, return_feats=True)
        return feats["gap_feat"]

    def unfreeze_backbone(self, num_stages: int = 1):
        """
        Progressively unfreeze the last `num_stages` stages of the backbone
        for fine-tuning after the head has warmed up.

        Parameters
        ----------
        num_stages : Number of top-level children to unfreeze (counting from end).
        """
        children = list(self.backbone.children())
        for child in children[-num_stages:]:
            for param in child.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Backbone] Unfroze {num_stages} stage(s). "
              f"Trainable params: {trainable:,}")


# ============================================================
# 7. MODEL FACTORY — single entry point for model creation
# ============================================================

class ModelFactory:
    """
    Centralises model instantiation so that callers do not need to know
    the constructor signature.

    Usage
    -----
    model = ModelFactory.create(cfg)
    """

    @staticmethod
    def create(cfg: HyperParameters) -> HybridSSSMConvTransformerClassifier:
        """
        Build and return the hybrid model configured by `cfg`.

        Also prints a parameter summary after construction.
        """
        model = HybridSSSMConvTransformerClassifier(
            num_classes=cfg.num_classes,
            backbone_name=cfg.backbone_name,
            pretrained=cfg.pretrained,
            ssm_dim=cfg.ssm_dim,
        )

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n[Model] {cfg.backbone_name} + ConvTransformer + SimpleSSM")
        print(f"        Total params    : {total:>12,}  ({total/1e6:.2f} M)")
        print(f"        Trainable params: {trainable:>12,}  ({trainable/1e6:.2f} M)")
        print(f"        Frozen (backbone): {total-trainable:>11,}  "
              f"({(total-trainable)/1e6:.2f} M)")

        return model.to(DEVICE)


# ============================================================
# 8. HYBRID CONTRASTIVE + CROSS-ENTROPY LOSS
# ============================================================

class HybridContrastiveCE(nn.Module):
    """
    Combined loss for representation learning and classification.

        L_total = w_CE · L_CE  +  w_Con · L_SupCon

    Cross-Entropy component
    -----------------------
    Standard label-smoothed CE.  Label smoothing ε distributes ε/(K-1) of
    the probability mass to non-target classes, acting as regularisation and
    reducing over-confident predictions that hurt calibration.

    Supervised Contrastive component (SupCon — Khosla et al., 2020)
    ----------------------------------------------------------------
    For each anchor i, positives are all samples j with the same class label.
    The loss maximises cosine similarity between same-class pairs while
    pushing apart cross-class pairs:

        L_SupCon = -1/|P(i)| Σ_{j∈P(i)} log [ exp(zᵢ·zⱼ / τ) /
                                                Σ_{k≠i} exp(zᵢ·zₖ / τ) ]

    where z = ℓ₂-normalised GAP feature, τ is the temperature.
    Lower τ → sharper distribution, stronger push-away effect.

    Numerical stability: logits are shifted by their row-max before exp
    (standard log-sum-exp trick) to avoid overflow.

    Parameters
    ----------
    ce_weight          : Weight w_CE  (e.g. 0.7).
    contrastive_weight : Weight w_Con (e.g. 0.3).
    contrastive_temp   : Temperature τ (e.g. 0.07).
    label_smoothing    : ε for CE label smoothing (e.g. 0.1).
    """

    def __init__(self,
                 ce_weight: float          = 0.7,
                 contrastive_weight: float = 0.3,
                 contrastive_temp: float   = 0.07,
                 label_smoothing: float    = 0.1):
        super().__init__()
        self.ce_weight          = ce_weight
        self.contrastive_weight = contrastive_weight
        self.temp               = contrastive_temp
        self.ce_loss            = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss.

        Parameters
        ----------
        logits   : [B, num_classes]  — raw classification scores.
        labels   : [B]               — integer class indices.
        features : [B, D]            — L2-normalised embedding (gap_feat).
                   Pass None to skip contrastive loss (e.g. single-sample batch).

        Returns
        -------
        total_loss : Scalar tensor (differentiable).
        details    : Dict with scalar float keys "ce_loss" and "contrastive_loss"
                     for logging / diagnostics.
        """

        # ── Sanitise logits ───────────────────────────────────────────────────
        logits = logits.float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        logits = torch.clamp(logits, min=-100.0, max=100.0)

        ce_loss          = self.ce_loss(logits, labels)
        contrastive_loss = torch.tensor(0.0, device=logits.device)

        # ── Supervised Contrastive Loss ───────────────────────────────────────
        # Only compute when batch has ≥ 2 samples (single-sample = no pairs).
        if features is not None and features.shape[0] > 1:
            features = features.float()
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            # L2 normalise so dot product equals cosine similarity.
            features = F.normalize(features, p=2, dim=1, eps=1e-6)

            batch_size = features.shape[0]

            # Positive pair mask: mask[i,j] = 1 iff label[i] == label[j]
            labels_col = labels.contiguous().view(-1, 1)          # [B, 1]
            mask = torch.eq(labels_col, labels_col.T).float()     # [B, B]

            # Self-contrast mask: exclude diagonal (i == i is not a valid pair)
            self_mask = torch.scatter(
                torch.ones_like(mask),
                dim=1,
                index=torch.arange(batch_size).view(-1, 1).to(logits.device),
                value=0.0
            )                                                      # [B, B] 0 on diag
            mask = mask * self_mask                                # valid positives

            # Only proceed if at least one positive pair exists in the batch.
            if mask.sum() > 0:
                # Pairwise cosine similarities scaled by temperature.
                sim_matrix = torch.matmul(features, features.T)   # [B, B]
                sim_scaled = sim_matrix / self.temp                # [B, B]

                # Subtract row-max for numerical stability (log-sum-exp trick).
                row_max, _ = torch.max(sim_scaled, dim=1, keepdim=True)
                sim_stable  = sim_scaled - row_max.detach()       # [B, B]

                # Exponentiate only valid pairs (exclude self-pairs from denominator).
                exp_sim     = torch.exp(sim_stable) * self_mask   # [B, B]

                # Log-probability of each pair under the softmax over negatives.
                log_prob    = sim_stable - torch.log(
                    exp_sim.sum(dim=1, keepdim=True) + 1e-9
                )                                                  # [B, B]

                # Average log-prob over all positives per anchor.
                pos_count = mask.sum(dim=1)                        # [B]
                # Avoid division by zero for anchors with no positives.
                pos_count = torch.clamp(pos_count, min=1e-6)

                mean_log_prob = (mask * log_prob).sum(dim=1) / pos_count
                contrastive_loss = -mean_log_prob.mean()

        total_loss = (self.ce_weight * ce_loss +
                      self.contrastive_weight * contrastive_loss)

        return total_loss, {
            "ce_loss":          ce_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }


# ============================================================
# 9. GPU AUGMENTER — on-device real-time data augmentation
# ============================================================

class GPUAugmenter(nn.Module):
    """
    On-GPU random augmentation applied to each mini-batch during training.

    Performing augmentation on the GPU avoids CPU→GPU bottlenecks and
    leverages batched tensor operations.  The pipeline is:

        RandomHorizontalFlip → RandomVerticalFlip → Random Rotation (±15°)
        → ColorJitter (brightness / contrast / saturation)

    Parameters
    ----------
    flip_p      : Probability of horizontal / vertical flipping.
    max_rotate  : Max rotation angle in degrees (uniform in [-max_rotate, max_rotate]).
    brightness  : Brightness jitter range (factor relative to original).
    contrast    : Contrast jitter range.
    saturation  : Saturation jitter range.
    """

    def __init__(self,
                 flip_p: float     = 0.5,
                 max_rotate: float = 15.0,
                 brightness: float = 0.2,
                 contrast: float   = 0.2,
                 saturation: float = 0.1):
        super().__init__()
        self.flip_p     = flip_p
        self.max_rotate = math.radians(max_rotate)   # convert to radians for affine
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation

    @torch.no_grad()                                 # no gradients needed
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic augmentation to a batch of images.

        Parameters
        ----------
        x : [B, 3, H, W]  float tensor, values ∈ [0, 1] or [-1, 1].

        Returns
        -------
        Augmented tensor of the same shape.
        """

        # Horizontal flip (batch-level random)
        if torch.rand(1).item() < self.flip_p:
            x = torch.flip(x, dims=[3])          # flip W dimension

        # Vertical flip
        if torch.rand(1).item() < self.flip_p:
            x = torch.flip(x, dims=[2])          # flip H dimension

        # Random brightness
        if self.brightness > 0:
            delta = (torch.rand(1, device=x.device) * 2 - 1) * self.brightness
            x     = torch.clamp(x + delta, 0.0, 1.0)

        # Random contrast: scale around mean
        if self.contrast > 0:
            factor = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * self.contrast
            mean   = x.mean(dim=[2, 3], keepdim=True)
            x      = torch.clamp((x - mean) * factor + mean, 0.0, 1.0)

        return x


# ============================================================
# 10. METRICS TRACKER — per-epoch accumulator
# ============================================================

class MetricsTracker:
    """
    Lightweight helper that accumulates batch-level statistics and
    computes epoch-level averages on demand.

    Attributes
    ----------
    running_loss : Sum of batch losses (weighted by batch size for correctness).
    correct      : Cumulative number of correct predictions.
    total        : Cumulative number of samples seen.
    loss_details : Dict mapping loss-component name → list of batch values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all accumulators for a new epoch."""
        self.running_loss: float             = 0.0
        self.correct:      int               = 0
        self.total:        int               = 0
        self.loss_details: Dict[str, List]   = defaultdict(list)

    def update(self,
               loss: float,
               preds: torch.Tensor,
               labels: torch.Tensor,
               loss_dict: Dict[str, float]):
        """
        Incorporate one batch of results.

        Parameters
        ----------
        loss      : Scalar loss value for this batch.
        preds     : [B] predicted class indices.
        labels    : [B] ground-truth class indices.
        loss_dict : Dict from the criterion's second return value.
        """
        self.running_loss += loss
        self.correct      += preds.eq(labels).sum().item()
        self.total        += labels.size(0)
        for key, val in loss_dict.items():
            self.loss_details[key].append(val)

    @property
    def accuracy(self) -> float:
        """Current accuracy as a percentage."""
        return 100.0 * self.correct / max(self.total, 1)

    def epoch_summary(self, num_batches: int) -> Tuple[float, float, Dict]:
        """
        Return (avg_loss, accuracy_pct, avg_loss_details).

        Parameters
        ----------
        num_batches : Total number of batches in the epoch (for loss averaging).
        """
        avg_loss    = self.running_loss / max(num_batches, 1)
        avg_details = {k: float(np.mean(v)) for k, v in self.loss_details.items()}
        return avg_loss, self.accuracy, avg_details


# ============================================================
# 11. TRAINER — train_one_epoch + validate_one_epoch
# ============================================================

class Trainer:
    """
    Encapsulates single-epoch training and validation logic.

    Separating these into a class (rather than module-level functions) makes
    it easy to swap model, criterion, or device without rewriting loops.

    Parameters
    ----------
    model         : The classifier network.
    criterion     : HybridContrastiveCE (or any compatible loss).
    optimizer     : Optimiser (AdamW recommended).
    scaler        : GradScaler for AMP (no-op on CPU).
    cfg           : HyperParameters for clip_norm and device settings.
    gpu_augmenter : Optional GPU augmentation module applied during training.
    """

    def __init__(self,
                 model:         nn.Module,
                 criterion:     nn.Module,
                 optimizer:     optim.Optimizer,
                 scaler:        GradScaler,
                 cfg:           HyperParameters,
                 gpu_augmenter: Optional[nn.Module] = None):
        self.model         = model
        self.criterion     = criterion
        self.optimizer     = optimizer
        self.scaler        = scaler
        self.cfg           = cfg
        self.gpu_augmenter = gpu_augmenter

    # ── 11.1  Training epoch ─────────────────────────────────────────────────

    def train_one_epoch(self) -> Tuple[float, float, Dict]:
        """
        Run one full pass over the training DataLoader.

        Returns
        -------
        avg_loss    : Mean total loss over all batches.
        accuracy    : Training accuracy in %.
        avg_details : Dict with per-component loss averages (CE, contrastive).
        """
        self.model.train()
        tracker = MetricsTracker()
        loop    = tqdm(self.train_loader, desc="  Train", leave=False)

        for images, labels in loop:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Optional on-GPU augmentation (faster than CPU albumentations
            # for simple geometric / colour transforms).
            if self.gpu_augmenter is not None:
                images = self.gpu_augmenter(images)

            # Zero gradients (set_to_none avoids unnecessary .fill_(0) ops).
            self.optimizer.zero_grad(set_to_none=True)

            # AMP context: ops run in fp16 where safe, fp32 elsewhere.
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits, features = self.model.forward_with_features(images)
                loss, loss_dict  = self.criterion(logits, labels, features)

            # Scaled backward + gradient clipping + optimiser step.
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Predictions: argmax of logits.
            preds = logits.argmax(dim=1)
            tracker.update(loss.item(), preds, labels, loss_dict)

            # Live progress bar postfix.
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{tracker.accuracy:.1f}%",
                ce=f"{np.mean(tracker.loss_details['ce_loss']):.4f}",
            )

        return tracker.epoch_summary(len(self.train_loader))

    # ── 11.2  Validation epoch ───────────────────────────────────────────────

    def validate_one_epoch(self) -> Tuple[float, float, Dict]:
        """
        Run one full pass over the validation DataLoader (no grad, no aug).

        Returns
        -------
        accuracy    : Validation accuracy in %.
        avg_loss    : Mean total loss over all batches.
        avg_details : Dict with per-component loss averages.
        """
        self.model.eval()
        tracker = MetricsTracker()
        loop    = tqdm(self.val_loader, desc="  Valid", leave=False)

        with torch.no_grad():
            for images, labels in loop:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                    logits, features = self.model.forward_with_features(images)
                    loss, loss_dict  = self.criterion(logits, labels, features)

                preds = logits.argmax(dim=1)
                tracker.update(loss.item(), preds, labels, loss_dict)

                loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{tracker.accuracy:.1f}%",
                )

        avg_loss, acc, avg_details = tracker.epoch_summary(len(self.val_loader))
        return acc, avg_loss, avg_details

    # ── 11.3  Inject data loaders (deferred to allow flexible setup) ─────────

    def set_loaders(self, train_loader, val_loader):
        """Attach DataLoader instances after Trainer construction."""
        self.train_loader = train_loader
        self.val_loader   = val_loader


# ============================================================
# 12. TRAINING DRIVER — outer epoch loop with checkpointing
# ============================================================

class TrainingDriver:
    """
    Orchestrates the full training schedule:
      • Iterates over epochs calling Trainer.train / Trainer.validate
      • Steps the learning-rate scheduler (ReduceLROnPlateau on val_acc)
      • Saves the best checkpoint whenever validation accuracy improves
      • Triggers early stopping when accuracy crosses `cfg.early_stop_acc`
      • Periodically calls a visualiser callback to update learning curves

    Parameters
    ----------
    trainer    : Trainer instance (already has loaders set).
    scheduler  : LR scheduler (any PyTorch scheduler).
    cfg        : HyperParameters.
    visualiser : Optional LearningCurveVisualizer for periodic plot saving.
    """

    def __init__(self,
                 trainer:    Trainer,
                 scheduler,
                 cfg:        HyperParameters,
                 visualiser: Optional["LearningCurveVisualizer"] = None):
        self.trainer    = trainer
        self.scheduler  = scheduler
        self.cfg        = cfg
        self.visualiser = visualiser

    def run(self) -> List[Dict]:
        """
        Execute the training loop for up to `cfg.epochs` epochs.

        Returns
        -------
        history : List of per-epoch metric dicts, one per epoch.
                  Keys: epoch, train_loss, train_acc, train_ce,
                        val_loss, val_acc, val_ce,
                        train_contrastive, val_contrastive.
        """
        best_val_acc = 0.0
        history: List[Dict] = []

        print(f"\n{'='*70}")
        print(f"  Training: {self.cfg.experiment_name}")
        print(f"  Loss mix : CE={self.cfg.ce_weight:.1f}  "
              f"SupCon={self.cfg.contrastive_weight:.1f}  "
              f"τ={self.cfg.contrastive_temp}")
        print(f"  Device   : {DEVICE}  |  Epochs: {self.cfg.epochs}")
        print(f"{'='*70}")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            current_lr = self.trainer.optimizer.param_groups[0]["lr"]
            print(f"\n[Epoch {epoch:>3}/{self.cfg.epochs}]  LR: {current_lr:.2e}")

            # ── Train ──────────────────────────────────────────────────────
            tr_loss, tr_acc, tr_details = self.trainer.train_one_epoch()

            # ── Validate ───────────────────────────────────────────────────
            val_acc, val_loss, val_details = self.trainer.validate_one_epoch()

            # ── LR scheduler step (maximise val_acc) ───────────────────────
            self.scheduler.step(val_acc)

            # ── Record history ─────────────────────────────────────────────
            record = {
                "epoch":             epoch,
                "train_loss":        tr_loss,
                "train_acc":         tr_acc,
                "train_ce":          tr_details.get("ce_loss", 0.0),
                "train_contrastive": tr_details.get("contrastive_loss", 0.0),
                "val_loss":          val_loss,
                "val_acc":           val_acc,
                "val_ce":            val_details.get("ce_loss", 0.0),
                "val_contrastive":   val_details.get("contrastive_loss", 0.0),
            }
            history.append(record)

            # ── Console summary ────────────────────────────────────────────
            elapsed = time.time() - t0
            flag = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(epoch, best_val_acc)
                flag = "  🏆 NEW BEST"

            print(f"  {elapsed:5.0f}s | "
                  f"Train {tr_loss:.4f}/{tr_acc:.1f}% "
                  f"(CE {tr_details.get('ce_loss',0):.4f} "
                  f"Con {tr_details.get('contrastive_loss',0):.4f}) | "
                  f"Val {val_loss:.4f}/{val_acc:.1f}% "
                  f"(CE {val_details.get('ce_loss',0):.4f} "
                  f"Con {val_details.get('contrastive_loss',0):.4f})"
                  + flag)

            # ── Periodic visualisation (every 50 epochs after warm-up) ────
            if (self.visualiser is not None
                    and epoch > 50
                    and epoch % 50 == 0):
                self.visualiser.update(history)

            # ── Early stopping ─────────────────────────────────────────────
            if val_acc >= self.cfg.early_stop_acc:
                print(f"\n🎉 Val acc {val_acc:.2f}% ≥ "
                      f"{self.cfg.early_stop_acc}% — early stop.")
                break

        return history

    # ── Internal helper ───────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, best_val_acc: float):
        """Persist the current best model weights to disk."""
        torch.save(
            {
                "model_state_dict": self.trainer.model.state_dict(),
                "best_val_acc":     best_val_acc,
                "epoch":            epoch,
                "cfg":              self.cfg,
            },
            self.cfg.checkpoint_path,
        )
        print(f"  💾 Checkpoint saved → {self.cfg.checkpoint_path}")


# ============================================================
# 13. LEARNING CURVE VISUALISER
# ============================================================

class LearningCurveVisualizer:
    """
    Saves a 2×2 matplotlib grid of training / validation curves.

    Panels
    ------
    [0,0] Total loss (train + val)
    [0,1] Cross-entropy component loss
    [1,0] Accuracy with 95% target line
    [1,1] Supervised contrastive component loss

    Parameters
    ----------
    experiment_name : Prefix for the saved PNG filename.
    save_dir        : Directory for output figures (default: current dir).
    dpi             : Resolution of saved figure.
    """

    def __init__(self,
                 experiment_name: str = "experiment",
                 save_dir: str        = ".",
                 dpi: int             = 300):
        self.experiment_name = experiment_name
        self.save_dir        = save_dir
        self.dpi             = dpi

    def update(self, history: List[Dict]):
        """
        Redraw and save the learning curves from the current history.

        Parameters
        ----------
        history : List of dicts as produced by TrainingDriver.run().
        """
        if len(history) < 2:
            return   # not enough points yet

        epochs    = [h["epoch"]            for h in history]
        tr_loss   = [h["train_loss"]       for h in history]
        val_loss  = [h["val_loss"]         for h in history]
        tr_acc    = [h["train_acc"]        for h in history]
        val_acc   = [h["val_acc"]          for h in history]
        tr_ce     = [h["train_ce"]         for h in history]
        val_ce    = [h["val_ce"]           for h in history]
        tr_con    = [h["train_contrastive"] for h in history]
        val_con   = [h["val_contrastive"]  for h in history]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{self.experiment_name}  —  Epoch {epochs[-1]}",
                     fontsize=14, fontweight="bold")

        # ── Panel (0,0): Total loss ───────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(epochs, tr_loss,  "b-", lw=2, label="Train total")
        ax.plot(epochs, val_loss, "r-", lw=2, label="Val total")
        ax.set_title("Total Loss  (CE + SupCon)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)

        # ── Panel (0,1): Cross-entropy component ──────────────────────────────
        ax = axes[0, 1]
        ax.plot(epochs, tr_ce,  "b-", lw=2, label="Train CE")
        ax.plot(epochs, val_ce, "r-", lw=2, label="Val CE")
        ax.set_title("Cross-Entropy Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)

        # ── Panel (1,0): Accuracy ─────────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(epochs, tr_acc,  "b-", lw=2, label="Train Acc")
        ax.plot(epochs, val_acc, "r-", lw=2, label="Val Acc")
        ax.axhline(95.0, color="green", ls="--", alpha=0.7,
                   label="95 % target")
        ax.set_title("Accuracy (%)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100); ax.legend(); ax.grid(True, alpha=0.3)

        # ── Panel (1,1): Supervised contrastive component ─────────────────────
        ax = axes[1, 1]
        ax.plot(epochs, tr_con,  "b-", lw=2, label="Train SupCon")
        ax.plot(epochs, val_con, "r-", lw=2, label="Val SupCon")
        ax.set_title("Supervised Contrastive Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = (f"{self.save_dir}/{self.experiment_name}"
                 f"_curves_ep{epochs[-1]:04d}.png")
        plt.savefig(fname, dpi=self.dpi, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"  📈 Curves saved → {fname}")


# ============================================================
# 14. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("  🧠 HYBRID CONTRASTIVE VISION NET")
    print("  Backbone: ConvNeXt-Tiny  |  Head: CBAM + ConvTransformer + SSM")
    print("  Task: CT-Kidney 4-class  (Cyst / Normal / Stone / Tumor)")
    print("  Loss: CE(0.7) + Supervised-Contrastive(0.3)")
    print("=" * 70)

    # ── Configuration ─────────────────────────────────────────────────────────
    cfg = HyperParameters(
        num_classes        = 4,
        backbone_name      = "convnext_tiny",
        pretrained         = True,
        ssm_dim            = 256,
        epochs             = 100,
        lr                 = 1e-4,
        weight_decay       = 0.01,
        ce_weight          = 0.7,
        contrastive_weight = 0.3,
        contrastive_temp   = 0.07,
        label_smoothing    = 0.1,
        clip_norm          = 1.0,
        lr_patience        = 5,
        lr_factor          = 0.5,
        early_stop_acc     = 95.0,
        checkpoint_path    = "./best_hybrid_convnext_ssm.pth",
        experiment_name    = "HybridConvNeXt_SSM_Contrastive_CTKidney",
    )

    # ── Data loaders (must be defined before this block) ──────────────────────
    # Expected interface:
    #   train_loader, val_loader, test_loader  →  DataLoader of (image, label)
    #   class_names                            →  List[str] of length num_classes
    #
    # Example call (supply your own):
    #   train_loader, val_loader, test_loader, class_names = load_ct_kidney_data()

    try:
        sample_batch = next(iter(train_loader))
        print(f"\n[Data] Batch shape : {sample_batch[0].shape}")
        print(f"       Label shape : {sample_batch[1].shape}")
        print(f"       Class counts: {torch.bincount(sample_batch[1]).tolist()}")
    except NameError:
        raise NameError(
            "Please define train_loader, val_loader, and test_loader "
            "before running __main__."
        )

    # ── Build model ───────────────────────────────────────────────────────────
    model = ModelFactory.create(cfg)

    # ── Loss function ─────────────────────────────────────────────────────────
    criterion = HybridContrastiveCE(
        ce_weight          = cfg.ce_weight,
        contrastive_weight = cfg.contrastive_weight,
        contrastive_temp   = cfg.contrastive_temp,
        label_smoothing    = cfg.label_smoothing,
    ).to(DEVICE)

    # ── Optimiser & LR scheduler ──────────────────────────────────────────────
    # AdamW decouples weight decay from the gradient update (unlike Adam),
    # which prevents scale-dependent regularisation.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ReduceLROnPlateau: halves LR if val_acc doesn't improve for `patience` epochs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",          # we are maximising accuracy
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        verbose=True,
    )

    # ── Mixed-precision scaler ────────────────────────────────────────────────
    # GradScaler is a no-op on CPU; safe to always instantiate.
    scaler = GradScaler()

    # ── GPU augmenter ─────────────────────────────────────────────────────────
    gpu_aug = GPUAugmenter(
        flip_p=0.5, max_rotate=15.0,
        brightness=0.2, contrast=0.2, saturation=0.1
    )

    # ── Assemble Trainer ──────────────────────────────────────────────────────
    trainer = Trainer(
        model         = model,
        criterion     = criterion,
        optimizer     = optimizer,
        scaler        = scaler,
        cfg           = cfg,
        gpu_augmenter = gpu_aug,
    )
    trainer.set_loaders(train_loader, val_loader)

    # ── Visualiser ────────────────────────────────────────────────────────────
    visualiser = LearningCurveVisualizer(
        experiment_name = cfg.experiment_name,
        save_dir        = ".",
        dpi             = 300,
    )

    # ── Run training ──────────────────────────────────────────────────────────
    driver = TrainingDriver(
        trainer    = trainer,
        scheduler  = scheduler,
        cfg        = cfg,
        visualiser = visualiser,
    )
    history = driver.run()

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🏆 FINAL TEST EVALUATION")
    print("=" * 70)

    # Restore the best checkpoint saved during training.
    ckpt = torch.load(cfg.checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded best checkpoint from epoch {ckpt['epoch']} "
          f"(val acc = {ckpt['best_val_acc']:.2f} %)")

    # Reuse the validate loop on the held-out test set.
    trainer.set_loaders(test_loader, test_loader)   # val_loader ← test_loader
    test_acc, test_loss, test_details = trainer.validate_one_epoch()

    print(f"\n  Test Accuracy  : {test_acc:.2f} %")
    print(f"  Test Loss      : {test_loss:.4f}")
    print(f"  Test CE        : {test_details.get('ce_loss', 0):.4f}")
    print(f"  Test SupCon    : {test_details.get('contrastive_loss', 0):.4f}")
    print(f"  Best Val Acc   : {ckpt['best_val_acc']:.2f} %")
    print("=" * 70)

    # ── Save final learning curves ────────────────────────────────────────────
    visualiser.update(history)
