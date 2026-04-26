# =============================================================================
# FULL XAI EVALUATION FOR DHCANet STUDENT (ColonCancerCT)
# Shows 10 random samples: top row originals, bottom row Grad-CAM overlays
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE

from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
from captum.attr import Saliency


class UltimateColonCancerEvaluator:
    def __init__(self, model, test_loader, class_names, device='cuda', shap_bg_samples=64):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.class_names = list(class_names)
        self.device = device

        self.images = []
        self.labels = []
        self.preds = []
        self.probs = []
        self.features = []

        self._collect_predictions()

        # SHAP background
        bg_batch = next(iter(test_loader))[0][:shap_bg_samples].to(device)
        self.shap_explainer = shap.DeepExplainer(self.model, bg_batch)

        # Captum Saliency
        self.saliency = Saliency(self.model)

    # -------- collect predictions --------
    def _collect_predictions(self):
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                logits = self.model(x)
                prob = F.softmax(logits, dim=1)
                pred = prob.argmax(1)

                self.images.extend(x.cpu().numpy())
                self.labels.extend(y.numpy())
                self.preds.extend(pred.cpu().numpy())
                self.probs.extend(prob.cpu().numpy())
                self.features.extend(logits.cpu().numpy())

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.preds = np.array(self.preds)
        self.probs = np.array(self.probs)
        self.features = np.array(self.features)

    # -------- Grad-CAM-like map --------
    def grad_cam_like(self, img_chw, target_class=None):
        x = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)  # [1,C,H,W]
        x.requires_grad_(True)
        out = self.model(x)
        if target_class is None:
            target_class = out.argmax(1).item()
        self.model.zero_grad()
        out[0, target_class].backward()
        grad = x.grad[0]  # [C,H,W]
        g = grad.abs().mean(0).cpu().numpy()
        g = cv2.resize(g, (224, 224))
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        return g

    # -------- Saliency (Captum) --------
    def saliency_map(self, img_chw, target_class=None):
        x = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        x.requires_grad_(True)
        with torch.no_grad():
            logits = self.model(x)
        if target_class is None:
            target_class = logits.argmax(1).item()
        attr = self.saliency.attribute(x, target=target_class)
        s = attr.abs().max(1)[0].cpu().numpy()[0]
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        return s

    # -------- LIME --------
    def lime_explanation(self, img_chw, num_samples=1000):
        img_hwc = img_chw.transpose(1, 2, 0)

        def pred_fn(imgs):
            imgs = np.stack(imgs, axis=0) if isinstance(imgs, list) else imgs
            tens = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(self.device)
            with torch.no_grad():
                logits = self.model(tens)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            return probs

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_hwc,
            classifier_fn=pred_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            hide_rest=False
        )
        m = (mask.astype(np.float32) > 0).astype(np.float32)
        m = cv2.resize(m, (224, 224))
        return m

    # -------- SHAP (DeepExplainer with additivity disabled) --------
    def shap_explanation(self, img_chw, target_class=1):
        x = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        shap_vals = self.shap_explainer.shap_values(x, check_additivity=False)
        sv = shap_vals[target_class][0]  # [C,H,W]
        s = np.abs(sv).mean(0)
        s = cv2.resize(s, (224, 224))
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        return s

    # -------- 10 random samples: originals + overlays --------
    def show_random_overlays(self, n_samples=10, method="gradcam"):
        """
        Show n_samples random images in one row and the same images
        with an XAI heatmap overlay (Grad-CAM-like or saliency) in the second row.
        """
        n_samples = min(n_samples, len(self.images))
        idxs = random.sample(range(len(self.images)), n_samples)

        fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
        if n_samples == 1:
            axes = np.array(axes).reshape(2, 1)

        for col, idx in enumerate(idxs):
            img_chw = self.images[idx]
            img_hwc = img_chw.transpose(1, 2, 0)
            t = self.class_names[self.labels[idx]]
            p = self.class_names[self.preds[idx]]
            pr = self.probs[idx, 1]

            # top row: original
            ax_top = axes[0, col]
            ax_top.imshow(img_hwc)
            ax_top.set_title(f"T:{t}\nP:{p}\nProb1:{pr:.3f}", fontsize=9)
            ax_top.axis("off")

            # choose heatmap
            if method.lower() == "saliency":
                heatmap = self.saliency_map(img_chw)
            elif method.lower() == "shap":
                heatmap = self.shap_explanation(img_chw)
            elif method.lower() == "lime":
                heatmap = self.lime_explanation(img_chw)
            else:
                heatmap = self.grad_cam_like(img_chw)

            # bottom row: overlay
            ax_bot = axes[1, col]
            ax_bot.imshow(img_hwc)
            ax_bot.imshow(heatmap, cmap="jet", alpha=0.5)
            ax_bot.set_title(f"{method.upper()} overlay", fontsize=9)
            ax_bot.axis("off")

        plt.suptitle(f"Random {n_samples} samples with {method.upper()} overlays", fontsize=14)
        plt.tight_layout()
        plt.show()

    # -------- main entry --------
    def run_full_evaluation(self, n_samples=10, bin_size=0.1):
        self.show_random_overlays(n_samples=n_samples, method="gradcam")



evaluator.run_full_evaluation(n_samples=10)
