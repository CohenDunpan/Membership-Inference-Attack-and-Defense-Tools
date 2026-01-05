"""
Label-only membership inference attack utilities consolidated into a single file.

This module re-implements the core decision-based membership inference attacks
from the CCS 2021 paper "Membership Leakage in Label-Only Exposures" (Zheng Li,
Yang Zhang) using a minimal, dependency-light interface. The implementation is
adapted from the original Label-Only-MIA repository and mirrors the structure of
NN-based attacks in this project so it can be dropped into the attack/defense
library directly.

Key ideas
---------
- Decision-only access: the attacker observes only predicted labels (no
  probabilities or losses) of the target model.
- Black-box adversarial example crafting (HopSkipJump by default) is used to
  measure how much perturbation is needed to change the model's decision; member
  examples typically require smaller perturbations than non-members.
- Distances between original and adversarial examples serve as scores for a
  membership classifier; ROC-AUC is used as the main evaluation metric.

Usage sketch
------------
>>> attack = LabelOnlyMIA(target_model, input_shape=(3, 32, 32), num_classes=10)
>>> results = attack.run(member_loader, nonmember_loader, max_iter=50)
>>> print(results["auc"])  # distance-based AUCs

Dependencies
------------
- PyTorch
- scikit-learn
- adversarial-robustness-toolbox (ART) for HopSkipJump
- foolbox (distance utilities)

If optional dependencies are missing, a clear ImportError will be raised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve

try:
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import PyTorchClassifier
    from art.utils import compute_success
except ImportError as exc:  # pragma: no cover - runtime check
    raise ImportError(
        "adversarial-robustness-toolbox is required for LabelOnlyMIA; "
        "install via pip install adversarial-robustness-toolbox"
    ) from exc

try:
    from foolbox.distances import l0, l1, l2, linf
except ImportError as exc:  # pragma: no cover - runtime check
    raise ImportError(
        "foolbox is required for distance calculations; install via pip install foolbox"
    ) from exc


def _prediction(logits: np.ndarray) -> Tuple[np.ndarray, int]:
    """Compute softmax and predicted class index for a single-sample batch."""
    scores = logits[0]
    scores_shifted = scores - np.max(scores)
    exp = np.exp(scores_shifted)
    softmax = exp / np.sum(exp)
    return softmax, int(np.argmax(softmax))


def _to_numpy(batch: torch.Tensor) -> np.ndarray:
    """Detach a tensor to CPU and convert to numpy with batch dimension preserved."""
    return batch.detach().cpu().numpy()


@dataclass
class AttackSample:
    """Distance record for one evaluated sample."""

    l0: float
    l1: float
    l2: float
    linf: float
    is_member: bool


class LabelOnlyMIA:
    """Decision-based label-only membership inference attack (HopSkipJump).

    The class accepts a PyTorch model and computes distance-based membership
    scores by crafting adversarial examples with HopSkipJump. AUC values are
    computed for each distance metric (L0/L1/L2/L_inf).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: Sequence[int],
        num_classes: int,
        device: Optional[torch.device] = None,
        clip_values: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.input_shape = tuple(input_shape)
        self.num_classes = num_classes
        self.clip_values = clip_values
        self.classifier = self._build_classifier()

    def _build_classifier(self) -> PyTorchClassifier:
        """Wrap the torch model as an ART classifier for HopSkipJump."""
        return PyTorchClassifier(
            model=self.model,
            clip_values=self.clip_values,
            loss=F.cross_entropy,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
        )

    def _build_attacker(self, max_iter: int, max_eval: int) -> HopSkipJump:
        """Instantiate HopSkipJump with the provided budget."""
        return HopSkipJump(
            classifier=self.classifier,
            targeted=False,
            max_iter=max_iter,
            max_eval=max_eval,
        )

    def _evaluate_loader(
        self,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        attacker: HopSkipJump,
        treat_as_member: bool,
        random_data: bool,
    ) -> List[AttackSample]:
        """Run the attacker over a loader and collect distance statistics."""
        results: List[AttackSample] = []
        for data, target in loader:
            data = data.to(self.device)
            target = target.to(self.device)
            x_np = _to_numpy(data)

            # Use predicted label when working with random data; otherwise rely on the true label.
            logits = self.classifier.predict(x_np)
            _, predicted = _prediction(logits)
            target_label = int(target.item())

            if predicted != target_label and not random_data:
                adv_np = x_np
                success = True
            else:
                adv_np = attacker.generate(x=x_np)
                if random_data:
                    success_score = compute_success(self.classifier, x_np, [predicted], adv_np)
                else:
                    success_score = compute_success(self.classifier, x_np, [target_label], adv_np)
                success = bool(np.mean(success_score) > 0)

            if not success:
                continue

            l0_dist = float(l0(x_np, adv_np))
            l1_dist = float(l1(x_np, adv_np))
            l2_dist = float(l2(x_np, adv_np))
            linf_dist = float(linf(x_np, adv_np))
            results.append(
                AttackSample(
                    l0=l0_dist,
                    l1=l1_dist,
                    l2=l2_dist,
                    linf=linf_dist,
                    is_member=treat_as_member,
                )
            )
        return results

    def _compute_auc(self, samples: List[AttackSample]) -> Dict[str, float]:
        """Compute ROC-AUC for each distance metric; returns 0.0 if not computable."""
        if not samples:
            return {"L0": 0.0, "L1": 0.0, "L2": 0.0, "Linf": 0.0}

        labels = np.array([1 if s.is_member else 0 for s in samples])
        distances = {
            "L0": np.array([s.l0 for s in samples]),
            "L1": np.array([s.l1 for s in samples]),
            "L2": np.array([s.l2 for s in samples]),
            "Linf": np.array([s.linf for s in samples]),
        }

        aucs: Dict[str, float] = {}
        for name, dist in distances.items():
            fpr, tpr, _ = roc_curve(labels, dist, pos_label=1, drop_intermediate=False)
            aucs[name] = float(round(auc(fpr, tpr), 4)) if len(fpr) > 1 else 0.0
        return aucs

    def run(
        self,
        member_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        nonmember_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        max_iter: int = 50,
        max_eval: int = 10000,
        random_data: bool = False,
    ) -> Dict[str, object]:
        """Execute the label-only attack.

        Args:
            member_loader: Data loader yielding member samples (attacker assumes label 1).
            nonmember_loader: Data loader yielding non-member samples (assumes label 0).
            max_iter: Iteration budget for HopSkipJump.
            max_eval: Query budget for HopSkipJump.
            random_data: If True, use predicted labels as the reference class (matches
                the "random data" variant from the original code).

        Returns:
            A dictionary containing per-metric AUCs and raw per-sample distances.
        """
        attacker = self._build_attacker(max_iter=max_iter, max_eval=max_eval)
        member_stats = self._evaluate_loader(member_loader, attacker, True, random_data)
        nonmember_stats = self._evaluate_loader(nonmember_loader, attacker, False, random_data)
        samples = member_stats + nonmember_stats
        return {
            "auc": self._compute_auc(samples),
            "distances": samples,
        }


__all__ = ["LabelOnlyMIA", "AttackSample"]
