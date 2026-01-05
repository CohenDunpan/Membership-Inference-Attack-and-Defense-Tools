"""
Single-file implementation of the augmentation-driven MIA defense pipeline
from the ICML 2021 paper "When Does Data Augmentation Help With Membership
Inference Attacks?" by Kaya & Dumitras.

This module consolidates the original project structure (models.py,
aux_funcs.py, train_models.py, mi_attacks.py, loss_rank_correlation.py) into a
single importable file. It exposes utilities to

- load datasets (FMNIST, CIFAR-10, CIFAR-100) into in-memory tensors;
- build classifiers with optional differential privacy (Opacus);
- train models with label augmentation (smooth, distillation, disturb label)
  and data augmentation (crop, noise, cutout, mixup);
- evaluate standard and augmentation-aware membership inference baselines;
- compute loss-rank correlation (LRC) between trained models.

Intended usage: place this file under a defense library (e.g.
Membership_Inference_Attack_and_Defense_Tools/Defense) and import the
high-level helpers defined at the bottom of the file.
"""

from __future__ import annotations

import os
import sys
import math
import random
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    from opacus import PrivacyEngine
except Exception as exc:  # pragma: no cover - dependency guard
    PrivacyEngine = None  # type: ignore
    _OPACUS_IMPORT_ERROR = exc
else:
    _OPACUS_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Reproducibility and simple file helpers
# ---------------------------------------------------------------------------

def set_random_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_path(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def file_exists(filename: str) -> bool:
    return os.path.isfile(filename)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

class Cutout:
    """Randomly masks out square patches from images."""

    def __init__(self, n_holes: int, length: int, device: str):
        self.n_holes = n_holes
        self.length = length
        self.device = device

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 4:
            h, w = img.size(2), img.size(3)
        else:
            h, w = img.size(1), img.size(2)

        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).to(self.device)
        mask = mask.expand_as(img)
        return img * mask


class ManualData(Dataset):
    """A minimal tensor-backed Dataset with optional on-the-fly augmentation."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, device: str = "cpu"):
        self.data = torch.from_numpy(data).to(device, dtype=torch.float)
        self.labels = torch.from_numpy(labels).to(device, dtype=torch.long)
        self.device = device
        self.train = True
        self.transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self.gaussian_std: Optional[float] = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        if self.train:
            if self.transforms is not None:
                data = self.transforms(data)
            if self.gaussian_std is not None:
                noise = torch.randn(data.size(), device=self.device) * self.gaussian_std
                data = torch.clamp(data + noise, min=0, max=1)
        return data, self.labels[idx]

    # dataset-level augmentation hooks
    def add_crop(self, padding_size: int) -> None:
        if padding_size > 0:
            self.data = self.data.to("cpu")
            self.labels = self.labels.to("cpu")
            self.device = "cpu"
            self.transforms = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomCrop(self.data.shape[-1], padding=padding_size),
                 transforms.ToTensor()]
            )

    def add_cutout(self, cutout_size: int) -> None:
        if cutout_size > 0:
            self.transforms = transforms.Compose([Cutout(n_holes=1, length=cutout_size, device=self.device)])

    def add_gaussian_aug(self, std_dev: float) -> None:
        self.gaussian_std = std_dev


def get_subset_data(ds: ManualData, idx: Optional[np.ndarray] = None) -> ManualData:
    idx = idx if idx is not None else np.arange(len(ds.data))
    np_data = ds.data[idx].cpu().detach().numpy()
    np_labels = ds.labels[idx].cpu().detach().numpy()
    return ManualData(np_data, np_labels, ds.device)


def get_pytorch_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using PyTorch version:", torch.__version__, "CUDA:", torch.cuda.is_available())
    return device


def get_loader(dataset: Dataset, shuffle: bool = True, batch_size: int = 128, device: str = "cpu") -> DataLoader:
    num_workers = 0 if device != "cpu" else 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_loaders(ds: Tuple[Dataset, Dataset], shuffle: bool = True, batch_size: int = 128, device: str = "cpu") -> Tuple[DataLoader, DataLoader]:
    train_ds, test_ds = ds
    return get_loader(train_ds, shuffle, batch_size, device), get_loader(test_ds, False, batch_size, device)


def _load_cifar(split: str, num_classes: int, root: str, device: str) -> ManualData:
    t = transforms.Compose([transforms.ToTensor()])
    dataset_cls = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100
    dataset = dataset_cls(root, train=(split == "train"), download=True, transform=t)
    data = (dataset.data / 255).transpose((0, 3, 1, 2))
    labels = np.array(dataset.targets)
    return ManualData(data, labels, device)


def get_cifar10_datasets(device: str = "cpu") -> Tuple[ManualData, ManualData]:
    create_path("data/cifar10")
    return _load_cifar("train", 10, "data/cifar10", device), _load_cifar("test", 10, "data/cifar10", device)


def get_cifar100_datasets(device: str = "cpu") -> Tuple[ManualData, ManualData]:
    create_path("data/cifar100")
    return _load_cifar("train", 100, "data/cifar100", device), _load_cifar("test", 100, "data/cifar100", device)


def get_fmnist_datasets(device: str = "cpu") -> Tuple[ManualData, ManualData]:
    create_path("data/fmnist")
    t = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST("data/fmnist", train=True, download=True, transform=t)
    test_dataset = datasets.FashionMNIST("data/fmnist", train=False, download=True, transform=t)
    train_data, test_data = train_dataset.data.numpy() / 255, test_dataset.data.numpy() / 255
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
    return ManualData(train_data, train_labels, device), ManualData(test_data, test_labels, device)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple layer
        return x.view(x.size(0), -1)


class FMNISTClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, dp: bool = False, device: str = "cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = 28
        if dp:
            self.dp_best_alphas: List[float] = []
            self.dp_epsilons: List[float] = []
            BN = lambda nf: nn.GroupNorm(min(32, nf), nf, affine=True)
        else:
            BN = lambda nf: nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.model = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            BN(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BN(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(256 * 49, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.num_classes),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.model(x), dim=1)

    def forward_w_temperature(self, x: torch.Tensor, T: float = 1) -> torch.Tensor:
        logits = self.model(x)
        return F.softmax(logits / T, dim=1)


class CIFARClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, dp: bool = False, device: str = "cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = 32
        if dp:
            self.dp_best_alphas: List[float] = []
            self.dp_epsilons: List[float] = []
            BN = lambda nf: nn.GroupNorm(min(32, nf), nf, affine=True)
        else:
            BN = lambda nf: nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            BN(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            BN(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BN(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BN(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            BN(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BN(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BN(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BN(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.num_classes),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.model(x), dim=1)

    def forward_w_temperature(self, x: torch.Tensor, T: float = 1) -> torch.Tensor:
        logits = self.model(x)
        return F.softmax(logits / T, dim=1)


# ---------------------------------------------------------------------------
# Optimizers and schedulers
# ---------------------------------------------------------------------------

class NullScheduler:
    def step(self) -> None:  # pragma: no cover - trivial scheduler
        return None


def get_std_optimizer(model: nn.Module, milestones: Optional[List[int]] = None, wd: float = 1e-6, optim_type: str = "adam", lr: Optional[float] = None) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    if optim_type == "adam":
        lr = 0.001 if lr is None else lr
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999), amsgrad=True, weight_decay=wd)
    elif optim_type == "sgd":
        lr = 0.1 if lr is None else lr
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_type}")

    if milestones is None:
        scheduler: torch.optim.lr_scheduler._LRScheduler = NullScheduler()
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    return optimizer, scheduler


def modify_optim_for_dp(model: nn.Module, optimizer: torch.optim.Optimizer, norm_clip: float = 1.0, noise_mult: float = 0.01, batch_size: int = 64, accumulation_steps: int = 4, training_size: int = 50000) -> PrivacyEngine:
    if PrivacyEngine is None:
        raise ImportError("Opacus is required for DP training") from _OPACUS_IMPORT_ERROR
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch_size * accumulation_steps,
        sample_size=training_size,
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=noise_mult,
        max_grad_norm=norm_clip,
    )
    privacy_engine.attach(optimizer)
    return privacy_engine


# ---------------------------------------------------------------------------
# Training primitives
# ---------------------------------------------------------------------------

def SoftLabelNLL(predicted: torch.Tensor, target: torch.Tensor, reduce: bool = False) -> torch.Tensor:
    if reduce:
        return -(target * predicted).sum(dim=1).mean()
    return -(target * predicted).sum(dim=1)


def clf_std_training_step(clf: nn.Module, optimizer: torch.optim.Optimizer, data: torch.Tensor, labels: torch.Tensor, device: str = "cpu") -> None:
    clf.train()
    loss_fn = nn.NLLLoss()
    b_x = data.to(device, dtype=torch.float)
    b_y = labels.to(device, dtype=torch.long)
    output = clf(b_x)
    loss = loss_fn(output, b_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def clf_dp_training_step(clf: nn.Module, optimizer: torch.optim.Optimizer, data: torch.Tensor, labels: torch.Tensor, batch_idx: int, accumulation_steps: int, tot_batches: int, device: str = "cpu") -> None:
    clf.train()
    loss_fn = nn.NLLLoss()
    b_x = data.to(device, dtype=torch.float)
    b_y = labels.to(device, dtype=torch.long)
    output = clf(b_x)
    loss = loss_fn(output, b_y)
    optimizer.zero_grad()
    loss.backward()
    if (batch_idx % accumulation_steps == 0) or (batch_idx == tot_batches):
        optimizer.step()
    else:
        optimizer.virtual_step()  # type: ignore[attr-defined]


def clf_mixup_training_step(clf: nn.Module, optimizer: torch.optim.Optimizer, data_1: torch.Tensor, labels_1: torch.Tensor, data_2: torch.Tensor, labels_2: torch.Tensor, alpha: float, device: str = "cpu") -> None:
    clf.train()
    lam = 0.5 if alpha == math.inf else (1 if alpha == 0 else np.random.beta(alpha, alpha))
    b_x = lam * data_1.to(device, dtype=torch.float) + (1 - lam) * data_2.to(device, dtype=torch.float)
    labels_1_one_hot = torch.zeros(data_1.shape[0], clf.num_classes, dtype=torch.float, device=device).scatter_(1, labels_1.view(-1, 1), 1)
    labels_2_one_hot = torch.zeros(data_2.shape[0], clf.num_classes, dtype=torch.float, device=device).scatter_(1, labels_2.view(-1, 1), 1)
    b_y = lam * labels_1_one_hot + (1 - lam) * labels_2_one_hot
    output = clf(b_x)
    loss = SoftLabelNLL(output, b_y, reduce=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def clf_disturblabel_training_step(clf: nn.Module, optimizer: torch.optim.Optimizer, data: torch.Tensor, labels: torch.Tensor, alpha: float, device: str = "cpu") -> None:
    clf.train()
    C = clf.num_classes
    p_c = 1 - ((C - 1) / C) * alpha
    p_i = (1 / C) * alpha
    loss_fn = nn.NLLLoss()
    b_x = data.to(device, dtype=torch.float)
    b_y = labels.to(device, dtype=torch.long).view(-1, 1)
    b_y_one_hot = (torch.ones(b_y.shape[0], C, device=device) * p_i)
    b_y_one_hot.scatter_(1, b_y, p_c)
    distribution = torch.distributions.OneHotCategorical(b_y_one_hot)
    b_y_disturbed = distribution.sample().max(dim=1)[1]
    output = clf(b_x)
    loss = loss_fn(output, b_y_disturbed)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def clf_distillation_training_step(clf: nn.Module, optimizer: torch.optim.Optimizer, data: torch.Tensor, teacher: nn.Module, temperature: float, device: str = "cpu") -> None:
    teacher.eval()
    clf.train()
    b_x = data.to(device, dtype=torch.float)
    with torch.no_grad():
        soft_targets = teacher.forward_w_temperature(b_x, temperature)
    output = clf(b_x)
    loss = SoftLabelNLL(output, soft_targets, reduce=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def clf_smoothlabel_training_step(clf: nn.Module, optimizer: torch.optim.Optimizer, data: torch.Tensor, labels: torch.Tensor, smoothing_coef: float, device: str = "cpu") -> None:
    clf.train()
    b_x = data.to(device, dtype=torch.float)
    b_y = labels.to(device, dtype=torch.long)
    b_y_one_hot = torch.zeros(data.shape[0], clf.num_classes, dtype=torch.float, device=device).scatter_(1, b_y.view(-1, 1), 1)
    smoothed = (1 - smoothing_coef) * b_y_one_hot + (smoothing_coef / clf.num_classes)
    output = clf(b_x)
    loss = SoftLabelNLL(output, smoothed, reduce=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _epoch_milestones(num_epochs: int) -> List[int]:
    if num_epochs == 3:
        return [1, 2]
    if num_epochs == 4:
        return [2, 3]
    if num_epochs == 7:
        return [3, 6]
    if num_epochs == 35:
        return [20, 30]
    return [int(num_epochs / 2), int(2 * num_epochs / 3)]


def _acc_topk(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res: List[float] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / batch_size)))
        return res


def test_clf(clf: nn.Module, loader: DataLoader, device: str = "cpu") -> Tuple[float, float]:
    clf.eval()
    top1_vals, top5_vals = [], []
    with torch.no_grad():
        for x, y in loader:
            b_x = x.to(device, dtype=torch.float)
            b_y = y.to(device, dtype=torch.long)
            output = clf(b_x)
            if clf.num_classes < 5:
                accs = _acc_topk(output, b_y, (1,))
                top5_vals.append(100.0)
            else:
                accs = _acc_topk(output, b_y, (1, 5))
                top5_vals.append(accs[1])
            top1_vals.append(accs[0])
    return float(np.mean(top1_vals)), float(np.mean(top5_vals))


def get_clf_losses(clf: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    loss_fn = nn.NLLLoss(reduction="none")
    losses = np.zeros(sum(len(batch[1]) for batch in loader))
    cur_idx = 0
    clf.eval()
    with torch.no_grad():
        for data, labels in loader:
            b_x = data.to(device, dtype=torch.float)
            b_y = labels.to(device, dtype=torch.long)
            output = clf(b_x)
            batch_losses = loss_fn(output, b_y).flatten().cpu().numpy()
            losses[cur_idx : cur_idx + len(b_x)] = batch_losses
            cur_idx += len(b_x)
    return losses


def get_clf_losses_w_aug(clf: nn.Module, loader: DataLoader, aug_type: str, aug_param, num_repeat: int = 25, device: str = "cpu") -> np.ndarray:
    """Losses under augmentation (augmentation-aware attack helper)."""
    if aug_type == "distillation":
        aug_param, teacher = aug_param
        teacher.eval()
    if aug_type in ["distillation", "smooth", "mixup"]:
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda pred, target: SoftLabelNLL(pred, target, reduce=False)
    else:
        loss_fn = nn.NLLLoss(reduction="none")
    if aug_type == "mixup":
        aug_param, mixing_data, mixing_labels = aug_param
        mixing_labels = torch.zeros(len(mixing_labels), clf.num_classes, dtype=torch.float, device=device).scatter_(1, mixing_labels.view(-1, 1), 1)

    total = sum(len(batch[1]) for batch in loader)
    losses = np.zeros((total, num_repeat))
    clf.eval()
    cur_idx = 0
    for data, labels in loader:
        b_x = data.to(device, dtype=torch.float)
        b_y = labels.to(device, dtype=torch.long)
        output = clf(b_x)
        for rep in range(num_repeat):
            if aug_type == "distillation":
                b_y_aug = teacher.forward_w_temperature(b_x, aug_param)
                losses[cur_idx : cur_idx + len(b_x), rep] = loss_fn(output, b_y_aug).flatten().cpu().numpy()
            elif aug_type == "smooth":
                b_y_one_hot = torch.zeros(b_x.shape[0], clf.num_classes, dtype=torch.float, device=device).scatter_(1, b_y.view(-1, 1), 1)
                b_y_aug = (1 - aug_param) * b_y_one_hot + (aug_param / clf.num_classes)
                losses[cur_idx : cur_idx + len(b_x), rep] = loss_fn(output, b_y_aug).flatten().cpu().numpy()
            elif aug_type == "disturblabel":
                C = clf.num_classes
                p_c = 1 - ((C - 1) / C) * aug_param
                p_i = (1 / C) * aug_param
                b_y_view = b_y.view(-1, 1)
                b_y_one_hot = (torch.ones(b_y_view.shape[0], C, device=device) * p_i)
                b_y_one_hot.scatter_(1, b_y_view, p_c)
                distribution = torch.distributions.OneHotCategorical(b_y_one_hot)
                b_y_aug = distribution.sample().max(dim=1)[1]
                losses[cur_idx : cur_idx + len(b_x), rep] = loss_fn(output, b_y_aug).flatten().cpu().numpy()
            elif aug_type == "noise":
                b_x_aug = torch.clamp(b_x + torch.randn_like(b_x) * aug_param, min=0, max=1)
                losses[cur_idx : cur_idx + len(b_x), rep] = loss_fn(clf(b_x_aug), b_y).flatten().cpu().numpy()
            elif aug_type == "crop":
                dim = b_x.shape[-1]
                padding = (int(aug_param),) * 4
                b_x_aug = F.pad(b_x, padding)
                i = torch.randint(0, int(aug_param) * 2 + 1, size=(1,)).item()
                j = torch.randint(0, int(aug_param) * 2 + 1, size=(1,)).item()
                b_x_aug = b_x_aug[:, :, i : i + dim, j : j + dim]
                losses[cur_idx : cur_idx + len(b_x), rep] = loss_fn(clf(b_x_aug), b_y).flatten().cpu().numpy()
            elif aug_type == "cutout":
                cutout = Cutout(n_holes=1, length=int(aug_param), device=device)
                b_x_aug = cutout(b_x.detach().clone())
                losses[cur_idx : cur_idx + len(b_x), rep] = loss_fn(clf(b_x_aug), b_y).flatten().cpu().numpy()
            elif aug_type == "mixup":
                indices = np.random.choice(len(mixing_data), size=len(b_x), replace=len(mixing_data) < len(b_x))
                lam = np.random.beta(aug_param, aug_param) if aug_param > 0 else 1
                b_x_aug = lam * b_x + (1 - lam) * mixing_data[indices]
                b_y_aug = torch.zeros(len(b_x), clf.num_classes, dtype=torch.float, device=device).scatter_(1, b_y.view(-1, 1), 1)
                b_y_aug = lam * b_y_aug + (1 - lam) * mixing_labels[indices]
                losses[cur_idx : cur_idx + len(b_x), rep] = loss_fn(clf(b_x_aug), b_y_aug).flatten().cpu().numpy()
            else:
                losses[cur_idx : cur_idx + len(b_x), rep] = loss_fn(output, b_y).flatten().cpu().numpy()
        cur_idx += len(b_x)
    return losses[:, 0] if num_repeat == 1 else losses


def get_clf_preds(clf: nn.Module, loader: DataLoader, logits: bool = True, temperature: float = 1, device: str = "cpu") -> np.ndarray:
    total = sum(len(batch[1]) for batch in loader)
    preds = np.zeros((total, clf.num_classes))
    cur_idx = 0
    clf.eval()
    with torch.no_grad():
        for data, _ in loader:
            b_x = data.to(device, dtype=torch.float)
            output = clf.model(b_x) if logits else clf.forward_w_temperature(b_x, T=temperature)
            preds[cur_idx : cur_idx + len(b_x)] = output.cpu().numpy()
            cur_idx += len(b_x)
    return preds


def get_correctly_classified_idx(clf: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    idx: List[np.ndarray] = []
    cur_idx = 0
    clf.eval()
    with torch.no_grad():
        for data, labels in loader:
            b_x = data.to(device, dtype=torch.float)
            b_y = labels.to(device, dtype=torch.long)
            preds = clf(b_x).max(dim=1)[1]
            correct_idx = torch.where(b_y == preds)[0].cpu().numpy()
            idx.append(correct_idx + cur_idx)
            cur_idx += len(b_x)
    return np.concatenate(idx)


# ---------------------------------------------------------------------------
# Membership inference helpers (Yeom et al.)
# ---------------------------------------------------------------------------

def yeom_mi_attack(losses: np.ndarray, threshold: float) -> np.ndarray:
    return (losses < threshold).astype(int)


def mi_success(train_memberships: np.ndarray, test_memberships: np.ndarray, print_details: bool = True) -> float:
    tp = np.sum(train_memberships)
    fp = np.sum(test_memberships)
    fn = len(train_memberships) - tp
    tn = len(test_memberships) - fp
    acc = 100 * (tp + tn) / (tp + fp + tn + fn)
    advantage = 2 * (acc - 50)
    if print_details:
        precision = 100 * (tp / (tp + fp)) if (tp + fp) > 0 else 0
        recall = 100 * (tp / (tp + fn)) if (tp + fn) > 0 else 0
        print(
            f"Adversary Advantage: {advantage:.3f}%, Accuracy: {acc:.3f}%, Precision: {precision:.3f}%, Recall: {recall:.3f}%"
        )
        print(f"In training: {tp}/{len(train_memberships)}, In testing: {tn}/{len(test_memberships)}")
    return advantage


def yeom_w_get_best_threshold(train_losses: np.ndarray, test_losses: np.ndarray) -> float:
    mean_loss = np.mean(train_losses)
    std_dev = np.std(train_losses)
    coeffs = np.linspace(-5, 5, num=1001, endpoint=True)
    advantages = []
    for coeff in coeffs:
        cur_threshold = mean_loss + std_dev * coeff
        cur_adv = mi_success(yeom_mi_attack(train_losses, cur_threshold), yeom_mi_attack(test_losses, cur_threshold), False)
        advantages.append(cur_adv)
    best_threshold = mean_loss + std_dev * coeffs[int(np.argmax(advantages))]
    return best_threshold


def apply_avg_and_best_attacks(train_losses: np.ndarray, test_losses: np.ndarray, idx: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[Tuple[float, np.ndarray, np.ndarray, float], Tuple[float, np.ndarray, np.ndarray, float]]:
    train_in_atk_train_idx, train_in_atk_test_idx, test_in_atk_train_idx, test_in_atk_test_idx = idx
    avg_loss_train = np.mean(train_losses[train_in_atk_train_idx])
    avg_train_memberships = yeom_mi_attack(train_losses[train_in_atk_test_idx], avg_loss_train)
    avg_test_memberships = yeom_mi_attack(test_losses[test_in_atk_test_idx], avg_loss_train)
    avg_adv = mi_success(avg_train_memberships, avg_test_memberships, print_details=False)
    best_threshold = yeom_w_get_best_threshold(train_losses[train_in_atk_train_idx], test_losses[test_in_atk_train_idx])
    best_train_memberships = yeom_mi_attack(train_losses[train_in_atk_test_idx], best_threshold)
    best_test_memberships = yeom_mi_attack(test_losses[test_in_atk_test_idx], best_threshold)
    best_adv = mi_success(best_train_memberships, best_test_memberships, print_details=False)
    return (avg_loss_train, avg_train_memberships, avg_test_memberships, avg_adv), (
        best_threshold,
        best_train_memberships,
        best_test_memberships,
        best_adv,
    )


# ---------------------------------------------------------------------------
# High-level configuration and training wrappers
# ---------------------------------------------------------------------------

@dataclass
class DefenseConfig:
    dataset: str
    label_aug_type: str = "no"  # no | smooth | distillation | disturblabel
    label_aug_param: float = 0.0
    data_aug_type: str = "no"  # no | crop | noise | cutout | mixup
    data_aug_param: float = 0.0
    epochs: int = 35
    batch_size: int = 128
    optimizer: str = "adam"
    learning_rate: Optional[float] = None
    weight_decay: float = 1e-6
    milestones: Optional[List[int]] = None
    dp_norm_clip: float = 0.0
    dp_noise_mult: float = 0.0
    dp_batch_size: int = 64
    dp_accumulation_steps: int = 4
    save_dir: Optional[str] = None
    mixup_seed: int = 0

    def is_dp(self) -> bool:
        return self.dp_norm_clip > 0 and self.dp_noise_mult > 0


@dataclass
class TrainResult:
    model: nn.Module
    top1: float
    top5: float
    dp_epsilon: Optional[float]
    path: Optional[str]


_DATASET_LOADERS: Dict[str, Callable[[str], Tuple[ManualData, ManualData]]] = {
    "fmnist": get_fmnist_datasets,
    "cifar10": get_cifar10_datasets,
    "cifar100": get_cifar100_datasets,
}


_MODEL_FACTORIES: Dict[str, Callable[[bool, str], nn.Module]] = {
    "fmnist": lambda is_dp, device: FMNISTClassifier(num_classes=10, dp=is_dp, device=device),
    "cifar10": lambda is_dp, device: CIFARClassifier(num_classes=10, dp=is_dp, device=device),
    "cifar100": lambda is_dp, device: CIFARClassifier(num_classes=100, dp=is_dp, device=device),
}


def _build_model_and_data(cfg: DefenseConfig, device: str) -> Tuple[nn.Module, Tuple[ManualData, ManualData], List[int]]:
    if cfg.dataset not in _DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    datasets = _DATASET_LOADERS[cfg.dataset](device)
    milestones = cfg.milestones if cfg.milestones is not None else _epoch_milestones(cfg.epochs)
    model = _MODEL_FACTORIES[cfg.dataset](cfg.is_dp(), device)
    return model, datasets, milestones


def _apply_dataset_aug(train_ds: ManualData, cfg: DefenseConfig) -> None:
    if cfg.data_aug_type == "crop":
        train_ds.add_crop(int(cfg.data_aug_param))
    elif cfg.data_aug_type == "noise":
        train_ds.add_gaussian_aug(float(cfg.data_aug_param))
    elif cfg.data_aug_type == "cutout":
        train_ds.add_cutout(int(cfg.data_aug_param))


def _train_step_selector(clf: nn.Module, optimizer: torch.optim.Optimizer, cfg: DefenseConfig, training_params, second_loader: Optional[DataLoader], device: str):
    if not cfg.is_dp():
        if cfg.label_aug_type == "no" and cfg.data_aug_type != "mixup":
            return lambda data, labels, batch_idx: clf_std_training_step(clf, optimizer, data, labels, device)
        if cfg.label_aug_type == "smooth":
            return lambda data, labels, batch_idx: clf_smoothlabel_training_step(clf, optimizer, data, labels, cfg.label_aug_param, device)
        if cfg.label_aug_type == "distillation":
            teacher, T = training_params
            return lambda data, labels, batch_idx: clf_distillation_training_step(clf, optimizer, data, teacher, T, device)
        if cfg.label_aug_type == "disturblabel":
            return lambda data, labels, batch_idx: clf_disturblabel_training_step(clf, optimizer, data, labels, cfg.label_aug_param, device)
        if cfg.data_aug_type == "mixup" and second_loader is not None:
            return lambda data1, labels1, data2, labels2: clf_mixup_training_step(clf, optimizer, data1, labels1, data2, labels2, cfg.data_aug_param, device)
        return lambda data, labels, batch_idx: clf_std_training_step(clf, optimizer, data, labels, device)
    accumulation_steps = cfg.dp_accumulation_steps
    tot_batches = sum(1 for _ in second_loader) if second_loader is not None else None  # type: ignore[arg-type]
    if tot_batches is None:
        raise ValueError("DP training expects a primary train loader")
    return lambda data, labels, batch_idx: clf_dp_training_step(clf, optimizer, data, labels, batch_idx, accumulation_steps, tot_batches, device)


def train_defended_model(cfg: DefenseConfig, teacher: Optional[nn.Module] = None, device: Optional[str] = None) -> TrainResult:
    device = device or get_pytorch_device()
    set_random_seeds(cfg.mixup_seed)
    clf, datasets, milestones = _build_model_and_data(cfg, device)
    _apply_dataset_aug(datasets[0], cfg)
    loader_batch_size = cfg.dp_batch_size if cfg.is_dp() else cfg.batch_size
    train_loader = get_loader(datasets[0], shuffle=True, batch_size=loader_batch_size, device=device)
    test_loader = get_loader(datasets[1], shuffle=False, batch_size=loader_batch_size, device=device)
    optimizer, scheduler = get_std_optimizer(clf, milestones=milestones, wd=cfg.weight_decay, optim_type=cfg.optimizer, lr=cfg.learning_rate)
    priv_engine = None
    if cfg.is_dp():
        priv_engine = modify_optim_for_dp(
            clf,
            optimizer,
            norm_clip=cfg.dp_norm_clip,
            noise_mult=cfg.dp_noise_mult,
            batch_size=cfg.dp_batch_size,
            accumulation_steps=cfg.dp_accumulation_steps,
            training_size=len(datasets[0].data),
        )
        clf.is_dp = True  # record on model
        training_params = cfg.dp_accumulation_steps
    else:
        clf.is_dp = False
        training_params = None

    second_loader = None
    if cfg.data_aug_type == "mixup":
        second_loader = get_loader(datasets[0], shuffle=True, batch_size=loader_batch_size, device=device)

    if cfg.label_aug_type == "distillation":
        if teacher is None:
            raise ValueError("Teacher model is required for distillation")
        training_params = (teacher, cfg.label_aug_param)

    step_func = _train_step_selector(clf, optimizer, cfg, training_params, second_loader if second_loader else train_loader, device)

    for epoch in range(1, cfg.epochs + 1):
        print(f"Epoch: {epoch}/{cfg.epochs}")
        top1_test, top5_test = test_clf(clf, test_loader, device)
        print(f"Top1 Test accuracy: {top1_test:.2f}")
        print(f"Top5 Test accuracy: {top5_test:.2f}")
        if cfg.data_aug_type == "mixup" and second_loader is not None:
            for (x1, y1), (x2, y2) in zip(train_loader, second_loader):
                step_func(x1, y1, x2, y2)
        else:
            batch_idx = 1
            for x, y in train_loader:
                step_func(x, y, batch_idx)
                batch_idx += 1
        if cfg.is_dp() and hasattr(optimizer, "privacy_engine"):
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(1e-5)  # type: ignore[attr-defined]
            clf.dp_best_alphas.append(best_alpha)
            clf.dp_epsilons.append(epsilon)
            print(f"(ε = {epsilon:.2f}, δ = 1e-5) for α = {best_alpha}")
        scheduler.step()

    top1_test, top5_test = test_clf(clf, test_loader, device)
    print(f"End - Top1 Test accuracy: {top1_test:.2f}")
    print(f"End - Top5 Test accuracy: {top5_test:.2f}")

    model_path = None
    if cfg.save_dir is not None:
        create_path(cfg.save_dir)
        model_path = os.path.join(cfg.save_dir, "clf.dat")
        with open(model_path, "wb") as f:
            torch.save(clf, f)

    dp_eps = clf.dp_epsilons[-1] if cfg.is_dp() and hasattr(clf, "dp_epsilons") and clf.dp_epsilons else None
    if priv_engine is not None:
        priv_engine.detach()

    return TrainResult(model=clf, top1=top1_test, top5=top5_test, dp_epsilon=dp_eps, path=model_path)


# ---------------------------------------------------------------------------
# Evaluation and correlation helpers
# ---------------------------------------------------------------------------

def apply_mi_attack(model: nn.Module, loaders: Tuple[DataLoader, DataLoader], idx: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], save_path: str, n_attacker_train: int = 100, seed: int = 0, device: str = "cpu") -> Dict:
    results: Dict = {}
    results_path = os.path.join(save_path, f"mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle")
    if file_exists(results_path):
        with open(results_path, "rb") as handle:
            results = pickle.load(handle)
    else:
        train_top1, train_top5 = test_clf(model, loaders[0], device)
        test_top1, test_top5 = test_clf(model, loaders[1], device)
        train_losses = get_clf_losses(model, loaders[0], device=device)
        test_losses = get_clf_losses(model, loaders[1], device=device)
        avg_results, best_results = apply_avg_and_best_attacks(train_losses, test_losses, idx)
        avg_loss_train, avg_train_memberships, avg_test_memberships, avg_adv = avg_results
        best_threshold, best_train_memberships, best_test_memberships, best_adv = best_results
        results = {
            "train_top1": train_top1,
            "train_top5": train_top5,
            "test_top1": test_top1,
            "test_top5": test_top5,
            "avg_yeom_adv": avg_adv,
            "best_yeom_adv": best_adv,
            "avg_threshold": avg_loss_train,
            "best_threshold": best_threshold,
            "avg_train_memberships": avg_train_memberships,
            "avg_test_memberships": avg_test_memberships,
            "best_train_memberships": best_train_memberships,
            "best_test_memberships": best_test_memberships,
            "std_train_losses": train_losses,
            "std_test_losses": test_losses,
            "attack_idx": idx,
        }
        with open(results_path, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        "Train Top1: {0:.3f}%, Train Top5: {1:.3f}%, Test Top1: {2:.3f}%, Test Top5: {3:.3f}%".format(
            results["train_top1"], results["train_top5"], results["test_top1"], results["test_top5"]
        )
    )
    print("Avg Yeom MI Advantage: {0:.2f}".format(results["avg_yeom_adv"]))
    print("Best Yeom MI Advantage: {0:.2f}".format(results["best_yeom_adv"]))
    return results


def get_lrc_score(first_model: Dict, second_model: Dict, print_details: bool = False, tol: float = 1e-6) -> Tuple[float, float]:
    from scipy.stats import spearmanr
    first_train, second_train = first_model["std_train_losses"], second_model["std_train_losses"]
    coef, p = spearmanr(first_train, second_train)
    if print_details:
        print(f"Spearmans correlation coefficient: {coef:.2f}, p-value: {p:.2f}")
    return float(coef), float(p)


def get_pairwise_lrc(first_paths: List[str], second_paths: List[str], n_attacker_train: int = 100, seed: int = 0) -> float:
    from itertools import product, combinations
    pairs = combinations(first_paths, 2) if first_paths == second_paths else product(first_paths, second_paths)
    scores = []
    for fpath, spath in pairs:
        with open(os.path.join(fpath, f"mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle"), "rb") as handle:
            first_results = pickle.load(handle)
        with open(os.path.join(spath, f"mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle"), "rb") as handle:
            second_results = pickle.load(handle)
        scores.append(get_lrc_score(first_results, second_results, False)[0])
    return float(np.mean(scores))


__all__ = [
    "DefenseConfig",
    "TrainResult",
    "train_defended_model",
    "apply_mi_attack",
    "get_lrc_score",
    "get_pairwise_lrc",
    "yeom_mi_attack",
    "yeom_w_get_best_threshold",
    "mi_success",
    "get_clf_losses_w_aug",
    "get_clf_losses",
    "get_clf_preds",
    "get_correctly_classified_idx",
    "Cutout",
    "ManualData",
    "CIFARClassifier",
    "FMNISTClassifier",
    "set_random_seeds",
    "get_pytorch_device",
]
