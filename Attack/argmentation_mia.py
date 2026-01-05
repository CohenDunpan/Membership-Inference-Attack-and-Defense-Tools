"""
Single-file implementation of augmentation-aware membership inference attacks
from the ICML 2021 paper "When Does Data Augmentation Help With Membership
Inference Attacks?".

The goal is to provide a drop-in attack module (mirrors the original
augmentation_mia repository) that can live under Attack/ in the
Membership_Inference_Attack_and_Defense_Tools project. The file includes the
core Yeom threshold attack, the augmentation-aware variant, and the minimal
helpers (data utilities, loss helpers, and model evaluation routines) needed to
run the attacks without importing the original multi-file project.

Expected usage patterns:
- Direct function calls (apply_mi_attack/apply_aware_attack) with a trained
  classifier and explicit loaders.
- Dataset sampling via take_subset_from_datasets to build attacker-held data.
- Optional config-driven execution via the __main__ entrypoint, matching the
  original script signature: python argmentation_mia.py path/to/config.json

Assumptions:
- Victim classifiers output log-probabilities (i.e., nn.LogSoftmax in the
  final layer). This matches the training code used in the source project.
- Saved victim models are stored with torch.save(..., path+'.dat').
"""

import json
import os
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def file_exists(filename: str) -> bool:
    return os.path.isfile(filename)


def set_random_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_path(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_pytorch_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using PyTorch version:", torch.__version__, "CUDA:", torch.cuda.is_available())
    return device


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class Cutout:
    """Randomly mask out square patches from an image tensor."""

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
    """Tensor-backed dataset used for sampling attacker-held subsets."""

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
        if self.train and self.transforms is not None:
            data = self.transforms(data)
        if self.train and self.gaussian_std is not None:
            noise = torch.randn(data.size(), device=self.device) * self.gaussian_std
            data = torch.clamp(data + noise, min=0, max=1)
        return data, self.labels[idx]

    def add_crop(self, padding_size: int) -> None:
        if padding_size > 0:
            self.data = self.data.to("cpu")
            self.labels = self.labels.to("cpu")
            self.device = "cpu"
            self.transforms = transforms.Compose(
                [transforms.ToPILImage(), transforms.RandomCrop(self.data.shape[-1], padding=padding_size), transforms.ToTensor()]
            )

    def add_cutout(self, cutout_size: int) -> None:
        if cutout_size > 0:
            self.transforms = transforms.Compose([Cutout(n_holes=1, length=cutout_size, device=self.device)])

    def add_gaussian_aug(self, std_dev: float) -> None:
        self.gaussian_std = std_dev


def get_loader(dataset: Dataset, shuffle: bool = True, batch_size: int = 128, device: str = "cpu") -> DataLoader:
    num_workers = 0 if device != "cpu" else 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_loaders(datasets_pair: Tuple[Dataset, Dataset], shuffle: bool = True, batch_size: int = 128, device: str = "cpu") -> Tuple[DataLoader, DataLoader]:
    train_ds, test_ds = datasets_pair
    return get_loader(train_ds, shuffle, batch_size, device), get_loader(test_ds, False, batch_size, device)


def get_subset_data(ds: ManualData, idx: Optional[np.ndarray] = None) -> ManualData:
    idx = idx if idx is not None else np.arange(len(ds.data))
    np_data = ds.data[idx].cpu().detach().numpy()
    np_labels = ds.labels[idx].cpu().detach().numpy()
    return ManualData(np_data, np_labels, ds.device)


def get_cifar10_datasets(device: str = "cpu") -> Tuple[ManualData, ManualData]:
    create_path("data/cifar10")
    t = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=t)
    test_dataset = datasets.CIFAR10("data/cifar10", train=False, download=True, transform=t)
    train_data, test_data = train_dataset.data / 255, test_dataset.data / 255
    train_data, test_data = train_data.transpose((0, 3, 1, 2)), test_data.transpose((0, 3, 1, 2))
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    return ManualData(train_data, train_labels, device), ManualData(test_data, test_labels, device)


def get_cifar100_datasets(device: str = "cpu") -> Tuple[ManualData, ManualData]:
    create_path("data/cifar100")
    t = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100("data/cifar100", train=True, download=True, transform=t)
    test_dataset = datasets.CIFAR100("data/cifar100", train=False, download=True, transform=t)
    train_data, test_data = train_dataset.data / 255, test_dataset.data / 255
    train_data, test_data = train_data.transpose((0, 3, 1, 2)), test_data.transpose((0, 3, 1, 2))
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    return ManualData(train_data, train_labels, device), ManualData(test_data, test_labels, device)


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


def get_ds(ds_name: str, device: str = "cpu") -> Tuple[ManualData, ManualData]:
    if ds_name == "fmnist":
        return get_fmnist_datasets(device)
    if ds_name == "cifar10":
        return get_cifar10_datasets(device)
    if ds_name == "cifar100":
        return get_cifar100_datasets(device)
    raise ValueError(f"Unsupported dataset: {ds_name}")


# ---------------------------------------------------------------------------
# Model-eval helpers
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> Sequence[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            acc = float(correct_k.mul_(100.0 / batch_size))
            res.append(acc)
        return res


def loader_inst_counter(loader: DataLoader) -> int:
    num_instances = 0
    for batch in loader:
        num_instances += len(batch[1])
    return num_instances


def SoftLabelNLL(predicted: torch.Tensor, target: torch.Tensor, reduce: bool = False) -> torch.Tensor:
    if reduce:
        return -(target * predicted).sum(dim=1).mean()
    return -(target * predicted).sum(dim=1)


def test_clf(clf: nn.Module, loader: DataLoader, device: str = "cpu") -> Tuple[float, float]:
    clf.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for x, y in loader:
            b_x = x.to(device, dtype=torch.float)
            b_y = y.to(device, dtype=torch.long)
            clf_output = clf(b_x)
            if clf.num_classes < 5:
                accs = accuracy(clf_output, b_y, topk=(1,))
                top1.update(accs[0], b_x.size(0))
                top5.update(100.0, b_x.size(0))
            else:
                accs = accuracy(clf_output, b_y, topk=(1, 5))
                top1.update(accs[0], b_x.size(0))
                top5.update(accs[1], b_x.size(0))
    return top1.avg, top5.avg


def get_clf_losses(clf: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    clf_loss_func = nn.NLLLoss(reduction="none")
    losses = np.zeros(loader_inst_counter(loader))
    cur_idx = 0
    clf.eval()
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            b_y = batch[1].to(device, dtype=torch.long)
            output = clf(b_x)
            losses[cur_idx : cur_idx + len(b_x)] = clf_loss_func(output, b_y).flatten().cpu().detach().numpy()
            cur_idx += len(b_x)
    return losses


def get_clf_losses_w_aug(
    clf: nn.Module,
    loader: DataLoader,
    aug_type: str,
    aug_param,
    num_repeat: int = 25,
    device: str = "cpu",
) -> np.ndarray:
    with torch.no_grad():
        return _get_clf_losses_w_aug_impl(clf, loader, aug_type, aug_param, num_repeat=num_repeat, device=device)


def _get_clf_losses_w_aug_impl(
    clf: nn.Module,
    loader: DataLoader,
    aug_type: str,
    aug_param,
    num_repeat: int = 25,
    device: str = "cpu",
) -> np.ndarray:
    if aug_type == "distillation":
        aug_param, teacher = aug_param
        teacher.eval()

    if aug_type in ["distillation", "smooth", "mixup"]:
        clf_loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda pred, target: SoftLabelNLL(pred, target, reduce=False)
    else:
        clf_loss_func = nn.NLLLoss(reduction="none")

    if aug_type == "mixup":
        aug_param, mixing_data, mixing_labels = aug_param
        mixing_labels = ((torch.zeros(len(mixing_labels), clf.num_classes, dtype=torch.float)).to(device)).scatter_(
            1, mixing_labels.view(-1, 1), 1
        )

    losses = np.zeros((loader_inst_counter(loader), num_repeat))
    clf.eval()
    cur_idx = 0
    for batch in loader:
        b_x = batch[0].to(device, dtype=torch.float)
        b_y = batch[1].to(device, dtype=torch.long)
        output = clf(b_x)
        for ii in range(num_repeat):
            if aug_type == "distillation":
                b_y_aug = teacher.forward_w_temperature(b_x, aug_param)
                losses[cur_idx : cur_idx + len(b_x), ii] = clf_loss_func(output, b_y_aug).flatten().cpu().detach().numpy()

            elif aug_type == "smooth":
                b_y_one_hot = (torch.zeros(b_x.shape[0], clf.num_classes, dtype=torch.float).to(device)).scatter_(
                    1, b_y.view(-1, 1), 1
                )
                b_y_aug = (1 - aug_param) * b_y_one_hot + (aug_param / clf.num_classes)
                losses[cur_idx : cur_idx + len(b_x), ii] = clf_loss_func(output, b_y_aug).flatten().cpu().detach().numpy()

            elif aug_type == "disturblabel":
                C = clf.num_classes
                p_c = 1 - ((C - 1) / C) * aug_param
                p_i = (1 / C) * aug_param
                b_y_view = b_y.view(-1, 1)
                b_y_one_hot = (torch.ones(b_y_view.shape[0], C) * p_i).to(device)
                b_y_one_hot.scatter_(1, b_y_view, p_c)
                distribution = torch.distributions.OneHotCategorical(b_y_one_hot)
                b_y_aug = distribution.sample().max(dim=1)[1]
                losses[cur_idx : cur_idx + len(b_x), ii] = clf_loss_func(output, b_y_aug).flatten().cpu().detach().numpy()

            elif aug_type == "noise":
                b_x_aug = torch.clamp(b_x + torch.randn(b_x.shape, device=device) * aug_param, min=0, max=1)
                losses[cur_idx : cur_idx + len(b_x), ii] = clf_loss_func(clf(b_x_aug), b_y).flatten().cpu().detach().numpy()

            elif aug_type == "crop":
                dim = b_x.shape[-1]
                padding = tuple([int(aug_param)] * 4)
                b_x_aug = F.pad(b_x, padding)
                i = torch.randint(0, int(aug_param) * 2 + 1, size=(1,)).item()
                j = torch.randint(0, int(aug_param) * 2 + 1, size=(1,)).item()
                b_x_aug = b_x_aug[:, :, i : (i + dim), j : (j + dim)]
                losses[cur_idx : cur_idx + len(b_x), ii] = clf_loss_func(clf(b_x_aug), b_y).flatten().cpu().detach().numpy()

            elif aug_type == "cutout":
                cutout = Cutout(n_holes=1, length=int(aug_param), device=device)
                b_x_aug = cutout(b_x.detach().clone().to(device))
                losses[cur_idx : cur_idx + len(b_x), ii] = clf_loss_func(clf(b_x_aug), b_y).flatten().cpu().detach().numpy()

            elif aug_type == "mixup":
                if len(mixing_data) < len(b_x):
                    indices = np.random.choice(len(mixing_data), size=len(b_x), replace=True)
                else:
                    indices = np.random.choice(len(mixing_data), size=len(b_x), replace=False)
                b_x_aug = b_x.detach().clone().to(device)
                lam = np.random.beta(aug_param, aug_param) if aug_param > 0 else 1
                b_x_aug = (lam * b_x_aug) + ((1 - lam) * mixing_data[indices])
                b_y_aug = (torch.zeros(len(b_x), clf.num_classes, dtype=torch.float).to(device)).scatter_(
                    1, b_y.view(-1, 1), 1
                )
                b_y_aug = (lam * b_y_aug) + ((1 - lam) * mixing_labels[indices])
                losses[cur_idx : cur_idx + len(b_x), ii] = clf_loss_func(clf(b_x_aug), b_y_aug).flatten().cpu().detach().numpy()

        cur_idx += len(b_x)

    return losses[:, 0] if num_repeat == 1 else losses


# ---------------------------------------------------------------------------
# Attack logic
# ---------------------------------------------------------------------------

def get_reduction_params() -> Tuple[Sequence[str], Sequence[Callable]]:
    names = ["median", "min", "max", "std", "mean"]
    funcs = [np.median, np.min, np.max, np.std, np.mean]
    return names, funcs


def yeom_mi_attack(losses: np.ndarray, avg_loss: float) -> np.ndarray:
    return (losses < avg_loss).astype(int)


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
            "Adversary Advantage: {0:.3f}%, Accuracy: {1:.3f}%, Precision : {2:.3f}%, Recall: {3:.3f}%".format(
                advantage, acc, precision, recall
            )
        )
        print("In training: {}/{}, In testing: {}/{}".format(tp, len(train_memberships), tn, len(test_memberships)))
    return advantage


def yeom_w_get_best_threshold(train_losses: np.ndarray, test_losses: np.ndarray) -> float:
    advantages = []
    mean_loss = np.mean(train_losses)
    std_dev = np.std(train_losses)
    coeffs = np.linspace(-5, 5, num=1001, endpoint=True)
    for coeff in coeffs:
        cur_threshold = mean_loss + std_dev * coeff
        cur_adv = mi_success(yeom_mi_attack(train_losses, cur_threshold), yeom_mi_attack(test_losses, cur_threshold), False)
        advantages.append(cur_adv)
    best_threshold = mean_loss + std_dev * coeffs[np.argmax(advantages)]
    return best_threshold


def split_indices(indices: np.ndarray, first_split_size: int) -> Tuple[np.ndarray, np.ndarray]:
    first_split_indices = np.random.choice(indices, size=first_split_size, replace=False)
    second_split_indices = np.array([x for x in indices if x not in first_split_indices])
    return first_split_indices, second_split_indices


def apply_avg_and_best_attacks(train_losses: np.ndarray, test_losses: np.ndarray, idx: Tuple[np.ndarray, ...]):
    train_in_atk_train_idx, train_in_atk_test_idx, test_in_atk_train_idx, test_in_atk_test_idx = idx
    avg_loss_train = np.mean(train_losses[train_in_atk_train_idx])
    avg_train_memberships = yeom_mi_attack(train_losses[train_in_atk_test_idx], avg_loss_train)
    avg_test_memberships = yeom_mi_attack(test_losses[test_in_atk_test_idx], avg_loss_train)
    avg_yeom_mi_advantage = mi_success(avg_train_memberships, avg_test_memberships, print_details=False)
    avg_results = (avg_loss_train, avg_train_memberships, avg_test_memberships, avg_yeom_mi_advantage)

    best_threshold = yeom_w_get_best_threshold(train_losses[train_in_atk_train_idx], test_losses[test_in_atk_train_idx])
    best_train_memberships = yeom_mi_attack(train_losses[train_in_atk_test_idx], best_threshold)
    best_test_memberships = yeom_mi_attack(test_losses[test_in_atk_test_idx], best_threshold)
    best_yeom_mi_advantage = mi_success(best_train_memberships, best_test_memberships, print_details=False)
    best_results = (best_threshold, best_train_memberships, best_test_memberships, best_yeom_mi_advantage)
    return avg_results, best_results


def take_subset_from_datasets(
    datasets_pair: Tuple[ManualData, ManualData],
    seed: int,
    n_attacker_train: int,
    n_attacker_test: int,
    batch_size: int = 1000,
    device: str = "cpu",
) -> Tuple[Tuple[DataLoader, DataLoader], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    np.random.seed(seed)
    train_indices = np.random.choice(len(datasets_pair[0].data), size=n_attacker_train + n_attacker_test, replace=False)
    test_indices = np.random.choice(len(datasets_pair[1].data), size=n_attacker_train + n_attacker_test, replace=False)
    train_in_atk_test_idx, train_in_atk_train_idx = split_indices(train_indices, n_attacker_test)
    test_in_atk_test_idx, test_in_atk_train_idx = split_indices(test_indices, n_attacker_test)

    train_data = datasets_pair[0].data[np.concatenate((train_in_atk_train_idx, train_in_atk_test_idx))].cpu().detach().numpy()
    train_labels = datasets_pair[0].labels[np.concatenate((train_in_atk_train_idx, train_in_atk_test_idx))].cpu().detach().numpy()
    test_data = datasets_pair[1].data[np.concatenate((test_in_atk_train_idx, test_in_atk_test_idx))].cpu().detach().numpy()
    test_labels = datasets_pair[1].labels[np.concatenate((test_in_atk_train_idx, test_in_atk_test_idx))].cpu().detach().numpy()

    train_ds = ManualData(train_data, train_labels)
    train_ds.train = False
    test_ds = ManualData(test_data, test_labels)
    test_ds.train = False

    train_loader = get_loader(train_ds, shuffle=False, batch_size=batch_size, device=device)
    test_loader = get_loader(test_ds, shuffle=False, batch_size=batch_size, device=device)

    train_in_atk_train_idx = np.arange(len(train_in_atk_train_idx))
    train_in_atk_test_idx = np.arange(len(train_in_atk_train_idx), len(train_data))
    test_in_atk_train_idx = np.arange(len(test_in_atk_train_idx))
    test_in_atk_test_idx = np.arange(len(test_in_atk_train_idx), len(test_data))
    idx = (train_in_atk_train_idx, train_in_atk_test_idx, test_in_atk_train_idx, test_in_atk_test_idx)
    return (train_loader, test_loader), idx


def apply_mi_attack(
    model: nn.Module,
    loaders: Tuple[DataLoader, DataLoader],
    idx: Tuple[np.ndarray, ...],
    save_path: Optional[str],
    n_attacker_train: int = 100,
    seed: int = 0,
    device: str = "cpu",
) -> Dict:
    results: Dict = {}
    results_path = os.path.join(save_path, f"mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle") if save_path else None
    if results_path and file_exists(results_path):
        with open(results_path, "rb") as handle:
            results = pickle.load(handle)
    else:
        train_top1, train_top5 = test_clf(model, loaders[0], device)
        test_top1, test_top5 = test_clf(model, loaders[1], device)
        train_losses = get_clf_losses(model, loaders[0], device=device)
        test_losses = get_clf_losses(model, loaders[1], device=device)
        avg_results, best_results = apply_avg_and_best_attacks(train_losses, test_losses, idx)
        avg_loss_train, avg_train_memberships, avg_test_memberships, avg_yeom_adv = avg_results
        best_threshold, best_train_memberships, best_test_memberships, best_yeom_adv = best_results

        results = {
            "train_top1": train_top1,
            "train_top5": train_top5,
            "test_top1": test_top1,
            "test_top5": test_top5,
            "avg_yeom_adv": avg_yeom_adv,
            "best_yeom_adv": best_yeom_adv,
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
        if results_path:
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


def apply_aware_attack(
    model: nn.Module,
    params: Dict,
    loaders: Tuple[DataLoader, DataLoader],
    idx: Tuple[np.ndarray, ...],
    save_path: Optional[str],
    n_attacker_train: int = 100,
    n_repeat: int = 25,
    seed: int = 0,
    teacher: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Dict:
    results: Dict = {}
    results_path = os.path.join(
        save_path, f"aware_mi_results_ntrain_{n_attacker_train}_numrepeat_{n_repeat}_randseed_{seed}.pickle"
    ) if save_path else None
    if results_path and file_exists(results_path):
        with open(results_path, "rb") as handle:
            results = pickle.load(handle)
    else:
        laug_type, laug_param = params["laug_type"], params["laug_param"]
        daug_type, daug_param = params["daug_type"], params["daug_param"]
        train_in_atk_train_idx, _, test_in_atk_train_idx, _ = idx

        if daug_type == "mixup":
            mixing_data = loaders[0].dataset.data[train_in_atk_train_idx].to(device)
            mixing_labels = loaders[0].dataset.labels[train_in_atk_train_idx].to(device)
            aug_type, aug_param = daug_type, (daug_param, mixing_data, mixing_labels)
        elif laug_type == "distillation":
            aug_type, aug_param = laug_type, (laug_param, teacher)
        elif laug_type != "no":
            aug_type, aug_param = laug_type, laug_param
        elif daug_type != "no":
            aug_type, aug_param = daug_type, daug_param
        else:
            aug_type, aug_param = "no", None

        train_losses = get_clf_losses_w_aug(model, loaders[0], aug_type, aug_param, num_repeat=n_repeat, device=device)
        test_losses = get_clf_losses_w_aug(model, loaders[1], aug_type, aug_param, num_repeat=n_repeat, device=device)

        if n_repeat == 1:
            _, aware_results = apply_avg_and_best_attacks(train_losses, test_losses, idx)
            threshold, train_memberships, test_memberships, adv = aware_results
            reduction = "none"
            train_losses_all, test_losses_all = train_losses, test_losses
        else:
            best_local_adv = -100
            for name, func in zip(*get_reduction_params()):
                cur_train_losses, cur_test_losses = func(train_losses, axis=1), func(test_losses, axis=1)
                _, aware_results = apply_avg_and_best_attacks(cur_train_losses, cur_test_losses, idx)
                cur_threshold, cur_train_memberships, cur_test_memberships, cur_adv = aware_results
                adv_local = mi_success(
                    yeom_mi_attack(cur_train_losses[train_in_atk_train_idx], cur_threshold),
                    yeom_mi_attack(cur_test_losses[test_in_atk_train_idx], cur_threshold),
                    False,
                )
                if best_local_adv < adv_local:
                    best_local_adv = adv_local
                    adv = cur_adv
                    train_memberships = cur_train_memberships
                    test_memberships = cur_test_memberships
                    train_losses_all = cur_train_losses
                    test_losses_all = cur_test_losses
                    threshold = cur_threshold
                    reduction = name

        results = {
            "threshold": threshold,
            "adv": adv,
            "train_memberships": train_memberships,
            "test_memberships": test_memberships,
            "train_losses": train_losses_all,
            "test_losses": test_losses_all,
            "num_repeat": n_repeat,
            "reduction": reduction,
        }
        if results_path:
            with open(results_path, "wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Aware MI Advantage: {0:.2f} - Reduction: {1}".format(results["adv"], results.get("reduction", "")))
    return results


# ---------------------------------------------------------------------------
# Model discovery helpers (for config-driven execution)
# ---------------------------------------------------------------------------

def parse_model_path(model_path: str) -> Optional[Dict]:
    try:
        sections = model_path.split("_")
        params: Dict = {}
        params["dset_name"] = sections[0]
        params["laug_type"] = sections[2]
        params["laug_param"] = eval(sections[3])
        params["daug_type"] = sections[5]
        params["daug_param"] = eval(sections[6])
        params["dp_norm_clip"] = eval(sections[9])
        params["dp_noise"] = eval(sections[11])
        params["num_epochs"] = int(sections[13])
        params["path_suffix"] = sections[15]
        return params
    except Exception:
        return None


def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    return torch.load(model_path + ".dat", map_location=device)


def collect_all_models(models_path: str) -> Sequence[Dict]:
    model_params = []
    for dir_name in [os.path.join(models_path, d) for d in os.listdir(models_path)]:
        params = parse_model_path(os.path.basename(dir_name))
        if params is None:
            continue
        mpath = os.path.join(dir_name, "clf")
        if not file_exists(mpath + ".dat"):
            continue
        params["model_path"] = mpath
        params["dir"] = dir_name
        model_params.append(params)
        if params["dp_norm_clip"] != 0 and params["dp_noise"] != 0:
            params["dp"] = True
            params["epsilon"] = load_model(mpath).dp_epsilons[-1]
        else:
            params["dp"] = False
            params["epsilon"] = np.inf
    return model_params


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def attack_wrapper(
    mi_loaders: Tuple[DataLoader, DataLoader],
    idx: Tuple[np.ndarray, ...],
    n_attacker_train: int,
    seed: int,
    params: Dict,
    device: str,
    n_repeat: int,
    regular_train_epochs: Optional[int],
) -> None:
    save_dir = params["dir"]
    print(f"Attacking {os.path.basename(save_dir)} - |A|: {n_attacker_train} - S: {seed}...")

    results_path = os.path.join(save_dir, f"mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle")
    if file_exists(results_path):
        apply_mi_attack(None, None, None, save_dir, n_attacker_train, seed, device)  # replays cached prints
    else:
        clf = load_model(params["model_path"], device)
        apply_mi_attack(clf, mi_loaders, idx, save_dir, n_attacker_train=n_attacker_train, seed=seed, device=device)

    if params["laug_type"] != "no" or params["daug_type"] != "no":
        num_repeat = 1 if params["laug_type"] in ["distillation", "smooth"] else n_repeat
        results_path = os.path.join(
            save_dir, f"aware_mi_results_ntrain_{n_attacker_train}_numrepeat_{num_repeat}_randseed_{seed}.pickle"
        )
        if file_exists(results_path):
            apply_aware_attack(None, None, None, None, save_dir, n_attacker_train, num_repeat, seed, None, None)
        else:
            if params["laug_type"] == "distillation":
                teacher_dir = os.path.dirname(save_dir)
                path_suffix = params["path_suffix"]
                teacher_path = f"{params['dset_name']}_laug_no_0_daug_no_0_dp_nc_0_nm_0_epochs_{regular_train_epochs}_run_{path_suffix}"
                teacher = load_model(os.path.join(teacher_dir, teacher_path, "clf"), device)
            else:
                teacher = None
            clf = load_model(os.path.join(save_dir, "clf"), device)
            apply_aware_attack(
                clf,
                params,
                mi_loaders,
                idx,
                save_dir,
                n_attacker_train=n_attacker_train,
                n_repeat=num_repeat,
                seed=seed,
                teacher=teacher,
                device=device,
            )
    print("--------------------------------------------")


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    n_attacker_train = cfg["attack"]["n_attacker_train"]
    n_attacker_test = cfg["attack"]["n_attacker_test"]
    seeds = cfg["attack"]["sampling_random_seeds"]
    n_repeat = cfg["attack"]["n_aware_repeat"]
    ds_names = cfg["training_datasets"]
    models_path = cfg["models_path"]
    device = get_pytorch_device()
    regular_train_epochs = cfg["training_num_epochs"]

    for ds_name in ds_names:
        path = os.path.join(models_path, ds_name)
        all_model_params = collect_all_models(path)
        print(f"There are {len(all_model_params)} models in {path}.")
        datasets_pair = get_ds(ds_name, device)
        for seed in seeds:
            mi_loaders, idx = take_subset_from_datasets(datasets_pair, seed, n_attacker_train, n_attacker_test, device=device)
            for params in all_model_params:
                attack_wrapper(mi_loaders, idx, n_attacker_train, seed, params, device=device, n_repeat=n_repeat, regular_train_epochs=regular_train_epochs)


if __name__ == "__main__":
    set_random_seeds()
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python argmentation_mia.py path/to/config.json")
    main(sys.argv[1])
