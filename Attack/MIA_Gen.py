#!/usr/bin/env python3
"""
PFAMI: Probabilistic Fluctuation based Membership Inference Attack (single-file version).

This file consolidates the original MIA-Gen implementation (attack_model_PFAMI, utils,
resnet, data.prepare, and the attacker entry) into one self-contained module. It can be
imported as a library or executed directly. The code keeps the original algorithmic
behavior while reducing cross-file dependencies to simplify reuse in other projects.
"""
import logging
import os
import random
import time
from copy import deepcopy
from typing import Any, Dict as TypingDict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # heavy but only used for visualization helpers
except Exception:  # pragma: no cover - plotting is optional
    plt = None
    sns = None

try:
    from diffusers import DiffusionPipeline, UNet2DModel, DDPMScheduler, DDPMPipeline
except Exception:  # pragma: no cover - only needed for diffusion targets
    DiffusionPipeline = None
    UNet2DModel = None
    DDPMScheduler = None
    DDPMPipeline = None

try:
    from pythae.models import AutoModel
except Exception:  # pragma: no cover - only needed for VAE targets
    AutoModel = None

try:
    from datasets import Image, Dataset
except Exception:  # pragma: no cover - HF datasets are optional
    Image = None
    Dataset = None

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:
    roc_auc_score = None
    roc_curve = None

# ---------------------------------------------------------------------------
# Logging and device helpers
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("MIA_Gen")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Minimal Dict helper (kept for backward compatibility)
# ---------------------------------------------------------------------------
class Dict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"Dict has no attribute {name}")

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)


# ---------------------------------------------------------------------------
# Utility functions (merged from utils.py)
# ---------------------------------------------------------------------------
def check_files_exist(*file_paths: str) -> bool:
    return all(os.path.isfile(fp) for fp in file_paths)


def create_folder(folder_path: str) -> None:
    os.makedirs(folder_path, exist_ok=True)


def save_dict_to_npz(my_dict: TypingDict[str, np.ndarray], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        np.savez(f, **my_dict)


def load_dict_from_npz(file_path: str) -> Dict:
    with np.load(file_path) as data:
        return Dict({k: v for k, v in data.items() if isinstance(v, np.ndarray)})


def ndarray_to_tensor(*ndarrays: np.ndarray) -> Tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(arr).float().to(DEVICE) for arr in ndarrays)


def tensor_to_ndarray(*tensors: torch.Tensor) -> Tuple[np.ndarray, ...]:
    return tuple(t.detach().cpu().numpy() for t in tensors)


def convert_labels_to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


def get_file_names(folder_path: str) -> list:
    return [os.path.join(folder_path, fn) for fn in sorted(os.listdir(folder_path)) if os.path.isfile(os.path.join(folder_path, fn))]


def extract(v: torch.Tensor, t: torch.Tensor, x_shape: Iterable[int]) -> torch.Tensor:
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# ---------------------------------------------------------------------------
# Tiny ResNet (merged from resnet.py)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, st))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feat: bool = False, return_mid_feat: bool = False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        mid_feat = out
        out = F.avg_pool2d(out, 4)
        feat = out.view(out.size(0), -1)
        out = self.linear(feat)
        if return_feat:
            return out, feat
        if return_mid_feat:
            return out, mid_feat
        return out


def ResNet18(num_channels: int = 3, num_classes: int = 10, num_blocks=(2, 2, 2, 2)) -> ResNet:
    return ResNet(BasicBlock, num_blocks=num_blocks, num_channels=num_channels, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Data preparation (merged from data/prepare.py)
# ---------------------------------------------------------------------------
def _augmentations(norm: bool = False):
    return transforms.Compose(
        [
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) if norm else transforms.Lambda(lambda x: x),
        ]
    )


def transform_images_dataset(examples, norm: bool = False):
    augment = _augmentations(norm=norm)
    images = [augment(img.convert("RGB")).numpy() for img in examples["image"]]
    return {"input": images}


def data_prepare(dataset_name: str = "celeba", mode: str = "datasets", data_root: Optional[str] = None, max_count: Optional[int] = None):
    import datasets as hf_datasets
    from torchvision.datasets import CIFAR10

    default_root = os.environ.get("MIA_DATA_ROOT", "/data/coding/data")
    data_root = data_root or default_root

    if dataset_name == "celeba":
        files = get_file_names(os.path.expanduser("~/MIA/MIA-Gen/VAEs/data/celeba64/total"))
    elif dataset_name == "tinyin":
        files = get_file_names(os.path.expanduser("~/MIA/MIA-Gen/VAEs/data/Tiny-IN"))
    elif dataset_name == "cifar10":
        train_ds = CIFAR10(root=data_root, train=True, download=False)
        test_ds = CIFAR10(root=data_root, train=False, download=False)
        imgs = [img for img, _ in train_ds] + [img for img, _ in test_ds]
        if max_count is not None:
            imgs = imgs[:max_count]
        full_ds = hf_datasets.Dataset.from_dict({"image": imgs}).cast_column("image", hf_datasets.Image())
        return full_ds if mode == "datasets" else np.stack([transform_images_dataset({"image": [im]})["input"][0] for im in imgs])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    full_dataset = hf_datasets.Dataset.from_dict({"image": files}).cast_column("image", hf_datasets.Image())
    if max_count is not None:
        full_dataset = full_dataset.select(range(min(len(full_dataset), max_count)))

    if mode == "datasets":
        return full_dataset
    tensor_path = "/mnt/data0/fuwenjie/MIA/MIA-Gen/data/datasets/celeba/celeba64.npz"
    if not check_files_exist(tensor_path):
        LOGGER.info("Generating cached tensor file for dataset...")
        full_dataset.set_transform(lambda ex: transform_images_dataset(ex, norm=False))
        images = [item["input"][None, :, :, :] for item in full_dataset]
        full_dataset = np.concatenate(images, axis=0)
        np.savez(tensor_path, full_dataset)
    else:
        full_dataset = np.load(tensor_path)["arr_0"]
    return full_dataset


# ---------------------------------------------------------------------------
# Attack implementation (merged from attack_model_PFAMI.py)
# ---------------------------------------------------------------------------
class MLAttckerModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(3, 8, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(8, 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=1)
        self.output_layer = nn.Linear(5, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        output = self.output_layer(x)
        return output.squeeze(1)


class AttackModel:
    def __init__(self, target_model: Any, datasets: dict, reference_model: Any, shadow_model: Any, cfg: dict, device: torch.device = DEVICE):
        self.device = device
        self.target_model = target_model
        self.datasets = datasets
        self.kind = cfg["attack_kind"]
        self.cfg = cfg
        self.base_dir = cfg.get("base_dir", BASE_DIR)
        self.shadow_model = shadow_model if shadow_model is not None and cfg.get("attack_kind") == "nn" else None
        self.reference_model = reference_model
        self.is_model_training = False
        if cfg["target_model"] == "vae" and target_model is not None:
            self.target_model_revision(target_model)

    # --------------------------- basic losses ---------------------------
    @staticmethod
    def loss_fn(logp, target, length, mean, logv):
        target = target[:, : torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        nll_loss = torch.nn.NLLLoss(ignore_index=0, reduction="none")(logp, target)
        nll_loss = nll_loss.reshape(100, -1).sum(-1)
        kl_loss = -0.5 * (1 + logv - mean.pow(2) - logv.exp()).sum(-1)
        kl_weight = 0.5
        return nll_loss + kl_loss * kl_weight

    @staticmethod
    def ddpm_loss(pipeline, clean_images, timestep):
        model = pipeline.unet
        model.eval()
        noise_scheduler = pipeline.scheduler
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        timesteps = torch.full((clean_images.shape[0],), timestep, device=clean_images.device)
        noise_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        model_output = model(noise_images, timesteps).sample
        loss = F.mse_loss(model_output, noise, reduction="none").mean(dim=(1, 2, 3))
        return tensor_to_ndarray(loss)[0]

    @staticmethod
    def ddim_singlestep(pipeline, x, t_c, t_target):
        model = pipeline.unet
        model.eval()
        noise_scheduler = pipeline.scheduler
        x = x.to(x.device)
        t_c = x.new_ones([x.shape[0]], dtype=torch.long) * t_c
        t_target = x.new_ones([x.shape[0]], dtype=torch.long) * t_target
        betas = noise_scheduler.betas.to(x.device)
        alphas = torch.cumprod(1.0 - betas, dim=0)
        alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
        alphas_t_target = extract(alphas, t=t_target, x_shape=x.shape)
        with torch.no_grad():
            epsilon = model(x, t_c).sample
        pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / (alphas_t_c.sqrt())
        x_t_target = alphas_t_target.sqrt() * pred_x_0 + (1 - alphas_t_target).sqrt() * epsilon
        return {"x_t_target": x_t_target, "epsilon": epsilon}

    def ddim_multistep(self, pipeline, x, t_c, target_steps, clip: bool = False):
        for t_target in target_steps:
            result = self.ddim_singlestep(pipeline, x, t_c, t_target)
            x = result["x_t_target"]
            t_c = t_target
        if clip:
            result["x_t_target"] = torch.clip(result["x_t_target"], -1, 1)
        return result

    def ddim_loss(self, model, x, timestep: int = 10, t_sec: int = 100):
        target_steps = list(range(0, t_sec, timestep))[1:]
        x_sec = self.ddim_multistep(model, x, t_c=0, target_steps=target_steps)["x_t_target"]
        recon = self.ddim_singlestep(model, x_sec, t_c=target_steps[-1], t_target=target_steps[-1] + timestep)
        recon = self.ddim_singlestep(model, recon["x_t_target"], t_c=target_steps[-1] + timestep, t_target=target_steps[-1])["x_t_target"]
        loss = (recon - x_sec) ** 2
        loss = loss.flatten(1).sum(dim=-1)
        return tensor_to_ndarray(loss)[0]

    # --------------------------- evaluation ---------------------------
    def diffusion_eval(self, model, input_dataset, cfg):
        outputs = []
        data_loader = DataLoader(
            dataset=input_dataset,
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
        )
        pipeline = model
        loss_function = getattr(self, cfg["loss_kind"] + "_loss")
        sample_steps = cfg["diffusion_sample_steps"]
        for _, batch in enumerate(data_loader):
            clean_images = batch["input"].to(self.device)
            batch_loss = np.zeros((clean_images.shape[0], cfg["diffusion_sample_number"]))
            for i, timestep in enumerate(sample_steps):
                if cfg["loss_kind"] == "ddpm":
                    loss = self.ddpm_loss(pipeline, clean_images, timestep)
                else:
                    loss = self.ddim_loss(pipeline, clean_images, t_sec=timestep)
                batch_loss[:, i] = loss
            outputs.append(batch_loss)
        return np.concatenate(outputs, axis=0)

    def vae_eval(self, model, input_dataset, cfg):
        outputs = []
        data_loader = DataLoader(
            dataset=input_dataset,
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
        )
        for _, batch in enumerate(data_loader):
            batch_loss = np.zeros((batch["input"].shape[0], cfg["extensive_per_num"]))
            for i in range(cfg["extensive_per_num"]):
                input_dict = {"data": batch["input"].to(self.device)}
                output_batch = self.output_reformat(model(input_dict)).loss
                batch_loss[:, i] = output_batch
            outputs.append(batch_loss)
        return np.concatenate(outputs, axis=0)

    def eval_perturb(self, model, dataset, cfg):
        model_eval = getattr(self, cfg["target_model"] + "_eval")
        perturb_fn = self.image_dataset_perturbation if cfg["target_model"] == "vae" else self.norm_image_dataset_perturbation
        ori_dataset = deepcopy(dataset)
        ori_dataset.set_transform(self.transform_images if cfg["target_model"] == "vae" else self.norm_transform_images)
        ori_losses = model_eval(model, ori_dataset, cfg)
        ref_ori_losses = model_eval(self.reference_model, ori_dataset, cfg) if cfg.get("calibration") else None
        strengths = np.linspace(cfg["start_strength"], cfg["end_strength"], cfg["perturbation_number"])
        per_losses, ref_per_losses = [], []
        for strength in strengths:
            per_dataset = perturb_fn(dataset, strength=strength)
            per_loss = model_eval(model, per_dataset, cfg)
            per_losses.append(np.expand_dims(per_loss, -1))
            if cfg.get("calibration"):
                ref_loss = model_eval(self.reference_model, per_dataset, cfg)
                ref_per_losses.append(np.expand_dims(ref_loss, -1))
        per_losses = np.concatenate(per_losses, axis=-1)
        var_losses = per_losses - np.expand_dims(ori_losses, -1)
        ref_per_losses = np.concatenate(ref_per_losses, axis=-1) if cfg.get("calibration") else None
        ref_var_losses = ref_per_losses - np.expand_dims(ref_ori_losses, -1) if cfg.get("calibration") else None
        return (
            Dict(per_losses=per_losses, ori_losses=ori_losses, var_losses=var_losses),
            Dict(ref_per_losses=ref_per_losses, ref_ori_losses=ref_ori_losses, ref_var_losses=ref_var_losses),
        )

    def gen_data_diffusion(self, model, img_path, sample_numbers: int = 100, batch_size: int = 100):
        if DiffusionPipeline is None:
            raise ImportError("diffusers is required for diffusion data generation")
        pipeline = model
        generated_samples = []
        for i in range(0, sample_numbers, batch_size):
            generated_samples.extend(pipeline(batch_size).images)
        create_folder(img_path)
        for idx, img in enumerate(generated_samples):
            img.save(os.path.join(img_path, f"{idx}.jpg"))
        files = get_file_names(img_path)
        return Dataset.from_dict({"image": files}).cast_column("image", Image())

    def data_prepare(self, kind: str, cfg):
        LOGGER.info("Preparing data (%s)...", kind)
        data_path = os.path.join(self.base_dir, cfg["attack_data_path"], f"attack_data_{cfg['target_model']}@{cfg['dataset']}")
        mem_data = self.datasets[kind]["train"]
        nonmem_data = self.datasets[kind]["valid"]
        mem_path = os.path.join(data_path, kind, "mem_feat.npz")
        nonmem_path = os.path.join(data_path, kind, "nonmen_feat.npz")
        ref_mem_path = os.path.join(data_path, kind, "ref_mem_feat.npz")
        ref_nonmem_path = os.path.join(data_path, kind, "ref_nonmen_feat.npz")
        paths = (mem_path, nonmem_path, ref_mem_path, ref_nonmem_path) if cfg.get("calibration") else (mem_path, nonmem_path)
        if not check_files_exist(*paths) or not cfg.get("load_attack_data", False):
            LOGGER.info("Generating features for %s...", kind)
            mem_feat, ref_mem_feat = self.eval_perturb(self.target_model if kind == "target" else self.shadow_model, mem_data, cfg)
            nonmem_feat, ref_nonmem_feat = self.eval_perturb(self.target_model if kind == "target" else self.shadow_model, nonmem_data, cfg)
            save_dict_to_npz(mem_feat, mem_path)
            save_dict_to_npz(nonmem_feat, nonmem_path)
            if cfg.get("calibration"):
                save_dict_to_npz(ref_mem_feat, ref_mem_path)
                save_dict_to_npz(ref_nonmem_feat, ref_nonmem_path)
        else:
            LOGGER.info("Loading cached features for %s", kind)
            mem_feat = load_dict_from_npz(mem_path)
            nonmem_feat = load_dict_from_npz(nonmem_path)
            ref_mem_feat = load_dict_from_npz(ref_mem_path) if cfg.get("calibration") else None
            ref_nonmem_feat = load_dict_from_npz(ref_nonmem_path) if cfg.get("calibration") else None
        return Dict(mem_feat=mem_feat, nonmem_feat=nonmem_feat, ref_mem_feat=ref_mem_feat, ref_nonmem_feat=ref_nonmem_feat)

    def feat_prepare(self, info_dict: Dict, cfg):
        if cfg.get("calibration"):
            mem_feat = info_dict.mem_feat.var_losses / np.expand_dims(info_dict.mem_feat.ori_losses, -1) - info_dict.ref_mem_feat.ref_var_losses / np.expand_dims(info_dict.ref_mem_feat.ref_ori_losses, -1)
            nonmem_feat = info_dict.nonmem_feat.var_losses / np.expand_dims(info_dict.nonmem_feat.ori_losses, -1) - info_dict.ref_nonmem_feat.ref_var_losses / np.expand_dims(info_dict.ref_nonmem_feat.ref_ori_losses, -1)
        else:
            mem_feat = info_dict.mem_feat.var_losses / np.expand_dims(info_dict.mem_feat.ori_losses, -1)
            nonmem_feat = info_dict.nonmem_feat.var_losses / np.expand_dims(info_dict.nonmem_feat.ori_losses, -1)
        if cfg["attack_kind"] == "stat":
            mem_feat = mem_feat[:, :, 5]
            nonmem_feat = nonmem_feat[:, :, 5]
            mem_feat[np.isnan(mem_feat)] = 0
            nonmem_feat[np.isnan(nonmem_feat)] = 0
            feat = np.concatenate([mem_feat.mean(axis=-1), nonmem_feat.mean(axis=-1)])
            ground_truth = np.concatenate([np.zeros(mem_feat.shape[0]), np.ones(nonmem_feat.shape[0])]).astype(int)
        else:
            mem_tensor, nonmem_tensor = ndarray_to_tensor(mem_feat, nonmem_feat)
            if cfg["target_model"] == "vae":
                mem_tensor.sort(axis=1)
                nonmem_tensor.sort(axis=1)
            feat = torch.cat([mem_tensor, nonmem_tensor]).unsqueeze(1)
            feat[torch.isnan(feat)] = 0
            ground_truth = torch.cat([torch.zeros(mem_tensor.shape[0]), torch.ones(nonmem_tensor.shape[0])]).long().to(self.device)
        return feat, ground_truth

    def attack_model_training(self, cfg):
        save_path = os.path.join(self.base_dir, cfg["attack_data_path"], f"attack_model_{cfg['target_model']}@{cfg['dataset']}", "attack_model.pth")
        raw_info = self.data_prepare("shadow", cfg)
        eval_raw_info = self.data_prepare("target", cfg)
        feat, ground_truth = self.feat_prepare(raw_info, cfg)
        eval_feat, eval_ground_truth = self.feat_prepare(eval_raw_info, cfg)
        feature_dim = feat.shape[-1]
        attack_model = ResNet18(num_channels=1, num_classes=2).to(self.device)
        if cfg.get("load_trained") and check_files_exist(save_path):
            attack_model.load_state_dict(torch.load(save_path, map_location=self.device))
            self.attack_model = attack_model
            self.is_model_training = True
            return
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=0)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1], device=self.device).float())
        for epoch in range(cfg["epoch_number"]):
            attack_model.train()
            predict = attack_model(feat)
            optimizer.zero_grad()
            loss = criterion(predict, ground_truth)
            loss.backward()
            optimizer.step()
            if epoch % 1 == 0:
                attack_model.eval()
                with torch.no_grad():
                    eval_predict = attack_model(eval_feat)
                LOGGER.info("Epoch %d - Loss %.4f", epoch, loss.item())
                self.eval_attack(ground_truth, predict[:, 1], plot=False)
                self.eval_attack(eval_ground_truth, eval_predict[:, 1], plot=False)
        self.is_model_training = True
        self.attack_model = attack_model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(attack_model.state_dict(), save_path)

    def conduct_attack(self, cfg):
        save_path = os.path.join(self.base_dir, cfg["attack_data_path"], f"attack_data_{cfg['target_model']}@{cfg['dataset']}", f"roc_{cfg['attack_kind']}.npz")
        raw_info = self.data_prepare("target", cfg)
        feat, ground_truth = self.feat_prepare(raw_info, cfg)
        if cfg["attack_kind"] == "nn":
            if not self.is_model_training:
                self.attack_model_training(cfg)
            predict = self.attack_model(feat)
            self.eval_attack(ground_truth, predict[:, 1], path=save_path)
        else:
            self.eval_attack(ground_truth, -feat, path=save_path)

    # --------------------------- plotting & metrics ---------------------------
    @staticmethod
    def eval_attack(y_true, y_scores, plot: bool = True, path: Optional[str] = None):
        if roc_curve is None or roc_auc_score is None:
            LOGGER.warning("scikit-learn not available; skipping ROC computation")
            return None
        if isinstance(y_true, torch.Tensor):
            y_true, y_scores = tensor_to_ndarray(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        if path is not None:
            create_folder(os.path.dirname(path))
            np.savez(path, fpr=fpr, tpr=tpr)
        auc_score = roc_auc_score(y_true, y_scores)
        LOGGER.info("AUC: %.4f", auc_score)
        threshold_point = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]
        LOGGER.info("ASR: %.4f", threshold_point)
        tpr_1fpr = tpr[np.argmin(np.abs(fpr - 0.01))]
        LOGGER.info("TPR@1%%FPR: %.4f", tpr_1fpr)
        if plot and plt is not None:
            plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.2f}; ASR={threshold_point:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.show()
        return auc_score

    @staticmethod
    def heatmap_plot(data, **kwargs):  # pragma: no cover - visualization helper
        if sns is None or plt is None:
            LOGGER.warning("seaborn/matplotlib not available; skip heatmap")
            return
        ax = sns.heatmap(data, annot=True, fmt=".2f", **kwargs)
        ax.set_ylabel(r"Perturbation Strength Factor $\lambda$", fontsize=18)
        ax.set_xlabel("Time step $t$", fontsize=18)
        ax.set_yticklabels([f"{label:.2f}" for label in np.linspace(0.95, 0.7, 10)], fontsize=14, rotation=45)
        ax.set_xticklabels([0, 50, 100, 150, 200, 250, 300, 350, 400, 450], fontsize=14, rotation=45)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def vec_heatmap(vec, **kwargs):  # pragma: no cover - visualization helper
        if sns is None or plt is None:
            LOGGER.warning("seaborn/matplotlib not available; skip heatmap")
            return
        vec = vec.reshape((1, -1))
        ax = sns.heatmap(vec, annot=True, yticklabels=False, square=True, cbar=False, annot_kws={"fontsize": 14}, fmt=".2f", **kwargs)
        ax.set_xticklabels([0, 50, 100, 150, 200, 250, 300, 350, 400, 450], fontsize=14)
        ax.set_xlabel("Time step $t$", fontsize=18)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def distinguishability_plot(mem, non_mem):  # pragma: no cover - visualization helper
        if sns is None or plt is None:
            LOGGER.warning("seaborn/matplotlib not available; skip plot")
            return
        sns.kdeplot(mem, fill=True, color="indianred", alpha=0.5)
        sns.kdeplot(non_mem, fill=True, color="forestgreen", alpha=0.5)
        mem_mean, mem_std = round(mem.mean(), 2), round(mem.std(), 2)
        non_mean, non_std = round(non_mem.mean(), 2), round(non_mem.std(), 2)
        plt.xlabel(r"$\Delta \widehat{p}_{\theta}$", fontsize=22, labelpad=10)
        plt.ylabel("Density", fontsize=22, labelpad=10)
        plt.legend(["Member", "Non-member"], fontsize=20, loc="upper right")
        plt.xlim([-0.6, 0.9])
        plt.tick_params(labelsize=16)
        plt.text(0.63, 0.25, f"mu_Mem={mem_mean:.2f}\nsigma_Mem={mem_std:.2f}", transform=plt.gca().transAxes, fontsize=20)
        plt.text(0.04, 0.6, f"mu_Non={non_mean:.2f}\nsigma_Non={non_std:.2f}", transform=plt.gca().transAxes, fontsize=20)
        plt.tight_layout()
        plt.show()

    # --------------------------- perturbations & misc ---------------------------
    @staticmethod
    def sentence_perturb(dataset, embedding, rate: float = 0.1):
        sim = torch.mm(embedding, embedding.T)
        prop = F.softmax(sim.fill_diagonal_(float("-inf")), dim=1).cpu().numpy()
        per_dataset = deepcopy(dataset)
        for idx in range(len(per_dataset)):
            ori_data = per_dataset[idx]
            sen_len = ori_data["length"]
            ori_sen = ori_data["input"][1:sen_len]
            per_sen = []
            for word in ori_sen:
                if random.random() < rate:
                    per_word = int(np.random.choice(len(prop[word, :]), p=prop[word, :]))
                    per_sen.append(per_word)
                else:
                    per_sen.append(word)
            input_sen = [2] + per_sen
            input_sen.extend([0] * (60 - sen_len))
            target_sen = per_sen + [3]
            target_sen.extend([0] * (60 - sen_len))
            per_dataset.data[str(idx)] = {"input": input_sen, "target": target_sen, "length": sen_len}
        return per_dataset

    @staticmethod
    def gaussian_noise_tensor(tensor, mean: float = 0.0, std: float = 0.1):
        noise = torch.randn(tensor.shape) * std + mean
        noisy_tensor = torch.clamp(tensor + noise, -1.0, 1.0)
        return noisy_tensor

    @staticmethod
    def norm_transform_images(examples):
        augment = transforms.Compose(
            [
                transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        images = [augment(img.convert("RGB")) for img in examples["image"]]
        return {"input": images}

    @staticmethod
    def transform_images(examples):
        augment = transforms.Compose(
            [
                transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        )
        images = [augment(img.convert("RGB")) for img in examples["image"]]
        return {"input": images}

    @staticmethod
    def norm_image_dataset_perturbation(dataset, strength: float):
        perturb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(size=int(64 * strength)),
                transforms.Resize(size=64),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        def transform_fn(examples):
            return {"input": [perturb(img.convert("RGB")) for img in examples["image"]]}
        per_dataset = deepcopy(dataset)
        per_dataset.set_transform(transform_fn)
        return per_dataset

    @staticmethod
    def image_dataset_perturbation(dataset, strength: float):
        perturb = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(size=int(64 * strength)),
            transforms.Resize(size=64),
        ])
        def transform_fn(examples):
            return {"input": [perturb(img.convert("RGB")) for img in examples["image"]]}
        per_dataset = deepcopy(dataset)
        per_dataset.set_transform(transform_fn)
        return per_dataset

    @staticmethod
    def output_reformat(output_dict: TypingDict[str, torch.Tensor]):
        for key in list(output_dict.keys()):
            output_dict[key] = output_dict[key].detach().cpu().numpy()
        return output_dict

    @staticmethod
    def loss_function(self, recon_x, x, mu, log_var, z):
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none").sum(dim=-1)
        else:
            recon_loss = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none").sum(dim=-1)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return (recon_loss + kld), recon_loss, kld

    @staticmethod
    def frequency(data: np.ndarray, interval: Tuple[float, float] = (-1, 1), split: int = 50):
        freq_vec = np.empty((data.shape[0], split))
        intervals = np.linspace(interval[0], interval[1], split + 1)
        for i in range(data.shape[0]):
            for j in range(split):
                freq_vec[i][j] = len(np.where(np.logical_and(data[i] >= intervals[j], data[i] <= intervals[j + 1]))[0])
        return freq_vec

    # --------------------------- misc ---------------------------
    def target_model_revision(self, model):
        if not hasattr(model.__class__, "loss_function"):
            model.__class__.loss_function = self.loss_function


# ---------------------------------------------------------------------------
# Model/data loading and runner
# ---------------------------------------------------------------------------
def build_toy_diffusion(sample_size: int = 64, device: torch.device = DEVICE):
    if UNet2DModel is None or DDPMScheduler is None or DDPMPipeline is None:
        raise ImportError("diffusers is required to build toy diffusion models")
    unet = UNet2DModel(
        sample_size=sample_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(num_train_timesteps=100)
    return DDPMPipeline(unet=unet, scheduler=scheduler).to(device)


def split_datasets(all_dataset, sample_number: int, seed: int = 42):
    rng = random.Random(seed)
    needed = sample_number * 6
    if len(all_dataset) < needed:
        raise ValueError(f"Not enough samples ({len(all_dataset)}) for requested splits ({needed}).")
    indices = rng.sample(range(len(all_dataset)), needed)
    def take(start):
        slice_idx = indices[start : start + sample_number]
        return all_dataset.select(slice_idx)
    return {
        "target": {"train": take(0), "valid": take(sample_number)},
        "shadow": {"train": take(sample_number * 2), "valid": take(sample_number * 3)},
        "reference": {"train": take(sample_number * 4), "valid": take(sample_number * 5)},
    }


def load_models_and_data(cfg: dict, device: torch.device = DEVICE):
    # Load target/shadow/reference models
    if cfg["target_model"] == "diffusion":
        if cfg["dataset"] == "cifar10":
            target_model = build_toy_diffusion(sample_size=64, device=device)
            shadow_model = build_toy_diffusion(sample_size=64, device=device) if cfg.get("attack_kind") == "nn" else None
            reference_model = build_toy_diffusion(sample_size=64, device=device) if cfg.get("calibration") else None
        else:
            if DiffusionPipeline is None:
                raise ImportError("diffusers is required for diffusion models")
            target_path = cfg.get("target_path")
            shadow_path = cfg.get("shadow_path")
            reference_path = cfg.get("reference_path")
            target_model = DiffusionPipeline.from_pretrained(target_path).to(device)
            shadow_model = DiffusionPipeline.from_pretrained(shadow_path).to(device) if shadow_path else None
            reference_model = DiffusionPipeline.from_pretrained(reference_path).to(device) if reference_path else None
    elif cfg["target_model"] == "vae":
        if AutoModel is None:
            raise ImportError("pythae is required for VAE models")
        target_path = cfg.get("target_path")
        reference_path = cfg.get("reference_path")
        shadow_path = cfg.get("shadow_path")
        target_model = AutoModel.load_from_folder(target_path).to(device)
        reference_model = AutoModel.load_from_folder(reference_path).to(device) if cfg.get("calibration") else None
        shadow_model = AutoModel.load_from_folder(shadow_path).to(device) if cfg.get("attack_kind") == "nn" else None
    else:
        raise ValueError(f"Unsupported target_model: {cfg['target_model']}")

    # Load datasets
    all_dataset = data_prepare(cfg["dataset"], mode="datasets", max_count=cfg.get("max_data_count"))
    if cfg["dataset"] == "cifar10":
        datasets = split_datasets(all_dataset, cfg["sample_number"], seed=cfg.get("random_seed", 42))
    elif cfg["dataset"] == "celeba":
        datasets = {
            "target": {
                "train": Dataset.from_dict(all_dataset[random.sample(range(0, 50000), cfg["sample_number"])] , features=all_dataset.features),
                "valid": Dataset.from_dict(all_dataset[random.sample(range(50000, 60000), cfg["sample_number"])] , features=all_dataset.features),
            },
            "shadow": {
                "train": Dataset.from_dict(all_dataset[random.sample(range(60000, 110000), cfg["sample_number"])] , features=all_dataset.features),
                "valid": Dataset.from_dict(all_dataset[random.sample(range(110000, 120000), cfg["sample_number"])] , features=all_dataset.features),
            },
            "reference": {
                "train": Dataset.from_dict(all_dataset[random.sample(range(120000, 170000), cfg["sample_number"])] , features=all_dataset.features),
                "valid": Dataset.from_dict(all_dataset[random.sample(range(170000, 180000), cfg["sample_number"])] , features=all_dataset.features),
            },
        }
    elif cfg["dataset"] == "tinyin":
        datasets = {
            "target": {
                "train": Dataset.from_dict(all_dataset[random.sample(range(0, 30000), cfg["sample_number"])] , features=all_dataset.features),
                "valid": Dataset.from_dict(all_dataset[random.sample(range(30000, 35000), cfg["sample_number"])] , features=all_dataset.features),
            },
            "shadow": {
                "train": Dataset.from_dict(all_dataset[random.sample(range(35000, 65000), cfg["sample_number"])] , features=all_dataset.features),
                "valid": Dataset.from_dict(all_dataset[random.sample(range(65000, 70000), cfg["sample_number"])] , features=all_dataset.features),
            },
            "reference": {
                "train": Dataset.from_dict(all_dataset[random.sample(range(70000, 100000), cfg["sample_number"])] , features=all_dataset.features),
                "valid": Dataset.from_dict(all_dataset[random.sample(range(100000, 105000), cfg["sample_number"])] , features=all_dataset.features),
            },
        }
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")

    return target_model, reference_model, shadow_model, datasets


# ---------------------------------------------------------------------------
# Entrypoint helpers
# ---------------------------------------------------------------------------
def load_config(config_path: Optional[str] = None) -> dict:
    import yaml
    cfg_path = os.environ.get("MIA_CONFIG", config_path or os.path.join(BASE_DIR, "configs", "config.yaml"))
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("base_dir", BASE_DIR)
    return cfg


def run_pfami(cfg: dict):
    seed = cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    target_model, reference_model, shadow_model, datasets = load_models_and_data(cfg, device=DEVICE)
    attack_model = AttackModel(target_model, datasets, reference_model, shadow_model, cfg, device=DEVICE)
    attack_model.conduct_attack(cfg)


if __name__ == "__main__":
    configuration = load_config()
    run_pfami(configuration)
