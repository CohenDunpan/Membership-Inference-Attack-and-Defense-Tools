"""
Single-file MemGuard defense implementation.

This module consolidates the original MemGuard codebase into a reusable
class that can be dropped into other membership inference attack/defense
projects. It keeps the original training and defense logic intact and uses
only PyTorch, NumPy, and configparser as dependencies.

Key components
--------------
- DatasetConfig: convenience wrapper for reading dataset-specific settings.
- InputData: dataset loader mirroring input_data_class.InputData.
- Model definitions: target model, defense discriminator, optimization model,
  and the attack NN used in the MemGuard paper.
- MemGuardDefense: end-to-end pipeline for training the target model, training
  the defense model, generating adversarial noise on outputs, and (optionally)
  evaluating an attack model against defended vs. original outputs.

Notes
-----
- All paths are resolved relative to the directory containing this file. The
  default config values follow the original "location" dataset setup. If you
  want to adapt to another dataset, update the config file or pass overrides
  to DatasetConfig.
- The API is designed so that callers can train once and reuse saved weights.
  Heavy steps (training and noise generation) are separated into methods.
- This file intentionally avoids any non-ASCII characters for portability.
"""
from __future__ import annotations

import argparse
import configparser
import dataclasses
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_dataloader(x_data: np.ndarray, y_data: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)
    denom = max(total_samples, 1)
    return total_loss / denom, total_correct / denom


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DatasetConfig:
    dataset: str = "location"
    all_data_path: str = "./data/location/data_complete.npz"
    shuffle_index: str = "./data/location/shuffle_index.npz"
    result_folder: str = "./result/location/code_publish/"
    result_file_publish: str = "result_publish_location_publish.ini"
    user_training_data_index_range: Dict[str, str] = dataclasses.field(default_factory=lambda: {"start": "0", "end": "1000"})
    user_testing_data_index_range: Dict[str, str] = dataclasses.field(default_factory=lambda: {"start": "1000", "end": "2000"})
    defense_member_data_index_range: Dict[str, str] = dataclasses.field(default_factory=lambda: {"start": "0", "end": "1000"})
    defense_nonmember_data_index_range: Dict[str, str] = dataclasses.field(default_factory=lambda: {"start": "1000", "end": "2000"})
    attacker_evaluate_member_data_range: Dict[str, str] = dataclasses.field(default_factory=lambda: {"start": "0", "end": "1000"})
    attacker_evaluate_nonmember_data_range: Dict[str, str] = dataclasses.field(default_factory=lambda: {"start": "2000", "end": "3000"})
    attacker_train_member_data_range: Dict[str, str] = dataclasses.field(default_factory=lambda: {"start": "3000", "end": "3500"})
    attacker_train_nonmember_data_range: Dict[str, str] = dataclasses.field(default_factory=lambda: {"start": "4000", "end": "4500"})
    num_classes: int = 30
    user_epochs: int = 200
    batch_size: int = 64
    defense_epochs: int = 400
    defense_batch_size: int = 64
    attack_epochs: int = 400
    attack_shallow_model_epochs: int = 200
    attack_shallow_model_batch_size: int = 64

    @staticmethod
    def from_config(config_path: str, dataset: str = "location") -> "DatasetConfig":
        parser = configparser.ConfigParser()
        parser.read(config_path)
        section = parser[dataset]
        return DatasetConfig(
            dataset=dataset,
            all_data_path=section.get("all_data_path", DatasetConfig.all_data_path),
            shuffle_index=section.get("shuffle_index", DatasetConfig.shuffle_index),
            result_folder=section.get("result_folder", DatasetConfig.result_folder),
            result_file_publish=section.get("result_file_publish", DatasetConfig.result_file_publish),
            user_training_data_index_range=eval(section.get("user_training_data_index_range")),  # noqa: S307
            user_testing_data_index_range=eval(section.get("user_testing_data_index_range")),
            defense_member_data_index_range=eval(section.get("defense_member_data_index_range")),
            defense_nonmember_data_index_range=eval(section.get("defense_nonmember_data_index_range")),
            attacker_evaluate_member_data_range=eval(section.get("attacker_evaluate_member_data_range")),
            attacker_evaluate_nonmember_data_range=eval(section.get("attacker_evaluate_nonmember_data_range")),
            attacker_train_member_data_range=eval(section.get("attacker_train_member_data_range")),
            attacker_train_nonmember_data_range=eval(section.get("attacker_train_nonmember_data_range")),
            num_classes=section.getint("num_classes"),
            user_epochs=section.getint("user_epochs"),
            batch_size=section.getint("batch_size"),
            defense_epochs=section.getint("defense_epochs"),
            defense_batch_size=section.getint("defense_batch_size"),
            attack_epochs=section.getint("attack_epochs"),
            attack_shallow_model_epochs=section.getint("attack_shallow_model_epochs"),
            attack_shallow_model_batch_size=section.getint("attack_shallow_model_batch_size"),
        )

    def resolve_paths(self, base_dir: str) -> None:
        self.all_data_path = os.path.normpath(os.path.join(base_dir, self.all_data_path))
        self.shuffle_index = os.path.normpath(os.path.join(base_dir, self.shuffle_index))
        self.result_folder = os.path.normpath(os.path.join(base_dir, self.result_folder)) + "/"


# ---------------------------------------------------------------------------
# Data input helpers
# ---------------------------------------------------------------------------

class InputData:
    def __init__(self, cfg: DatasetConfig, base_dir: str):
        self.cfg = cfg
        self.base_dir = base_dir

    def _load_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        npzdata = np.load(self.cfg.all_data_path)
        x_data = npzdata["x"]
        y_data = npzdata["y"]
        npzdata_index = np.load(self.cfg.shuffle_index)
        index_data = npzdata_index["x"]
        return x_data, y_data, index_data

    def input_data_user(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        x_data, y_data, index_data = self._load_arrays()
        idx_train = index_data[int(self.cfg.user_training_data_index_range["start"]): int(self.cfg.user_training_data_index_range["end"])]
        idx_test = index_data[int(self.cfg.user_testing_data_index_range["start"]): int(self.cfg.user_testing_data_index_range["end"])]
        x_train = x_data[idx_train, :]
        x_test = x_data[idx_test, :]
        y_train = y_data[idx_train] - 1.0
        y_test = y_data[idx_test] - 1.0
        return (x_train, y_train), (x_test, y_test)

    def input_data_defender(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_data, y_data, index_data = self._load_arrays()
        idx_member = index_data[int(self.cfg.defense_member_data_index_range["start"]): int(self.cfg.defense_member_data_index_range["end"])]
        idx_nonmember = index_data[int(self.cfg.defense_nonmember_data_index_range["start"]): int(self.cfg.defense_nonmember_data_index_range["end"])]
        x_member = x_data[idx_member, :]
        x_nonmember = x_data[idx_nonmember, :]
        y_member = y_data[idx_member]
        y_nonmember = y_data[idx_nonmember]

        x_train = np.concatenate((x_member, x_nonmember), axis=0)
        y_train = np.concatenate((y_member, y_nonmember), axis=0) - 1.0
        label = np.zeros([x_train.shape[0]], dtype=int)
        label[0: x_member.shape[0]] = 1
        return x_train, y_train, label

    def input_data_attacker_evaluate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_data, y_data, index_data = self._load_arrays()
        idx_member = index_data[int(self.cfg.attacker_evaluate_member_data_range["start"]): int(self.cfg.attacker_evaluate_member_data_range["end"])]
        idx_nonmember = index_data[int(self.cfg.attacker_evaluate_nonmember_data_range["start"]): int(self.cfg.attacker_evaluate_nonmember_data_range["end"])]
        x_member = x_data[idx_member, :]
        x_nonmember = x_data[idx_nonmember, :]
        y_member = y_data[idx_member]
        y_nonmember = y_data[idx_nonmember]

        x_eval = np.concatenate((x_member, x_nonmember), axis=0)
        y_eval = np.concatenate((y_member, y_nonmember), axis=0) - 1.0
        label = np.zeros([x_eval.shape[0]], dtype=int)
        label[0: x_member.shape[0]] = 1
        return x_eval, y_eval, label

    def input_data_attacker_adv1(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_data, y_data, index_data = self._load_arrays()
        idx_member = index_data[int(self.cfg.attacker_train_member_data_range["start"]): int(self.cfg.attacker_train_member_data_range["end"])]
        idx_nonmember = index_data[int(self.cfg.attacker_train_nonmember_data_range["start"]): int(self.cfg.attacker_train_nonmember_data_range["end"])]
        x_member = x_data[idx_member, :]
        x_nonmember = x_data[idx_nonmember, :]
        y_member = y_data[idx_member]
        y_nonmember = y_data[idx_nonmember]

        x_train = np.concatenate((x_member, x_nonmember), axis=0)
        y_train = np.concatenate((y_member, y_nonmember), axis=0) - 1.0
        label = np.zeros([x_train.shape[0]], dtype=int)
        label[0: x_member.shape[0]] = 1
        return x_train, y_train, label

    def input_data_attacker_shallow_model_adv1(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        x_data, y_data, index_data = self._load_arrays()
        idx_member = index_data[int(self.cfg.attacker_train_member_data_range["start"]): int(self.cfg.attacker_train_member_data_range["end"])]
        idx_nonmember = index_data[int(self.cfg.attacker_train_nonmember_data_range["start"]): int(self.cfg.attacker_train_nonmember_data_range["end"])]
        x_member = x_data[idx_member, :]
        x_nonmember = x_data[idx_nonmember, :]
        y_member = y_data[idx_member] - 1.0
        y_nonmember = y_data[idx_nonmember] - 1.0
        return (x_member, y_member), (x_nonmember, y_nonmember)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def _flatten_input_dim(input_shape) -> int:
    return int(np.prod(input_shape))


class UserModel(nn.Module):
    def __init__(self, input_dim: int, labels_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(128, labels_dim)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class DefenseModel(nn.Module):
    def __init__(self, input_dim: int, labels_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, labels_dim),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DefenseOptimizeModel(nn.Module):
    def __init__(self, input_dim: int, labels_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, labels_dim),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        softmaxed = F.softmax(x, dim=1)
        return self.net(softmaxed)


class AttackNNModel(nn.Module):
    def __init__(self, input_dim: int, labels_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, labels_dim),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Core MemGuard pipeline
# ---------------------------------------------------------------------------

class MemGuardDefense:
    def __init__(self, config_path: str = None, dataset: str = "location", device: str | torch.device = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path or os.path.join(self.base_dir, "config.ini")
        self.cfg = DatasetConfig.from_config(self.config_path, dataset=dataset)
        self.cfg.resolve_paths(self.base_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.data = InputData(self.cfg, self.base_dir)

    # ------------------------------------------------------------------
    # Training routines
    # ------------------------------------------------------------------
    def train_user_model(self, save: bool = True) -> str:
        set_seed(1000)
        (x_train, y_train), (x_test, y_test) = self.data.input_data_user()
        input_shape = x_train.shape[1:]
        model = UserModel(input_dim=_flatten_input_dim(input_shape), labels_dim=self.cfg.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        train_loader = _build_dataloader(x_train, y_train, self.cfg.batch_size, shuffle=True)
        test_loader = _build_dataloader(x_test, y_test, self.cfg.batch_size, shuffle=False)

        for epoch in range(self.cfg.user_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).long()
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 150 == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1

            if (epoch + 1) % 100 == 0:
                train_loss, train_acc = self._evaluate_multiclass(model, train_loader, criterion)
                test_loss, test_acc = self._evaluate_multiclass(model, test_loader, criterion)
                print(f"[User] Epoch {epoch}: train loss {train_loss:.4f}, train acc {train_acc:.4f}, test loss {test_loss:.4f}, test acc {test_acc:.4f}")

        weights_path = os.path.join(self.cfg.result_folder, "models", f"epoch_{self.cfg.user_epochs}_weights_user.pt")
        if save:
            _ensure_dir(os.path.dirname(weights_path))
            torch.save({"state_dict": model.state_dict()}, weights_path)
            print(f"Saved user model to {weights_path}")
        return weights_path

    def train_defense_model(self, user_weights_path: str = None, save: bool = True) -> str:
        set_seed(1000)
        x_train, _, l_train = self.data.input_data_defender()
        input_shape = x_train.shape[1:]

        # Load user model to produce confidence outputs
        user_weights = user_weights_path or os.path.join(self.cfg.result_folder, "models", f"epoch_{self.cfg.user_epochs}_weights_user.pt")
        if not os.path.isfile(user_weights):
            raise FileNotFoundError(f"User weights not found at {user_weights}")
        user_model = UserModel(input_dim=_flatten_input_dim(input_shape), labels_dim=self.cfg.num_classes).to(self.device)
        user_state = torch.load(user_weights, map_location=self.device)
        user_model.load_state_dict(user_state["state_dict"])
        user_model.eval()

        with torch.no_grad():
            logits_list = []
            loader = DataLoader(torch.tensor(x_train, dtype=torch.float32), batch_size=self.cfg.defense_batch_size, shuffle=False)
            for batch in loader:
                batch = batch.to(self.device)
                logits = user_model(batch)
                probs = torch.softmax(logits, dim=1)
                logits_list.append(probs.cpu().numpy())
            f_train = np.concatenate(logits_list, axis=0)

        f_train = np.sort(f_train, axis=1)
        model = DefenseModel(input_dim=_flatten_input_dim(f_train.shape[1:]), labels_dim=1).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        label_train = l_train.astype(np.float32)
        train_loader = _build_dataloader(f_train, label_train, self.cfg.defense_batch_size, shuffle=True)

        for epoch in range(self.cfg.defense_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).float()
                optimizer.zero_grad()
                logits = model(batch_x).squeeze(-1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                loss_val, acc_val = _evaluate_binary(model, train_loader, self.device)
                print(f"[Defense] Epoch {epoch}: loss {loss_val:.4f}, acc {acc_val:.4f}")

        weights_path = os.path.join(self.cfg.result_folder, "models", f"epoch_{self.cfg.defense_epochs}_weights_defense.pt")
        if save:
            _ensure_dir(os.path.dirname(weights_path))
            torch.save({"state_dict": model.state_dict()}, weights_path)
            print(f"Saved defense model to {weights_path}")
        return weights_path

    # ------------------------------------------------------------------
    # Noise generation (MemGuard defense)
    # ------------------------------------------------------------------
    def generate_adversarial_outputs(
        self,
        user_weights_path: str = None,
        defense_weights_path: str = None,
        c1: float = 1.0,
        c2: float = 10.0,
        c3_initial: float = 0.1,
        step_size: float = 0.1,
        max_iteration: int = 300,
        patience_c3: float = 1e5,
        tag: str = "evaluation",
    ) -> str:
        set_seed(1000)
        user_weights = user_weights_path or os.path.join(self.cfg.result_folder, "models", f"epoch_{self.cfg.user_epochs}_weights_user.pt")
        defense_weights = defense_weights_path or os.path.join(self.cfg.result_folder, "models", f"epoch_{self.cfg.defense_epochs}_weights_defense.pt")
        if not os.path.isfile(user_weights):
            raise FileNotFoundError(f"User weights not found at {user_weights}")
        if not os.path.isfile(defense_weights):
            raise FileNotFoundError(f"Defense weights not found at {defense_weights}")

        x_eval, _, l_eval = self.data.input_data_attacker_evaluate()
        input_shape = x_eval.shape[1:]

        # Load user model for logits on evaluation set
        user_model = UserModel(input_dim=_flatten_input_dim(input_shape), labels_dim=self.cfg.num_classes).to(self.device)
        user_state = torch.load(user_weights, map_location=self.device)
        user_model.load_state_dict(user_state["state_dict"])
        user_model.eval()

        f_eval_list = []
        logits_eval_list = []
        loader = DataLoader(torch.tensor(x_eval, dtype=torch.float32), batch_size=100, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = user_model(batch)
                probs = torch.softmax(logits, dim=1)
                logits_eval_list.append(logits.cpu().numpy())
                f_eval_list.append(probs.cpu().numpy())
        f_eval = np.concatenate(f_eval_list, axis=0)
        f_eval_logits = np.concatenate(logits_eval_list, axis=0)
        del user_model

        # Sorting and index bookkeeping
        sort_index = np.argsort(f_eval, axis=1)
        back_index = np.copy(sort_index)
        for i in np.arange(back_index.shape[0]):
            back_index[i, sort_index[i, :]] = np.arange(back_index.shape[1])
        f_eval_sorted = np.sort(f_eval, axis=1)
        f_eval_logits_sorted = np.sort(f_eval_logits, axis=1)

        model_opt = DefenseOptimizeModel(input_dim=_flatten_input_dim(f_eval_sorted.shape[1:]), labels_dim=1).to(self.device)
        defense_state = torch.load(defense_weights, map_location=self.device)
        model_opt.load_state_dict(defense_state["state_dict"])
        model_opt.eval()

        result_array = np.zeros_like(f_eval_sorted, dtype=np.float32)
        result_array_logits = np.zeros_like(f_eval_sorted, dtype=np.float32)
        success_fraction = 0.0

        for idx in range(f_eval_sorted.shape[0]):
            if idx % 100 == 0:
                print(f"[Noise] sample {idx}")

            origin_prob = torch.tensor(f_eval_sorted[idx, :], dtype=torch.float32, device=self.device).view(1, -1)
            origin_logit = torch.tensor(f_eval_logits_sorted[idx, :], dtype=torch.float32, device=self.device).view(1, -1)
            max_label = int(torch.argmax(origin_prob).item())

            label_mask = torch.zeros((1, self.cfg.num_classes), dtype=torch.float32, device=self.device)
            label_mask[0, max_label] = 1.0

            def _compute_loss(sample_f: torch.Tensor, c3_value: float):
                sample_f.retain_grad()
                output_logit = model_opt(sample_f).squeeze(-1)
                correct_label = torch.sum(label_mask * sample_f, dim=1)
                wrong_label = torch.max((1 - label_mask) * sample_f - 1e8 * label_mask, dim=1)[0]
                loss1 = torch.abs(output_logit)
                loss2 = F.relu(wrong_label - correct_label)
                loss3 = torch.sum(torch.abs(F.softmax(sample_f, dim=1) - origin_prob))
                return c1 * loss1 + c2 * loss2 + c3_value * loss3, output_logit

            sample_f = origin_logit.clone().detach().requires_grad_(True)
            with torch.no_grad():
                initial_score = torch.sigmoid(model_opt(sample_f)).item()

            if abs(initial_score - 0.5) <= 1e-5:
                success_fraction += 1.0
                result_array[idx, :] = origin_prob.cpu().numpy()[0, back_index[idx, :]]
                result_array_logits[idx, :] = origin_logit.cpu().numpy()[0, back_index[idx, :]]
                continue

            last_softmax = origin_prob.cpu().numpy()[0, back_index[idx, :]].copy()
            last_logits = origin_logit.cpu().numpy()[0, back_index[idx, :]].copy()

            success = True
            c3_value = c3_initial
            iterate_time = 1

            while success:
                sample_f = origin_logit.clone().detach().requires_grad_(True)
                result_max_label = -1
                result_score = initial_score
                step = 1
                while step < max_iteration and (result_max_label != max_label or (result_score - 0.5) * (initial_score - 0.5) > 0):
                    loss, _ = _compute_loss(sample_f, c3_value)
                    loss.backward()
                    grad = sample_f.grad.detach()
                    grad_norm = torch.norm(grad)
                    if grad_norm > 0:
                        grad = grad / grad_norm
                    sample_f = (sample_f - step_size * grad).detach().requires_grad_(True)
                    with torch.no_grad():
                        result_score = torch.sigmoid(model_opt(sample_f)).item()
                        result_max_label = int(torch.argmax(sample_f).item())
                    step += 1

                if max_label != result_max_label:
                    if iterate_time == 1:
                        print(f"[Noise] failed label mismatch at sample {idx}, c3 {c3_value}")
                        success_fraction -= 1.0
                    break

                if (result_score - 0.5) * (initial_score - 0.5) > 0:
                    if iterate_time == 1:
                        with torch.no_grad():
                            max_score = torch.max(F.softmax(sample_f, dim=1)).item()
                        print(f"[Noise] max iteration reached sample {idx}, score {result_score:.6f}, softmax max {max_score:.6f}, c3 {c3_value}")
                    break

                with torch.no_grad():
                    softmax_sample = F.softmax(sample_f, dim=1).cpu().numpy()[0, back_index[idx, :]]
                    sample_logits_cpu = sample_f.cpu().numpy()[0, back_index[idx, :]]
                    last_softmax[:] = softmax_sample
                    last_logits[:] = sample_logits_cpu

                iterate_time += 1
                c3_value *= 10
                if c3_value > patience_c3:
                    break

            success_fraction += 1.0
            result_array[idx, :] = last_softmax[:]
            result_array_logits[idx, :] = last_logits[:]

        print(f"Success fraction: {success_fraction / float(f_eval_sorted.shape[0])}")

        _ensure_dir(os.path.join(self.cfg.result_folder, "attack"))
        np.savez(
            os.path.join(self.cfg.result_folder, "attack", f"noise_data_{tag}.npz"),
            defense_output=result_array,
            defense_output_logits=result_array_logits,
            tc_output=f_eval,
            tc_output_logits=f_eval_logits,
        )
        return os.path.join(self.cfg.result_folder, "attack", f"noise_data_{tag}.npz")

    # ------------------------------------------------------------------
    # Optional attack evaluation (shadow attack model)
    # ------------------------------------------------------------------
    def evaluate_attack(
        self,
        noise_npz_path: str,
        shadow_weights_path: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        set_seed(10000)
        x_eval, _, l_eval = self.data.input_data_attacker_evaluate()
        label_test = l_eval.astype(np.float32)

        if not os.path.isfile(noise_npz_path):
            raise FileNotFoundError(noise_npz_path)
        npz_defense = np.load(noise_npz_path)
        f_evaluate_noise = npz_defense["defense_output"]
        f_evaluate_origin = npz_defense["tc_output"]

        (x_train, y_train), (x_test, y_test) = self.data.input_data_attacker_shallow_model_adv1()
        shadow_path = shadow_weights_path or os.path.join(self.cfg.result_folder, "models", f"epoch_{self.cfg.user_epochs}_weights_attack_shallow_model_adv1.pt")
        if not os.path.isfile(shadow_path):
            raise FileNotFoundError(f"Shadow model weights not found at {shadow_path}")
        shadow_model = UserModel(input_dim=_flatten_input_dim(x_train.shape[1:]), labels_dim=self.cfg.num_classes).to(self.device)
        shadow_state = torch.load(shadow_path, map_location=self.device)
        shadow_model.load_state_dict(shadow_state["state_dict"])
        shadow_model.eval()

        with torch.no_grad():
            logits_list = []
            loader_shadow = DataLoader(torch.tensor(np.concatenate([x_train, x_test], axis=0), dtype=torch.float32), batch_size=self.cfg.defense_batch_size, shuffle=False)
            for batch in loader_shadow:
                batch = batch.to(self.device)
                logits = shadow_model(batch)
                logits_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            f_train = np.concatenate(logits_list, axis=0)
        del shadow_model

        f_train = np.sort(f_train, axis=1)
        f_evaluate_defense = np.sort(f_evaluate_noise, axis=1)
        f_evaluate_origin = np.sort(f_evaluate_origin, axis=1)

        attack_model = AttackNNModel(input_dim=_flatten_input_dim(f_train.shape[1:]), labels_dim=1).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(attack_model.parameters(), lr=0.01)

        label_train = np.concatenate((np.ones(x_train.shape[0]), np.zeros(x_test.shape[0]))).astype(np.float32)
        train_loader = _build_dataloader(f_train, label_train, self.cfg.defense_batch_size, shuffle=True)
        test_loader_defense = _build_dataloader(f_evaluate_defense, label_test, self.cfg.defense_batch_size, shuffle=False)
        test_loader_origin = _build_dataloader(f_evaluate_origin, label_test, self.cfg.defense_batch_size, shuffle=False)

        for epoch in range(self.cfg.attack_epochs):
            attack_model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).float()
                optimizer.zero_grad()
                logits = attack_model(batch_x).squeeze(-1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 300 == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1

        loss_defense, acc_defense = _evaluate_binary(attack_model, test_loader_defense, self.device)
        loss_origin, acc_origin = _evaluate_binary(attack_model, test_loader_origin, self.device)
        print(f"[Attack] defense loss {loss_defense:.4f}, acc {acc_defense:.4f}")
        print(f"[Attack] origin loss {loss_origin:.4f}, acc {acc_origin:.4f}")
        return np.array([loss_defense, acc_defense]), np.array([loss_origin, acc_origin])

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _evaluate_multiclass(self, model: nn.Module, loader: DataLoader, criterion) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).long()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_x.size(0)
        denom = max(total_samples, 1)
        return total_loss / denom, total_correct / denom


# ---------------------------------------------------------------------------
# CLI entry points for quick experimentation
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="MemGuard single-file defense")
    parser.add_argument("--config", default=None, help="Path to config.ini; defaults to config beside this file")
    parser.add_argument("--dataset", default="location")
    parser.add_argument("--mode", choices=["train_user", "train_defense", "generate_noise"], default="generate_noise")
    args = parser.parse_args()

    mg = MemGuardDefense(config_path=args.config, dataset=args.dataset)

    if args.mode == "train_user":
        mg.train_user_model(save=True)
    elif args.mode == "train_defense":
        mg.train_defense_model(save=True)
    else:
        mg.generate_adversarial_outputs(tag="evaluation")


if __name__ == "__main__":
    _cli()
