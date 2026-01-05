# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# This file consolidates the PrivGAN defense (originally released as a
# research prototype) into a single, library-friendly module. The goal is to
# make the membership-inference-aware GAN training loop and its utilities easy
# to import from a single file inside the broader MIA attack/defense toolkit.
#
# The implementation mirrors the reference PyTorch code from the privGAN
# repository (PETS 2021), with light wrapping for convenience and small safety
# guards (e.g., batching when dataset splits are small). No algorithmic changes
# are intended.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats  # kept for compatibility; not used in the core loop
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


########################################
# Generic helpers
########################################


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    """Return a CUDA device when available unless an explicit device is given."""

    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(data: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32, device=device)


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _init_weights(module: nn.Module) -> None:
    # Match the reference Keras RandomNormal(stddev=0.02)
    if isinstance(module, (nn.Linear, nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_optimizer(model: nn.Module, lr: float = 0.0002, beta1: float = 0.5) -> torch.optim.Optimizer:
    """Adam optimizer used throughout the GANs."""

    return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))


########################################
# Dataset-specific building blocks
########################################


class MNIST_Generator(nn.Module):
    """Generator for MNIST (fully-connected, 784 output with tanh)."""

    def __init__(self, randomDim: int = 100):
        super().__init__()
        self.randomDim = randomDim
        self.net = nn.Sequential(
            nn.Linear(randomDim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MNIST_Discriminator(nn.Module):
    """Discriminator for MNIST (binary classification with sigmoid)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class MNIST_DiscriminatorPrivate(nn.Module):
    """Classifier to guess which generator produced the sample."""

    def __init__(self, OutSize: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, OutSize),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class CIFAR_Generator(nn.Module):
    """Generator for CIFAR-10 (DCGAN-style)."""

    def __init__(self, randomDim: int = 100):
        super().__init__()
        self.randomDim = randomDim
        self.fc = nn.Sequential(
            nn.Linear(randomDim, 2 * 2 * 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.size(0), 512, 2, 2)
        return self.net(x)


class CIFAR_Discriminator(nn.Module):
    """Discriminator for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1),
            nn.Sigmoid(),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CIFAR_DiscriminatorPrivate(nn.Module):
    """Classifier that predicts which generator produced the sample."""

    def __init__(self, OutSize: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, OutSize),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LFW_Generator(nn.Module):
    """Generator for LFW (fully-connected)."""

    def __init__(self, randomDim: int = 100):
        super().__init__()
        self.randomDim = randomDim
        self.net = nn.Sequential(
            nn.Linear(randomDim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2914),
            nn.Tanh(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LFW_Discriminator(nn.Module):
    """Discriminator for LFW (binary classification)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2914, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class LFW_DiscriminatorPrivate(nn.Module):
    """Classifier that predicts generator index."""

    def __init__(self, OutSize: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2914, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, OutSize),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


@dataclass
class _ModelSpec:
    generator_factory: Callable[[int], nn.Module]
    discriminator_factory: Callable[[], nn.Module]
    privacy_disc_factory: Callable[[int], nn.Module]


_MODEL_REGISTRY = {
    "mnist": _ModelSpec(
        generator_factory=lambda rd: MNIST_Generator(randomDim=rd),
        discriminator_factory=MNIST_Discriminator,
        privacy_disc_factory=lambda out: MNIST_DiscriminatorPrivate(OutSize=out),
    ),
    "cifar": _ModelSpec(
        generator_factory=lambda rd: CIFAR_Generator(randomDim=rd),
        discriminator_factory=CIFAR_Discriminator,
        privacy_disc_factory=lambda out: CIFAR_DiscriminatorPrivate(OutSize=out),
    ),
    "lfw": _ModelSpec(
        generator_factory=lambda rd: LFW_Generator(randomDim=rd),
        discriminator_factory=LFW_Discriminator,
        privacy_disc_factory=lambda out: LFW_DiscriminatorPrivate(OutSize=out),
    ),
}


def _build_default_models(dataset: str, randomDim: int, n_pairs: int) -> Tuple[List[nn.Module], List[nn.Module], nn.Module]:
    if dataset not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from: {list(_MODEL_REGISTRY)}")

    spec = _MODEL_REGISTRY[dataset]
    generators = [spec.generator_factory(randomDim) for _ in range(n_pairs)]
    discriminators = [spec.discriminator_factory() for _ in range(n_pairs)]
    pdisc = spec.privacy_disc_factory(n_pairs)
    return generators, discriminators, pdisc


########################################
# Core GAN training loops
########################################


def SimpGAN(
    X_train: np.ndarray,
    generator: Optional[nn.Module] = None,
    discriminator: Optional[nn.Module] = None,
    randomDim: int = 100,
    epochs: int = 200,
    batchSize: int = 128,
    lr: float = 0.0002,
    beta1: float = 0.5,
    verbose: int = 1,
    lSmooth: float = 0.9,
    SplitTF: bool = False,
    device: Optional[torch.device] = None,
):
    """Single GAN training loop implemented in PyTorch."""

    device = _get_device(device)
    generator = generator or MNIST_Generator(randomDim=randomDim)
    discriminator = discriminator or MNIST_Discriminator()
    generator.to(device)
    discriminator.to(device)

    g_opt = make_optimizer(generator, lr=lr, beta1=beta1)
    d_opt = make_optimizer(discriminator, lr=lr, beta1=beta1)
    bce = nn.BCELoss()

    dataset = TensorDataset(_to_tensor(X_train, device))
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, drop_last=True)

    dLosses: List[float] = []
    gLosses: List[float] = []

    print("Epochs:", epochs)
    print("Batch size:", batchSize)
    print("Batches per epoch:", len(loader))

    for e in range(1, epochs + 1):
        g_t: List[float] = []
        d_t: List[float] = []

        for i, (real_batch,) in enumerate(loader):
            noise = torch.randn(batchSize, randomDim, device=device)
            fake_batch = generator(noise)

            # discriminator step
            _set_requires_grad(discriminator, True)
            d_opt.zero_grad()
            real_labels = torch.full((batchSize, 1), lSmooth, device=device)
            fake_labels = torch.zeros(batchSize, 1, device=device)

            d_loss_real = bce(discriminator(real_batch), real_labels)
            d_loss_fake = bce(discriminator(fake_batch.detach()), fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_opt.step()
            _set_requires_grad(discriminator, False)

            # generator step
            g_opt.zero_grad()
            gen_labels = torch.ones(batchSize, 1, device=device)
            g_loss = bce(discriminator(fake_batch), gen_labels)
            g_loss.backward()
            g_opt.step()

            if verbose == 1:
                print(
                    f"epoch = {e}/{epochs}, batch = {i+1}/{len(loader)}, d_loss={d_loss.item():.3f}, g_loss={g_loss.item():.3f}",
                    end="\r",
                )

            d_t.append(d_loss.item())
            g_t.append(g_loss.item())

        dLosses.append(float(np.mean(d_t)))
        gLosses.append(float(np.mean(g_t)))

        if verbose == 1:
            print(f"epoch = {e}/{epochs}, d_loss={dLosses[-1]:.3f}, g_loss={gLosses[-1]:.3f}")

    return generator, discriminator, dLosses, gLosses


def TrainDiscriminator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    discriminator: Optional[nn.Module] = None,
    epochs: int = 200,
    batchSize: int = 128,
    lr: float = 0.0002,
    beta1: float = 0.5,
    device: Optional[torch.device] = None,
):
    """Train the privacy discriminator (multi-class classifier)."""

    device = _get_device(device)
    discriminator = discriminator or MNIST_DiscriminatorPrivate(OutSize=len(np.unique(y_train)))
    discriminator.to(device)

    opt = make_optimizer(discriminator, lr=lr, beta1=beta1)
    ce = nn.CrossEntropyLoss()

    dataset = TensorDataset(_to_tensor(X_train, device), torch.tensor(y_train, dtype=torch.long, device=device))
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, drop_last=True)

    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = discriminator(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

    return discriminator


def privGAN(
    X_train: np.ndarray,
    generators: Optional[Sequence[nn.Module]] = None,
    discriminators: Optional[Sequence[nn.Module]] = None,
    pDisc: Optional[nn.Module] = None,
    randomDim: int = 100,
    disc_epochs: int = 50,
    epochs: int = 200,
    dp_delay: int = 100,
    batchSize: int = 128,
    lr: float = 0.0002,
    beta1: float = 0.5,
    verbose: int = 1,
    lSmooth: float = 0.95,
    privacy_ratio: float = 1.0,
    SplitTF: bool = False,
    device: Optional[torch.device] = None,
):
    """PrivGAN training loop in PyTorch."""

    device = _get_device(device)

    if generators is None:
        generators = [MNIST_Generator(randomDim=randomDim), MNIST_Generator(randomDim=randomDim)]
    if discriminators is None:
        discriminators = [MNIST_Discriminator(), MNIST_Discriminator()]
    if len(generators) != len(discriminators):
        raise ValueError("Different number of generators and discriminators")

    n_reps = len(generators)
    if n_reps == 1:
        raise ValueError("You cannot have only one generator-discriminator pair")

    pDisc = pDisc or MNIST_DiscriminatorPrivate(OutSize=n_reps)

    for g, d in zip(generators, discriminators):
        g.to(device)
        d.to(device)
    pDisc.to(device)

    g_opts = [make_optimizer(g, lr=lr, beta1=beta1) for g in generators]
    d_opts = [make_optimizer(d, lr=lr, beta1=beta1) for d in discriminators]
    p_opt = make_optimizer(pDisc, lr=lr, beta1=beta1)

    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()

    # split dataset across generators
    X_splits: List[torch.Tensor] = []
    y_train = []
    t = len(X_train) // n_reps
    for i in range(n_reps):
        if i < n_reps - 1:
            X_splits.append(_to_tensor(X_train[i * t : (i + 1) * t], device))
            y_train.extend([i] * t)
        else:
            X_splits.append(_to_tensor(X_train[i * t :], device))
            y_train.extend([i] * len(X_train[i * t :]))
    y_train = np.array(y_train)

    # pretrain privacy discriminator
    TrainDiscriminator(X_train, y_train, discriminator=pDisc, epochs=disc_epochs, batchSize=batchSize, lr=lr, beta1=beta1, device=device)
    with torch.no_grad():
        logits = pDisc(_to_tensor(X_train, device))
        yp = logits.argmax(dim=1).cpu().numpy()
        print("dp-Accuracy:", np.mean(y_train == yp))

    # Ensure at least one batch per epoch to avoid empty loops when t < batchSize
    batchCount = max(1, int(np.ceil(t / batchSize)))
    print("Epochs:", epochs)
    print("Batch size:", batchSize)
    print("Batches per epoch:", batchCount)

    dLosses = np.zeros((n_reps, epochs))
    dpLosses = np.zeros(epochs)
    gLosses = np.zeros(epochs)

    for e in range(epochs):
        d_t = np.zeros((n_reps, batchCount))
        dp_t = np.zeros(batchCount)
        g_t = np.zeros(batchCount)

        for i in range(batchCount):
            noise = torch.randn(batchSize, randomDim, device=device)
            generatedImages: List[torch.Tensor] = []
            yDis2: List[int] = []
            yDis2f: List[np.ndarray] = []

            # discriminator updates
            for j in range(n_reps):
                real_batch = X_splits[j][torch.randint(0, len(X_splits[j]), (batchSize,), device=device)]
                fake_batch = generators[j](noise)
                generatedImages.append(fake_batch)

                real_labels = torch.full((batchSize, 1), lSmooth, device=device)
                fake_labels = torch.zeros(batchSize, 1, device=device)

                _set_requires_grad(discriminators[j], True)
                d_opts[j].zero_grad()
                d_loss_real = bce(discriminators[j](real_batch), real_labels)
                d_loss_fake = bce(discriminators[j](fake_batch.detach()), fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_opts[j].step()
                _set_requires_grad(discriminators[j], False)

                d_t[j, i] = d_loss.item()
                labels_other = list(range(n_reps))
                labels_other.remove(j)
                yDis2.extend([j] * batchSize)
                yDis2f.append(np.random.choice(labels_other, size=batchSize))

            yDis2_arr = np.array(yDis2)
            all_generated = torch.cat(generatedImages, dim=0)

            # privacy discriminator update
            if e >= dp_delay:
                _set_requires_grad(pDisc, True)
                p_opt.zero_grad()
                logits = pDisc(all_generated.detach())
                dp_loss = ce(logits, torch.tensor(yDis2_arr, device=device))
                dp_loss.backward()
                p_opt.step()
                _set_requires_grad(pDisc, False)
                dp_t[i] = dp_loss.item()

            # generator updates
            _set_requires_grad(pDisc, False)
            for j in range(n_reps):
                g_opts[j].zero_grad()
                adv_labels = torch.ones(batchSize, 1, device=device)
                adv_loss = bce(discriminators[j](generatedImages[j]), adv_labels)
                priv_targets = torch.tensor(yDis2f[j], device=device)
                priv_loss = ce(pDisc(generatedImages[j]), priv_targets)
                g_loss = adv_loss + privacy_ratio * priv_loss
                g_loss.backward()
                g_opts[j].step()
                g_t[i] += g_loss.item()

            if verbose == 1:
                print(f"epoch = {e}/{epochs}, batch = {i+1}/{batchCount}", end="\r")

        dLosses[:, e] = np.mean(d_t, axis=1)
        dpLosses[e] = np.mean(dp_t) if np.any(dp_t) else 0.0
        gLosses[e] = np.mean(g_t)

        if verbose == 1:
            print("epoch =", e)
            print("dLosses =", np.mean(d_t, axis=1))
            print("dpLosses =", dpLosses[e])
            print("gLosses =", gLosses[e])
            with torch.no_grad():
                yp = pDisc(all_generated).argmax(dim=1).cpu().numpy()
                print("dp-Accuracy:", np.mean(yDis2_arr == yp))

    return generators, discriminators, pDisc, dLosses, dpLosses, gLosses


########################################
# Ancillary functions (visualization and attacks)
########################################


def DisplayImages(
    generator: nn.Module,
    randomDim: int = 100,
    NoImages: int = 100,
    figSize: Tuple[int, int] = (10, 10),
    TargetShape: Tuple[int, ...] = (28, 28),
    device: Optional[torch.device] = None,
):
    if (len(figSize) != 2) or (figSize[0] * figSize[1] < NoImages):
        print("Invalid Figure Size")
        return

    device = _get_device(device)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(NoImages, randomDim, device=device)
        generatedImages = generator(noise).cpu().numpy()

    TargetShape = tuple([NoImages] + list(TargetShape))
    generatedImages = generatedImages.reshape(TargetShape)

    for i in range(generatedImages.shape[0]):
        plt.subplot(figSize[0], figSize[1], i + 1)
        plt.imshow(generatedImages[i], interpolation="nearest", cmap="gray_r")
        plt.axis("off")
    plt.tight_layout()


def _predict_disc(discriminator: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    discriminator.eval()
    with torch.no_grad():
        preds = discriminator(_to_tensor(X, device)).cpu().numpy().squeeze()
    return preds


def WBattack(X: np.ndarray, X_comp: np.ndarray, discriminator: nn.Module, device: Optional[torch.device] = None):
    device = _get_device(device)
    Dat = np.concatenate([X, X_comp])
    p = _predict_disc(discriminator, Dat, device)
    In = np.argsort(-p)[: len(X)]
    Accuracy = np.sum(1.0 * (In < len(X))) / len(X)
    print("White-box attack accuracy:", Accuracy)
    return Accuracy


def WBattack_priv(
    X: np.ndarray,
    X_comp: np.ndarray,
    discriminators: Sequence[nn.Module],
    device: Optional[torch.device] = None,
):
    device = _get_device(device)
    Dat = np.concatenate([X, X_comp])
    Pred = [
        _predict_disc(discriminators[i], Dat, device)
        for i in range(len(discriminators))
    ]
    p_mean = np.mean(Pred, axis=0)
    p_max = np.max(Pred, axis=0)

    In_mean = np.argsort(-p_mean)[: len(X)]
    In_max = np.argsort(-p_max)[: len(X)]

    Acc_max = np.sum(1.0 * (In_max < len(X))) / len(X)
    Acc_mean = np.sum(1.0 * (In_mean < len(X))) / len(X)

    print("White-box attack accuracy (max):", Acc_max)
    print("White-box attack accuracy (mean):", Acc_mean)
    return Acc_max, Acc_mean


def WBattack_TVD(X: np.ndarray, X_comp: np.ndarray, discriminator: nn.Module, device: Optional[torch.device] = None):
    device = _get_device(device)
    n1, _ = np.histogram(_predict_disc(discriminator, X, device), bins=50, density=True, range=[0, 1])
    n2, _ = np.histogram(_predict_disc(discriminator, X_comp, device), bins=50, density=True, range=[0, 1])
    tvd = 0.5 * np.linalg.norm(n1 - n2, 1) / 50.0
    print("Total Variational Distance:", tvd)
    return tvd


def WBattack_TVD_priv(
    X: np.ndarray,
    X_comp: np.ndarray,
    discriminators: Sequence[nn.Module],
    device: Optional[torch.device] = None,
):
    device = _get_device(device)
    tvd = []
    for disc in discriminators:
        n1, _ = np.histogram(_predict_disc(disc, X, device), bins=50, density=True, range=[0, 1])
        n2, _ = np.histogram(_predict_disc(disc, X_comp, device), bins=50, density=True, range=[0, 1])
        tvd.append(0.5 * np.linalg.norm(n1 - n2, 1) / 50.0)
    print("Total Variational Distance - max:", max(tvd))
    print("Total Variational Distance - mean:", np.mean(tvd))
    return float(np.max(tvd)), float(np.mean(tvd))


def _generate_fake(generator: nn.Module, N: int, randomDim: int, device: torch.device) -> np.ndarray:
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(N, randomDim, device=device)
        X_fake = generator(noise).cpu().numpy()
    return X_fake


def MC_eps_attack(
    X: np.ndarray,
    X_comp: np.ndarray,
    X_ho: np.ndarray,
    generator: nn.Module,
    N: int = 100000,
    M: int = 100,
    n_pc: int = 40,
    reps: int = 10,
    randomDim: int = 100,
    device: Optional[torch.device] = None,
):
    device = _get_device(device)
    sh = int(np.prod(X.shape[1:]))
    X = np.reshape(X, (len(X), sh))
    X_comp = np.reshape(X_comp, (len(X_comp), sh))
    X_ho = np.reshape(X_ho, (len(X_ho), sh))

    pca = PCA(n_components=n_pc)
    pca.fit(X_ho)

    res = []
    for _ in range(reps):
        X_fake = _generate_fake(generator, N, randomDim, device)
        X_fake = np.reshape(X_fake, (len(X_fake), sh))
        X_fake_dr = pca.transform(X_fake)

        idx1 = np.random.randint(len(X), size=M)
        M_x = pca.transform(np.reshape(X[idx1, :], (len(idx1), sh)))
        M_xc = pca.transform(np.reshape(X_comp[idx1, :], (len(idx1), sh)))

        min_x = []
        min_xc = []
        for i in range(M):
            temp_x = np.tile(M_x[i, :], (len(X_fake_dr), 1))
            temp_xc = np.tile(M_xc[i, :], (len(X_fake_dr), 1))
            D_x = np.sqrt(np.sum((temp_x - X_fake_dr) ** 2, axis=1))
            D_xc = np.sqrt(np.sum((temp_xc - X_fake_dr) ** 2, axis=1))
            min_x.append(np.min(D_x))
            min_xc.append(np.min(D_xc))

        eps = np.median(min_x + min_xc)
        s_x = []
        s_xc = []
        for i in range(M):
            temp_x = np.tile(M_x[i, :], (len(X_fake_dr), 1))
            temp_xc = np.tile(M_xc[i, :], (len(X_fake_dr), 1))
            D_x = np.sqrt(np.sum((temp_x - X_fake_dr) ** 2, axis=1))
            D_xc = np.sqrt(np.sum((temp_xc - X_fake_dr) ** 2, axis=1))
            s_x.append(np.sum(D_x <= eps) / len(X_fake_dr))
            s_xc.append(np.sum(D_xc <= eps) / len(X_fake_dr))

        s_x_xc = np.array(s_x + s_xc)
        In = np.argsort(-s_x_xc)[:M]
        res.append(1 if np.sum(In < M) >= 0.5 * M else 0)

    return float(np.mean(res))


def MC_eps_attack_priv(
    X: np.ndarray,
    X_comp: np.ndarray,
    X_ho: np.ndarray,
    generators: Sequence[nn.Module],
    N: int = 100000,
    M: int = 100,
    n_pc: int = 40,
    reps: int = 10,
    randomDim: int = 100,
    device: Optional[torch.device] = None,
):
    device = _get_device(device)
    sh = int(np.prod(X.shape[1:]))
    X = np.reshape(X, (len(X), sh))
    X_comp = np.reshape(X_comp, (len(X_comp), sh))
    X_ho = np.reshape(X_ho, (len(X_ho), sh))

    pca = PCA(n_components=n_pc)
    pca.fit(X_ho)

    res = []
    for _ in range(reps):
        n_g = len(generators)
        X_fake_dr = []
        for j in range(n_g):
            X_fake = _generate_fake(generators[j], int(N / n_g), randomDim, device)
            X_fake = np.reshape(X_fake, (len(X_fake), sh))
            X_fake_dr.append(pca.transform(X_fake))
        X_fake_dr = np.vstack(X_fake_dr)

        idx1 = np.random.randint(len(X), size=M)
        M_x = pca.transform(np.reshape(X[idx1, :], (len(idx1), sh)))
        M_xc = pca.transform(np.reshape(X_comp[idx1, :], (len(idx1), sh)))

        min_x = []
        min_xc = []
        for i in range(M):
            temp_x = np.tile(M_x[i, :], (len(X_fake_dr), 1))
            temp_xc = np.tile(M_xc[i, :], (len(X_fake_dr), 1))
            D_x = np.sqrt(np.sum((temp_x - X_fake_dr) ** 2, axis=1))
            D_xc = np.sqrt(np.sum((temp_xc - X_fake_dr) ** 2, axis=1))
            min_x.append(np.min(D_x))
            min_xc.append(np.min(D_xc))

        eps = np.median(min_x + min_xc)
        s_x = []
        s_xc = []
        for i in range(M):
            temp_x = np.tile(M_x[i, :], (len(X_fake_dr), 1))
            temp_xc = np.tile(M_xc[i, :], (len(X_fake_dr), 1))
            D_x = np.sqrt(np.sum((temp_x - X_fake_dr) ** 2, axis=1))
            D_xc = np.sqrt(np.sum((temp_xc - X_fake_dr) ** 2, axis=1))
            s_x.append(np.sum(D_x <= eps) / len(X_fake_dr))
            s_xc.append(np.sum(D_xc <= eps) / len(X_fake_dr))

        s_x_xc = np.array(s_x + s_xc)
        In = np.argsort(-s_x_xc)[:M]
        res.append(1 if np.sum(In < M) >= 0.5 * M else 0)

    return float(np.mean(res))


########################################
# Convenience wrapper class
########################################


class PrivGANDefense:
    """Minimal wrapper for training and sampling PrivGAN models.

    Example
    -------
    >>> defense = PrivGANDefense(dataset="mnist", n_pairs=2)
    >>> defense.train(x_train)
    >>> samples = defense.sample(16, generator_idx=0)
    """

    def __init__(
        self,
        dataset: str = "mnist",
        n_pairs: int = 2,
        randomDim: int = 100,
        device: Optional[torch.device] = None,
    ) -> None:
        if n_pairs < 2:
            raise ValueError("PrivGAN requires at least two generator-discriminator pairs.")
        self.dataset = dataset.lower()
        self.randomDim = randomDim
        self.n_pairs = n_pairs
        self.device = _get_device(device)

        self.generators, self.discriminators, self.pdisc = _build_default_models(
            self.dataset, randomDim=randomDim, n_pairs=n_pairs
        )
        for g, d in zip(self.generators, self.discriminators):
            g.to(self.device)
            d.to(self.device)
        self.pdisc.to(self.device)

        self.dLosses: Optional[np.ndarray] = None
        self.dpLosses: Optional[np.ndarray] = None
        self.gLosses: Optional[np.ndarray] = None

    def train(
        self,
        X_train: np.ndarray,
        disc_epochs: int = 50,
        epochs: int = 200,
        dp_delay: int = 100,
        batchSize: int = 128,
        lr: float = 0.0002,
        beta1: float = 0.5,
        verbose: int = 1,
        lSmooth: float = 0.95,
        privacy_ratio: float = 1.0,
    ) -> None:
        (
            self.generators,
            self.discriminators,
            self.pdisc,
            self.dLosses,
            self.dpLosses,
            self.gLosses,
        ) = privGAN(
            X_train=X_train,
            generators=self.generators,
            discriminators=self.discriminators,
            pDisc=self.pdisc,
            randomDim=self.randomDim,
            disc_epochs=disc_epochs,
            epochs=epochs,
            dp_delay=dp_delay,
            batchSize=batchSize,
            lr=lr,
            beta1=beta1,
            verbose=verbose,
            lSmooth=lSmooth,
            privacy_ratio=privacy_ratio,
            device=self.device,
        )

    def sample(self, n: int, generator_idx: int = 0) -> np.ndarray:
        if generator_idx >= len(self.generators):
            raise IndexError("generator_idx is out of range")
        gen = self.generators[generator_idx]
        gen.eval()
        with torch.no_grad():
            noise = torch.randn(n, self.randomDim, device=self.device)
            return gen(noise).cpu().numpy()

    def white_box_attack(self, X: np.ndarray, X_comp: np.ndarray) -> Tuple[float, float]:
        acc_max, acc_mean = WBattack_priv(X, X_comp, self.discriminators, device=self.device)
        return acc_max, acc_mean

    def white_box_tvd(self, X: np.ndarray, X_comp: np.ndarray) -> Tuple[float, float]:
        return WBattack_TVD_priv(X, X_comp, self.discriminators, device=self.device)

    def monte_carlo_eps(self, X: np.ndarray, X_comp: np.ndarray, X_ho: np.ndarray, **kwargs) -> float:
        return MC_eps_attack_priv(
            X=X,
            X_comp=X_comp,
            X_ho=X_ho,
            generators=self.generators,
            randomDim=self.randomDim,
            device=self.device,
            **kwargs,
        )


__all__ = [
    "SimpGAN",
    "TrainDiscriminator",
    "privGAN",
    "DisplayImages",
    "WBattack",
    "WBattack_priv",
    "WBattack_TVD",
    "WBattack_TVD_priv",
    "MC_eps_attack",
    "MC_eps_attack_priv",
    "MNIST_Generator",
    "MNIST_Discriminator",
    "MNIST_DiscriminatorPrivate",
    "CIFAR_Generator",
    "CIFAR_Discriminator",
    "CIFAR_DiscriminatorPrivate",
    "LFW_Generator",
    "LFW_Discriminator",
    "LFW_DiscriminatorPrivate",
    "PrivGANDefense",
]
