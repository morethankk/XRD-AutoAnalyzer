"""PyTorch implementation of the convolutional neural network used for
XRD/PDF phase identification.

This module mirrors the functionality of the original TensorFlow version
but relies exclusively on :mod:`torch`.  It exposes utilities for setting
up datasets, training, evaluation and saving models.  The network
architecture is kept identical to the previous implementation to preserve
accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class CustomDropout(nn.Module):
    """Dropout layer that is always active.

    In the original TensorFlow model a custom layer was used to apply
    dropout during both training and inference to enable Monte Carlo
    dropout sampling.  The same behaviour is reproduced here by always
    enabling dropout in the forward pass.
    """

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - short description inherited
        return nn.functional.dropout(x, p=self.rate, training=True)


class XRDModel(nn.Module):
    """Convolutional neural network for XRD/PDF analysis."""

    def __init__(
        self,
        n_phases: int,
        is_pdf: bool,
        n_dense: Sequence[int] = (3100, 1200),
        dropout_rate: float = 0.7,
    ) -> None:
        super().__init__()

        conv: List[nn.Module] = []
        if is_pdf:
            # Architecture optimised for PDF analysis
            conv.extend(
                [
                    nn.Conv1d(1, 64, 60, padding=30),
                    nn.ReLU(),
                    nn.MaxPool1d(3, stride=2, padding=1),
                    nn.MaxPool1d(3, stride=2, padding=1),
                    nn.MaxPool1d(2, stride=2, padding=1),
                    nn.MaxPool1d(1, stride=2, padding=0),
                    nn.MaxPool1d(1, stride=2, padding=0),
                    nn.MaxPool1d(1, stride=2, padding=0),
                ]
            )
        else:
            # Architecture optimised for XRD analysis
            conv.extend(
                [
                    nn.Conv1d(1, 64, 35, padding=17),
                    nn.ReLU(),
                    nn.MaxPool1d(3, stride=2, padding=1),
                    nn.Conv1d(64, 64, 30, padding=15),
                    nn.ReLU(),
                    nn.MaxPool1d(3, stride=2, padding=1),
                    nn.Conv1d(64, 64, 25, padding=12),
                    nn.ReLU(),
                    nn.MaxPool1d(2, stride=2, padding=1),
                    nn.Conv1d(64, 64, 20, padding=10),
                    nn.ReLU(),
                    nn.MaxPool1d(1, stride=2, padding=0),
                    nn.Conv1d(64, 64, 15, padding=7),
                    nn.ReLU(),
                    nn.MaxPool1d(1, stride=2, padding=0),
                    nn.Conv1d(64, 64, 10, padding=5),
                    nn.ReLU(),
                    nn.MaxPool1d(1, stride=2, padding=0),
                ]
            )

        self.features = nn.Sequential(*conv)

        # Determine the flatten dimension dynamically using a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 4501)
            flat_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            CustomDropout(dropout_rate),
            nn.Linear(flat_dim, n_dense[0]),
            nn.ReLU(),
            nn.BatchNorm1d(n_dense[0]),
            CustomDropout(dropout_rate),
            nn.Linear(n_dense[0], n_dense[1]),
            nn.ReLU(),
            nn.BatchNorm1d(n_dense[1]),
            CustomDropout(dropout_rate),
            nn.Linear(n_dense[1], n_phases),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - short description inherited
        x = self.features(x)
        return self.classifier(x)


@dataclass
class DataSetUp:
    """Utility class for organising XRD spectra for training."""

    xrd: np.ndarray
    testing_fraction: float = 0.0

    @property
    def num_phases(self) -> int:
        return len(self.xrd)

    @property
    def phase_indices(self) -> List[int]:
        return list(range(self.num_phases))

    @property
    def x(self) -> np.ndarray:
        intensities: List[np.ndarray] = []
        for augmented_spectra, _ in zip(self.xrd, self.phase_indices):
            for pattern in augmented_spectra:
                intensities.append(pattern)
        return np.array(intensities)

    @property
    def y(self) -> np.ndarray:
        one_hot_vectors: List[List[float]] = []
        for augmented_spectra, index in zip(self.xrd, self.phase_indices):
            for _ in augmented_spectra:
                vec = [0.0] * len(self.xrd)
                vec[index] = 1.0
                one_hot_vectors.append(vec)
        return np.array(one_hot_vectors)

    def split_training_testing(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        x = self.x
        y = self.y
        combined_xy = list(zip(x, y))
        np.random.shuffle(combined_xy)

        if self.testing_fraction == 0:
            train_x, train_y = zip(*combined_xy)
            return np.array(train_x), np.array(train_y), None, None

        total_samples = len(combined_xy)
        n_testing = int(self.testing_fraction * total_samples)

        train_xy = combined_xy[n_testing:]
        train_x, train_y = zip(*train_xy)

        test_xy = combined_xy[:n_testing]
        test_x, test_y = zip(*test_xy)

        return (
            np.array(train_x),
            np.array(train_y),
            np.array(test_x),
            np.array(test_y),
        )


def _prepare_dataloader(x: np.ndarray, y: np.ndarray) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 2, 1)
    y_indices = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_indices)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_phases: int,
    num_epochs: int,
    is_pdf: bool,
    n_dense: Sequence[int] = (3100, 1200),
    dropout_rate: float = 0.7,
) -> XRDModel:
    """Train the convolutional neural network."""

    model = XRDModel(n_phases, is_pdf, n_dense, dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    loader = _prepare_dataloader(x_train, y_train)
    model.train()
    for _ in range(num_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model


def test_model(model: XRDModel, test_x: np.ndarray, test_y: np.ndarray) -> None:
    """Evaluate the trained model on a test set and print accuracy."""

    loader = _prepare_dataloader(test_x, test_y)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            _, predicted = torch.max(out, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

    if total > 0:
        print(f"Test Accuracy: {100 * correct / total:.2f}%")


def main(
    xrd: np.ndarray,
    num_epochs: int,
    testing_fraction: float,
    is_pdf: bool,
    fmodel: str = "Model.pth",
) -> None:
    """Entry point used by example scripts to train a model."""

    obj = DataSetUp(xrd, testing_fraction)
    num_phases = obj.num_phases
    train_x, train_y, test_x, test_y = obj.split_training_testing()

    model = train_model(train_x, train_y, num_phases, num_epochs, is_pdf)

    torch.save(model.state_dict(), fmodel)

    if testing_fraction != 0 and test_x is not None and test_y is not None:
        test_model(model, test_x, test_y)

