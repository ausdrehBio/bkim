from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim
import torch
from torchmetrics import Accuracy, AUROC
from tqdm import tqdm

from .dataset import get_dataloaders
from .util import get_device


class FederatedCNN(nn.Module):
    """
    Federated CNN model.
    """

    def __init__(self, in_channels, num_classes,device=None):
        """
        Initialize the model.

        :param in_channels: number of input channels
        :param num_classes: number of output classes
        :param device: device to use for training. If None, the device is automatically detected.
        """
        super().__init__()

        if not device:
            self.device = get_device()
        else:
            self.device = device

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        #    nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: input tensor
        :return: output tensor
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_parameters(self):
        """
        Get the parameters of the model.

        :return: list of numpy arrays containing the parameters
        """
        with torch.no_grad():
            w = []
            for name, param in self.named_parameters():
                w.append(param.data.clone().detach().cpu().numpy())
        return w

    def set_parameters(self, w):
        """
        Set the parameters of the model to the given values.

        :param w: list of numpy arrays containing the parameters
        """
        with torch.no_grad():
            for i, (name, param) in enumerate(self.named_parameters()):
                p = w[i] if isinstance(w[i], np.ndarray) else np.array(w[i], dtype='float32')
                param.data = torch.from_numpy(p).to(device=torch.device)

    def train_model(self, epochs, optimizer, criterion, train_loader, test_loader):
        """
        Train the model for a given number of epochs.

        :param epochs: number of epochs to train for
        :param optimizer: optimizer to use for training
        :param criterion: loss function to use for training
        :param train_loader: data loader for the training data
        :param test_loader: data loader for the test data
        :return: tuple of dictionaries of metrics for training and testing
        """
        self.to(self.device)

        # Initialize metrics
        l_train_metrics, l_test_metrics = defaultdict(list), defaultdict(list)

        for _ in tqdm(range(epochs)):

            train_metrics = self._train_epoch(
                mode="train",
                data_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
            )
            test_metrics = self._train_epoch(
                mode="test",
                data_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
            )

            for k, v in train_metrics.items():
                l_train_metrics[k].append(v)
            for k, v in test_metrics.items():
                l_test_metrics[k].append(v)

        return l_train_metrics, l_test_metrics

    def _train_epoch(self, mode, data_loader, optimizer, criterion):
        """
        Train or test the model for one epoch.

        :param mode: "train" or "test"
        :param data_loader: data loader for the data to train/test on
        :param optimizer: optimizer to use for training
        :param criterion: loss function to use for training
        :return: dictionary of metrics
        """
        accuracy = Accuracy(task="binary").to(self.device)
        metrics = defaultdict(list)

        if mode == "train":
            self.train()
        elif mode == "test":
            self.eval()
        else:
            raise ValueError("mode must be either 'train' or 'test'")

        for data, target in data_loader:

            # predict
            data = data.to(self.device)
            target = target.to(self.device)
            output = self(data).squeeze(1)
            loss = criterion(output, target)

            # update metrics
            metrics["loss"].append(loss.item())
            metrics["acc"].append(accuracy(output, target).item())

            # update weights if training
            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # return mean of batch metrics
        return {k: np.mean(v) for k, v in metrics.items()}


if __name__ == '__main__':
    SRC_PATH = "./resources/pneumoniamnist.npz"

    model = FederatedCNN(in_channels=1, num_classes=1)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)
    ce_loss = nn.BCEWithLogitsLoss()

    train_dl, test_dl = get_dataloaders(SRC_PATH)

    train_metrics, test_metrics = model.train_model(
        epochs=20,
        train_loader=train_dl,
        test_loader=test_dl,
        optimizer=optimizer,
        criterion=ce_loss,
    )

    print(train_metrics)
