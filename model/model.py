from torch import nn
import torch
import numpy as np

from .util import get_device
from tqdm import tqdm
from collections import defaultdict


class FederatedCNN(nn.Module):

    def __init__(self, in_channels, num_classes, device=None):
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
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_parameters(self):
        with torch.no_grad():
            w = []
            for name, param in self.named_parameters():
                w.append(param.data.clone().detach().cpu().numpy())
        return w

    def set_parameters(self):
        with torch.no_grad():
            for i, (name, param) in enumerate(self.named_parameters()):
                p = w[i] if isinstance(w[i], np.ndarray) else np.array(w[i], dtype='float32')
                param.data = torch.from_numpy(p).to(device=torch.device)

    def train_model(self, epochs, optimizer, criterion, train_loader, test_loader):
        self.to(self.device)
        train_loss, test_loss = [], []

        # auroc = AUROC(task="binary")
        for _ in tqdm(range(epochs)):

            train_metrics = defaultdict(list)
            self.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                data = data.unsqueeze(0).transpose(0, 1).to(self.device)
                target = target.float().squeeze(1).to(self.device)
                output = self(data).squeeze(1)
                loss = criterion(output, target)

                # _, predicted = torch.max(output, 1)
                train_metrics["train loss"].append(loss.item())
                # train_metrics["train acc"].append(metrics.accuracy_score(target, predicted))
                # train_metrics["train auc"].append(auroc(output, target))

                loss.backward()
                optimizer.step()

            test_metrics = defaultdict(list)
            self.eval()
            for data, target in test_loader:
                data = data.unsqueeze(0).transpose(0, 1).to(self.device)
                target = target.float().squeeze(1).to(self.device)
                output = self(data).squeeze(1)
                loss = criterion(output, target)

                # _, predicted = torch.max(output, 1)
                test_metrics["test loss"].append(loss.item())
                # test_metrics["test acc"].append(metrics.accuracy_score(target, predicted))
                # test_metrics["test auc"].append(auroc(output, target))

            train_loss.append(np.mean(train_metrics["train loss"]))
            test_loss.append(np.mean(test_metrics["test loss"]))

        return train_loss, test_loss
