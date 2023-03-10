import pathlib
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


def read_medmnist(path):
    """
    Read MedMNIST datasets in numpy zip file format .npz
    containing images and their labels.

    :param path: OS path to file
    :param concat_split: MedMNIST contains predefined test-,
        train- and validation-split. If True, concatenate them.
    :return: Numpy array
    """
    fp = pathlib.Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"'{path}' not found.")
    if not fp.suffix != "npz":
        raise ValueError(f"File needs to be of type *.npz. Given: {fp.suffix}")
    dataset = np.load(path, allow_pickle=True)
    if "train_images" in dataset.keys():
        images = np.concatenate([
            dataset["train_images"],
            dataset["test_images"],
            dataset["val_images"],
        ])
        labels = np.concatenate([
            dataset["train_labels"],
            dataset["test_labels"],
            dataset["val_labels"],
        ])
        return images, labels
    return dataset["images"], dataset["labels"]


def get_dataloaders(path, train_val_test_split=(0.75, 0.12, 0.13), batch_size=32, seed=42):
    """
    Return train- and test dataloaders.

    :param path: Path to data.
    :param train_val_test_split: Train-test ratio
    :param batch_size: Batch size
    :param seed: Random seed
    :return: train_dataloader, test_dataloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
    dataset = ImageDataset(path, transform=transform)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        train_val_test_split,
        generator=torch.Generator().manual_seed(seed)
    )
    print(len(train_set), len(val_set), len(test_set))
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


class ImageDataset(Dataset):
    """
    DataLoader for MedMNIST datasets.
    """

    def __init__(self, path, transform=None, target_transform=None):
        data, labels = read_medmnist(path)
        self.data = torch.from_numpy(data).float().unsqueeze(0).transpose(0, 1)
        self.labels = torch.from_numpy(labels).float().squeeze(1)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        feature = self.data[idx]
        return feature, label
