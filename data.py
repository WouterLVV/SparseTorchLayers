from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import os
import pandas as pd


# Input: flattened 768 float array with values between -1 and 1
# Output: one-hot encoded class vector
def load_mnist(dataloader_kwargs, dataset_dir):
    norm1 = norm2 = (0.5,)  # Must be an iterable of the number of channels (for MNIST only 1)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm1, norm2),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    output_transform = transforms.Compose([
        transforms.Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), src=torch.tensor(1.)))
    ])

    traindata = datasets.MNIST(root=dataset_dir, train=True, download=True, transform=input_transform,
                               target_transform=output_transform)
    testdata = datasets.MNIST(root=dataset_dir, train=False, download=True, transform=input_transform,
                              target_transform=output_transform)

    train_loader = DataLoader(traindata, **dataloader_kwargs)
    test_loader = DataLoader(testdata, **dataloader_kwargs)

    return traindata, testdata, train_loader, test_loader


# Input: flattened 768 float array with values between -1 and 1
# Output: one-hot encoded class vector
def load_fashion_mnist(dataloader_kwargs, dataset_dir):
    norm1 = norm2 = (0.5,)  # Must be an iterable of the number of channels (for MNIST only 1)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm1, norm2),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    output_transform = transforms.Compose([
        transforms.Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), src=torch.tensor(1.)))
    ])

    traindata = datasets.FashionMNIST(root=dataset_dir, train=True, download=True, transform=input_transform,
                                      target_transform=output_transform)
    testdata = datasets.FashionMNIST(root=dataset_dir, train=False, download=True, transform=input_transform,
                                     target_transform=output_transform)

    train_loader = DataLoader(traindata, **dataloader_kwargs)
    test_loader = DataLoader(testdata, **dataloader_kwargs)

    return traindata, testdata, train_loader, test_loader


# Input: flattened 3092 float array with values between -1 and 1
# Output: one-hot encoded class vector
def load_cifar10(dataloader_kwargs, dataset_dir):
    norm1 = norm2 = (.5, .5, .5)

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm1, norm2),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    output_transform = transforms.Compose([
        transforms.Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), src=torch.tensor(1.)))
    ])
    traindata = datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=input_transform,
                                 target_transform=output_transform)
    testdata = datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=input_transform,
                                target_transform=output_transform)

    train_loader = DataLoader(traindata, **dataloader_kwargs)
    test_loader = DataLoader(testdata, **dataloader_kwargs)

    return traindata, testdata, train_loader, test_loader


# Input:
def load_ucihar(dataloader_kwargs, dataset_dir):
    traindata = UCI_HAR(root=dataset_dir, train=True, download=True)
    testdata = UCI_HAR(root=dataset_dir, train=False, download=True)

    train_loader = DataLoader(traindata, **dataloader_kwargs)
    test_loader = DataLoader(testdata, **dataloader_kwargs)

    return traindata, testdata, train_loader, test_loader


# Input: 500 integer array, with values around 500 (only 20 features are informative)
# Output: binary classification of either -1 or 1
def load_madelon(dataloader_kwargs, dataset_dir):
    traindata = MADELON(root=dataset_dir, train=True, download=True)
    testdata = MADELON(root=dataset_dir, train=False, download=True)

    train_loader = DataLoader(traindata, **dataloader_kwargs)
    test_loader = DataLoader(testdata, **dataloader_kwargs)

    return traindata, testdata, train_loader, test_loader


def load_higgs(dataloader_kwargs, dataset_dir):
    traindata = HIGGS(root=dataset_dir, train=True, download=True)
    testdata = HIGGS(root=dataset_dir, train=False, download=True)

    train_loader = DataLoader(traindata, **dataloader_kwargs)
    test_loader = DataLoader(testdata, **dataloader_kwargs)

    return traindata, testdata, train_loader, test_loader


class UCI_HAR(Dataset):
    name = "UCI_HAR"
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

    train_files = (os.path.join("train", "X_train.txt"), os.path.join("train", "y_train.txt"))
    test_files = (os.path.join("test", "X_test.txt"), os.path.join("test", "y_test.txt"))

    main_files = {os.path.join("train", "X_train.txt"), os.path.join("train", "y_train.txt"),
                  os.path.join("test", "X_test.txt"), os.path.join("test", "y_test.txt")}

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        self.root = os.path.join(root, "")
        if not os.path.exists(root) or not os.path.isdir(root):
            raise FileNotFoundError(f"Root is not a valid folder")
        if download:
            self._download(self.root)

        self.folder = os.path.join(self.root, self.name)

        self.transform = transform
        self.target_transform = target_transform

        self.xdata, self.ydata = self._load_data(train=train)
        self.num_items = self.xdata.shape[0]

    def _download(self, root):
        p = os.path.join(root, self.name)
        if not os.path.exists(p) or not os.path.isdir(p):
            os.makedirs(p)
        else:
            if all([os.path.exists(os.path.join(p, f)) for f in self.main_files]):
                return

        import urllib.request
        import zipfile
        zf = os.path.join(p, "tmp.zip")
        print(f"Downloading zip from {self.link}")
        urllib.request.urlretrieve(self.link, zf)
        print("Extracting files...")
        with zipfile.ZipFile(zf, 'r') as f:
            f.extractall(p)
        for fname in os.listdir(os.path.join(p, "UCI HAR Dataset")):
            os.rename(os.path.join(p, "UCI HAR Dataset", fname), os.path.join(p, fname))
        import shutil
        shutil.rmtree(os.path.join(p, "UCI HAR Dataset"))
        shutil.rmtree(os.path.join(p, "__MACOSX"))
        os.remove(os.path.join(p, "tmp.zip"))

    def _load_data(self, train):
        if train:
            fx, fy = self.train_files
        else:
            fx, fy = self.test_files
        fx = os.path.join(self.folder, fx)
        fy = os.path.join(self.folder, fy)
        xdata = torch.tensor(pd.read_csv(fx, delim_whitespace=True, header=None).to_numpy())
        ydata = torch.tensor(pd.read_csv(fy, delim_whitespace=True, header=None).to_numpy())
        if xdata.shape[0] != ydata.shape[0]:
            raise Exception(f"Files {fx} and {fy} do not have the same number of entries: {xdata.shape} and {ydata.shape}")
        return xdata, ydata

    def __len__(self):
        return self.num_items

    def __getitem__(self, item):
        x = self.xdata[item] if self.transform is None else self.transform(self.xdata[item])
        y = self.ydata[item] if self.target_transform is None else self.target_transform(self.ydata[item])
        return x, y


class MADELON(Dataset):
    name = "MADELON"
    links = [("https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels", "madelon_valid.labels"),
             ("https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data",  "madelon_train.data"),
             ("https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels", "madelon_train.labels"),
             ("https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data", "madelon_valid.data"),
             ("https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon.param", "madelon.param"),
             ("https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data", "madelon_test.data")]

    train_files = ("madelon_train.data", "madelon_train.labels")
    test_files = ("madelon_valid.data", "madelon_valid.labels")

    main_files = {"madelon_train.data", "madelon_train.labels", "madelon_valid.data", "madelon_valid.labels"}

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        self.root = os.path.join(root, "")
        if not os.path.exists(root) or not os.path.isdir(root):
            raise FileNotFoundError(f"Root is not a valid folder")
        if download:
            self._download(self.root)

        self.folder = os.path.join(self.root, self.name)

        self.transform = transform
        self.target_transform = target_transform

        self.xdata, self.ydata = self._load_data(train=train)
        self.num_items = self.xdata.shape[0]

    def _download(self, root):
        p = os.path.join(root, self.name)
        if not os.path.exists(p) or not os.path.isdir(p):
            os.makedirs(p)
        else:
            if all([os.path.exists(os.path.join(p, f)) for f in self.main_files]):
                return

        import urllib.request
        for link in self.links:
            print(f"Downloading {link[1]} from {link[0]}")
            urllib.request.urlretrieve(link[0], os.path.join(p, link[1]))
        print("Files downloaded.")

    def _load_data(self, train):
        if train:
            fx, fy = self.train_files
        else:
            fx, fy = self.test_files
        fx = os.path.join(self.folder, fx)
        fy = os.path.join(self.folder, fy)
        xdata = torch.tensor(pd.read_csv(fx, delim_whitespace=True, header=None).to_numpy())
        ydata = torch.tensor(pd.read_csv(fy, delim_whitespace=True, header=None).to_numpy())
        if xdata.shape[0] != ydata.shape[0]:
            raise Exception(f"Files {fx} and {fy} do not have the same number of entries: {xdata.shape} and {ydata.shape}")
        return xdata, ydata

    def __len__(self):
        return self.num_items

    def __getitem__(self, item):
        x = self.xdata[item] if self.transform is None else self.transform(self.xdata[item])
        y = self.ydata[item] if self.target_transform is None else self.target_transform(self.ydata[item])
        return x, y


class HIGGS(Dataset):
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    name = "HIGGS"

    main_files = {"higgs.csv"}

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        self.root = os.path.join(root, "")
        if not os.path.exists(root) or not os.path.isdir(root):
            raise FileNotFoundError(f"Root is not a valid folder")
        if download:
            self._download(self.root)

        self.folder = os.path.join(self.root, self.name)

        self.transform = transform
        self.target_transform = target_transform

        self.xdata, self.ydata = self._load_data(train=train)
        self.num_items = self.xdata.shape[0]

    def _download(self, root):
        p = os.path.join(root, self.name)
        if not os.path.exists(p) or not os.path.isdir(p):
            os.makedirs(p)
        else:
            if all([os.path.exists(os.path.join(p, f)) for f in self.main_files]):
                return

        import urllib.request
        import gzip
        import shutil
        zf = os.path.join(p, "tmp.gz")
        print(f"Downloading zip from {self.link}")
        urllib.request.urlretrieve(self.link, zf)
        print("Extracting files...")
        with gzip.open(zf, 'rt') as f_in:
            with open(os.path.join(p, "higgs.csv"), 'x') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(os.path.join(p, "tmp.gz"))

    def _load_data(self, train):
        f = os.path.join(self.folder, "higgs.csv")

        data = pd.read_csv(f, delimiter=',', header=None)

        if train:
            data = data[:-500000]
        else:
            data = data[-500000:]

        data = data.to_numpy()
        xdata = torch.tensor(data[:, 1:])
        ydata = torch.tensor(data[:, 0])
        return xdata, ydata

    def __len__(self):
        return self.num_items

    def __getitem__(self, item):
        x = self.xdata[item] if self.transform is None else self.transform(self.xdata[item])
        y = self.ydata[item] if self.target_transform is None else self.target_transform(self.ydata[item])
        return x, y


if __name__ == "__main__":
    x = load_higgs({}, "../data")
    print(x)
    print("aaaaa")
