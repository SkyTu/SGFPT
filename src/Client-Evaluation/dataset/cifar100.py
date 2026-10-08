import torch
import torchvision
import os
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, DataLoader
import argparse
#---------------------------------------------- device:cpu dtype:float32-----------------------------------------------


class Cifar_FewshotDataset(Dataset):
    def __init__(self, args):
        self.shots = args["shots"]
        self.prepocess = args["preprocess"]
        self.all_train = CIFAR100(os.path.expanduser("../dataset"), transform=None,download=True,train=True)
        self.new_train_data = self.construct_few_shot_data()
        pass

    def __len__(self):
        return len(self.new_train_data)

    def __getitem__(self, idx):

        return {"image": self.new_train_data[idx][0], "label": self.new_train_data[idx][1]}

    def construct_few_shot_data(self):
        new_train_data = []
        train_shot_count={}
        all_indices = [_ for _ in range(len(self.all_train))]
        np.random.shuffle(all_indices)

        for index in all_indices:
            label = self.all_train[index][1]
            if label not in train_shot_count:
                train_shot_count[label]=0

            if train_shot_count[label]<self.shots:
                tmp = self.all_train[index]
                # convert img to compatible tensors
                tmp_data = [self.prepocess(tmp[0]),tmp[1]]
                new_train_data.append(tmp_data)
                train_shot_count[label] += 1

        return new_train_data

class Cifar_TrainDataset(Dataset):
    def __init__(self, args):
        self.prepocess = args["preprocess"]
        root = args.get("root", os.path.expanduser("../dataset"))
        self.all_train = CIFAR100(root, transform=self.prepocess, download=True, train=True)
        self.classes = self.all_train.classes

    def __len__(self):
        return len(self.all_train)

    def __getitem__(self, idx):
        return {"image": self.all_train[idx][0], "label": self.all_train[idx][1]}


def load_train_cifar100(batch_size=1, shots=None, preprocess=None, root=None, shuffle=True):
    """
    Load CIFAR-100 train split.

    - shots is None: use full train set (50,000 images).
    - shots is int: use few-shot subset with `shots` samples per class.
    """
    if shots is None:
        args = {"preprocess": preprocess, "root": root or os.path.expanduser("../dataset")}
        train_data = Cifar_TrainDataset(args)
    else:
        args = {"shots": shots, "preprocess": preprocess}
        train_data = Cifar_FewshotDataset(args)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return train_data, train_loader

class Cifar_TestDataset(Dataset):
    def __init__(self, args):
        self.prepocess = args["preprocess"]
        root = args.get("root", os.path.expanduser("../dataset"))
        self.all_test = CIFAR100(root, transform=self.prepocess, download=True, train=False)

    def __len__(self):
        return len(self.all_test)

    def __getitem__(self, idx):
        return {"image": self.all_test[idx][0], "label": self.all_test[idx][1]}


def load_test_cifar100(batch_size=1, preprocess=None, root=None, shuffle=True):
    args = {"preprocess": preprocess,
            "root": root or os.path.expanduser("../dataset")}
    test_data = Cifar_TestDataset(args)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return test_data, test_loader
