import torch
from torchvision.datasets import DTD as TorchvisionDTD
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class DTDDataset(Dataset):
    """DTD数据集包装器，直接使用torchvision"""
    def __init__(self, root, split='train', transform=None, download=True):
        self.dataset = TorchvisionDTD(root=root, split=split, download=download)
        self.transform = transform
        self.classes = self.dataset.classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label}


def load_dtd_train(batch_size=1, seed=42, preprocess=None, root="./data", **kwargs):
    """加载DTD训练集"""
    train_data = DTDDataset(root=root, split='train', transform=preprocess, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_data, train_loader


def load_dtd_test(batch_size=1, seed=42, preprocess=None, root="./data", shuffle=True, **kwargs):
    """加载DTD测试集"""
    test_data = DTDDataset(root=root, split='test', transform=preprocess, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return test_data, test_loader
