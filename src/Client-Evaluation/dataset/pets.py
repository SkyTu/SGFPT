import torch
from torchvision.datasets import OxfordIIITPet as TorchvisionPets
from torch.utils.data import Dataset, DataLoader


class PetsDataset(Dataset):
    """Oxford-IIIT Pets数据集包装器，直接使用torchvision"""
    def __init__(self, root, split='trainval', transform=None, download=True):
        # torchvision的Pets使用trainval和test
        self.dataset = TorchvisionPets(root=root, split=split, download=download)
        self.transform = transform
        self.classes = self.dataset.classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label}


def load_pets_train(batch_size=1, seed=42, preprocess=None, root="./data", **kwargs):
    """加载Pets训练集"""
    train_data = PetsDataset(root=root, split='trainval', transform=preprocess, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_data, train_loader


def load_pets_test(batch_size=1, seed=42, preprocess=None, root="./data", shuffle=True, **kwargs):
    """加载Pets测试集"""
    test_data = PetsDataset(root=root, split='test', transform=preprocess, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return test_data, test_loader
