import torch
from torchvision.datasets import Flowers102 as TorchvisionFlowers102
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class Flowers102Dataset(Dataset):
    """Flowers-102数据集包装器，直接使用torchvision"""
    def __init__(self, root, split='train', transform=None, download=True):
        self.dataset = TorchvisionFlowers102(root=root, split=split, download=download)
        self.transform = transform
        # torchvision的 Flowers102 数据集不会直接提供“类别名字符串”。
        # 但 B2TPT 的 PromptCLIP 需要用这些 strings 生成文本 prompt，
        # 否则会退化成“语义无关的类别编码”，acc 接近随机。
        # 这里使用 Oxford-102 官方 102 类目名称（label id=1..102 的顺序）。
        self.classes = [
            "pink primrose",
            "hard-leaved pocket orchid",
            "canterbury bells",
            "sweet pea",
            "english marigold",
            "tiger lily",
            "moon orchid",
            "bird of paradise",
            "monkshood",
            "globe thistle",
            "snapdragon",
            "colt's foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe-flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby-lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "pink-yellow dahlia?",
            "cautleya spicata",
            "japanese anemone",
            "black-eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert-rose",
            "tree mallow",
            "magnolia",
            "cyclamen",
            "watercress",
            "canna lily",
            "hippeastrum",
            "bee balm",
            "ball moss",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily",
        ]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label}


def load_flowers102_train(batch_size=1, seed=42, preprocess=None, root="./data", **kwargs):
    """加载Flowers-102训练集"""
    # torchvision 的 Flowers102：train/val/test 都是不同 split，其中 train 和 val 大小相近。
    # 你希望用更大的监督训练集做优化：这里合并 train + val。
    download = bool(kwargs.get("download", True))
    train_split = Flowers102Dataset(root=root, split='train', transform=preprocess, download=download)
    val_split = Flowers102Dataset(root=root, split='val', transform=preprocess, download=download)
    train_data = ConcatDataset([train_split, val_split])
    # B2TPT 的 server 代码会读取 train_data.classes；ConcatDataset 没有该字段，
    # 所以这里把类名从 train_split 继承挂上去。
    setattr(train_data, "classes", getattr(train_split, "classes", None))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_data, train_loader


def load_flowers102_test(batch_size=1, seed=42, preprocess=None, root="./data", shuffle=True, **kwargs):
    """加载Flowers-102测试集"""
    test_data = Flowers102Dataset(root=root, split='test', transform=preprocess, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return test_data, test_loader
