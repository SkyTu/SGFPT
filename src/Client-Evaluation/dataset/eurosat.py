import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import EuroSAT as TorchvisionEuroSAT


def _split_indices(n_samples: int, train_ratio: float = 0.8, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=g).tolist()
    n_train = int(n_samples * train_ratio)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return train_idx, test_idx


def _build_subset(root: str, preprocess, split: str, seed: int, download: bool = True):
    # Some environments miss CA certificates which breaks torchvision's download
    # via urllib (SSL certificate verify failed). certifi provides a bundle.
    import os
    if download and os.environ.get("SSL_CERT_FILE") is None:
        try:
            import certifi  # type: ignore
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except Exception:
            pass
    full = TorchvisionEuroSAT(root=root, transform=preprocess, download=download)
    train_idx, test_idx = _split_indices(len(full), train_ratio=0.8, seed=seed)
    indices = train_idx if split == "train" else test_idx

    subset = Subset(full, indices)

    class _DictWrapper(Dataset):
        # Return dict batch entries expected by PromptCLIP.
        def __init__(self, ds):
            self.ds = ds
            self.classes = getattr(ds, "classes", None)
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]  # torchvision returns (image, target)
            return {"image": img, "label": label}

    wrapped = _DictWrapper(subset)
    wrapped.classes = full.classes
    return wrapped


def load_eurosat_train(batch_size=1, seed=42, preprocess=None, root="./data", **kwargs):
    train_data = _build_subset(
        root=root,
        preprocess=preprocess,
        split="train",
        seed=seed,
        download=bool(kwargs.get("download", True)),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_data, train_loader


def load_eurosat_test(batch_size=1, seed=42, preprocess=None, root="./data", shuffle=True, **kwargs):
    test_data = _build_subset(
        root=root,
        preprocess=preprocess,
        split="test",
        seed=seed,
        download=bool(kwargs.get("download", True)),
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return test_data, test_loader
