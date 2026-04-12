import torch
from torch.utils.data import Dataset


class GpuDataset(Dataset):
    def __init__(self, dataset: Dataset, device: torch.device):
        all_imgs = []
        all_lbls = []

        for i in range(len(dataset)):
            img, lbl = dataset[i]
            all_imgs.append(img.unsqueeze(0))
            all_lbls.append(lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl))

        self.images = torch.cat(all_imgs, dim=0).to(device)
        self.labels = torch.stack(all_lbls).to(device).long()

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
