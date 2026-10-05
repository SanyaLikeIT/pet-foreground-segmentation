import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PetSegDataset(Dataset):
    def __init__(self, root, split='train', transform=None, download=False):
        self.transform = transform
        if split == 'train':
            self.dataset = OxfordIIITPet(root=root, split='trainval',
                                         target_types='segmentation', download=download)
            with open('data/splits/train.txt') as f:
                self.indices = [int(line) for line in f]
        elif split == 'val':
            self.dataset = OxfordIIITPet(root=root, split='trainval',
                                         target_types='segmentation', download=download)
            with open('data/splits/val.txt') as f:
                self.indices = [int(line) for line in f]
        elif split == 'test':
            self.dataset = OxfordIIITPet(root=root, split='test',
                                         target_types='segmentation', download=download)
            with open('data/splits/test.txt') as f:
                self.indices = [int(line) for line in f]
        else:
            raise ValueError("split must be train/val/test")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, mask = self.dataset[real_idx]
        # mask: PIL, значения 1=pet, 2=background, 3=border
        mask = np.array(mask)
        binary_mask = (mask == 1).astype(np.float32)  # 1 для животного, 0 для фона и границы

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=binary_mask)
            image = augmented['image']
            mask = augmented['mask']
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(binary_mask).unsqueeze(0)
        return image, mask


def get_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])