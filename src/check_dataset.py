import matplotlib.pyplot as plt
import numpy as np
from src.dataset import PetSegDataset, get_train_transform, get_val_transform


def denormalize(img):
    img = img.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return img.clip(0, 1)


def show_samples(split, transform, n=3, save_path=None):
    ds = PetSegDataset('data/raw', split=split, transform=transform)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    for i in range(n):
        img, mask = ds[i]
        img_np = denormalize(img)
        mask_np = mask[0].numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'{split} image {i}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('binary mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img_np)
        axes[i, 2].imshow(mask_np, cmap='Reds', alpha=0.5)
        axes[i, 2].set_title('overlay')
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'saved: {save_path}')
    plt.close()


if __name__ == '__main__':
    show_samples('train', get_train_transform(), n=3,
                 save_path='report/example_train.png')
    show_samples('val', get_val_transform(), n=3,
                 save_path='report/example_val.png')