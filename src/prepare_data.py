import os
import random
from torchvision.datasets import OxfordIIITPet

def main():
    root = 'data/raw'
    os.makedirs(root, exist_ok=True)
    os.makedirs('data/splits', exist_ok=True)

    # Скачиваем trainval и test
    trainval = OxfordIIITPet(root=root, split='trainval',
                             target_types='segmentation', download=True)
    test = OxfordIIITPet(root=root, split='test',
                         target_types='segmentation', download=True)

    n_trainval = len(trainval)
    indices = list(range(n_trainval))
    random.seed(42)
    random.shuffle(indices)

    split = int(0.8 * n_trainval)
    train_idx = indices[:split]
    val_idx = indices[split:]

    with open('data/splits/train.txt', 'w') as f:
        f.write('\n'.join(map(str, train_idx)))
    with open('data/splits/val.txt', 'w') as f:
        f.write('\n'.join(map(str, val_idx)))
    with open('data/splits/test.txt', 'w') as f:
        f.write('\n'.join(map(str, range(len(test)))))

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test)}")

if __name__ == '__main__':
    main()