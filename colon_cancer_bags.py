"""
colon_cancer_bags.py
Colon Cancer MIL 데이터셋 로더
폴더 구조: Patches/<label>/<img_id>/<patch 파일들>
- label: 0(negative) 또는 1(positive)
- img_id 폴더 하나 = bag 하나
- 그 안의 개별 파일들 = instance(patch)
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ColonCancerBags(Dataset):
    def __init__(self, root="Patches", transform=None):
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])

        self.bag_paths = []   # 각 원소: 해당 bag(img 폴더) 안 patch 파일 경로 리스트
        self.labels = []

        for label_str in sorted(os.listdir(root)):        # "0", "1"
            label_dir = os.path.join(root, label_str)
            if not os.path.isdir(label_dir):
                continue
            label = int(label_str)

            for img_id in sorted(os.listdir(label_dir)):
                img_dir = os.path.join(label_dir, img_id)
                if not os.path.isdir(img_dir):
                    continue
                patch_files = [
                    os.path.join(img_dir, f)
                    for f in sorted(os.listdir(img_dir))
                    if os.path.isfile(os.path.join(img_dir, f))
                ]
                if len(patch_files) == 0:
                    continue

                self.bag_paths.append(patch_files)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch_files = self.bag_paths[idx]
        label = self.labels[idx]

        instances = []
        for f in patch_files:
            img = Image.open(f).convert("RGB")
            instances.append(self.transform(img))

        bag = torch.stack(instances)   # (N, 3, H, W)
        return bag, label


if __name__ == "__main__":
    ds = ColonCancerBags(root="Patches")
    print(f"총 bag 수: {len(ds)}")
    print(f"positive bag 비율: {sum(ds.labels) / len(ds):.2f}")

    bag, label = ds[0]
    print(f"첫 bag shape: {bag.shape}, label: {label}")
    print(f"patch 파일 예시 경로: {ds.bag_paths[0][0]}")
