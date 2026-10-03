"""
mnist_bags.py
MIL 학습 파이프라인 검증을 위한 합성 데이터셋 (MNIST-bags)
참고: Ilse et al. (2018), "Attention-based Deep Multiple Instance Learning"
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MnistBags(Dataset):
    """
    각 bag(가방)은 여러 장의 MNIST 숫자 이미지(instance)로 구성됨.
    bag 안에 target_number(기본값 9)가 하나라도 있으면 bag label = 1, 없으면 0.
    -> 실제 병리 슬라이드(bag) 안에 암세포 패치(instance)가 있는지 여부를
       가장 단순한 형태로 흉내낸 합성 데이터.

    instance_labels_list: 각 instance의 실제 숫자 레이블 (디버깅/attention 검증용).
    실전 MIL(예: 병리 슬라이드)에서는 instance 단위 레이블에 접근할 수 없는 게 보통이지만,
    합성 데이터에서는 "attention이 정말 target에 집중했는지" 검증하는 데 쓸 수 있음.
    """

    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2,
                 num_bags=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bags = num_bags
        self.train = train

        self.r = np.random.RandomState(seed)

        loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                './data', train=self.train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=60000 if train else 10000, shuffle=False
        )
        self.all_imgs, self.all_labels = next(iter(loader))

        self.bags_list, self.labels_list, self.instance_labels_list = self._form_bags()

    def _form_bags(self):
        bags_list = []
        labels_list = []
        instance_labels_list = []

        for _ in range(self.num_bags):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length))
            bag_length = max(bag_length, 1)

            indices = torch.LongTensor(
                self.r.randint(0, len(self.all_imgs), bag_length)
            )
            bag_imgs = self.all_imgs[indices]
            bag_digit_labels = self.all_labels[indices]

            # bag label: target 숫자가 하나라도 포함되어 있으면 1
            bag_label = int((bag_digit_labels == self.target_number).any())

            bags_list.append(bag_imgs)
            labels_list.append(bag_label)
            instance_labels_list.append(bag_digit_labels)

        return bags_list, labels_list, instance_labels_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        bag = self.bags_list[idx]          # shape: (bag_length, 1, 28, 28)
        label = self.labels_list[idx]      # 0 또는 1
        return bag, label


if __name__ == "__main__":
    train_ds = MnistBags(num_bags=250, train=True, seed=1)
    test_ds = MnistBags(num_bags=50, train=False, seed=2)

    print(f"train bags: {len(train_ds)}, test bags: {len(test_ds)}")
    bag, label = train_ds[0]
    print(f"첫 번째 bag shape: {bag.shape}, label: {label}")
    print(f"positive bag 비율 (train): {sum(train_ds.labels_list) / len(train_ds):.2f}")
