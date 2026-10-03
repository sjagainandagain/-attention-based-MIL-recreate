"""
train_colon.py
Colon Cancer 데이터셋에서 Attention MIL 모델 학습
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import transforms

from colon_cancer_bags import ColonCancerBags
from mil_model import AttentionMIL


def train(model, train_ds, optimizer, num_epochs=30):
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        correct = 0

        for bag, label in train_ds:
            label = torch.tensor([float(label)])

            optimizer.zero_grad()
            prob, logit, _ = model(bag)

            loss = F.binary_cross_entropy(prob.view(-1), label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = (prob.view(-1) > 0.5).float()
            correct += (pred == label).sum().item()

        train_loss = total_loss / len(train_ds)
        train_acc = correct / len(train_ds)
        print(f"[Epoch {epoch:02d}] train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")


def evaluate(model, test_ds):
    model.eval()
    correct = 0
    with torch.no_grad():
        for bag, label in test_ds:
            label = torch.tensor([float(label)])
            prob, _, _ = model(bag)
            pred = (prob.view(-1) > 0.5).float()
            correct += (pred == label).sum().item()

    acc = correct / len(test_ds)
    print(f"Test accuracy: {acc:.4f} (error: {1 - acc:.4f})")
    return acc


if __name__ == "__main__":
    torch.manual_seed(1)

    # 학습용: flip/rotation/color jitter로 증강 -> "패치가 뒤집히거나 돌아가도
    # 세포 종류는 그대로"라는 사실을 모델이 배우게 함
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    # 평가용: 증강 없이 원본 그대로 (공정한 평가를 위해)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # bag 개수와 순서만 확인하기 위한 메타데이터용 인스턴스
    meta_ds = ColonCancerBags(root="Patches")
    n_total = len(meta_ds)

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(1)).tolist()
    n_train = int(n_total * 0.8)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    train_full = ColonCancerBags(root="Patches", transform=train_transform)
    test_full = ColonCancerBags(root="Patches", transform=eval_transform)

    train_ds = Subset(train_full, train_idx)
    test_ds = Subset(test_full, test_idx)
    print(f"train bags: {len(train_ds)}, test bags: {len(test_ds)}")

    model = AttentionMIL(in_channels=3)
    optimizer = optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4
    )

    train(model, train_ds, optimizer, num_epochs=40)
    evaluate(model, test_ds)

    torch.save(model.state_dict(), "attention_mil_colon.pt")
    print("Saved trained model to attention_mil_colon.pt")
