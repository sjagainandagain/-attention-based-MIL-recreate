"""
train.py
MNIST-bags에서 Attention MIL 모델 학습 (Ilse et al. 2018 sanity check)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from mnist_bags import MnistBags
from mil_model import AttentionMIL


def train(model, train_ds, optimizer, num_epochs=20):
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

    train_ds = MnistBags(num_bags=250, train=True, seed=1)
    test_ds = MnistBags(num_bags=50, train=False, seed=2)

    model = AttentionMIL()
    optimizer = optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4
    )

    train(model, train_ds, optimizer, num_epochs=20)
    evaluate(model, test_ds)

    torch.save(model.state_dict(), "attention_mil.pt")
    print("Saved trained model to attention_mil.pt")
