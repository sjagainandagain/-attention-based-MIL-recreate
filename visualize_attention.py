"""
visualize_attention.py
학습된 Attention MIL 모델이 실제로 target 숫자(9)에 높은 attention을
부여했는지 검증한다.
"""

import torch

from mnist_bags import MnistBags
from mil_model import AttentionMIL


def inspect_bag(model, bag, label, instance_labels, target_number=9):
    model.eval()
    with torch.no_grad():
        prob, _, attn = model(bag)

    attn = attn.view(-1)  # (N,)
    order = torch.argsort(attn, descending=True)

    print(f"bag label: {label}, predicted prob: {prob.item():.4f}")
    print(f"{'rank':<5}{'digit':<8}{'attention':<10}{'is target?'}")
    for rank, idx in enumerate(order):
        digit = instance_labels[idx].item()
        weight = attn[idx].item()
        is_target = "★" if digit == target_number else ""
        print(f"{rank:<5}{digit:<8}{weight:<10.4f}{is_target}")


if __name__ == "__main__":
    test_ds = MnistBags(num_bags=50, train=False, seed=2)

    model = AttentionMIL()
    model.load_state_dict(torch.load("attention_mil.pt", map_location="cpu"))

    # positive bag(라벨 1) 하나를 찾아서 검증
    for i in range(len(test_ds)):
        bag, label = test_ds[i]
        if label == 1:
            instance_labels = test_ds.instance_labels_list[i]
            print(f"=== test bag #{i} ===")
            inspect_bag(model, bag, label, instance_labels)
            break
