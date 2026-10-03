"""
visualize_attention_colon.py
학습된 Attention MIL 모델이 실제로 epithelial(상피) 세포 patch에
높은 attention을 부여했는지 검증한다.
파일명 형식: imgN-xposX-yposY-<celltype>.bmp 에서 celltype을 그대로 이용.
"""

import os
import torch

from colon_cancer_bags import ColonCancerBags
from mil_model import AttentionMIL


def cell_type_from_filename(path):
    # img1-xpos116-ypos95-fibroblast.bmp -> "fibroblast"
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    return name.split("-")[-1]


def inspect_bag(model, bag, label, patch_paths):
    model.eval()
    with torch.no_grad():
        prob, _, attn = model(bag)

    attn = attn.view(-1)
    order = torch.argsort(attn, descending=True)

    print(f"bag label: {label}, predicted prob: {prob.item():.4f}")
    print(f"{'rank':<5}{'cell type':<14}{'attention':<10}{'epithelial?'}")
    for rank, idx in enumerate(order):
        cell_type = cell_type_from_filename(patch_paths[idx])
        weight = attn[idx].item()
        mark = "★" if cell_type == "epithelial" else ""
        print(f"{rank:<5}{cell_type:<14}{weight:<10.4f}{mark}")


if __name__ == "__main__":
    ds = ColonCancerBags(root="Patches")

    model = AttentionMIL(in_channels=3)
    model.load_state_dict(torch.load("attention_mil_colon.pt", map_location="cpu"))

    # positive bag(라벨 1) 하나를 찾아서 검증
    for i in range(len(ds)):
        bag, label = ds[i]
        if label == 1:
            patch_paths = ds.bag_paths[i]
            print(f"=== bag #{i} ({ds.bag_paths[i][0].split('/')[1]}) ===")
            inspect_bag(model, bag, label, patch_paths)
            break
