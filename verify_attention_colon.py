"""
verify_attention_colon.py
test에 속한 positive bag 전부에 대해, attention이 epithelial patch에
기준선(무작위로 뽑았을 때의 기대치)보다 더 몰리는지 확인한다.
"""

import os
import torch

from colon_cancer_bags import ColonCancerBags
from mil_model import AttentionMIL


def cell_type_from_filename(path):
    # img1-xpos116-ypos95-fibroblast.bmp -> "fibroblast"
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split("-")[-1]


if __name__ == "__main__":
    ds = ColonCancerBags(root="Patches")
    n_total = len(ds)

    # [1] train_colon.py와 똑같은 셔플(시드 1)로 test에 속한 bag 번호를 복원
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(1)).tolist()
    n_train = int(n_total * 0.8)
    test_idx = indices[n_train:]

    model = AttentionMIL(in_channels=3)
    model.load_state_dict(torch.load("attention_mil_colon.pt", map_location="cpu"))
    model.eval()

    print(f"{'bag':<6}{'n_patch':<9}{'baseline':<10}{'model_share':<13}{'diff'}")

    diffs = []
    for i in test_idx:
        bag, label = ds[i]
        if label != 1:            # [2] positive bag만 사용
            continue

        patch_paths = ds.bag_paths[i]
        is_epi = torch.tensor(
            [cell_type_from_filename(p) == "epithelial" for p in patch_paths]
        )

        with torch.no_grad():
            _, _, attn = model(bag)
        attn = attn.view(-1)

        baseline = is_epi.float().mean().item()   # [3] 기준선: epithelial 비율
        share = attn[is_epi].sum().item()          # [4] 모델의 몫: epithelial의 attention 합
        diff = share - baseline
        diffs.append(diff)

        print(f"{i:<6}{len(patch_paths):<9}{baseline:<10.2f}{share:<13.2f}{diff:+.2f}")

    # [5] 요약 숫자 두 개
    n_higher = sum(d > 0 for d in diffs)
    print(f"\n모델의 몫이 기준선보다 높은 bag: {len(diffs)}개 중 {n_higher}개")
    print(f"평균 차이: {sum(diffs) / len(diffs):+.3f}")
