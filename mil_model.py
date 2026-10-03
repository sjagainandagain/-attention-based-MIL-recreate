"""
mil_model.py
Attention-based Deep MIL (Ilse et al., 2018) 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    def __init__(self, in_channels=1, feature_dim=500, attention_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim

        # 1. Instance feature extractor (작은 CNN)
        #    bag 안의 이미지 한 장(instance)을 벡터로 변환하는 역할
        #    AdaptiveAvgPool2d로 patch 크기가 달라져도(27x27, 32x32 등) 항상 4x4로 고정
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.feature_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.feature_dim),
            nn.ReLU(),
        )

        # 2. Attention 메커니즘: instance 임베딩마다 "중요도 점수"를 계산
        #    a_k = softmax( w^T tanh(V h_k) )
        self.attention_V = nn.Linear(self.feature_dim, self.attention_dim)
        self.attention_w = nn.Linear(self.attention_dim, 1)

        # 3. Bag-level classifier
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, bag):
        # bag: (num_instances, 1, 28, 28)  <- 가방 하나에 들어있는 이미지들
        H = self.feature_extractor(bag)                # (N, 50, 4, 4)
        H = H.view(H.size(0), -1)                       # (N, 50*4*4)
        H = self.feature_fc(H)                           # (N, feature_dim) <- instance 임베딩

        # attention score 계산
        A = torch.tanh(self.attention_V(H))               # (N, attention_dim)
        A = self.attention_w(A)                            # (N, 1)
        A = torch.transpose(A, 1, 0)                        # (1, N)
        A = F.softmax(A, dim=1)                              # 가중치 합 = 1

        # attention 가중평균 -> bag을 대표하는 벡터 하나로 집계
        Z = torch.mm(A, H)                                    # (1, feature_dim)

        # bag-level 예측
        Y_logit = self.classifier(Z)                           # (1, 1)
        Y_prob = torch.sigmoid(Y_logit)

        return Y_prob, Y_logit, A


if __name__ == "__main__":
    from mnist_bags import MnistBags

    ds = MnistBags(num_bags=5, train=True, seed=1)
    model = AttentionMIL()

    bag, label = ds[0]
    prob, logit, attn = model(bag)

    print(f"bag shape: {bag.shape}, label: {label}")
    print(f"predicted prob: {prob.item():.4f}")
    print(f"attention weights shape: {attn.shape}, sum: {attn.sum().item():.4f}")
