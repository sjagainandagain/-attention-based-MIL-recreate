# Attention-based Deep MIL — Reproduction

PyTorch reproduction of Ilse et al. (2018), [*Attention-based Deep Multiple Instance Learning*](https://arxiv.org/abs/1802.04712).

- 목표: 새 모델 제안 X. 논문 핵심 주장 확인 — "attention 가중치가 bag 라벨을 결정하는 instance를 짚어낸다"
- 데이터: MNIST-bags (파이프라인 검증) → Colon Cancer (실제 병리 패치)
- 구현 범위: 기본 attention (`tanh`)만. Gated Attention 미구현

## Table of Contents

- [Background](#background)
- [Results](#results)
- [Method](#method)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Notes on AI Assistance](#notes-on-ai-assistance)
- [References](#references)

## Background

**Multiple Instance Learning (MIL)**
- bag 하나에 instance 여러 개, 라벨은 bag 단위로만 존재 (weakly supervised)
- 예: 병리 슬라이드(bag) 안 패치(instance) 여러 개, 암세포 패치 1개라도 있으면 양성

**Attention pooling**

```
a_k = softmax_k( wᵀ tanh(V h_k) )
z   = Σ_k a_k h_k
```

- `h_k`: instance 임베딩, `a_k`: 학습된 가중치, `z`: bag 벡터
- `a_k` = 모델이 어느 instance를 근거로 판단했는지 보여주는 값

## Results

> 작성자 실행 결과 (`PROJECT_NOTES.md` 기준). README 작성 시 재실행 확인 X. 시드·환경에 따라 달라질 수 있음.

### 1. MNIST-bags — 파이프라인 검증

- 라벨: bag 안에 숫자 `9`가 1개 이상이면 1

| 항목 | 값 |
|---|---|
| Train / Test bags | 250 / 50 |
| Epochs | 20 |
| Test error | **0.06** (acc 94%) |

- positive bag 1개 확인: `9` instance 2개에 attention 약 99.95% 집중 (0.7418 + 0.2577), 나머지 숫자 ≈ 0

### 2. Colon Cancer — 분류 성능

- H&E 조직 이미지 27×27 패치, 99 bags (positive 52%)
- Train / Test = 79 / 20 (8:2, seed 고정)

| 실험 | 설정 | Train acc | Test acc | 격차 |
|---|---|---|---|---|
| 1 | 증강 없음, 20 epochs | 94.94% | 75.00% | 19.94%p |
| 2 | 증강 추가, 40 epochs | 86.08% | 75.00% | 11.08%p |

- 증강 (train만): horizontal/vertical flip, rotation 180°, ColorJitter 0.2
- 증강 후 격차 감소, test acc 변화 없음

### 3. Attention 검증 — epithelial 세포에 집중하는가

- 근거: 이 데이터셋에서 양성의 생물학적 근거 = epithelial(상피) 세포
- 대상: test positive bag 9개 전체
- 기준선: bag 내 epithelial 비율 (무작위 선택 시 epithelial일 확률)
- 모델의 몫: epithelial instance attention 합

| bag | patches | 기준선 | 모델의 몫 | 차이 |
|---|---|---|---|---|
| 80 | 326 | 0.73 | 0.96 | +0.23 |
| 92 | 332 | 0.85 | 0.96 | +0.11 |
| 50 | 207 | 0.68 | 1.00 | +0.32 |
| 57 | 110 | 0.80 | 1.00 | +0.20 |
| 61 | 84 | 0.30 | 0.99 | +0.70 |
| 95 | 359 | 0.38 | 1.00 | +0.62 |
| 69 | 187 | 0.37 | 1.00 | +0.63 |
| 91 | 692 | 0.14 | 0.74 | +0.60 |
| 93 | 505 | 0.50 | 0.98 | +0.48 |

- 9/9 bag에서 모델의 몫 > 기준선, 평균 차이 **+0.431**
- 기준선 낮은 bag (61: 0.30, 91: 0.14)에서도 attention 대부분 epithelial
- test acc(75%)는 논문보다 낮음. 정확도와 attention 근거는 별도 평가 필요

## Method

```
patch ×N → Conv(5×5)-ReLU-MaxPool → Conv(5×5)-ReLU-MaxPool → AdaptiveAvgPool(4×4)
         → FC(800→500)-ReLU               # instance 임베딩 H
         → attention(500→128→1) → softmax  # 가중치 A
         → Z = A·H                        # bag 벡터
         → Linear(500→1) → sigmoid        # bag 확률
```

- Optimizer: Adam (lr 1e-4, weight decay 1e-4)
- Batch: bag 1개
- Loss: binary cross-entropy
- `in_channels`로 MNIST(1) / Colon(3) 대응, `AdaptiveAvgPool2d`로 패치 크기 무관

## Getting Started

### Requirements

- Python 3
- `torch`, `torchvision`, `numpy`, `pillow`, `gdown` (Colon 데이터 다운로드)

```bash
pip install torch torchvision numpy pillow gdown
```

- GPU 권장 (Colon 학습). CPU도 동작

### Data

| 데이터 | 출처 |
|---|---|
| MNIST | torchvision, 첫 실행 시 `./data`에 자동 다운로드 |
| Colon Cancer | 패치 단위 전처리 공개본 `Patches.tar.gz` (Google Drive, gdown). 원본(Warwick)은 로그인 필요로 접근 실패. 저장소에 데이터 미포함 |

- Colon 폴더 구조: `Patches/<0|1>/<img_id>/*.bmp` (프로젝트 루트에 압축 해제)

### Usage

MNIST-bags

```bash
python mnist_bags.py            # 데이터셋 생성 확인
python train.py                 # 학습 → attention_mil.pt
python visualize_attention.py
```

Colon Cancer

```bash
python colon_cancer_bags.py     # 데이터 로드 확인
python train_colon.py           # 학습 → attention_mil_colon.pt
python verify_attention_colon.py
```

- `verify_attention_colon.py`: `train_colon.py`와 같은 시드(1) 셔플로 test bag 복원 → 학습 후 같은 데이터로 실행
- 현재 `train_colon.py` = 실험 2 설정. 실험 1 재현 시 transform·epoch 직접 수정

## Project Structure

```
.
├── mil_model.py                  # Attention MIL 모델
├── mnist_bags.py                 # MNIST-bags 합성 데이터셋 (instance별 숫자 라벨 포함)
├── train.py                      # MNIST-bags 학습·평가
├── visualize_attention.py        # MNIST 단일 bag attention (초기 버전)
├── colon_cancer_bags.py          # Colon Cancer 로더
├── train_colon.py                # Colon 학습·평가 (train만 증강)
├── visualize_attention_colon.py  # Colon 단일 bag attention (초기 버전)
├── verify_attention_colon.py     # test positive bag 전체 attention 검증 (Results 3)
└── PROJECT_NOTES.md              # 진행 기록·판단 근거
```

## Limitations

- test set 20 bags → 1 bag = 5%p. 75% 자체가 불안정
- 교차검증 X, 단일 시드
- 증강으로 과적합 감소, test acc 개선 X. 데이터 양(79 bags) 한계 추정 — 추가 실험으로 확인 X
- attention 검증: positive test bag 9개, 기준선 대비 단순 비교
- bag 91: 모델의 몫 0.74 (다른 bag 0.96–1.00). 패치 최다(692)·기준선 최저(0.14). 패치 수와 attention 분산 관계 미확인
- Colon 데이터: 전처리 공개본 사용 → 논문 전처리·분할과 다를 수 있음, 논문 수치와 직접 비교 어려움

## Roadmap

- [ ] 교차검증
- [ ] 다중 시드 반복, 평균·분산 보고
- [ ] 패치 수 vs attention 분산 (bag 91)
- [ ] 특징 추출기 CNN → ViT 교체
- [ ] (선택) Gated Attention, ACMIL

## Notes on AI Assistance

- 의대 예과생 자기주도 학습 포트폴리오
- 코드 대부분 Claude(AI 어시스턴트) 도움으로 작성
- 작성자 역할: 논문·코드 단계별 이해, 실험 실행, 결과 해석
- 진행 과정·판단 기록: [`PROJECT_NOTES.md`](PROJECT_NOTES.md)

## References

- Ilse M, Tomczak JM, Welling M. *Attention-based Deep Multiple Instance Learning*. ICML 2018. [arXiv](https://arxiv.org/abs/1802.04712) · [Official code](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)
