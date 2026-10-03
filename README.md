# Attention-based Deep MIL 재현 프로젝트

Ilse et al. (2018), [*Attention-based Deep Multiple Instance Learning*](https://arxiv.org/abs/1802.04712)를 PyTorch로 직접 재현한 학습 프로젝트입니다.

새로운 모델을 제안하는 것이 아니라, 논문의 핵심 주장인 **"attention 가중치가 bag 라벨을 결정짓는 instance를 정확히 짚어낸다"** 를 작은 벤치마크에서 직접 확인하는 데 초점을 맞췄습니다.

## 배경과 AI 도구 사용에 대한 안내

- 의과대학 예과생이 자기주도 학습으로 진행한 포트폴리오 프로젝트입니다.
- **코드 작성의 대부분은 Claude(AI 어시스턴트)의 도움을 받았습니다.** 작성자는 논문과 코드를 단계별로 읽으며 이해하고, 실험을 직접 실행하며 결과를 해석하는 방식으로 진행했습니다.
- 진행 과정과 판단의 기록은 [`PROJECT_NOTES.md`](PROJECT_NOTES.md)에 있습니다.

## MIL이란?

Multiple Instance Learning(MIL)은 **bag(가방) 하나에 여러 instance가 들어 있고, 라벨은 bag 전체에만 붙어 있는** 약지도학습(weakly supervised) 설정입니다.

> 예) 병리 슬라이드 한 장(bag) 안에 패치 여러 개(instance)가 있고, 암세포 패치가 하나라도 있으면 슬라이드가 양성.

### Attention pooling

각 instance 임베딩 `h_k`마다 중요도 점수를 학습하고, softmax로 가중치 `a_k`를 만든 뒤 가중평균으로 bag 벡터 `z`를 만듭니다.

```
a_k = softmax_k( wᵀ tanh(V h_k) )
z   = Σ_k a_k h_k
```

`a_k`는 "모델이 어떤 instance를 보고 판단했는지"를 보여 주는 해석 가능성의 근거가 됩니다.

> 이 저장소는 논문의 기본 attention(`tanh`만 사용)을 구현했고, Gated Attention은 구현하지 않았습니다.

## 결과 요약

아래 수치는 작성자가 직접 실행한 결과이며(`PROJECT_NOTES.md` 기준), 이 README를 쓰면서 다시 실행해 확인한 값은 아닙니다. 시드와 환경에 따라 달라질 수 있습니다.

### 1단계: MNIST-bags (파이프라인 검증)

MNIST 숫자를 bag으로 묶은 합성 데이터에서, bag 안에 숫자 `9`가 하나라도 있으면 라벨 1입니다.

| 항목 | 값 |
|---|---|
| 학습/테스트 bag 수 | 250 / 50 |
| 에폭 | 20 |
| Test error | **0.06** (정확도 94%) |

단일 positive bag을 확인했을 때, `9`가 들어 있는 instance 2개에 attention의 약 99.95%(0.7418 + 0.2577)가 모였고 다른 숫자는 거의 0이었습니다.

### 2단계: Colon Cancer (실제 병리 데이터)

H&E 염색 조직 이미지에서 추출한 27×27 패치 데이터입니다. 99 bags(positive 52%), train/test = 79/20(8:2, seed 고정).

| 실험 | 설정 | Train acc | Test acc |
|---|---|---|---|
| 1. 증강 없음 | 20 epoch | 94.94% | 75.00% |
| 2. 증강 추가 | 40 epoch | 86.08% | 75.00% |

증강은 학습 데이터에만 적용했습니다(수평/수직 flip, 180° 회전, ColorJitter 0.2). 증강으로 train-test 격차는 19.94%p에서 11.08%p로 줄었지만 test 정확도는 그대로였습니다.

### 3단계: Attention이 올바른 근거를 보는가

정확도만으로는 "옳은 이유로 판단했는지" 알 수 없어서, attention이 이 데이터셋에서 양성의 생물학적 근거인 **epithelial(상피) 세포**에 모이는지 검증했습니다.

test의 positive bag 9개 전체에서 두 값을 비교했습니다.

- **기준선**: bag에서 무작위로 patch를 뽑았을 때 epithelial일 확률 (= bag 내 epithelial 비율)
- **모델의 몫**: epithelial instance들이 차지한 attention 가중치의 합

| bag | patch 수 | 기준선 | 모델의 몫 | 차이 |
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

**9개 bag 모두에서 모델의 몫이 기준선보다 높았고, 평균 차이는 +0.431입니다.** 기준선이 낮은 bag(61: 0.30, 91: 0.14)에서도 attention이 대부분 epithelial에 집중됐으므로, 우연으로 설명하기 어렵습니다.

test 정확도(75%)는 논문보다 낮지만, 모델이 제시하는 판단 근거는 일관되게 타당했습니다. 정확도와 해석 가능성은 따로 평가해야 한다는 점을 확인했습니다.

## 한계점

- **test set이 작습니다.** 20 bags라 1개 차이가 5%p입니다. 75%라는 수치 자체가 불안정합니다.
- **교차검증을 하지 않았고**, 시드도 하나만 사용했습니다.
- 데이터 증강으로 과적합은 줄었지만 **test 정확도는 개선되지 않았습니다.** 79개 bag이라는 데이터 양이 한계로 보이지만, 이 해석은 추가 실험으로 확인하지 못했습니다.
- attention 검증은 positive test bag 9개에 대한 것이고, 기준선 대비 비교라는 단순한 지표입니다.
- **bag 91**은 model share가 0.74로 다른 bag(0.96~1.00)보다 낮았습니다. patch 수가 가장 많고(692) 기준선이 가장 낮은(0.14) bag입니다. patch 수가 많을수록 attention이 분산되는지는 확인하지 못했고, 후속 질문으로 남겨 두었습니다.
- Colon Cancer 데이터는 원본(Warwick)에 접근할 수 없어 **이미 patch 단위로 전처리된 공개 버전**을 사용했습니다. 논문의 전처리·분할과 동일하지 않을 수 있어서 논문 수치와 직접 비교하기 어렵습니다.
- 실험 1(증강 없음, 20 epoch)을 다시 하려면 `train_colon.py`의 transform과 epoch 수를 직접 수정해야 합니다. 현재 코드는 실험 2 설정입니다.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `mil_model.py` | Attention MIL 모델. `in_channels`로 MNIST(1)/Colon(3) 대응, `AdaptiveAvgPool2d`로 patch 크기와 무관하게 동작 |
| `mnist_bags.py` | MNIST-bags 합성 데이터셋 (검증용으로 instance별 숫자 라벨도 저장) |
| `train.py` | MNIST-bags 학습 및 평가 |
| `colon_cancer_bags.py` | Colon Cancer 데이터셋 로더 (`Patches/<label>/<img_id>/*.bmp`) |
| `train_colon.py` | Colon Cancer 학습 및 평가 (train에만 증강 적용) |
| `visualize_attention.py` | MNIST 단일 bag의 attention 확인 (초기 버전) |
| `visualize_attention_colon.py` | Colon 단일 bag의 attention 확인 (초기 버전) |
| `verify_attention_colon.py` | test positive bag 전체에 대한 체계적 attention 검증 (3단계 결과 생성) |

모델 구성은 다음과 같습니다.

```
patch (N개) → Conv(5×5)-ReLU-MaxPool → Conv(5×5)-ReLU-MaxPool → AdaptiveAvgPool(4×4)
            → FC(800→500)-ReLU            # instance 임베딩 H
            → attention(500→128→1) → softmax   # 가중치 A
            → Z = A·H                     # bag 벡터
            → Linear(500→1) → sigmoid     # bag 확률
```

학습은 Adam(lr=1e-4, weight_decay=1e-4), bag 1개씩(batch=1), 손실은 binary cross-entropy입니다.

## 재현 방법

환경: Python 3, `torch`, `torchvision`, `numpy`, `pillow` (Colon 데이터 다운로드에 `gdown`)

```bash
pip install torch torchvision numpy pillow gdown
```

### MNIST-bags

```bash
python mnist_bags.py        # 데이터셋 생성 확인
python train.py             # 학습 → attention_mil.pt 저장
python visualize_attention.py
```

MNIST는 처음 실행할 때 `./data`로 자동 다운로드됩니다.

### Colon Cancer

1. patch 단위로 전처리된 `Patches.tar.gz`를 받아 프로젝트 루트에서 압축을 풉니다. 폴더 구조는 `Patches/<0|1>/<img_id>/*.bmp`여야 합니다. 이 저장소에는 데이터가 포함되어 있지 않습니다.
2. 실행합니다.

```bash
python colon_cancer_bags.py     # 데이터 로드 확인
python train_colon.py           # 학습 → attention_mil_colon.pt 저장
python verify_attention_colon.py
```

`verify_attention_colon.py`는 `train_colon.py`와 같은 시드(1)의 셔플로 test bag을 복원하므로, 학습 후에 같은 데이터로 실행해야 합니다.

CPU로도 동작하지만 Colon Cancer 학습은 GPU가 있는 환경(예: Colab)에서 하는 편이 빠릅니다.

## 앞으로 할 일

- 교차검증으로 test set 불안정성 완화
- 시드를 바꿔 여러 번 반복해 평균과 분산 보고
- patch 수와 attention 분산의 관계 확인(bag 91 관찰)
- (선택) Gated Attention, ACMIL 등으로 확장

## 참고 문헌

- M. Ilse, J. M. Tomczak, M. Welling. *Attention-based Deep Multiple Instance Learning*. ICML 2018. ([논문](https://arxiv.org/abs/1802.04712), [공식 구현](https://github.com/AMLab-Amsterdam/AttentionDeepMIL))
