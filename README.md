# PICL: 물리 정보 기반 연속 학습을 통한 광학 역문제 해결

## 🎯 연구 개요

이 저장소는 **물리 정보 기반 연속 학습(Physics-Informed Continual Learning, PICL)** 프레임워크를 구현하여, **VMamba**와 **1D Mamba** 아키텍처를 사용해 시간 게이트 광학 산란 이미지로부터 재료의 굴절률을 추정하는 광학 역문제를 해결합니다.

## 🏗️ 아키텍처 개요

### 데이터 흐름 파이프라인

```
5장의 시간 게이트 이미지 → 2D VMamba → 1D Mamba → 재료 분류
(B, 5, 3, 224, 224) → (B, 5, 1024) → (B, 512) → (B, 5)
```

### 1. **2D VMamba 백본** (공간 특징 추출기)
- **입력**: `(B, 5, 3, 224, 224)` - 5장의 시간 게이트 이미지 (0-1ns, 1-2ns, 2-3ns, 3-4ns, 4-5ns)
- **출력**: `(B, 5, 1024)` - 각 시간 스텝의 공간 특징 벡터
- **역할**: 각 개별 이미지 프레임에서 공간 특징을 추출
- **구현**: `vmamba_backbone.py`

### 2. **1D Mamba** (시간 시퀀스 모델러)
- **입력**: `(B, 5, 1024)` - 공간 특징 벡터들의 시퀀스
- **출력**: `(B, 5, 1024)` - 시간적 의존성을 포함한 처리된 시퀀스
- **역할**: 연속된 시간 스텝 간의 시간적 관계를 모델링
- **구현**: `mamba_1d_temporal.py` (state-spaces/mamba의 공식 Mamba1D)

### 3. **시퀀스-투-밸류** (최종 예측)
- **입력**: `(B, 5, 1024)` - 1D Mamba에서 처리된 시퀀스
- **출력**: `(B, 512)` - 최종 특징 벡터 (h_5는 모든 시간 정보를 포함)
- **역할**: 마지막 시간 스텝을 사용하여 시퀀스를 단일 예측으로 변환
- **구현**: `mamba_1d_temporal.py`의 `SequenceToValue` 클래스

## 📊 데이터셋 구조

### 재료 및 굴절률
- **공기**: 1.0
- **물**: 1.33
- **아크릴**: 1.49
- **유리**: 1.52
- **사파이어**: 1.77

### 데이터 구성
```
train/
├── air_4D/images/          # 100개 샘플 × 5장의 시간 게이트 이미지
├── water_4D/images/         # 100개 샘플 × 5장의 시간 게이트 이미지
├── acrylic_4D/images/       # 100개 샘플 × 5장의 시간 게이트 이미지
├── glass_4D/images/         # 100개 샘플 × 5장의 시간 게이트 이미지
├── sapphire_4D/images/      # 100개 샘플 × 5장의 시간 게이트 이미지
└── dataset_labels.json      # 재료 라벨 및 굴절률

test/
├── air_4D_test/images/      # 50개 샘플 × 5장의 시간 게이트 이미지
├── water_4D_test/images/    # 50개 샘플 × 5장의 시간 게이트 이미지
├── acrylic_4D_test/images/  # 50개 샘플 × 5장의 시간 게이트 이미지
├── glass_4D_test/images/    # 50개 샘플 × 5장의 시간 게이트 이미지
├── sapphire_4D_test/images/ # 50개 샘플 × 5장의 시간 게이트 이미지
└── dataset_labels_test.json # 테스트 라벨
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 서브모듈과 함께 저장소 클론
git clone --recursive https://github.com/your-username/PICL.git
cd PICL

# 의존성 설치
pip install torch torchvision
pip install einops
pip install causal-conv1d
pip install mamba-ssm
```

### 2. 기본 사용법

#### 2D VMamba 백본 사용
```python
from vmamba_backbone import VMambaBackbone

# 백본 초기화
backbone = VMambaBackbone(
    model_name='vmamba_base_s2l15',
    out_indices=(3,),
    channel_first=True
)

# 5장의 시간 게이트 이미지 처리
images = torch.randn(2, 5, 3, 224, 224)  # (B, T, C, H, W)
features = backbone(images)  # (B, 5, 1024)
```

#### 시퀀스 모델링을 위한 1D Mamba 사용
```python
from mamba_1d_temporal import Mamba1D, SequenceToValue

# 1D Mamba (공식 구현)
mamba1d = Mamba1D(d_model=512, layer_idx=0, device='cuda')
sequence = torch.randn(2, 5, 512)  # (B, T, D)
processed = mamba1d(sequence)  # (B, 5, 512)

# 시퀀스-투-밸류 변환
seq2val = SequenceToValue(input_dim=1024, d_model=512, device='cuda')
features = torch.randn(2, 5, 1024)  # 2D VMamba에서
final_features = seq2val(features)  # (B, 512) - 모든 시간 정보를 포함한 h_5
```

#### 완전한 PICL 파이프라인
```python
# 완전한 파이프라인: 이미지 → 분류
from picl_classification_experiment import PICLClassifier

model = PICLClassifier(num_classes=5)
images = torch.randn(2, 5, 3, 224, 224)  # 5장의 시간 게이트 이미지
logits = model(images)  # (B, 5) - 재료 분류
```

### 3. 훈련
```bash
# 분류 실험 실행
python picl_classification_experiment.py
```

## 🔬 연구 맥락

### 물리 정보 기반 신경망 (PINN)
- 관측 데이터와 물리 법칙(헬름홀츠 방정식)을 통합
- 이중 손실 함수: 데이터 손실 + 물리 손실
- 숨겨진 물리적 특성의 정확한 추정 가능

### 연속 학습
- 순차 학습에서의 파괴적 망각 문제 해결
- 안정성(기존 지식 유지)과 가소성(새로운 작업 학습)의 균형
- 실제 광학 측정 시나리오에 필수적

### 광학 역문제
- **순방향 문제**: 굴절률 주어짐 → 광학 산란 예측
- **역방향 문제**: 광학 산란 주어짐 → 굴절률 추정
- **도전과제**: 비선형, 부적절한 문제, 시간적 동역학 모델링 필요

## 📁 저장소 구조

```
PICL/
├── FSCIL/                          # 서브모듈: FSCIL 저장소
│   ├── mmfscil/models/
│   │   └── vmamba_backbone.py      # 원본 VMamba 구현
│   └── VMamba/                     # VMamba 소스 코드
├── mamba/                          # 서브모듈: 공식 Mamba 저장소
│   └── mamba_ssm/                  # Mamba 소스 코드
├── vmamba_backbone.py              # 시간 게이트 처리를 위한 수정된 VMamba
├── mamba_1d_temporal.py            # 1D Mamba + 시퀀스-투-밸류
├── picl_classification_experiment.py # 완전한 PICL 실험
├── train/                          # 훈련 데이터셋
├── test/                           # 테스트 데이터셋
└── README.md                       # 이 파일
```

## 🧪 주요 특징

### 1. **시간 게이트 처리**
- 5장의 연속된 시간 게이트 이미지 처리 (0-1ns ~ 4-5ns)
- 시간적 시퀀스 정보 유지
- 물리 정보 기반 시간적 모델링 가능

### 2. **공식 Mamba 구현**
- state-spaces/mamba 공식 구현 사용
- 긴 시퀀스에 대한 선형 복잡도
- Transformer 대비 우수한 성능

### 3. **시퀀스-투-밸류 아키텍처**
- 시간적 시퀀스를 단일 예측으로 변환
- 마지막 시간 스텝이 모든 이전 정보 포함
- 재료 분류 작업에 최적화

### 4. **모듈화 설계**
- 2D VMamba: 공간 특징 추출
- 1D Mamba: 시간적 시퀀스 모델링
- 분류 헤드: 재료 예측
- 확장 및 수정 용이

## 📈 성능

- **데이터셋**: 5개 재료 × 100개 훈련 샘플 × 50개 테스트 샘플
- **입력**: 샘플당 5장의 시간 게이트 이미지 (224×224×3)
- **출력**: 재료 분류 (5개 클래스)
- **아키텍처**: VMamba + 1D Mamba + MLP

## 🔮 향후 연구

1. **물리 손실 통합**: 헬름홀츠 방정식 제약 조건 추가
2. **연속 학습**: 순차적 재료 학습을 위한 FSCIL 구현
3. **회귀**: 연속적인 굴절률 추정으로 확장
4. **실시간 처리**: 실제 응용을 위한 최적화

## 📚 참고문헌

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)
- [FSCIL: Few-Shot Class-Incremental Learning](https://arxiv.org/abs/2004.10956)
- [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10561)

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🤝 기여

기여를 환영합니다! Pull Request를 자유롭게 제출해 주세요.

## 📧 연락처

질문이나 협업에 대해서는 연구팀에 연락해 주세요.

---

**참고**: 이 저장소는 핵심 아키텍처와 구현에 중점을 둡니다. 완전한 물리 정보 기반 연속 학습 실험을 위해서는 추가 구성 요소(물리 손실, 연속 학습 전략)가 향후 버전에서 통합될 예정입니다.
