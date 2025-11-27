# PICL (Physics-Informed Contrastive Learning) Configuration
# 원본 이미지 -> VMamba Backbone -> Feature Maps 파이프라인

# ========================
# Model Configuration
# ========================
model = dict(
    type='PICLModel',
    
    # VMamba Backbone (2D Spatial Feature Extractor)
    backbone=dict(
        model_name='vmamba_base_s2l15',  # 또는 'vmamba_tiny_s2l5', 'vmamba_small_s2l15'
        pretrained_path='./VMamba/vssm_base_0229_ckpt_epoch_237.pth',  # VMamba pretrained weight 경로
        out_indices=(3,),  # 마지막 stage만 사용 (1024 dim)
        frozen_stages=-1,  # -1: 모든 stage 학습 가능, 0~3: 해당 stage까지 freeze
        channel_first=True
    ),
    
    # 1D Mamba Temporal (시간적 시퀀스 모델링)
    temporal=dict(
        input_dim=1024,  # VMamba backbone output dim (정보 손실 방지를 위해 1024 유지)
    ),
    
    # Physics Loss Configuration
    physics=dict(
        c=3e8,                      # 광속 (m/s)
        physics_weight=1.0,         # Physics loss 가중치 (PDE 제약)
        data_weight=1.0,            # Data loss 가중치 (굴절률 예측)
        classification_weight=10.0, # Classification loss 가중치 (주 목표)
        # 물리적 단위 (MCX 설정 기준)
        dx=1e-3,                    # 공간 스텝 x 방향 (1.0 mm = 1e-3 m)
        dy=1e-3,                    # 공간 스텝 y 방향 (1.0 mm = 1e-3 m)
        dt=1e-9,                    # 시간 스텝 (1.0 ns = 1e-9 s)
    ),
    
    # Classification Configuration
    num_classes=5,  # 5개 재료: air, water, acrylic, glass, sapphire
)

# ========================
# Dataset Configuration
# ========================
data = dict(
    # Training dataset
    train=dict(
        data_root='./train',
        label_file='./train/dataset_labels.json',  # JSON 파일에서 레이블 로드
        image_size=224,
        batch_size=8,
        num_workers=4,
        shuffle=True
    ),
    
    # Test dataset
    test=dict(
        data_root='./test',
        label_file='./test/dataset_labels_test.json',  # JSON 파일에서 레이블 로드
        image_size=224,
        batch_size=4,
        num_workers=4,
        shuffle=False
    )
)

# ========================
# Training Configuration
# ========================
train = dict(
    epochs=25,
    learning_rate=0.0005,
    weight_decay=0.0001,
    
    # Optimizer
    optimizer='Adam',  # 'Adam', 'SGD', 'AdamW'
    
    # Learning rate scheduler
    scheduler='ReduceLROnPlateau',  # 'CosineAnnealing', 'ReduceLROnPlateau', 'StepLR', 'MultiStepLR'
    scheduler_params=dict(
        factor=0.5,      # 학습률을 0.5배로 감소
        patience=2,      # 5 epoch 동안 개선 없으면 감소
        min_lr=1e-6      # 최소 학습률
    ),
    
    # Checkpoint
    save_interval=10,  # 10 epoch마다 저장
    save_best=True,    # Best model 저장
    
    # Logging
    log_interval=10,   # 10 iteration마다 로그 출력
)

# ========================
# Runtime Configuration
# ========================
work_dir = './work_dirs'  # 작업 디렉토리
device = 'cuda'  # 'cuda' or 'cpu'
seed = 42        # Random seed
deterministic = True  # 재현성을 위한 deterministic 모드

# GPU 설정
gpu_ids = [0]  # 사용할 GPU ID 리스트

# Logging
log_level = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
log_file = None     # None이면 work_dir에 자동 생성

# Resume training
resume_from = None  # checkpoint 경로 (None이면 처음부터 학습)
load_from = None    # pretrained model 경로 (fine-tuning용)

