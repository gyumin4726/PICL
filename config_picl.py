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
        physics_weight=0.01,         # Physics loss 가중치 (PDE 제약)
        data_weight=10.0,            # Data loss 가중치 (굴절률 예측)
        classification_weight=10.0, # Classification loss 가중치 (주 목표)
        # CombinedLoss 가중치 (DRLoss + CrossEntropyLoss)
        dr_weight=1.0,              # DRLoss 가중치 (0.0: 사용 안 함, 1.0: DRLoss만 사용)
        ce_weight=0.0,              # CrossEntropyLoss 가중치 (1.0: CE Loss만 사용)
        # 물리적 단위 (MCX 설정 기준)
        dx=1e-3,                    # 공간 스텝 x 방향 (1.0 mm = 1e-3 m)
        dy=1e-3,                    # 공간 스텝 y 방향 (1.0 mm = 1e-3 m)
        dt=1e-9,                    # 시간 스텝 (1.0 ns = 1e-9 s)
    ),
    
    # Classification Configuration
    num_classes=5,  # Base 5종 조직: epidermis, dermis, subcutaneous_fat, muscle, whole_blood
)

# ========================
# Dataset Configuration
# ========================
data = dict(
    # Training dataset
    train=dict(
        data_root='./train',
        label_file='./train/dataset_labels_train.json',  # JSON 파일에서 레이블 로드
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
    
)

# ========================
# Runtime Configuration
# ========================
work_dir = './work_dirs'  # 작업 디렉토리
device = 'cuda'  # 'cuda' or 'cpu'
seed = 42        # Random seed
deterministic = True  # 재현성을 위한 deterministic 모드


