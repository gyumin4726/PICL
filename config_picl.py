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
        input_dim=1024,  # VMamba backbone output dim
        d_model=512,     # Temporal feature dimension
    ),
    
    # Physics Loss Configuration
    physics=dict(
        c=3e8,              # 광속 (m/s)
        physics_weight=10.0, # Physics loss 가중치
        data_weight=1.0,    # Data loss 가중치
    )
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
    epochs=10,
    learning_rate=0.001,
    weight_decay=0.0001,
    
    # Optimizer
    optimizer='Adam',  # 'Adam', 'SGD', 'AdamW'
    
    # Learning rate scheduler
    scheduler='CosineAnnealing',  # 'CosineAnnealing', 'StepLR', 'MultiStepLR'
    scheduler_params=dict(
        T_max=100,
        eta_min=1e-6
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

