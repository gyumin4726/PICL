"""
PICL Configuration for Material Classification
Simple configuration without Continual Learning components
"""

# model settings
model = dict(
    backbone=dict(
        type='VMambaBackbone',
        model_name='vmamba_base_s2l15',
        pretrained_path='./vssm_base_0229_ckpt_epoch_237.pth',
        out_indices=(3,),  # Only last stage for classification
        frozen_stages=0,
        channel_first=True
    ),
    temporal_model=dict(
        type='SequenceToValue',
        input_dim=1024,
        d_model=512,
        device='cuda'
    ),
    head=dict(
        type='ClassificationHead',
        in_channels=512,
        num_classes=5,  # 5 materials: air, water, acrylic, glass, sapphire
        hidden_dim=256,
        dropout=0.1
    )
)

# dataset settings
dataset = dict(
    type='PICLDataset',
    train_data_dir='train',
    test_data_dir='test',
    train_labels='train/dataset_labels.json',
    test_labels='test/dataset_labels_test.json',
    materials=['air', 'water', 'acrylic', 'glass', 'sapphire'],
    refractive_indices={
        'air': 1.0,
        'water': 1.33,
        'acrylic': 1.49,
        'glass': 1.52,
        'sapphire': 1.77
    }
)

# data pipeline
data_pipeline = dict(
    train=dict(
        type='Compose',
        transforms=[
            dict(type='Resize', size=(224, 224)),
            dict(type='ToTensor'),
            dict(type='Normalize', 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225])
        ]
    ),
    test=dict(
        type='Compose',
        transforms=[
            dict(type='Resize', size=(224, 224)),
            dict(type='ToTensor'),
            dict(type='Normalize', 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225])
        ]
    )
)

# training settings
train = dict(
    batch_size=8,
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-4,
    optimizer=dict(type='Adam'),
    scheduler=dict(type='StepLR', step_size=10, gamma=0.1),
    loss=dict(type='CrossEntropyLoss')
)

# evaluation settings
eval = dict(
    batch_size=8,
    metrics=['accuracy', 'precision', 'recall', 'f1_score']
)

# experiment settings
experiment = dict(
    name='picl_material_classification',
    work_dir='./work_dirs/picl',
    log_level='INFO',
    save_checkpoint=True,
    checkpoint_interval=10,
    eval_interval=5
)
