"""
PICL Training Script
원본 이미지 -> VMamba Backbone -> Feature Maps 학습 파이프라인
"""

import os
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau
from PIL import Image
import numpy as np
from tqdm import tqdm

from picl_model import PICLModel


class OpticalScatteringDataset(Dataset):
    """
    4D 광학 산란 이미지 데이터셋
    각 샘플: 5개의 시간 게이트 이미지 (B, 5, 3, 224, 224)
    JSON 파일에서 레이블 정보를 로드합니다.
    """
    
    # Material name to class index mapping
    MATERIAL_TO_IDX = {
        'air': 0,
        'water': 1,
        'acrylic': 2,
        'glass': 3,
        'sapphire': 4
    }
    
    IDX_TO_MATERIAL = {v: k for k, v in MATERIAL_TO_IDX.items()}
    
    def __init__(self, data_root, label_file, image_size=224, transform=None):
        self.data_root = Path(data_root)
        self.label_file = label_file
        self.image_size = image_size
        self.transform = transform
        
        # JSON 파일에서 데이터 로드
        self.samples = []
        self._load_from_json()
    
    def _load_from_json(self):
        """JSON 파일에서 데이터 샘플 로드"""
        with open(self.label_file, 'r') as f:
            dataset_info = json.load(f)
        
        # 각 샘플에 대해 이미지 경로 구성
        for sample in dataset_info['samples']:
            sample_id = sample['sample_id']
            material = sample['material']
            n_value = sample['refractive_index']
            base_path = sample['base_path']
            
            # 5개의 시간 게이트 이미지 경로 생성
            # 실제 구조: air_4D/images/air001/air001_1.png ~ air001_5.png
            time_gates = []
            for gate_idx in range(1, 6):  # 1~5
                img_path = self.data_root / base_path / f"{sample_id}_{gate_idx}.png"
                time_gates.append(img_path)
            
            # 모든 이미지가 존재하는지 확인
            if all(img_path.exists() for img_path in time_gates):
                self.samples.append({
                    'images': time_gates,
                    'class': material,
                    'n_true': n_value,
                    'sample_id': sample_id
                })
        
        print(f"Loaded {len(self.samples)} samples from {self.label_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 5개의 시간 게이트 이미지 로드 (시퀀스)
        images = []
        for img_path in sample['images']:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
        
        images = np.stack(images, axis=0)  # (5, H, W, 3) - 5개 시퀀스
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)  # (5, 3, H, W) - [시퀀스, 채널, H, W]
        
        n_true = torch.tensor(sample['n_true'], dtype=torch.float32)
        
        # Material class to index
        material_class = sample['class']
        material_idx = self.MATERIAL_TO_IDX.get(material_class, 0)
        material_label = torch.tensor(material_idx, dtype=torch.long)
        
        # 5개 시퀀스 => 1개 예측값
        return images, n_true, material_label


def load_config(config_path):
    """Config 파일 로드"""
    config = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config)
    
    # dict 타입만 추출
    config = {k: v for k, v in config.items() if isinstance(v, dict) and not k.startswith('_')}
    return config


def build_optimizer(model, config):
    """Optimizer 생성"""
    opt_type = config['train']['optimizer']
    lr = config['train']['learning_rate']
    wd = config['train']['weight_decay']
    
    if opt_type == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    elif opt_type == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    return optimizer


def build_scheduler(optimizer, config):
    """Learning rate scheduler 생성"""
    sched_type = config['train']['scheduler']
    params = config['train']['scheduler_params']
    
    if sched_type == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=params['T_max'], eta_min=params['eta_min'])
    elif sched_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',  # loss가 감소하는 방향
            factor=params.get('factor', 0.5),  # 학습률을 0.5배로 감소
            patience=params.get('patience', 5),  # 5 epoch 동안 개선 없으면 감소
            verbose=True,  # 학습률 변경 시 출력
            min_lr=params.get('min_lr', 1e-6)  # 최소 학습률
        )
    elif sched_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    elif sched_type == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=params['milestones'], gamma=params['gamma'])
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")
    
    return scheduler


def train_epoch(model, dataloader, optimizer, device, epoch):
    """1 epoch 학습"""
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0
    total_cls_loss = 0
    correct = 0
    total_samples = 0
    
    # 가중치 가져오기
    w_cls = model.physics_config.get('classification_weight', 10.0)
    w_data = model.physics_config.get('data_weight', 1.0)
    w_physics = model.physics_config.get('physics_weight', 1.0)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for images, n_true, material_label in pbar:
        images = images.to(device)
        n_true = n_true.to(device)
        material_label = material_label.to(device)
        
        # Forward pass
        results = model(images, n_true, material_label)
        
        loss = results['total_loss']
        data_loss = results['data_loss']
        physics_loss = results['physics_loss']
        cls_loss = results['classification_loss']
        class_pred = results['etf_class_pred']  # ETF Classifier의 분류 결과
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계
        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_physics_loss += physics_loss.item()
        total_cls_loss += cls_loss.item()
        
        # Accuracy
        correct += (class_pred == material_label).sum().item()
        total_samples += material_label.size(0)
        
        # 가중치가 곱해진 loss 값 표시
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{w_cls * cls_loss.item():.4f}",
            'data': f"{w_data * data_loss.item():.4f}",
            'physics': f"{w_physics * physics_loss.item():.4f}",
            'acc': f"{100.0 * correct / total_samples:.1f}%"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_data_loss = total_data_loss / len(dataloader)
    avg_physics_loss = total_physics_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    accuracy = 100.0 * correct / total_samples
    
    return avg_loss, avg_data_loss, avg_physics_loss, avg_cls_loss, accuracy


def validate(model, dataloader, device):
    """검증"""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    predictions = []
    ground_truths = []
    class_predictions = []
    class_labels = []
    
    with torch.no_grad():
        for images, n_true, material_label in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            n_true = n_true.to(device)
            material_label = material_label.to(device)
            
            # Forward pass
            results = model(images, n_true, material_label)
            
            total_loss += results['total_loss'].item()
            total_cls_loss += results['classification_loss'].item()
            predictions.extend(results['n_pred'].cpu().numpy())
            ground_truths.extend(n_true.cpu().numpy())
            class_predictions.extend(results['etf_class_pred'].cpu().numpy())  # ETF Classifier의 분류 결과
            class_labels.extend(material_label.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    class_predictions = np.array(class_predictions)
    class_labels = np.array(class_labels)
    
    # MSE 계산
    mse = np.mean((predictions - ground_truths) ** 2)
    
    # Accuracy 계산
    accuracy = 100.0 * np.mean(class_predictions == class_labels)
    
    return avg_loss, avg_cls_loss, mse, accuracy, predictions, ground_truths, class_predictions, class_labels


def main():
    parser = argparse.ArgumentParser(description='Train PICL Model')
    parser.add_argument('config', type=str, help='Config file path')
    parser.add_argument('--work-dir', type=str, default=None, help='Working directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Config 로드
    config = load_config(args.config)
    
    # Random seed 설정 (재현성을 위해)
    seed = config.get('seed', 42)
    deterministic = config.get('deterministic', True)
    
    def set_seed(seed_value):
        """재현성을 위한 seed 설정 함수"""
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # CUDA 비결정적 연산 방지 (성능 저하 가능하지만 재현성 보장)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA 10.2+
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except:
                pass  # PyTorch 버전에 따라 지원되지 않을 수 있음
    
    set_seed(seed)
    print(f"Random seed: {seed}, Deterministic: {deterministic}")
    
    # DataLoader용 generator 생성 (재현성을 위해)
    def worker_init_fn(worker_id):
        """DataLoader worker의 seed 설정"""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Generator for DataLoader (재현성 보장)
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Work directory 설정
    work_dir = args.work_dir if args.work_dir else config['work_dir']
    os.makedirs(work_dir, exist_ok=True)
    
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋 생성
    print("\n=== Loading Dataset ===")
    train_dataset = OpticalScatteringDataset(
        data_root=config['data']['train']['data_root'],
        label_file=config['data']['train']['label_file'],
        image_size=config['data']['train']['image_size']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['train']['batch_size'],
        shuffle=config['data']['train']['shuffle'],
        num_workers=config['data']['train']['num_workers'],
        generator=generator,  # 재현성을 위한 generator
        worker_init_fn=worker_init_fn  # Worker seed 설정
    )
    
    # Test 데이터셋을 validation으로 사용 (별도 validation dataset 없음)
    test_dataset = OpticalScatteringDataset(
        data_root=config['data']['test']['data_root'],
        label_file=config['data']['test']['label_file'],
        image_size=config['data']['test']['image_size']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['test']['batch_size'],
        shuffle=False,  # Validation/Test는 shuffle 안 함
        num_workers=config['data']['test']['num_workers']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples (used for validation): {len(test_dataset)}")
    
    # 모델 생성 (seed 설정 이후에 생성하여 ETF 가중치 초기화 시 재현성 보장)
    print("\n=== Building Model ===")
    # 모델 생성 전에 seed 재확인 (ETF 가중치 초기화를 위해)
    set_seed(seed)
    model = PICLModel(
        backbone_model=config['model']['backbone']['model_name'],
        pretrained_path=config['model']['backbone']['pretrained_path'],
        temporal_config={
            'input_dim': config['model']['temporal']['input_dim'],
            'device': device
        },
        physics_config=config['model']['physics'],
        num_classes=config['model']['num_classes']
    )
    model = model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # Resume
    start_epoch = 0
    best_accuracy = 0.0  # Accuracy 기준으로 best model 저장
    best_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
        print(f"Best accuracy so far: {best_accuracy:.2f}%")
    
    # 가중치 가져오기
    w_cls = config['model']['physics'].get('classification_weight', 10.0)
    w_data = config['model']['physics'].get('data_weight', 1.0)
    w_physics = config['model']['physics'].get('physics_weight', 1.0)
    
    # 학습
    print("\n" + "="*70)
    print("=== Training ===")
    print("="*70)
    for epoch in range(start_epoch, config['train']['epochs']):
        # Epoch마다 generator seed 업데이트 (재현성 보장)
        # FSCIL의 DistSamplerSeedHook과 동일한 방식: seed + epoch
        generator.manual_seed(seed + epoch)
        
        # 1. Train
        avg_loss, avg_data_loss, avg_physics_loss, avg_cls_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # 2. Evaluate (test 데이터로 평가)
        eval_loss, eval_cls_loss, eval_mse, eval_accuracy, _, _, _, _ = validate(
            model, test_loader, device
        )
        
        # 출력
        print("\n" + "-"*70)
        print(f"Epoch {epoch + 1}/{config['train']['epochs']}")
        print("-"*70)
        print(f"  [Train]")
        print(f"    Loss:      {avg_loss:.4f}")
        print(f"    Cls Loss:  {w_cls * avg_cls_loss:.4f} (weighted)")
        print(f"    Data Loss: {w_data * avg_data_loss:.4f} (weighted)")
        print(f"    Phys Loss: {w_physics * avg_physics_loss:.4f} (weighted)")
        print(f"    Accuracy:  {train_accuracy:.2f}%")
        print(f"  [Test/Validation]")
        print(f"    Loss:      {eval_loss:.4f}")
        print(f"    Cls Loss:  {eval_cls_loss:.4f}")
        print(f"    MSE:       {eval_mse:.6f}")
        print(f"    Accuracy:  {eval_accuracy:.2f}%")
        print(f"  [Learning Rate]")
        print(f"    LR:        {optimizer.param_groups[0]['lr']:.6f}")
        
        # Scheduler step
        # ReduceLROnPlateau는 eval loss를 인자로 받음
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(eval_loss)  # eval loss 기반
        else:
            scheduler.step()  # epoch 기반
        
        # 3. Checkpoint 저장
        # 3.1. Latest model 저장 (매 epoch마다)
        latest_checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_accuracy': best_accuracy,
            'best_loss': best_loss,
            'eval_accuracy': eval_accuracy,
            'eval_loss': eval_loss,
            'train_accuracy': train_accuracy,
            'train_loss': avg_loss
        }
        torch.save(latest_checkpoint, os.path.join(work_dir, 'latest.pth'))
        
        # 3.2. Best model 저장 (Accuracy 기준, 갱신될 때만)
        is_best = eval_accuracy > best_accuracy
        if is_best:
            best_accuracy = eval_accuracy
            best_loss = eval_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'best_loss': best_loss,
                'eval_accuracy': eval_accuracy,
                'eval_loss': eval_loss,
                'train_accuracy': train_accuracy,
                'train_loss': avg_loss
            }
            torch.save(checkpoint, os.path.join(work_dir, 'best.pth'))
            print(f"  [Best Model]")
            print(f"    ✓ Saved! (Accuracy: {best_accuracy:.2f}%)")
        else:
            print(f"  [Best Model]")
            print(f"    Best: {best_accuracy:.2f}% | Current: {eval_accuracy:.2f}%")
        
        # 3.3. Last model 저장 (마지막 epoch)
        if epoch + 1 == config['train']['epochs']:
            torch.save(latest_checkpoint, os.path.join(work_dir, 'last.pth'))
            print(f"  [Last Model]")
            print(f"    ✓ Saved! (Final epoch: {epoch + 1})")
        
        print("-"*70)
    
    print("\n=== Training Complete ===")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"\nSaved checkpoints:")
    print(f"  - best.pth:  Best model (Accuracy: {best_accuracy:.2f}%)")
    print(f"  - latest.pth: Latest model (Epoch {config['train']['epochs']})")
    print(f"  - last.pth:  Final model (Epoch {config['train']['epochs']})")
    print(f"\nAll saved in: {work_dir}")


if __name__ == '__main__':
    main()

