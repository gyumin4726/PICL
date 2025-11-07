#!/usr/bin/env python3
"""
PICL Inference Script
학습된 모델로 물리 계수 예측 및 피처맵 추출
"""

import os
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

from picl_model import PICLModel
from train_picl import load_config, OpticalScatteringDataset


def load_model(config, checkpoint_path, device):
    """학습된 모델 로드"""
    model = PICLModel(
        backbone_model=config['model']['backbone']['model_name'],
        pretrained_path=None,  # Checkpoint에서 로드할 것이므로 None
        temporal_config={
            'input_dim': config['model']['temporal']['input_dim'],
            'd_model': config['model']['temporal']['d_model'],
            'device': device
        },
        physics_config=config['model']['physics']
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Loss: {checkpoint['best_loss']:.4f}")
    
    return model


def extract_features(model, images, device):
    """피처맵 추출"""
    with torch.no_grad():
        images = images.to(device)
        B, T, C, H, W = images.shape
        
        # VMamba backbone으로 공간 특징 추출
        x_flat = images.view(B * T, C, H, W)
        features = model.backbone(x_flat)
        
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, (tuple, list)):
            features = features[0]
        
        # (B*T, C, H', W') -> (B, T, C, H', W')
        _, C_feat, H_feat, W_feat = features.shape
        features = features.view(B, T, C_feat, H_feat, W_feat)
        
        return features


def predict_coefficients(model, images, device):
    """물리 계수 예측"""
    with torch.no_grad():
        images = images.to(device)
        n_pred, mu_a_pred, mu_s_prime_pred = model.predict_coefficients(images)
    
    return {
        'n': n_pred.cpu().numpy(),
        'mu_a': mu_a_pred.cpu().numpy(),
        'mu_s_prime': mu_s_prime_pred.cpu().numpy()
    }


def inference_on_dataset(model, dataset, device, save_dir):
    """데이터셋 전체에 대해 추론"""
    results = []
    
    print("\n=== Running Inference ===")
    for idx in tqdm(range(len(dataset))):
        images, n_true, class_name = dataset[idx]
        images = images.unsqueeze(0)  # Add batch dimension
        
        # 물리 계수 예측
        coeffs = predict_coefficients(model, images, device)
        
        # 피처맵 추출
        features = extract_features(model, images, device)
        
        result = {
            'idx': idx,
            'class': class_name,
            'n_true': float(n_true.numpy()),
            'n_pred': float(coeffs['n'][0]),
            'mu_a_pred': float(coeffs['mu_a'][0]),
            'mu_s_prime_pred': float(coeffs['mu_s_prime'][0]),
            'feature_shape': list(features.shape)
        }
        results.append(result)
    
    # 통계 계산
    n_true_list = [r['n_true'] for r in results]
    n_pred_list = [r['n_pred'] for r in results]
    
    mse = np.mean([(t - p)**2 for t, p in zip(n_true_list, n_pred_list)])
    mae = np.mean([abs(t - p) for t, p in zip(n_true_list, n_pred_list)])
    
    print(f"\n=== Results ===")
    print(f"Total samples: {len(results)}")
    print(f"MSE (n): {mse:.6f}")
    print(f"MAE (n): {mae:.6f}")
    
    # 클래스별 통계
    print(f"\n=== Class-wise Results ===")
    classes = set([r['class'] for r in results])
    for cls in sorted(classes):
        cls_results = [r for r in results if r['class'] == cls]
        cls_n_true = [r['n_true'] for r in cls_results]
        cls_n_pred = [r['n_pred'] for r in cls_results]
        cls_mse = np.mean([(t - p)**2 for t, p in zip(cls_n_true, cls_n_pred)])
        cls_mae = np.mean([abs(t - p) for t, p in zip(cls_n_true, cls_n_pred)])
        
        print(f"{cls:12s}: n_true={cls_n_true[0]:.2f}, "
              f"n_pred_avg={np.mean(cls_n_pred):.4f}±{np.std(cls_n_pred):.4f}, "
              f"MSE={cls_mse:.6f}, MAE={cls_mae:.6f}")
    
    # 결과 저장
    output_file = os.path.join(save_dir, 'inference_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return results


def inference_single_sample(model, image_paths, device):
    """단일 샘플 추론 (5개 이미지 경로)"""
    print(f"\n=== Single Sample Inference ===")
    
    # 이미지 로드
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        images.append(img)
    
    images = np.stack(images, axis=0)  # (5, H, W, 3)
    images = torch.from_numpy(images).float()
    images = images.permute(0, 3, 1, 2)  # (5, 3, H, W)
    images = images.unsqueeze(0)  # (1, 5, 3, H, W)
    
    # 물리 계수 예측
    coeffs = predict_coefficients(model, images, device)
    
    # 피처맵 추출
    features = extract_features(model, images, device)
    
    print(f"Predicted coefficients:")
    print(f"  Refractive index (n):            {coeffs['n'][0]:.4f}")
    print(f"  Absorption coefficient (μa):     {coeffs['mu_a'][0]:.4f}")
    print(f"  Reduced scattering coeff (μs'):  {coeffs['mu_s_prime'][0]:.4f}")
    print(f"\nFeature map shape: {features.shape}")
    print(f"  (Batch, Time, Channels, Height, Width)")
    
    return coeffs, features


def main():
    parser = argparse.ArgumentParser(description='PICL Inference')
    parser.add_argument('config', type=str, help='Config file path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('--mode', type=str, default='dataset', choices=['dataset', 'single'],
                       help='Inference mode: dataset or single sample')
    parser.add_argument('--images', nargs='+', type=str, default=None,
                       help='Image paths for single mode (5 images)')
    parser.add_argument('--output-dir', type=str, default='./inference_output',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config 로드
    config = load_config(args.config)
    
    # 모델 로드
    model = load_model(config, args.checkpoint, device)
    
    if args.mode == 'dataset':
        # 데이터셋 전체에 대해 추론
        dataset = OpticalScatteringDataset(
            data_root=config['data']['test']['data_root'],
            label_file=config['data']['test']['label_file'],
            image_size=config['data']['test']['image_size']
        )
        
        results = inference_on_dataset(model, dataset, device, args.output_dir)
        
    elif args.mode == 'single':
        # 단일 샘플 추론
        if args.images is None or len(args.images) != 5:
            raise ValueError("Single mode requires exactly 5 image paths (--images)")
        
        coeffs, features = inference_single_sample(model, args.images, device)
        
        # 피처맵 저장 (numpy)
        features_np = features.cpu().numpy()
        output_file = os.path.join(args.output_dir, 'features.npy')
        np.save(output_file, features_np)
        print(f"\n✓ Features saved to {output_file}")


if __name__ == '__main__':
    main()

