"""
PICL Inference Script
학습된 모델로 물리 계수 예측 및 피처맵 추출
"""

import os
import argparse
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
            'device': device
        },
        physics_config=config['model']['physics'],
        num_classes=config['model']['num_classes']
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


def predict_material(model, images, device):
    """재료 분류 및 물리 계수 예측"""
    with torch.no_grad():
        images = images.to(device)
        class_logits, class_pred, physical_coeffs = model.predict_material(images)
    
    # Softmax for probabilities
    import torch.nn.functional as F
    class_probs = F.softmax(class_logits, dim=1)
    
    return {
        'class_pred': class_pred.cpu().numpy(),
        'class_probs': class_probs.cpu().numpy(),
        'n': physical_coeffs[:, 0].cpu().numpy(),
        'mu_a': physical_coeffs[:, 1].cpu().numpy(),
        'mu_s_prime': physical_coeffs[:, 2].cpu().numpy()
    }


def inference_on_dataset(model, dataset, device, save_dir):
    """데이터셋 전체에 대해 추론"""
    results = []
    
    print("\n=== Running Inference ===")
    for idx in tqdm(range(len(dataset))):
        images, n_true, mu_a_true, mu_s_true, g_true, tissue_label = dataset[idx]
        images = images.unsqueeze(0)  # Add batch dimension
        
        # 재료 분류 및 물리 계수 예측
        preds = predict_material(model, images, device)
        
        # 피처맵 추출
        features = extract_features(model, images, device)
        
        # Tissue name mapping
        tissue_label_idx = int(tissue_label.numpy())
        tissue_name = OpticalScatteringDataset.IDX_TO_TISSUE.get(tissue_label_idx, 'unknown')
        pred_tissue_idx = int(preds['class_pred'][0])
        pred_tissue_name = OpticalScatteringDataset.IDX_TO_TISSUE.get(pred_tissue_idx, 'unknown')
        
        result = {
            'idx': idx,
            'tissue_true': tissue_name,
            'tissue_pred': pred_tissue_name,
            'class_probs': preds['class_probs'][0].tolist(),
            'n_true': float(n_true.numpy()),
            'n_pred': float(preds['n'][0]),
            'mu_a_pred': float(preds['mu_a'][0]),
            'mu_s_prime_pred': float(preds['mu_s_prime'][0]),
            'feature_shape': list(features.shape)
        }
        results.append(result)
    
    # 통계 계산
    n_true_list = [r['n_true'] for r in results]
    n_pred_list = [r['n_pred'] for r in results]
    
    mse = np.mean([(t - p)**2 for t, p in zip(n_true_list, n_pred_list)])
    mae = np.mean([abs(t - p) for t, p in zip(n_true_list, n_pred_list)])
    
    # Classification accuracy
    correct = sum(1 for r in results if r['tissue_true'] == r['tissue_pred'])
    accuracy = 100.0 * correct / len(results)
    
    print(f"\n=== Results ===")
    print(f"Total samples: {len(results)}")
    print(f"\n📊 Classification:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"\n📐 Regression (Refractive Index):")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # 클래스별 통계
    print(f"\n=== Tissue-wise Results ===")
    tissues = sorted(set([r['tissue_true'] for r in results]))
    
    # Header
    print(f"{'Tissue':<20} {'Count':<7} {'Accuracy':<10} {'n_true':<8} {'n_pred':<15} {'MSE':<10} {'MAE':<10}")
    print("-" * 100)
    
    for tissue in tissues:
        tissue_results = [r for r in results if r['tissue_true'] == tissue]
        tissue_correct = sum(1 for r in tissue_results if r['tissue_true'] == r['tissue_pred'])
        tissue_accuracy = 100.0 * tissue_correct / len(tissue_results)
        
        tissue_n_true = [r['n_true'] for r in tissue_results]
        tissue_n_pred = [r['n_pred'] for r in tissue_results]
        tissue_mse = np.mean([(t - p)**2 for t, p in zip(tissue_n_true, tissue_n_pred)])
        tissue_mae = np.mean([abs(t - p) for t, p in zip(tissue_n_true, tissue_n_pred)])
        
        print(f"{tissue:<20} {len(tissue_results):<7} {tissue_accuracy:>6.1f}%    "
              f"{tissue_n_true[0]:.2f}     {np.mean(tissue_n_pred):.4f}±{np.std(tissue_n_pred):.4f}  "
              f"{tissue_mse:.6f}  {tissue_mae:.6f}")
    
    # Confusion Matrix
    print(f"\n=== Confusion Matrix ===")
    from collections import defaultdict
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r['tissue_true']][r['tissue_pred']] += 1
    
    # Print header
    header_label = "True \\ Pred"
    print(f"{header_label:<20}", end="")
    for tissue in tissues:
        print(f"{tissue:<20}", end="")
    print()
    print("-" * (20 + 20 * len(tissues)))
    
    # Print rows
    for true_tissue in tissues:
        print(f"{true_tissue:<20}", end="")
        for pred_tissue in tissues:
            count = confusion[true_tissue][pred_tissue]
            print(f"{count:<20}", end="")
        print()
    
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
    
    # 재료 분류 및 물리 계수 예측
    preds = predict_material(model, images, device)
    
    # 피처맵 추출
    features = extract_features(model, images, device)
    
    # Tissue name
    pred_tissue_idx = int(preds['class_pred'][0])
    pred_tissue_name = OpticalScatteringDataset.IDX_TO_TISSUE.get(pred_tissue_idx, 'unknown')
    
    print(f"\n📊 Classification:")
    print(f"  Predicted Tissue: {pred_tissue_name} (class {pred_tissue_idx})")
    print(f"  Class Probabilities:")
    for idx, prob in enumerate(preds['class_probs'][0]):
        tissue_name = OpticalScatteringDataset.IDX_TO_TISSUE.get(idx, 'unknown')
        print(f"    {tissue_name:<20}: {prob:.4f} ({prob*100:.1f}%)")
    
    print(f"\n📐 Physical Coefficients:")
    print(f"  Refractive index (n):            {preds['n'][0]:.4f}")
    print(f"  Absorption coefficient (μa):     {preds['mu_a'][0]:.4f}")
    print(f"  Reduced scattering coeff (μs'):  {preds['mu_s_prime'][0]:.4f}")
    
    print(f"\n🗺️ Feature Map:")
    print(f"  Shape: {features.shape}")
    print(f"  (Batch, Time, Channels, Height, Width)")
    
    return preds, features


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

