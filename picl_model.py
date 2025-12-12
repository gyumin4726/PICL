"""
Complete PICL Model: Integrated Physics-Informed Neural Network
Combines 2D VMamba + 1D Mamba + Physics Constraints

This module implements the complete PICL pipeline that simultaneously:
1. Predicts physical coefficients (n, μa, μs') using neural networks
2. Computes physical quantities (Φ, ∂Φ/∂t, ∇²Φ) from original images
3. Validates physics constraints using PDE equations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

# Import our modules
from vmamba_backbone import VMambaBackbone
from mamba_1d_temporal import SequenceToValue
from pde_physics import PINNPhysicsLoss


def generate_random_orthogonal_matrix(feat_in: int, num_classes: int) -> torch.Tensor:
    """
    Generate a random orthogonal matrix using QR decomposition.
    
    This function is adopted from FSCIL's ETFHead implementation.
    
    Args:
        feat_in (int): Input feature dimension
        num_classes (int): Number of classes (output dimension)
        
    Returns:
        torch.Tensor: Orthogonal matrix of shape (feat_in, num_classes)
    """
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)  # QR decomposition
    orth_vec = torch.tensor(orth_vec).float()
    
    # Verify orthogonality
    assert torch.allclose(
        torch.matmul(orth_vec.T, orth_vec), 
        torch.eye(num_classes), 
        atol=1.e-7
    ), "The max irregular value is : {}".format(
        torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes)))
    )
    
    return orth_vec


class DRLoss(nn.Module):
    """
    Dot-Product Regression Loss (DRLoss) for ETF-based classification.
    
    This loss function is adopted from FSCIL's ETFHead implementation.
    It measures the squared difference between the dot product of features and targets,
    normalized by the target norm squared.
    
    Args:
        reduction (str): Reduction method ('mean' or 'sum'). Default: 'mean'
        loss_weight (float): Weight for the loss. Default: 1.0
        reg_lambda (float): Regularization lambda. Default: 0.0
    """
    
    def __init__(self, reduction='mean', loss_weight=1.0, reg_lambda=0.):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
        self,
        feat,
        target,
        h_norm2=None,
        m_norm2=None,
        avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2))**2) / h_norm2)

        return loss * self.loss_weight


class CombinedLoss(nn.Module):
    """
    Combined loss of DRLoss and CrossEntropyLoss.
    
    This loss function combines the geometric DRLoss with standard CrossEntropyLoss,
    allowing flexible weighting between the two losses.
    
    Args:
        dr_weight (float): Weight for DRLoss. Default: 1.0
        ce_weight (float): Weight for CrossEntropyLoss. Default: 1.0
    """
    
    def __init__(self, dr_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.dr_loss = DRLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dr_weight = dr_weight
        self.ce_weight = ce_weight

    def forward(self, feat, target, labels=None, etf_vec=None, **kwargs):
        """
        Forward pass for combined loss.
        
        Args:
            feat (torch.Tensor): Feature vectors (B, D) - normalized features
            target (torch.Tensor): Target ETF vectors (B, D) - ETF class vectors
            labels (torch.Tensor, optional): Class labels (B,) for CrossEntropyLoss
            etf_vec (torch.Tensor, optional): ETF weight matrix (D, C) for logits computation
            
        Returns:
            torch.Tensor: Combined loss value
        """
        dr_loss = self.dr_loss(feat, target)
        
        # Cross Entropy Loss 계산
        if etf_vec is not None:
            # ETF 벡터를 사용하여 로짓 계산
            logits = torch.matmul(feat, etf_vec)  # (batch_size, num_classes)
            if labels is None:
                # 레이블이 제공되지 않은 경우, target에서 가장 큰 값의 인덱스를 사용
                labels = torch.argmax(target, dim=1)
            ce_loss = self.ce_loss(logits, labels)
        else:
            # ETF 벡터가 없는 경우 CE Loss를 0으로 설정
            ce_loss = torch.tensor(0.0, device=feat.device)
        
        total_loss = self.dr_weight * dr_loss + self.ce_weight * ce_loss
        
        # Add individual losses to kwargs for logging (FSCIL과 동일)
        if 'log_vars' in kwargs:
            kwargs['log_vars'].update({
                'dr_loss': dr_loss.item(),
                'ce_loss': ce_loss.item()
            })
        
        return total_loss


class MLPRegressor(nn.Module):
    """
    MLP Regressor for Physical Coefficients Prediction with Separate Heads.
    
    Uses multi-task learning architecture:
    - Shared backbone: Learns common features from temporal representations
    - Separate heads: Independently predict each physical coefficient
    
    This design avoids gradient conflicts between coefficients with different:
    - Value scales: n ∈ [1.33, 1.50], μₐ ∈ [0.0005, 0.50], μₛ ∈ [0.005, 50.0], g ∈ [0.89, 0.95]
    - All coefficients have Data Loss supervision
    
    Architecture:
        Input (B, 1024)
            ↓
        Shared MLP: 1024 → 256 (common features)
            ↓
            ├─→ Head_n:  256 → 1 (refractive index)
            ├─→ Head_μₐ: 256 → 1 (absorption coefficient)
            ├─→ Head_μₛ: 256 → 1 (scattering coefficient)
            └─→ Head_g: 256 → 1 (anisotropy factor)
    
    Args:
        input_dim (int): Input feature dimension (default: 1024)
        hidden_dim (int): Shared backbone output dimension (default: 256)
        dropout (float): Dropout rate for regularization (default: 0.5)
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        
        # Shared backbone for common feature extraction
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for each physical coefficient
        self.head_n = nn.Linear(hidden_dim, 1)      # Refractive index
        self.head_mu_a = nn.Linear(hidden_dim, 1)   # Absorption coefficient
        self.head_mu_s = nn.Linear(hidden_dim, 1)   # Scattering coefficient
        self.head_g = nn.Linear(hidden_dim, 1)      # Anisotropy factor
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for physical coefficients prediction.
        
        Process:
        1. Extract common features via shared backbone
        2. Independently predict each coefficient via separate heads
        
        Args:
            x (torch.Tensor): Input features (B, input_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                - n_pred: (B,) Refractive index [1.33 ~ 1.50]
                - mu_a_pred: (B,) Absorption coefficient [0.0005 ~ 0.50]
                - mu_s_pred: (B,) Scattering coefficient [0.005 ~ 50.0]
                - g_pred: (B,) Anisotropy factor [0.89 ~ 0.95]
        """
        # Shared feature extraction
        shared_features = self.shared_backbone(x)  # (B, hidden_dim)
        
        # Independent predictions for each coefficient
        n_pred = self.head_n(shared_features).squeeze(-1)      # (B,)
        mu_a_pred = self.head_mu_a(shared_features).squeeze(-1)  # (B,)
        mu_s_pred = self.head_mu_s(shared_features).squeeze(-1)  # (B,)
        g_pred = self.head_g(shared_features).squeeze(-1)        # (B,)
        
        return n_pred, mu_a_pred, mu_s_pred, g_pred


class ETFClassifier(nn.Module):
    """
    Equiangular Tight Frame (ETF) Classifier for Material Classification.
    
    ETF Classifier는 고정된 기하학적 구조를 가진 classifier로,
    모든 클래스 간의 각도가 동일하여 few-shot learning에 유리합니다.
    
    This implementation follows FSCIL's ETFHead with proper ETF construction:
    W_ETF = √(C/(C-1)) × orth_vec × (I - 1/C × 11^T)
    
    Architecture:
    - Input: (B, input_dim) - Temporal features from 1D Mamba (default: 1024)
    - Pre-logits: L2 normalization (no dimension change)
    - ETF Head: Fixed orthogonal weights (non-learnable, bias-free)
    - Output: (B, num_classes) - Class logits
    
    Materials:
        0: air
        1: water
        2: acrylic
        3: glass
        4: sapphire
    
    Key Features:
    - Equiangular geometry: All class pairs have equal angular separation
    - QR decomposition: Ensures proper orthogonality
    - Scale factor: √(C/(C-1)) for correct simplex geometry
    - Few-shot friendly: Strong geometric prior reduces overfitting
    - FSCIL-compatible: Identical implementation to FSCIL's ETFHead
    
    Args:
        num_classes (int): Number of material classes (default: 5)
        input_dim (int): Input feature dimension (default: 1024)
    """
    
    def __init__(self, num_classes: int = 5, input_dim: int = 1024):
        """
        ETF Classifier (FSCIL 방식과 동일)
        
        Args:
            num_classes (int): Number of classes
            input_dim (int): Input feature dimension (FSCIL과 동일하게 차원 변화 없음)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # FSCIL과 동일: feature_extractor 없이 직접 사용
        # ETF Classifier Head (fixed, not learnable)
        # FSCIL과 완전히 동일: register_buffer 사용
        etf_vec = self._create_etf_weights(num_classes, input_dim)
        self.register_buffer('etf_vec', etf_vec)
        # etf_vec: (input_dim, num_classes) = (1024, 5) - FSCIL과 동일한 형태
        
    def _create_etf_weights(self, num_classes: int, feature_dim: int) -> nn.Parameter:
        """
        Create Equiangular Tight Frame (ETF) weights using FSCIL method.
        
        FSCIL ETF 구조:
        W_ETF = √(C/(C-1)) × orth_vec × (I - 1/C × 11^T)
        
        여기서:
        - C: 클래스 수
        - I: Identity matrix (C × C)
        - 1: ones matrix (C × C)
        - orth_vec: Random orthogonal matrix (D × C) from QR decomposition
        - 모든 클래스 쌍의 내적이 -1/(C-1)로 동일
        
        This implementation follows FSCIL's ETFHead for mathematical correctness.
        
        Args:
            num_classes (int): Number of classes (C)
            feature_dim (int): Feature dimension (D)
            
        Returns:
            nn.Parameter: Fixed ETF weights (feature_dim, num_classes)
        """
        # 1. Generate random orthogonal matrix using QR decomposition
        orth_vec = generate_random_orthogonal_matrix(feature_dim, num_classes)
        # orth_vec shape: (feature_dim, num_classes) = (D, C)
        
        # 2. Create ETF transformation matrix (FSCIL과 완전히 동일한 표현)
        i_nc_nc = torch.eye(num_classes)  # Identity (C × C)
        one_nc_nc = torch.mul(
            torch.ones(num_classes, num_classes),
            (1 / num_classes))  # FSCIL과 동일한 표현
        
        # 3. Apply ETF formula (FSCIL과 완전히 동일한 표현)
        # Scale factor: √(C/(C-1)) - CORRECT formula from FSCIL
        # ETF = √(C/(C-1)) × orth_vec × (I - 1/C × 11^T)
        # Shape: (D, C) @ (C, C) = (D, C)
        etf_vec = torch.mul(
            torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
            math.sqrt(num_classes / (num_classes - 1)))  # FSCIL과 동일한 표현
        
        # 4. Return etf_vec tensor (FSCIL과 완전히 동일)
        # register_buffer는 __init__에서 직접 호출해야 하므로,
        # 여기서는 tensor만 반환하고 __init__에서 register_buffer로 등록
        return etf_vec  # (D, C) - torch.Tensor
    
    def pre_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-logits processing (FSCIL과 동일)
        Normalize만 수행, 차원 변화 없음
        """
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x
    
    def forward(self, input_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for material classification using ETF geometry (FSCIL 방식).
        
        Process:
        1. Pre-logits: Normalize features (차원 변화 없음)
        2. Compute logits: features @ etf_vec
        3. Predict class: argmax over logits
        
        Args:
            input_features (torch.Tensor): Input features (B, input_dim) = (B, 1024)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - class_logits: (B, num_classes) - Class logits (not probabilities)
                - class_pred: (B,) - Predicted class indices [0-4]
        """
        # 1. Pre-logits: Normalize only (FSCIL과 동일)
        features = self.pre_logits(input_features)  # (B, input_dim), ||f|| = 1
        
        # 2. Compute logits using fixed ETF weights (FSCIL과 동일한 방식)
        # Dot product: cosine similarity with each class prototype
        class_logits = features @ self.etf_vec  # (B, 1024) @ (1024, 5) = (B, 5)
        
        # 3. Get predicted class (argmax)
        class_pred = torch.argmax(class_logits, dim=1)  # (B,)
        
        return class_logits, class_pred
    


class PICLModel(nn.Module):
    """
    Complete PICL Model: Physics-Informed Neural Network for Optical Scattering
    
    Architecture:
    1. 2D VMamba Backbone: (B, 5, 3, 224, 224) → (B, 5, 1024)
    2. 1D Mamba + SequenceToValue: (B, 5, 1024) → (B, 512)
    3. MLP Classifier: (B, 512) → (B, 3) - (n, μa, μs')
    4. Physics Validation: Original images + Predicted coefficients → PDE residual
    
    Args:
        backbone_model (str): VMamba model variant
        pretrained_path (str): Path to pretrained VMamba weights
        temporal_config (dict): Configuration for temporal modeling
        physics_config (dict): Configuration for physics constraints
    """
    
    def __init__(self, 
                 backbone_model: str = 'vmamba_base_s2l15',
                 pretrained_path: Optional[str] = None,
                 temporal_config: Optional[dict] = None,
                 physics_config: Optional[dict] = None,
                 num_classes: int = 5):
        super().__init__()
        
        # Default configurations
        if temporal_config is None:
            temporal_config = {
                'input_dim': 1024,  # VMamba 출력 차원 (정보 손실 방지)
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        
        if physics_config is None:
            physics_config = {
                'physics_weight': 1.0,
                'data_weight': 1.0,
                'classification_weight': 10.0,  # Classification is primary goal
                'c': 3e8,
                'dx': 1e-3,  # 1.0 mm = 1e-3 m (MCX 설정 기준)
                'dy': 1e-3,  # 1.0 mm = 1e-3 m
                'dt': 1e-9,  # 1.0 ns = 1e-9 s
            }
        
        # 1. 2D VMamba Backbone (Spatial Feature Extractor)
        self.backbone = VMambaBackbone(
            model_name=backbone_model,
            pretrained_path=pretrained_path,
            out_indices=(3,),  # Last stage only (1024 dim)
            frozen_stages=-1,  # All stages trainable
            channel_first=True
        )
        
        # 2. 1D Mamba + SequenceToValue (Temporal Sequence Modeler)
        self.temporal = SequenceToValue(
            input_dim=temporal_config['input_dim'],
            device=temporal_config['device']
        )
        
        # Temporal feature dimension: average pooling으로 1024 차원 유지
        temporal_feature_dim = temporal_config['input_dim']  # 1024
        
        # 3. MLP Regressor (Physical Coefficients Predictor)
        self.regressor = MLPRegressor(
            input_dim=temporal_feature_dim,  # 1024
            hidden_dim=256,
            dropout=0.5
        )
        
        # 4. ETF Classifier (Material Classification directly from temporal features)
        # FSCIL과 동일: feature_extractor 없이 직접 사용
        self.etf_classifier = ETFClassifier(
            num_classes=num_classes,  # 5 materials
            input_dim=temporal_feature_dim  # 1024 (FSCIL과 동일하게 차원 변화 없음)
        )
        
        # 5. Physics Loss Module
        self.physics_loss = PINNPhysicsLoss(
            c=physics_config['c'],
            physics_weight=physics_config['physics_weight'],
            data_weight=physics_config['data_weight']
        )
        
        # 6. Classification Loss (CombinedLoss: DRLoss + CrossEntropyLoss)
        dr_weight = physics_config.get('dr_weight', 0.0)  # DRLoss weight
        ce_weight = physics_config.get('ce_weight', 1.0)  # CrossEntropyLoss weight
        self.classification_loss = CombinedLoss(dr_weight=dr_weight, ce_weight=ce_weight)
        
        # Store configurations
        self.temporal_config = temporal_config
        self.physics_config = physics_config
        self.num_classes = num_classes
    
    def forward(self, images: torch.Tensor, n_true: torch.Tensor, 
                material_label: Optional[torch.Tensor] = None,
                mu_a_true: Optional[torch.Tensor] = None,
                mu_s_true: Optional[torch.Tensor] = None,
                g_true: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Complete PICL forward pass with simultaneous processing.
        
        Args:
            images (torch.Tensor): Original image sequence (B, 5, 3, 224, 224)
            n_true (torch.Tensor): True refractive index (B,)
            material_label (torch.Tensor, optional): True material labels (B,) [0-4]
            mu_a_true (torch.Tensor, optional): True absorption coefficient (B,)
            mu_s_true (torch.Tensor, optional): True scattering coefficient (B,)
            g_true (torch.Tensor, optional): True anisotropy factor (B,)
            
        Returns:
            Dict[str, torch.Tensor]: Complete results including predictions and losses
        """
        B, T, C, H, W = images.shape
        
        # ========================================
        # PATH 1: Neural Network Prediction
        # ========================================
        
        # 1. 2D VMamba: Extract spatial features
        x_flat = images.view(B * T, C, H, W)  # (B*T, C, H, W)
        features = self.backbone(x_flat)  # Returns tuple or list
        
        # Handle tuple/list output format
        if isinstance(features, (tuple, list)):
            features = features[0]  # Get the first element
        if isinstance(features, (tuple, list)):
            features = features[0]  # Sometimes nested, get again
        
        # Preserve spatial structure: Use adaptive pooling to reduce but not eliminate spatial info
        # Instead of (1,1), use (7,7) to keep some spatial structure
        # This allows us to model temporal relationships per spatial location
        _, C_feat, H_feat, W_feat = features.shape
        
        # Adaptive pooling to reduce spatial size but preserve structure
        # (B*T, 1024, H', W') -> (B*T, 1024, 7, 7) for manageable size
        target_size = min(7, H_feat, W_feat)  # Use smaller of 7 or actual size
        features = F.adaptive_avg_pool2d(features, (target_size, target_size))  # (B*T, 1024, 7, 7)
        
        # Reshape to (B, T, C, H, W) for temporal processing per spatial location
        features = features.view(B, T, C_feat, target_size, target_size)  # (B, T, 1024, 7, 7)
        
        # Process temporal sequence for each spatial location
        # Reshape: (B, T, C, H, W) -> (B*H*W, T, C) to process each spatial location independently
        B, T, C, H_sp, W_sp = features.shape
        features_reshaped = features.permute(0, 3, 4, 1, 2).contiguous()  # (B, H_sp, W_sp, T, C)
        features_reshaped = features_reshaped.view(B * H_sp * W_sp, T, C)  # (B*H_sp*W_sp, T, C)
        
        # 2. 1D Mamba: Process temporal sequence for each spatial location
        temporal_features_all = self.temporal(features_reshaped)  # (B*H_sp*W_sp, 5, 1024)
        
        # Average pooling over time steps: (B*H_sp*W_sp, 5, 1024) → (B*H_sp*W_sp, 1024)
        temporal_features = temporal_features_all.mean(dim=1)  # (B*H_sp*W_sp, 1024)
        
        # Reshape back and aggregate spatial information
        temporal_features = temporal_features.view(B, H_sp, W_sp, -1)  # (B, H_sp, W_sp, 1024)
        # Aggregate spatial information: average pooling
        temporal_features = temporal_features.permute(0, 3, 1, 2)  # (B, 1024, H_sp, W_sp)
        temporal_features = F.adaptive_avg_pool2d(temporal_features, (1, 1))  # (B, 1024, 1, 1)
        temporal_features = temporal_features.flatten(1)  # (B, 1024)
        
        # 3. 병렬 경로: 물리계수 예측 + ETF 분류
        # 경로 1: MLP Regressor → 4개 계수 예측
        n_pred, mu_a_pred, mu_s_pred, g_pred = self.regressor(temporal_features)
        
        # 경로 2: ETF Classifier로 분류 (temporal features에서 직접 분류)
        etf_class_logits, etf_class_pred = self.etf_classifier(temporal_features)  # (B, num_classes), (B,)
        
        # ========================================
        # PATH 2: Physics Quantities Computation
        # ========================================
        
        # Convert images to grayscale for physics computation (if needed)
        if images.shape[2] == 3:  # RGB to grayscale
            phi_sequence = torch.mean(images, dim=2)  # (B, T, H, W)
        else:
            phi_sequence = images  # Already grayscale
        
        # ========================================
        # PATH 3: Loss Computation
        # ========================================
        
        # 3.1. Data Loss (n, mu_a, mu_s, g 모두 라벨과 비교)
        # mu_s_prime = mu_s * (1 - g) 계산
        mu_s_prime_pred = mu_s_pred * (1 - g_pred)
        
        # 각 계수에 대한 data loss 계산
        data_loss_n = F.huber_loss(n_pred, n_true, delta=0.1, reduction='mean')
        data_loss_mu_a = torch.tensor(0.0, device=images.device)
        data_loss_mu_s = torch.tensor(0.0, device=images.device)
        data_loss_g = torch.tensor(0.0, device=images.device)
        
        if mu_a_true is not None:
            data_loss_mu_a = F.huber_loss(mu_a_pred, mu_a_true, delta=0.1, reduction='mean')
        if mu_s_true is not None:
            data_loss_mu_s = F.huber_loss(mu_s_pred, mu_s_true, delta=0.1, reduction='mean')
        if g_true is not None:
            data_loss_g = F.huber_loss(g_pred, g_true, delta=0.1, reduction='mean')
        
        # Total data loss (모든 계수의 평균)
        data_loss = (data_loss_n + data_loss_mu_a + data_loss_mu_s + data_loss_g) / 4.0
        
        # 3.2. Physics Loss (PDE residual)
        # 물리적 단위 사용 (MCX 설정: 픽셀당 1.0mm, 시간 게이트 간격 1ns)
        dx = self.physics_config.get('dx', 1e-3)  # 1.0 mm = 1e-3 m
        dy = self.physics_config.get('dy', 1e-3)  # 1.0 mm = 1e-3 m
        dt = self.physics_config.get('dt', 1e-9)  # 1.0 ns = 1e-9 s
        
        # Physics loss는 mu_s_prime 사용 (mu_s * (1 - g))
        physics_loss = self.physics_loss.pde.physics_loss(
            phi_sequence, n_pred, mu_a_pred, mu_s_prime_pred, 
            source=None, dx=dx, dy=dy, dt=dt
        )
        
        loss_dict = {
            'data_loss': data_loss,
            'data_loss_n': data_loss_n,
            'data_loss_mu_a': data_loss_mu_a,
            'data_loss_mu_s': data_loss_mu_s,
            'data_loss_g': data_loss_g,
            'physics_loss': physics_loss
        }
        
        # 3.2. Classification Loss (CombinedLoss: DRLoss + CrossEntropyLoss)
        # FSCIL과 동일한 방식
        etf_classification_loss = torch.tensor(0.0, device=images.device)
        if material_label is not None:
            # FSCIL 방식과 동일
            # 1. Pre-logits: Normalize only (차원 변화 없음)
            x = self.etf_classifier.pre_logits(temporal_features)  # (B, 1024)
            
            # 2. ETF target 벡터 선택 (FSCIL과 동일한 방식)
            # FSCIL: target = self.etf_vec[:, gt_label].t()
            # etf_vec shape: (D, C) = (1024, 5)
            # etf_vec[:, material_label] → (1024, B) → .t() → (B, 1024)
            etf_target = self.etf_classifier.etf_vec[:, material_label].t()  # (B, 1024)
            
            # 3. CombinedLoss 계산 (FSCIL과 동일)
            etf_vec = self.etf_classifier.etf_vec  # (1024, 5)
            etf_classification_loss = self.classification_loss(
                feat=x,
                target=etf_target,
                labels=material_label,
                etf_vec=etf_vec
            )
        
        # 3.3. Total Loss
        w_cls = self.physics_config.get('classification_weight', 10.0)
        w_data = self.physics_config.get('data_weight', 1.0)
        w_physics = self.physics_config.get('physics_weight', 1.0)
        
        total_loss = (w_cls * etf_classification_loss + 
                     w_data * loss_dict['data_loss'] + 
                     w_physics * loss_dict['physics_loss'])
        
        # ========================================
        # RETURN COMPLETE RESULTS
        # ========================================
        
        results = {
            # Predictions - Physical Coefficients
            'n_pred': n_pred,
            'mu_a_pred': mu_a_pred,
            'mu_s_pred': mu_s_pred,
            'g_pred': g_pred,
            'mu_s_prime_pred': mu_s_prime_pred,  # Computed: mu_s * (1 - g)
            
            # Predictions - ETF Classification (ETF 분류)
            'etf_class_logits': etf_class_logits,
            'etf_class_pred': etf_class_pred,
            
            # Features
            'temporal_features': temporal_features,
            
            # Losses
            'total_loss': total_loss,
            'classification_loss': etf_classification_loss,  # ETF 분류 loss
            'data_loss': loss_dict['data_loss'],
            'physics_loss': loss_dict['physics_loss'],
            
            # Additional info
            'phi_sequence': phi_sequence,  # For debugging/visualization
        }
        
        return results
    
    def predict_coefficients(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict physical coefficients without computing physics loss.
        
        Args:
            images (torch.Tensor): Image sequence (B, 5, 3, 224, 224)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (n_pred, mu_a_pred, mu_s_prime_pred)
        """
        with torch.no_grad():
            B, T, C, H, W = images.shape
            
            # 1. 2D VMamba
            x_flat = images.view(B * T, C, H, W)
            features = self.backbone(x_flat)
            if isinstance(features, (tuple, list)):
                features = features[0]
            if isinstance(features, (tuple, list)):
                features = features[0]
            
            # Preserve spatial structure: Use adaptive pooling to reduce but not eliminate spatial info
            _, C_feat, H_feat, W_feat = features.shape
            
            # Adaptive pooling to reduce spatial size but preserve structure
            target_size = min(7, H_feat, W_feat)  # Use smaller of 7 or actual size
            features = F.adaptive_avg_pool2d(features, (target_size, target_size))  # (B*T, 1024, 7, 7)
            
            # Reshape to (B, T, C, H, W) for temporal processing per spatial location
            features = features.view(B, T, C_feat, target_size, target_size)  # (B, T, 1024, 7, 7)
            
            # Process temporal sequence for each spatial location
            B, T, C, H_sp, W_sp = features.shape
            features_reshaped = features.permute(0, 3, 4, 1, 2).contiguous()  # (B, H_sp, W_sp, T, C)
            features_reshaped = features_reshaped.view(B * H_sp * W_sp, T, C)  # (B*H_sp*W_sp, T, C)
            
            # 2. 1D Mamba: Process temporal sequence for each spatial location
            temporal_features_all = self.temporal(features_reshaped)  # (B*H_sp*W_sp, 5, 1024)
            
            # Average pooling over time steps: (B*H_sp*W_sp, 5, 1024) → (B*H_sp*W_sp, 1024)
            temporal_features = temporal_features_all.mean(dim=1)  # (B*H_sp*W_sp, 1024)
            
            # Reshape back and aggregate spatial information
            temporal_features = temporal_features.view(B, H_sp, W_sp, -1)  # (B, H_sp, W_sp, 1024)
            temporal_features = temporal_features.permute(0, 3, 1, 2)  # (B, 1024, H_sp, W_sp)
            temporal_features = F.adaptive_avg_pool2d(temporal_features, (1, 1))  # (B, 1024, 1, 1)
            temporal_features = temporal_features.flatten(1)  # (B, 1024)
            
            # 3. MLP Regressor
            n_pred, mu_a_pred, mu_s_pred, g_pred = self.regressor(temporal_features)
            mu_s_prime_pred = mu_s_pred * (1 - g_pred)  # mu_s' = mu_s * (1 - g)
            
            return n_pred, mu_a_pred, mu_s_prime_pred
    
    def predict_material(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict material classification and physical coefficients.
        
        Args:
            images (torch.Tensor): Image sequence (B, 5, 3, 224, 224)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - class_logits: (B, num_classes) - ETF classification logits
                - class_pred: (B,) - Predicted class indices
                - physical_coeffs: (B, 3) - [n, μₐ, μₛ']
        """
        with torch.no_grad():
            B, T, C, H, W = images.shape
            
            # 1. 2D VMamba
            x_flat = images.view(B * T, C, H, W)
            features = self.backbone(x_flat)
            if isinstance(features, (tuple, list)):
                features = features[0]
            if isinstance(features, (tuple, list)):
                features = features[0]
            
            # Preserve spatial structure
            _, C_feat, H_feat, W_feat = features.shape
            target_size = min(7, H_feat, W_feat)
            features = F.adaptive_avg_pool2d(features, (target_size, target_size))
            
            # Reshape for temporal processing
            features = features.view(B, T, C_feat, target_size, target_size)
            B, T, C, H_sp, W_sp = features.shape
            features_reshaped = features.permute(0, 3, 4, 1, 2).contiguous()
            features_reshaped = features_reshaped.view(B * H_sp * W_sp, T, C)
            
            # 2. 1D Mamba: Process temporal sequence for each spatial location
            temporal_features_all = self.temporal(features_reshaped)  # (B*H_sp*W_sp, 5, 1024)
            
            # Average pooling over time steps: (B*H_sp*W_sp, 5, 1024) → (B*H_sp*W_sp, 1024)
            temporal_features = temporal_features_all.mean(dim=1)  # (B*H_sp*W_sp, 1024)
            
            # Reshape back and aggregate spatial information
            temporal_features = temporal_features.view(B, H_sp, W_sp, -1)  # (B, H_sp, W_sp, 1024)
            # Aggregate spatial information: average pooling
            temporal_features = temporal_features.permute(0, 3, 1, 2)  # (B, 1024, H_sp, W_sp)
            temporal_features = F.adaptive_avg_pool2d(temporal_features, (1, 1))  # (B, 1024, 1, 1)
            temporal_features = temporal_features.flatten(1)  # (B, 1024)
            
            # 3. Get physical coefficients
            n_pred, mu_a_pred, mu_s_pred, g_pred = self.regressor(temporal_features)
            mu_s_prime_pred = mu_s_pred * (1 - g_pred)  # mu_s' = mu_s * (1 - g)
            physical_coeffs = torch.stack([n_pred, mu_a_pred, mu_s_prime_pred], dim=1)  # (B, 3)
            
            # 4. ETF Classifier로 분류
            class_logits, class_pred = self.etf_classifier(temporal_features)  # (B, num_classes), (B,)
            
            return class_logits, class_pred, physical_coeffs


