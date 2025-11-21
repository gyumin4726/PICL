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


class MLPRegressor(nn.Module):
    """
    MLP Regressor for Physical Coefficients Prediction with Separate Heads.
    
    Uses multi-task learning architecture:
    - Shared backbone: Learns common features from temporal representations
    - Separate heads: Independently predict each physical coefficient
    
    This design avoids gradient conflicts between coefficients with different:
    - Value scales: n ∈ [1.0, 1.8], μₐ ∈ [0.001, 10], μₛ' ∈ [0.1, 10]
    - Supervision strength: n has Data Loss, μₐ/μₛ' only have Physics Loss
    
    Architecture:
        Input (B, 512)
            ↓
        Shared MLP: 512 → 256 (common features)
            ↓
            ├─→ Head_n:  256 → 1 (refractive index)
            ├─→ Head_μₐ: 256 → 1 (absorption coefficient)
            └─→ Head_μₛ': 256 → 1 (reduced scattering coefficient)
    
    Args:
        input_dim (int): Input feature dimension (default: 512)
        hidden_dim (int): Shared backbone output dimension (default: 256)
        dropout (float): Dropout rate for regularization (default: 0.5)
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.5):
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
        self.head_n = nn.Linear(hidden_dim, 1)           # Refractive index
        self.head_mu_a = nn.Linear(hidden_dim, 1)        # Absorption coefficient
        self.head_mu_s_prime = nn.Linear(hidden_dim, 1)  # Reduced scattering coefficient
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for physical coefficients prediction.
        
        Process:
        1. Extract common features via shared backbone
        2. Independently predict each coefficient via separate heads
        
        Args:
            x (torch.Tensor): Input features (B, input_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - n_pred: (B,) Refractive index [1.0 ~ 1.8]
                - mu_a_pred: (B,) Absorption coefficient [0.001 ~ 10]
                - mu_s_prime_pred: (B,) Reduced scattering coefficient [0.1 ~ 10]
        """
        # Shared feature extraction
        shared_features = self.shared_backbone(x)  # (B, hidden_dim)
        
        # Independent predictions for each coefficient
        n_pred = self.head_n(shared_features).squeeze(-1)              # (B,)
        mu_a_pred = self.head_mu_a(shared_features).squeeze(-1)        # (B,)
        mu_s_prime_pred = self.head_mu_s_prime(shared_features).squeeze(-1)  # (B,)
        
        return n_pred, mu_a_pred, mu_s_prime_pred


class ETFClassifier(nn.Module):
    """
    Equiangular Tight Frame (ETF) Classifier for Material Classification.
    
    ETF Classifier는 고정된 기하학적 구조를 가진 classifier로,
    모든 클래스 간의 각도가 동일하여 few-shot learning에 유리합니다.
    
    This implementation follows FSCIL's ETFHead with proper ETF construction:
    W_ETF = √(C/(C-1)) × orth_vec × (I - 1/C × 11^T)
    
    Architecture:
    - Input: (B, 3) - Physical coefficients [n, μₐ, μₛ']
    - Feature Extractor: 3 → 64 → 32 (learnable, for low-dim input expansion)
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
    
    Args:
        num_classes (int): Number of material classes (default: 5)
        input_dim (int): Dimension of physical coefficients (default: 3)
        hidden_dims (tuple): Hidden layer dimensions (default: (64, 32))
    """
    
    def __init__(self, num_classes: int = 5, input_dim: int = 3, 
                 hidden_dims: tuple = (64, 32)):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Feature extractor (learnable)
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = prev_dim
        
        # ETF Classifier Head (fixed, not learnable)
        self.etf_head = self._create_etf_weights(num_classes, self.feature_dim)
        
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
        
        # 2. Create ETF transformation matrix
        i_nc_nc = torch.eye(num_classes)  # Identity (C × C)
        one_nc_nc = torch.ones(num_classes, num_classes) / num_classes  # (C × C)
        
        # 3. Apply ETF formula
        # Scale factor: √(C/(C-1)) - CORRECT formula from FSCIL
        scale = math.sqrt(num_classes / (num_classes - 1))
        
        # ETF = orth_vec @ (I - 1/C × 11^T)
        # Shape: (D, C) @ (C, C) = (D, C)
        etf_vec = torch.matmul(orth_vec, i_nc_nc - one_nc_nc)
        etf_vec = etf_vec * scale
        
        # 4. Register as non-trainable parameter
        # Transpose to (num_classes, feature_dim) for F.linear
        etf_weights = nn.Parameter(etf_vec.T, requires_grad=False)
        
        return etf_weights
    
    def forward(self, input_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for material classification using ETF geometry.
        
        Process:
        1. Extract features: (B, input_dim) → (B, feature_dim) using learnable MLP
        2. L2 normalize: Required for ETF geometry (projects onto hypersphere)
        3. Compute logits: features @ W_ETF^T (bias-free linear)
        4. Predict class: argmax over logits
        
        Args:
            input_features (torch.Tensor): Input features (B, input_dim) - Can be temporal features or physical coefficients
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - class_logits: (B, num_classes) - Class logits (not probabilities)
                - class_pred: (B,) - Predicted class indices [0-4]
        """
        # 1. Extract features from input (input_dim → feature_dim dimensions)
        features = self.feature_extractor(input_features)  # (B, feature_dim)
        
        # 2. L2 normalize features (CRITICAL for ETF geometry)
        # Projects features onto unit hypersphere
        features = F.normalize(features, p=2, dim=1)  # (B, feature_dim), ||f|| = 1
        
        # 3. Compute logits using fixed ETF weights (bias-free)
        # Dot product: cosine similarity with each class prototype
        # logits = features @ W_ETF^T
        class_logits = F.linear(features, self.etf_head, bias=None)  # (B, num_classes)
        
        # 4. Get predicted class (argmax)
        class_pred = torch.argmax(class_logits, dim=1)  # (B,)
        
        return class_logits, class_pred
    
    def get_etf_structure(self) -> torch.Tensor:
        """
        Return the ETF weight matrix for visualization.
        
        Returns:
            torch.Tensor: ETF weights (num_classes, feature_dim)
        """
        return self.etf_head.data
    
    def compute_class_similarity(self) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between class vectors.
        
        For perfect ETF structure, all off-diagonal elements should be -1/(C-1).
        
        Returns:
            torch.Tensor: Similarity matrix (num_classes, num_classes)
        """
        W = self.etf_head.data  # (C, D)
        W_norm = F.normalize(W, p=2, dim=1)  # Normalize rows
        similarity = torch.mm(W_norm, W_norm.T)  # (C, C)
        
        return similarity


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
                'input_dim': 1024,
                'd_model': 512,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        
        if physics_config is None:
            physics_config = {
                'physics_weight': 1.0,
                'data_weight': 1.0,
                'classification_weight': 10.0,  # Classification is primary goal
                'c': 3e8
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
            d_model=temporal_config['d_model'],
            device=temporal_config['device']
        )
        
        # 3. MLP Regressor (Physical Coefficients Predictor)
        self.regressor = MLPRegressor(
            input_dim=temporal_config['d_model'],
            hidden_dim=256,
            dropout=0.5
        )
        
        # 4. ETF Classifier (Material Classification directly from temporal features)
        # 1D Mamba 출력을 ETF Classifier로 직접 분류
        self.direct_classifier = ETFClassifier(
            num_classes=num_classes,  # 5 materials
            input_dim=temporal_config['d_model'],  # 512 (1D Mamba 출력 차원)
            hidden_dims=(256, 128)  # Feature extractor dimensions
        )
        
        # 5. Physics Loss Module
        self.physics_loss = PINNPhysicsLoss(
            c=physics_config['c'],
            physics_weight=physics_config['physics_weight'],
            data_weight=physics_config['data_weight']
        )
        
        # Store configurations
        self.temporal_config = temporal_config
        self.physics_config = physics_config
        self.num_classes = num_classes
    
    def forward(self, images: torch.Tensor, n_true: torch.Tensor, 
                material_label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Complete PICL forward pass with simultaneous processing.
        
        Args:
            images (torch.Tensor): Original image sequence (B, 5, 3, 224, 224)
            n_true (torch.Tensor): True refractive index (B,)
            material_label (torch.Tensor, optional): True material labels (B,) [0-4]
            
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
        temporal_features = self.temporal(features_reshaped)  # (B*H_sp*W_sp, 512)
        
        # Reshape back and aggregate spatial information
        temporal_features = temporal_features.view(B, H_sp, W_sp, -1)  # (B, H_sp, W_sp, 512)
        # Aggregate spatial information: average pooling
        temporal_features = temporal_features.permute(0, 3, 1, 2)  # (B, 512, H_sp, W_sp)
        temporal_features = F.adaptive_avg_pool2d(temporal_features, (1, 1))  # (B, 512, 1, 1)
        temporal_features = temporal_features.flatten(1)  # (B, 512)
        
        # 3. 병렬 경로: 물리계수 예측 + 직접 분류
        # 경로 1: MLP Regressor → 3개 계수 예측
        n_pred, mu_a_pred, mu_s_prime_pred = self.regressor(temporal_features)
        
        # 경로 2: ETF Classifier로 직접 분류 (1D Mamba 출력에서 바로 분류)
        direct_class_logits, direct_class_pred = self.direct_classifier(temporal_features)  # (B, num_classes), (B,)
        
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
        
        # 3.1. Physics Loss (Data + PDE)
        pinn_loss, loss_dict = self.physics_loss(
            phi_original=phi_sequence,      # Original images for physics
            n_pred=n_pred,                  # Predicted refractive index
            n_true=n_true,                  # True refractive index
            mu_a_pred=mu_a_pred,            # Predicted absorption coefficient
            mu_s_prime_pred=mu_s_prime_pred, # Predicted reduced scattering coefficient
            dx=1.0, dy=1.0, dt=1.0         # Spatial and temporal steps
        )
        
        # 3.2. Classification Loss (직접 분류 경로만)
        direct_classification_loss = torch.tensor(0.0, device=images.device)
        if material_label is not None:
            # 직접 분류하는 loss
            direct_classification_loss = F.cross_entropy(direct_class_logits, material_label)
        
        # 3.3. Total Loss
        w_cls = self.physics_config.get('classification_weight', 10.0)
        w_data = self.physics_config.get('data_weight', 1.0)
        w_physics = self.physics_config.get('physics_weight', 1.0)
        
        total_loss = (w_cls * direct_classification_loss + 
                     w_data * loss_dict['data_loss'] + 
                     w_physics * loss_dict['physics_loss'])
        
        # ========================================
        # RETURN COMPLETE RESULTS
        # ========================================
        
        results = {
            # Predictions - Physical Coefficients
            'n_pred': n_pred,
            'mu_a_pred': mu_a_pred,
            'mu_s_prime_pred': mu_s_prime_pred,
            
            # Predictions - Direct Classification (직접 분류)
            'direct_class_logits': direct_class_logits,
            'direct_class_pred': direct_class_pred,
            
            # Features
            'temporal_features': temporal_features,
            
            # Losses
            'total_loss': total_loss,
            'classification_loss': direct_classification_loss,  # 직접 분류 loss
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
            temporal_features = self.temporal(features_reshaped)  # (B*H_sp*W_sp, 512)
            
            # Reshape back and aggregate spatial information
            temporal_features = temporal_features.view(B, H_sp, W_sp, -1)  # (B, H_sp, W_sp, 512)
            temporal_features = temporal_features.permute(0, 3, 1, 2)  # (B, 512, H_sp, W_sp)
            temporal_features = F.adaptive_avg_pool2d(temporal_features, (1, 1))  # (B, 512, 1, 1)
            temporal_features = temporal_features.flatten(1)  # (B, 512)
            
            # 3. MLP Regressor
            n_pred, mu_a_pred, mu_s_prime_pred = self.regressor(temporal_features)
            
            return n_pred, mu_a_pred, mu_s_prime_pred
    
    def predict_material(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict material classification and physical coefficients.
        
        Args:
            images (torch.Tensor): Image sequence (B, 5, 3, 224, 224)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - class_logits: (B, num_classes) - Class logits (직접 분류)
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
            
            # 2. 1D Mamba
            temporal_features = self.temporal(features_reshaped)
            temporal_features = temporal_features.view(B, H_sp, W_sp, -1)
            temporal_features = temporal_features.permute(0, 3, 1, 2)
            temporal_features = F.adaptive_avg_pool2d(temporal_features, (1, 1))
            temporal_features = temporal_features.flatten(1)  # (B, 512)
            
            # 3. Get physical coefficients
            n_pred, mu_a_pred, mu_s_prime_pred = self.regressor(temporal_features)
            physical_coeffs = torch.stack([n_pred, mu_a_pred, mu_s_prime_pred], dim=1)  # (B, 3)
            
            # 4. ETF Classifier로 직접 분류
            class_logits, class_pred = self.direct_classifier(temporal_features)  # (B, num_classes), (B,)
            
            return class_logits, class_pred, physical_coeffs


def test_picl_model():
    """Test the complete PICL model."""
    print("🧪 Testing Complete PICL Model")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    B, T, C, H, W = 2, 5, 3, 224, 224
    images = torch.randn(B, T, C, H, W, device=device)
    n_true = torch.tensor([1.33, 1.52], device=device)  # Water, Glass
    
    # Initialize model
    model = PICLModel(
        backbone_model='vmamba_base_s2l15',
        pretrained_path=None,  # No pretrained weights for testing
        temporal_config={
            'input_dim': 1024,
            'd_model': 512,
            'device': device
        },
        physics_config={
            'physics_weight': 1.0,
            'data_weight': 1.0,
            'c': 3e8
        }
    )
    model = model.to(device)
    
    # Test forward pass
    print("\n📋 Testing Complete Forward Pass...")
    results = model(images, n_true)
    
    print(f"✅ Input shape: {images.shape}")
    print(f"✅ n_pred shape: {results['n_pred'].shape}")
    print(f"✅ mu_a_pred shape: {results['mu_a_pred'].shape}")
    print(f"✅ mu_s_prime_pred shape: {results['mu_s_prime_pred'].shape}")
    print(f"✅ Total loss: {results['total_loss'].item():.6f}")
    print(f"✅ Data loss: {results['data_loss'].item():.6f}")
    print(f"✅ Physics loss: {results['physics_loss'].item():.6f}")
    
    # Test prediction only
    print("\n📋 Testing Prediction Only...")
    n_pred, mu_a_pred, mu_s_prime_pred = model.predict_coefficients(images)
    
    print(f"✅ Predicted n: {n_pred.cpu().numpy()}")
    print(f"✅ Predicted μa: {mu_a_pred.cpu().numpy()}")
    print(f"✅ Predicted μs': {mu_s_prime_pred.cpu().numpy()}")
    
    # Test parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,}")
    
    print("\n🎉 PICL Model test completed successfully!")
    print("\n📝 Usage:")
    print("   1. model(images, n_true) - Complete forward pass with physics loss")
    print("   2. model.predict_coefficients(images) - Prediction only")
    print("   3. Results include predictions, losses, and intermediate features")


if __name__ == "__main__":
    test_picl_model()
