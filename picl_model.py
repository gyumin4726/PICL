"""
Complete PICL Model: Integrated Physics-Informed Neural Network
Combines 2D VMamba + 1D Mamba + Physics Constraints

This module implements the complete PICL pipeline that simultaneously:
1. Predicts physical coefficients (n, μa, μs') using neural networks
2. Computes physical quantities (Φ, ∂Φ/∂t, ∇²Φ) from original images
3. Validates physics constraints using PDE equations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

# Import our modules
from vmamba_backbone import VMambaBackbone
from mamba_1d_temporal import SequenceToValue
from pde_physics import PINNPhysicsLoss


class MLPRegressor(nn.Module):
    """
    MLP Regressor for Physical Coefficients Prediction.
    
    Predicts 3 physical coefficients: refractive index (n), 
    absorption coefficient (μa), and reduced scattering coefficient (μs').
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, 
                 num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for physical coefficients prediction.
        
        Args:
            x (torch.Tensor): Input features (B, input_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (n_pred, mu_a_pred, mu_s_prime_pred)
        """
        output = self.classifier(x)  # (B, 3)
        
        # Split into individual coefficients
        n_pred = output[:, 0]           # Refractive index
        mu_a_pred = output[:, 1]        # Absorption coefficient  
        mu_s_prime_pred = output[:, 2]  # Reduced scattering coefficient
        
        return n_pred, mu_a_pred, mu_s_prime_pred


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
                 physics_config: Optional[dict] = None):
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
        self.classifier = MLPRegressor(
            input_dim=temporal_config['d_model'],
            hidden_dim=256,
            num_classes=3,  # n, μa, μs'
            dropout=0.5
        )
        
        # 4. Physics Loss Module
        self.physics_loss = PINNPhysicsLoss(
            c=physics_config['c'],
            physics_weight=physics_config['physics_weight'],
            data_weight=physics_config['data_weight']
        )
        
        # Store configurations
        self.temporal_config = temporal_config
        self.physics_config = physics_config
    
    def forward(self, images: torch.Tensor, n_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete PICL forward pass with simultaneous processing.
        
        Args:
            images (torch.Tensor): Original image sequence (B, 5, 3, 224, 224)
            n_true (torch.Tensor): True refractive index (B,)
            
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
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))  # (B*T, 1024, 1, 1)
        features = features.flatten(1)  # (B*T, 1024)
        
        # Reshape back to batch format
        features = features.view(B, T, -1)  # (B, T, 1024)
        
        # 2. 1D Mamba + SequenceToValue: Process temporal sequence
        temporal_features = self.temporal(features)  # (B, 512)
        
        # 3. MLP Regressor: Predict physical coefficients
        n_pred, mu_a_pred, mu_s_prime_pred = self.classifier(temporal_features)
        
        # ========================================
        # PATH 2: Physics Quantities Computation
        # ========================================
        
        # Convert images to grayscale for physics computation (if needed)
        if images.shape[2] == 3:  # RGB to grayscale
            phi_sequence = torch.mean(images, dim=2)  # (B, T, H, W)
        else:
            phi_sequence = images  # Already grayscale
        
        # ========================================
        # PATH 3: Physics Loss Computation
        # ========================================
        
        # Compute combined PINN loss
        total_loss, loss_dict = self.physics_loss(
            phi_original=phi_sequence,      # Original images for physics
            n_pred=n_pred,                  # Predicted refractive index
            n_true=n_true,                  # True refractive index
            mu_a_pred=mu_a_pred,            # Predicted absorption coefficient
            mu_s_prime_pred=mu_s_prime_pred, # Predicted reduced scattering coefficient
            dx=1.0, dy=1.0, dt=1.0         # Spatial and temporal steps
        )
        
        # ========================================
        # RETURN COMPLETE RESULTS
        # ========================================
        
        results = {
            # Predictions
            'n_pred': n_pred,
            'mu_a_pred': mu_a_pred,
            'mu_s_prime_pred': mu_s_prime_pred,
            
            # Features
            'temporal_features': temporal_features,
            
            # Losses
            'total_loss': total_loss,
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
            
            # Global pooling
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
            features = features.view(B, T, -1)
            
            # 2. 1D Mamba + SequenceToValue
            temporal_features = self.temporal(features)
            
            # 3. MLP Regressor
            n_pred, mu_a_pred, mu_s_prime_pred = self.classifier(temporal_features)
            
            return n_pred, mu_a_pred, mu_s_prime_pred


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
