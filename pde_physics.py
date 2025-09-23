"""
Physics-Informed Neural Network (PINN) for Optical Scattering
Time-dependent Diffusion Equation Implementation

This module implements the time-dependent diffusion equation for optical scattering
in turbid media, which will be used as physics constraints in PINN training.

Equation: c/n * ∂Φ/∂t + μa*Φ - ∇·(D∇Φ) = S(r,t)

Where:
- Φ(r,t): Fluence rate (pixel values in MCX simulation images)
- n: Refractive index (target prediction)
- μa: Absorption coefficient (target prediction)  
- D: Diffusion coefficient = 1/(3*(μa + μs'))
- c: Speed of light in vacuum
- S(r,t): Source term
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class TimeDependentDiffusionPDE:
    """
    Time-dependent Diffusion Equation for Optical Scattering
    
    Implements the physics equation:
    c/n * ∂Φ/∂t + μa*Φ - ∇·(D∇Φ) = S(r,t)
    
    This class provides methods to compute the PDE residual for PINN training.
    """
    
    def __init__(self, c: float = 3e8):
        """
        Initialize the PDE parameters.
        
        Args:
            c (float): Speed of light in vacuum (m/s)
        """
        self.c = c  # Speed of light in vacuum
        
    def diffusion_coefficient(self, mu_a: torch.Tensor, mu_s_prime: torch.Tensor) -> torch.Tensor:
        """
        Calculate diffusion coefficient D from absorption and reduced scattering coefficients.
        
        D = 1 / (3 * (μa + μs'))
        
        Args:
            mu_a (torch.Tensor): Absorption coefficient
            mu_s_prime (torch.Tensor): Reduced scattering coefficient (μs' = μs(1-g))
            
        Returns:
            torch.Tensor: Diffusion coefficient
        """
        return 1.0 / (3.0 * (mu_a + mu_s_prime))
    
    def compute_laplacian(self, phi: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
        """
        Compute Laplacian (∇²Φ) using finite differences.
        
        Args:
            phi (torch.Tensor): Fluence rate field (B, H, W)
            dx (float): Spatial step in x direction
            dy (float): Spatial step in y direction
            
        Returns:
            torch.Tensor: Laplacian of phi
        """
        # Compute second derivatives using central differences
        d2phi_dx2 = torch.zeros_like(phi)
        d2phi_dy2 = torch.zeros_like(phi)
        
        # ∂²Φ/∂x²
        d2phi_dx2[:, :, 1:-1] = (phi[:, :, 2:] - 2*phi[:, :, 1:-1] + phi[:, :, :-2]) / (dx**2)
        
        # ∂²Φ/∂y²  
        d2phi_dy2[:, 1:-1, :] = (phi[:, 2:, :] - 2*phi[:, 1:-1, :] + phi[:, :-2, :]) / (dy**2)
        
        return d2phi_dx2 + d2phi_dy2
    
    def compute_gradient_divergence(self, phi: torch.Tensor, D: torch.Tensor, 
                                  dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
        """
        Compute ∇·(D∇Φ) using finite differences.
        
        Args:
            phi (torch.Tensor): Fluence rate field (B, H, W)
            D (torch.Tensor): Diffusion coefficient field (B, H, W)
            dx (float): Spatial step in x direction
            dy (float): Spatial step in y direction
            
        Returns:
            torch.Tensor: Divergence of D times gradient of phi
        """
        # Compute gradients
        grad_phi_x = torch.zeros_like(phi)
        grad_phi_y = torch.zeros_like(phi)
        
        # ∂Φ/∂x (central difference)
        grad_phi_x[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dx)
        
        # ∂Φ/∂y (central difference)
        grad_phi_y[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
        
        # Compute D∇Φ
        D_grad_phi_x = D * grad_phi_x
        D_grad_phi_y = D * grad_phi_y
        
        # Compute ∇·(D∇Φ)
        div_D_grad_phi = torch.zeros_like(phi)
        
        # ∂(D∂Φ/∂x)/∂x
        div_D_grad_phi[:, :, 1:-1] += (D_grad_phi_x[:, :, 2:] - D_grad_phi_x[:, :, :-2]) / (2 * dx)
        
        # ∂(D∂Φ/∂y)/∂y
        div_D_grad_phi[:, 1:-1, :] += (D_grad_phi_y[:, 2:, :] - D_grad_phi_y[:, :-2, :]) / (2 * dy)
        
        return div_D_grad_phi
    
    def compute_time_derivative(self, phi_sequence: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Compute time derivative ∂Φ/∂t using finite differences.
        
        Args:
            phi_sequence (torch.Tensor): Fluence rate sequence (B, T, H, W)
            dt (float): Time step
            
        Returns:
            torch.Tensor: Time derivative (B, T-1, H, W)
        """
        if phi_sequence.shape[1] < 2:
            raise ValueError("Need at least 2 time steps to compute derivative")
            
        # Forward difference: ∂Φ/∂t ≈ (Φ(t+1) - Φ(t)) / dt
        dphi_dt = (phi_sequence[:, 1:] - phi_sequence[:, :-1]) / dt
        
        return dphi_dt
    
    def pde_residual(self, phi_sequence: torch.Tensor, n: torch.Tensor, mu_a: torch.Tensor, mu_s_prime: torch.Tensor,
                    source: Optional[torch.Tensor] = None, 
                    dx: float = 1.0, dy: float = 1.0, dt: float = 1.0) -> torch.Tensor:
        """
        Compute PDE residual for the time-dependent diffusion equation.
        
        Residual = n/c * ∂Φ/∂t + μa*Φ - ∇·(D∇Φ) - S(r,t)
        
        Args:
            phi_sequence (torch.Tensor): Fluence rate sequence (B, T, H, W)
            n (torch.Tensor): Refractive index (B,) or (B, 1, 1, 1)
            mu_a (torch.Tensor): Absorption coefficient (B,) or (B, 1, 1, 1)
            mu_s_prime (torch.Tensor): Reduced scattering coefficient (B,) or (B, 1, 1, 1)
            source (torch.Tensor, optional): Source term (B, T, H, W). Defaults to zero.
            dx (float): Spatial step in x direction
            dy (float): Spatial step in y direction  
            dt (float): Time step
            
        Returns:
            torch.Tensor: PDE residual (B, T-1, H, W)
        """
        B, T, H, W = phi_sequence.shape
        
        # Ensure n, mu_a, and mu_s_prime have correct shapes
        if n.dim() == 1:
            n = n.view(B, 1, 1, 1)
        if mu_a.dim() == 1:
            mu_a = mu_a.view(B, 1, 1, 1)
        if mu_s_prime.dim() == 1:
            mu_s_prime = mu_s_prime.view(B, 1, 1, 1)
            
        # Compute time derivative
        dphi_dt = self.compute_time_derivative(phi_sequence, dt)  # (B, T-1, H, W)
        
        # Compute diffusion coefficient
        D = self.diffusion_coefficient(mu_a, mu_s_prime)  # (B, 1, 1, 1)
        
        # Compute ∇·(D∇Φ) for representative time step (2nd image)
        center_idx = 1  # Use 2nd image (index 1) as representative
        phi_center = phi_sequence[:, center_idx]  # (B, H, W)
        div_D_grad_phi_center = self.compute_gradient_divergence(
            phi_center, D.squeeze(), dx, dy
        )  # (B, H, W)
        
        # Expand to match time dimension for residual calculation
        div_D_grad_phi = div_D_grad_phi_center.unsqueeze(1).expand(-1, T-1, -1, -1)
        
        # Compute source term (default to zero if not provided)
        if source is None:
            source = torch.zeros_like(dphi_dt)
        else:
            source = source[:, :-1]  # Match time dimension
        
        # Compute PDE residual
        residual = (n / self.c) * dphi_dt + mu_a * phi_sequence[:, :-1] - div_D_grad_phi - source
        
        return residual
    
    def physics_loss(self, phi_sequence: torch.Tensor, n: torch.Tensor, mu_a: torch.Tensor, mu_s_prime: torch.Tensor,
                    source: Optional[torch.Tensor] = None,
                    dx: float = 1.0, dy: float = 1.0, dt: float = 1.0,
                    weight: float = 1.0) -> torch.Tensor:
        """
        Compute physics loss (MSE of PDE residual).
        
        Args:
            phi_sequence (torch.Tensor): Fluence rate sequence (B, T, H, W)
            n (torch.Tensor): Refractive index (B,)
            mu_a (torch.Tensor): Absorption coefficient (B,)
            mu_s_prime (torch.Tensor): Reduced scattering coefficient (B,)
            source (torch.Tensor, optional): Source term
            dx, dy, dt (float): Spatial and temporal steps
            weight (float): Loss weight
            
        Returns:
            torch.Tensor: Physics loss scalar
        """
        residual = self.pde_residual(phi_sequence, n, mu_a, mu_s_prime, source, dx, dy, dt)
        loss = weight * torch.mean(residual**2)
        return loss


class PINNPhysicsLoss(nn.Module):
    """
    Physics-Informed Neural Network Loss Module
    
    Combines data loss and physics loss for training PINN models.
    """
    
    def __init__(self, c: float = 3e8, physics_weight: float = 1.0, data_weight: float = 1.0):
        """
        Initialize PINN loss module.
        
        Args:
            c (float): Speed of light in vacuum
            physics_weight (float): Weight for physics loss
            data_weight (float): Weight for data loss
        """
        super().__init__()
        self.pde = TimeDependentDiffusionPDE(c)
        self.physics_weight = physics_weight
        self.data_weight = data_weight
        
    def forward(self, phi_original: torch.Tensor, n_pred: torch.Tensor, n_true: torch.Tensor,
                mu_a_pred: torch.Tensor, mu_s_prime_pred: torch.Tensor,
                source: Optional[torch.Tensor] = None,
                dx: float = 1.0, dy: float = 1.0, dt: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined PINN loss.
        
        Args:
            phi_original (torch.Tensor): Original MCX simulation images (B, T, H, W)
            n_pred (torch.Tensor): Predicted refractive index (B,)
            n_true (torch.Tensor): True refractive index (B,)
            mu_a_pred (torch.Tensor): Predicted absorption coefficient (B,)
            mu_s_prime_pred (torch.Tensor): Predicted reduced scattering coefficient (B,)
            source (torch.Tensor, optional): Source term
            dx, dy, dt (float): Spatial and temporal steps
            
        Returns:
            Tuple[torch.Tensor, dict]: Total loss and loss components
        """
        # Data loss (MSE between predicted and true refractive index)
        data_loss = F.mse_loss(n_pred, n_true)
        
        # Physics loss (PDE residual using original images and predicted coefficients)
        physics_loss = self.pde.physics_loss(
            phi_original, n_pred, mu_a_pred, mu_s_prime_pred, source, dx, dy, dt
        )
        
        # Combined loss
        total_loss = self.data_weight * data_loss + self.physics_weight * physics_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss
        }
        
        return total_loss, loss_dict


def test_pde_implementation():
    """Test the PDE implementation with dummy data."""
    print("🧪 Testing Time-dependent Diffusion Equation Implementation")
    print("=" * 60)
    
    # Create dummy data
    B, T, H, W = 2, 5, 32, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy fluence rate sequence (time-gate images)
    phi_sequence = torch.randn(B, T, H, W, device=device)
    
    # Dummy material properties
    n = torch.tensor([1.33, 1.52], device=device)  # Water, Glass
    mu_a = torch.tensor([0.01, 0.02], device=device)  # Absorption coefficients
    mu_s_prime = torch.tensor([1.0, 1.5], device=device)  # Reduced scattering coefficients
    
    # Initialize PDE
    pde = TimeDependentDiffusionPDE()
    
    # Test diffusion coefficient calculation
    D = pde.diffusion_coefficient(mu_a, mu_s_prime)
    print(f"✅ Diffusion coefficients: {D.squeeze().cpu().numpy()}")
    
    # Test PDE residual calculation
    residual = pde.pde_residual(phi_sequence, n, mu_a, mu_s_prime)
    print(f"✅ PDE residual shape: {residual.shape}")
    print(f"✅ PDE residual mean: {residual.mean().item():.6f}")
    print(f"✅ PDE residual std: {residual.std().item():.6f}")
    
    # Test physics loss
    physics_loss = pde.physics_loss(phi_sequence, n, mu_a, mu_s_prime)
    print(f"✅ Physics loss: {physics_loss.item():.6f}")
    
    # Test PINN loss module
    pinl_loss = PINNPhysicsLoss(physics_weight=1.0, data_weight=1.0)
    n_true = n + 0.1 * torch.randn_like(n)  # Add noise to refractive index
    
    total_loss, loss_dict = pinl_loss(phi_sequence, n, n_true, mu_a, mu_s_prime)
    
    print(f"\n📊 PINN Loss Components:")
    print(f"   Total Loss: {loss_dict['total_loss'].item():.6f}")
    print(f"   Data Loss: {loss_dict['data_loss'].item():.6f}")
    print(f"   Physics Loss: {loss_dict['physics_loss'].item():.6f}")
    
    print("\n🎉 PDE implementation test completed successfully!")
    print("\n📝 Usage in PINN Training:")
    print("   1. Use TimeDependentDiffusionPDE for physics constraints")
    print("   2. Use PINNPhysicsLoss for combined data + physics loss")
    print("   3. Integrate with your neural network training loop")
    print("   4. Adjust physics_weight to balance data vs physics constraints")


if __name__ == "__main__":
    test_pde_implementation()
