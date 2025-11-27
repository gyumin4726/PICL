"""
Physics-Informed Neural Network (PINN) for Optical Scattering
Time-dependent Diffusion Equation Implementation

This module implements the time-dependent diffusion equation for optical scattering
in turbid media, which will be used as physics constraints in PINN training.

Equation: n/c * ∂Φ/∂t + μa*Φ - ∇·(D∇Φ) = S(r,t)

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
    n/c * ∂Φ/∂t + μa*Φ - ∇·(D∇Φ) = S(r,t)
    
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
        
        # Compute ∇·(D∇Φ) for ALL time steps (not just center)
        # This ensures PDE residual is computed accurately at each time step
        D_expanded = D.squeeze(1).squeeze(1).squeeze(1)  # (B,)
        D_expanded = D_expanded.view(B, 1, 1)  # (B, 1, 1) for broadcasting
        
        # Compute div_D_grad_phi for each time step in the residual calculation
        # Residual uses phi_sequence[:, :-1], so we compute for t=0, 1, 2, ..., T-2
        div_D_grad_phi_all = []
        for t in range(T - 1):  # For each time step used in residual (0 to T-2)
            phi_t = phi_sequence[:, t]  # (B, H, W) - image at time step t
            D_t = D_expanded.expand(B, H, W)  # (B, H, W) - expand D to match spatial dimensions
            div_D_grad_phi_t = self.compute_gradient_divergence(
                phi_t, D_t, dx, dy
            )  # (B, H, W)
            div_D_grad_phi_all.append(div_D_grad_phi_t)
        
        # Stack all time steps: (B, T-1, H, W)
        div_D_grad_phi = torch.stack(div_D_grad_phi_all, dim=1)
        
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
        # Use Huber Loss instead of MSE for robustness to outliers
        # delta=0.1: threshold between quadratic and linear regions
        loss = weight * F.huber_loss(residual, torch.zeros_like(residual), delta=0.1, reduction='mean')
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
        # Data loss (Huber Loss between predicted and true refractive index)
        # Huber Loss: robust to outliers, linear penalty for large errors
        # For |error| < delta: quadratic (like MSE)
        # For |error| >= delta: linear (like MAE)
        # delta=0.1: threshold between quadratic and linear regions
        data_loss = F.huber_loss(n_pred, n_true, delta=0.1, reduction='mean')
        
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


