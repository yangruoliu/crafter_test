#!/usr/bin/env python3
"""
Test script for fixed loss normalization functionality
"""

import torch
import numpy as np

def test_loss_normalization_fixed():
    """Test the fixed loss normalization functionality"""
    
    # Mock the key components of CustomPPO for testing
    class MockCustomPPO:
        def __init__(self):
            self.device = torch.device("cpu")
            self.loss_normalization = True
            self.norm_decay = 0.99
            self.vf_coef = 0.5
            self.ent_coef = 0.01
            self.direction_weight = 0.3
            
            # Initialize loss statistics
            self.loss_norm_steps = 0
            self.policy_loss_ma = None
            self.value_loss_ma = None
            self.entropy_loss_ma = None
            self.direction_loss_ma = None
            
            self.policy_loss_var = None
            self.value_loss_var = None
            self.entropy_loss_var = None
            self.direction_loss_var = None
        
        def _update_loss_statistics(self, policy_loss: float, value_loss: float, 
                                   entropy_loss: float, direction_loss: float) -> None:
            """Update moving averages and variances for loss normalization."""
            self.loss_norm_steps += 1
            
            losses = [policy_loss, value_loss, entropy_loss, direction_loss]
            moving_averages = [self.policy_loss_ma, self.value_loss_ma, 
                              self.entropy_loss_ma, self.direction_loss_ma]
            moving_variances = [self.policy_loss_var, self.value_loss_var,
                               self.entropy_loss_var, self.direction_loss_var]
            
            # Initialize on first step
            if self.loss_norm_steps == 1:
                self.policy_loss_ma = policy_loss
                self.value_loss_ma = value_loss
                self.entropy_loss_ma = entropy_loss
                self.direction_loss_ma = direction_loss
                
                self.policy_loss_var = 0.0
                self.value_loss_var = 0.0
                self.entropy_loss_var = 0.0
                self.direction_loss_var = 0.0
            else:
                # Update moving averages and variances using EWMA
                updated_mas = []
                updated_vars = []
                
                for loss, ma, var in zip(losses, moving_averages, moving_variances):
                    # Update moving average
                    new_ma = self.norm_decay * ma + (1 - self.norm_decay) * loss
                    # Update moving variance (Welford's online algorithm adaptation)
                    new_var = self.norm_decay * var + (1 - self.norm_decay) * (loss - ma) ** 2
                    
                    updated_mas.append(new_ma)
                    updated_vars.append(new_var)
                
                self.policy_loss_ma, self.value_loss_ma, self.entropy_loss_ma, self.direction_loss_ma = updated_mas
                self.policy_loss_var, self.value_loss_var, self.entropy_loss_var, self.direction_loss_var = updated_vars

        def _normalize_losses(self, policy_loss: torch.Tensor, value_loss: torch.Tensor,
                             entropy_loss: torch.Tensor, direction_loss: torch.Tensor) -> tuple:
            """Fixed normalize losses using moving averages to handle scale differences."""
            if not self.loss_normalization or self.loss_norm_steps < 10:
                # Skip normalization for first few steps to build statistics
                return policy_loss, value_loss, entropy_loss, direction_loss, 1.0, self.ent_coef, self.vf_coef, self.direction_weight
            
            # Normalize by moving averages (avoid division by zero)
            eps = 1e-8
            
            policy_loss_norm = policy_loss / (abs(self.policy_loss_ma) + eps)
            value_loss_norm = value_loss / (abs(self.value_loss_ma) + eps)
            entropy_loss_norm = entropy_loss / (abs(self.entropy_loss_ma) + eps)
            direction_loss_norm = direction_loss / (abs(self.direction_loss_ma) + eps)
            
            # Calculate coefficient of variation for adaptive weighting
            # Use absolute values to handle negative loss means (like entropy)
            policy_std = (self.policy_loss_var ** 0.5)
            value_std = (self.value_loss_var ** 0.5)
            entropy_std = (self.entropy_loss_var ** 0.5)
            direction_std = (self.direction_loss_var ** 0.5)
            
            policy_cov = policy_std / (abs(self.policy_loss_ma) + eps)
            value_cov = value_std / (abs(self.value_loss_ma) + eps)
            entropy_cov = entropy_std / (abs(self.entropy_loss_ma) + eps)
            direction_cov = direction_std / (abs(self.direction_loss_ma) + eps)
            
            # Ensure all coefficients are positive
            policy_cov = max(0.0, policy_cov)
            value_cov = max(0.0, value_cov)
            entropy_cov = max(0.0, entropy_cov)
            direction_cov = max(0.0, direction_cov)
            
            # Adaptive weights based on coefficient of variation
            # Apply smoothing to prevent extreme weight distributions
            policy_cov_smooth = policy_cov ** 0.5  # Take square root to reduce extreme values
            value_cov_smooth = value_cov ** 0.5
            entropy_cov_smooth = entropy_cov ** 0.5
            direction_cov_smooth = direction_cov ** 0.5
            
            total_cov = policy_cov_smooth + value_cov_smooth + entropy_cov_smooth + direction_cov_smooth + eps
            
            # Base weights from original coefficients with adaptive scaling
            base_policy_weight = 1.0
            base_value_weight = self.vf_coef
            base_entropy_weight = self.ent_coef
            base_direction_weight = self.direction_weight
            
            # Adaptive weights combining base weights with variation-based scaling
            adaptive_policy_weight = (policy_cov_smooth / total_cov) * base_policy_weight
            adaptive_value_weight = (value_cov_smooth / total_cov) * base_value_weight
            adaptive_entropy_weight = (entropy_cov_smooth / total_cov) * base_entropy_weight
            adaptive_direction_weight = (direction_cov_smooth / total_cov) * base_direction_weight
            
            # Ensure minimum weights to prevent complete suppression
            min_weight = 0.01
            adaptive_policy_weight = max(min_weight * base_policy_weight, adaptive_policy_weight)
            adaptive_value_weight = max(min_weight * base_value_weight, adaptive_value_weight)
            adaptive_entropy_weight = max(min_weight * base_entropy_weight, adaptive_entropy_weight)
            adaptive_direction_weight = max(min_weight * base_direction_weight, adaptive_direction_weight)
            
            return (policy_loss_norm, value_loss_norm, entropy_loss_norm, direction_loss_norm,
                    adaptive_policy_weight, adaptive_entropy_weight, adaptive_value_weight, adaptive_direction_weight)

    # Create test instance
    ppo = MockCustomPPO()
    
    # Test with realistic loss values including negative entropy
    test_losses = [
        (1.0, 48.5, -2.47, 1.58),  # Similar to your training log
        (0.8, 45.2, -2.39, 1.76),
        (1.2, 52.1, -2.55, 1.42),
        (0.9, 47.8, -2.41, 1.63),
        (1.1, 49.3, -2.48, 1.71),
        (0.85, 46.7, -2.44, 1.59),
        (1.05, 48.9, -2.46, 1.68),
        (0.95, 47.2, -2.42, 1.54),
        (1.15, 50.1, -2.49, 1.72),
        (0.88, 46.9, -2.43, 1.61),
        (1.08, 49.7, -2.47, 1.69),  # Step 11 - normalization kicks in
        (0.92, 47.5, -2.41, 1.57),
    ]
    
    print("Testing FIXED loss normalization...")
    print("=" * 60)
    
    for i, (policy_loss, value_loss, entropy_loss, direction_loss) in enumerate(test_losses):
        # Update statistics
        ppo._update_loss_statistics(policy_loss, value_loss, entropy_loss, direction_loss)
        
        # Test normalization
        policy_tensor = torch.tensor(policy_loss)
        value_tensor = torch.tensor(value_loss)
        entropy_tensor = torch.tensor(entropy_loss)
        direction_tensor = torch.tensor(direction_loss)
        
        results = ppo._normalize_losses(policy_tensor, value_tensor, entropy_tensor, direction_tensor)
        
        print(f"Step {i+1}:")
        print(f"  Original losses: P={policy_loss:.3f}, V={value_loss:.3f}, E={entropy_loss:.3f}, D={direction_loss:.3f}")
        
        if len(results) == 8:  # Normalized losses and adaptive weights
            policy_norm, value_norm, entropy_norm, direction_norm, w_p, w_e, w_v, w_d = results
            print(f"  Normalized losses: P={policy_norm:.3f}, V={value_norm:.3f}, E={entropy_norm:.3f}, D={direction_norm:.3f}")
            print(f"  Adaptive weights: P={w_p:.4f}, E={w_e:.6f}, V={w_v:.4f}, D={w_d:.4f}")
            print(f"  Weight sum: {w_p + w_e + w_v + w_d:.4f}")
            
            # Calculate and show coefficient of variations
            if hasattr(ppo, 'policy_loss_ma') and ppo.policy_loss_ma is not None:
                eps = 1e-8
                policy_cov = max(0.0, (ppo.policy_loss_var ** 0.5) / (abs(ppo.policy_loss_ma) + eps))
                value_cov = max(0.0, (ppo.value_loss_var ** 0.5) / (abs(ppo.value_loss_ma) + eps))
                entropy_cov = max(0.0, (ppo.entropy_loss_var ** 0.5) / (abs(ppo.entropy_loss_ma) + eps))
                direction_cov = max(0.0, (ppo.direction_loss_var ** 0.5) / (abs(ppo.direction_loss_ma) + eps))
                
                print(f"  CoV: P={policy_cov:.3f}, V={value_cov:.3f}, E={entropy_cov:.3f}, D={direction_cov:.3f}")
        else:
            print(f"  Normalization skipped (insufficient data)")
        
        if hasattr(ppo, 'policy_loss_ma') and ppo.policy_loss_ma is not None:
            print(f"  Moving averages: P={ppo.policy_loss_ma:.3f}, V={ppo.value_loss_ma:.3f}, E={ppo.entropy_loss_ma:.3f}, D={ppo.direction_loss_ma:.3f}")
        print()

if __name__ == "__main__":
    test_loss_normalization_fixed()