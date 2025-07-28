#!/usr/bin/env python3
"""
Test script for loss normalization functionality
"""

import torch
import numpy as np
from model_with_attn import CustomPPO

def test_loss_normalization():
    """Test the loss normalization functionality"""
    
    # Create a dummy CustomPPO instance for testing
    class DummyPPO:
        def __init__(self):
            self.device = torch.device("cpu")
            
    # Initialize the loss normalization components
    ppo = CustomPPO(
        "MlpPolicy",
        env=None,  # We'll mock this
        learning_rate=3e-4,
        direction_weight=0.3,
        loss_normalization=True,
        norm_decay=0.99
    )
    
    # Mock some loss values to test normalization
    test_losses = [
        (1.0, 0.5, 0.1, 2.0),  # policy, value, entropy, direction
        (0.8, 0.4, 0.08, 1.5),
        (1.2, 0.6, 0.12, 2.5),
        (0.9, 0.45, 0.09, 1.8),
        (1.1, 0.55, 0.11, 2.2),
    ]
    
    print("Testing loss normalization...")
    print("=" * 50)
    
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
            print(f"  Adaptive weights: P={w_p:.3f}, E={w_e:.3f}, V={w_v:.3f}, D={w_d:.3f}")
            print(f"  Weight sum: {w_p + w_e + w_v + w_d:.3f}")
        else:
            print(f"  Normalization skipped (insufficient data)")
        
        if hasattr(ppo, 'policy_loss_ma'):
            print(f"  Moving averages: P={ppo.policy_loss_ma:.3f}, V={ppo.value_loss_ma:.3f}, E={ppo.entropy_loss_ma:.3f}, D={ppo.direction_loss_ma:.3f}")
        print()

if __name__ == "__main__":
    test_loss_normalization()