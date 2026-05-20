# -*- coding: utf-8 -*-
"""
Test script to verify incremental learning modifications for G-NCDM.

Usage:
    cd GNCDM
    python experiments/test_incremental.py
"""
import os
import sys

# Add GNCDM directory to path for local imports
gncdm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gncdm_dir)

import numpy as np
import pandas as pd
import torch

def test_all_components():
    print("=" * 60)
    print("Testing Incremental Learning Components for G-NCDM")
    print("=" * 60)
    
    # Test 1: Import all modules
    print("\n[Test 1] Importing modules...")
    try:
        from core.model import GNCDM, truncated_normal_init
        from core.train import (
            generate_cognitive_biased_mnar_mask,
            apply_mnar_mask_to_data,
            AsymmetricHybridDataLoader,
            calculate_tmd,
            LinearWarmupScheduler,
            eval_incremental,
            train_incremental
        )
        from incremental.loss import TopologyAwareDecoupledLoss, IncrementalDecoupledLoss
        print("[OK] All modules imported successfully!")
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Create dummy data
    print("\n[Test 2] Creating dummy data...")
    try:
        n_user = 100
        n_item = 50
        n_know = 10
        
        # Create dummy Q-matrix
        Q_mat = np.random.randint(0, 2, size=(n_item, n_know))
        
        # Create dummy log data
        np.random.seed(42)
        user_ids = np.random.randint(0, n_user, 500)
        item_ids = np.random.randint(0, n_item, 500)
        scores = np.random.randint(0, 2, 500)
        
        train_data = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'score': scores
        })
        
        print("[OK] Dummy data created successfully!")
    except Exception as e:
        print(f"[FAIL] Data creation failed: {e}")
        return False
    
    # Test 3: Initialize original model
    print("\n[Test 3] Initializing original G-NCDM model...")
    try:
        device = torch.device('cpu')
        model_old = GNCDM(
            n_user=n_user,
            n_item=n_item,
            n_know=n_know,
            user_dim=32,
            item_dim=32,
            Q_mat=Q_mat,
            device=device,
            alpha=0.5,
            monotonicity_assumption=True
        )
        print("[OK] Original model initialized successfully!")
    except Exception as e:
        print(f"[FAIL] Model initialization failed: {e}")
        return False
    
    # Test 4: Test topology expansion
    print("\n[Test 4] Testing topology expansion...")
    try:
        # Create expanded Q-matrix
        delta_M = 10  # 10 new items
        delta_K = 5   # 5 new knowledge concepts
        
        Q_expanded = np.zeros((n_item + delta_M, n_know + delta_K))
        Q_expanded[:n_item, :n_know] = Q_mat
        Q_expanded[n_item:, n_know:] = np.random.randint(0, 2, size=(delta_M, delta_K))
        
        # Create dynamic model (copy of old model)
        model_dynamic = GNCDM(
            n_user=n_user,
            n_item=n_item,
            n_know=n_know,
            user_dim=32,
            item_dim=32,
            Q_mat=Q_mat,
            device=device,
            alpha=0.5,
            monotonicity_assumption=True
        )
        
        # Expand topology
        model_dynamic.expand_topology(delta_M, delta_K, Q_expanded)
        
        # Verify expansion
        assert model_dynamic.n_item == n_item + delta_M, "Item count mismatch"
        assert model_dynamic.n_know == n_know + delta_K, "Knowledge count mismatch"
        assert model_dynamic.is_expanded == True, "Expansion flag not set"
        
        # Check that old parameters are frozen
        old_params_frozen = all(not p.requires_grad for name, p in model_dynamic.named_parameters() 
                               if 'f_nn.' in name or 'g_nn.' in name)
        assert old_params_frozen, "Old parameters not frozen"
        
        # Check that new parameters are trainable
        new_params_trainable = all(p.requires_grad for name, p in model_dynamic.named_parameters() 
                                   if 'f_nn_new.' in name or 'g_nn_new.' in name)
        assert new_params_trainable, "New parameters not trainable"
        
        print("[OK] Topology expansion successful!")
        print(f"  - Items: {n_item} -> {model_dynamic.n_item}")
        print(f"  - Knowledge: {n_know} -> {model_dynamic.n_know}")
    except Exception as e:
        print(f"[FAIL] Topology expansion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4b: Test LoRA topology expansion
    print("\n[Test 4b] Testing LoRA topology expansion...")
    try:
        # Create LoRA model (copy of old model)
        model_lora = GNCDM(
            n_user=n_user,
            n_item=n_item,
            n_know=n_know,
            user_dim=32,
            item_dim=32,
            Q_mat=Q_mat,
            device=device,
            alpha=0.5,
            monotonicity_assumption=True
        )
        
        # Expand topology with LoRA
        model_lora.expand_topology_lora(delta_M, delta_K, Q_expanded, M_old=n_item, rank=4)
        
        # Verify expansion
        assert model_lora.n_item == n_item + delta_M, "LoRA Item count mismatch"
        assert model_lora.n_know == n_know + delta_K, "LoRA Knowledge count mismatch"
        assert model_lora.is_expanded == True, "LoRA Expansion flag not set"
        assert model_lora.use_lora == True, "LoRA flag not set"
        
        # Check that old parameters are frozen
        old_params_frozen_lora = all(not p.requires_grad for name, p in model_lora.named_parameters() 
                                     if 'f_nn.' in name or 'g_nn.' in name)
        assert old_params_frozen_lora, "LoRA Old parameters not frozen"
        
        # Check that LoRA parameters are trainable
        lora_params_trainable = all(p.requires_grad for name, p in model_lora.named_parameters() 
                                    if 'A_new_' in name or 'B_new_' in name or '_agg' in name)
        assert lora_params_trainable, "LoRA parameters not trainable"
        
        # Check LoRA parameters exist
        assert hasattr(model_lora, 'A_new_g'), "Missing A_new_g"
        assert hasattr(model_lora, 'B_new_g'), "Missing B_new_g"
        assert hasattr(model_lora, 'A_new_f'), "Missing A_new_f"
        assert hasattr(model_lora, 'B_new_f'), "Missing B_new_f"
        
        # Check B matrices are zero-initialized
        assert torch.all(model_lora.B_new_g == 0), "B_new_g not zero-initialized"
        assert torch.all(model_lora.B_new_f == 0), "B_new_f not zero-initialized"
        
        print("[OK] LoRA topology expansion successful!")
        print(f"  - Items: {n_item} -> {model_lora.n_item}")
        print(f"  - Knowledge: {n_know} -> {model_lora.n_know}")
        print(f"  - LoRA rank: {model_lora.lora_rank}")
    except Exception as e:
        print(f"[FAIL] LoRA topology expansion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test MNAR masking
    print("\n[Test 5] Testing MNAR masking...")
    try:
        Q_new = np.random.randint(0, 2, size=(10, 5))
        mask_prob, mask = generate_cognitive_biased_mnar_mask(Q_new, gamma=1.0, tau=0.5)
        
        assert mask_prob.shape == (10,), "Mask prob shape mismatch"
        assert mask.shape == (10,), "Mask shape mismatch"
        assert np.all(mask_prob >= 0) and np.all(mask_prob <= 1), "Mask prob out of range"
        
        print("[OK] MNAR masking test passed!")
    except Exception as e:
        print(f"[FAIL] MNAR masking failed: {e}")
        return False
    
    # Test 6: Test TMD calculation
    print("\n[Test 6] Testing TMD calculation...")
    try:
        theta_old = torch.randn(10, n_know)
        theta_new = torch.randn(10, n_know + delta_K)
        
        tmd = calculate_tmd(theta_old, theta_new, n_know)
        
        assert isinstance(tmd, float), "TMD should be float"
        assert tmd >= 0, "TMD should be non-negative"
        
        print(f"[OK] TMD calculation passed! TMD = {tmd:.4f}")
    except Exception as e:
        print(f"[FAIL] TMD calculation failed: {e}")
        return False
    
    # Test 7: Test Linear Warmup Scheduler
    print("\n[Test 7] Testing Linear Warmup Scheduler...")
    try:
        active_params = [p for p in model_dynamic.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(active_params, lr=0.001)
        scheduler = LinearWarmupScheduler(optimizer, warmup_epochs=5, warmup_factor=0.01)
        
        # Test warmup
        lrs = []
        for epoch in range(10):
            scheduler.step(epoch)
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
        
        # Check that learning rate increases during warmup
        assert lrs[0] == 0.001 * 0.01, "Initial LR incorrect"
        assert lrs[4] == 0.001 * (0.01 + 0.99 * 4/5), "LR at epoch 4 incorrect"
        assert lrs[5] == 0.001, "Final warmup LR should reach base LR at epoch 5"
        
        print("[OK] Linear Warmup Scheduler test passed!")
    except Exception as e:
        print(f"[FAIL] Linear Warmup Scheduler failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 8: Test Decoupled Loss
    print("\n[Test 8] Testing Decoupled Loss...")
    try:
        loss_fn = TopologyAwareDecoupledLoss(
            model_old=model_old,
            model_dynamic=model_dynamic,
            original_know_dim=n_know,
            device=device
        )
        
        # Create dummy batch
        batch_size = 8
        user_log = torch.randn(batch_size, model_dynamic.n_item).to(device)
        # item_log should be (batch_size, n_user) format based on IDCDataset
        item_log = torch.randn(batch_size, n_user).to(device)
        user_id = torch.randint(0, n_user, (batch_size,)).to(device)
        item_id = torch.randint(0, model_dynamic.n_item, (batch_size,)).to(device)
        score = torch.randint(0, 2, (batch_size,)).float().to(device)
        is_new = torch.tensor([False]*5 + [True]*3).to(device)  # 5 old, 3 new
        
        # Ensure old samples only use old items
        item_id[:5] = torch.randint(0, n_item, (5,)).to(device)
        
        # Calculate loss
        loss, components = loss_fn(
            user_log=user_log,
            item_log=item_log,
            user_id=user_id,
            item_id=item_id,
            score=score,
            is_new=is_new,
            epoch=0,
            total_epochs=10,
            V_old=Q_mat.sum(),
            V_new=Q_expanded[n_item:, n_know:].sum()
        )
        
        assert loss.requires_grad, "Loss should require grad"
        assert 'L_old' in components, "L_old missing from components"
        assert 'L_new' in components, "L_new missing from components"
        assert 'alpha' in components, "alpha missing from components"
        
        print(f"[OK] Decoupled Loss test passed!")
        print(f"  - Total Loss: {loss.item():.6f}")
        print(f"  - L_old: {components['L_old']:.6f}")
        print(f"  - L_new: {components['L_new']:.6f}")
        print(f"  - alpha: {components['alpha']:.4f}")
    except Exception as e:
        print(f"[FAIL] Decoupled Loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 9: Test forward pass with expanded model
    print("\n[Test 9] Testing forward pass with expanded model...")
    try:
        model_dynamic.eval()
        with torch.no_grad():
            user_log = torch.randn(2, model_dynamic.n_item).to(device)
            # item_log should be (batch_size, n_user) format
            item_log = torch.randn(2, n_user).to(device)
            user_id = torch.tensor([0, 1]).to(device)
            item_id = torch.tensor([0, n_item + 1]).to(device)  # One old, one new item
            
            pred = model_dynamic(user_log, item_log, user_id, item_id)
            
            assert pred.shape == (2, 1), "Output shape mismatch"
            assert torch.all(pred >= 0) and torch.all(pred <= 1), "Output out of [0,1] range"
            
            print("[OK] Forward pass test passed!")
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_all_components()
    exit(0 if success else 1)