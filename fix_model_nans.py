#!/usr/bin/env python3
"""
Fix NaN values in BatchNorm running statistics in a saved model checkpoint.
"""
import torch
import argparse
import shutil
from pathlib import Path

def fix_batchnorm_nans(model_state_dict):
    """Replace NaN values in BatchNorm running statistics with valid values."""
    fixed_count = 0
    
    for name, param in model_state_dict.items():
        if 'running_mean' in name or 'running_var' in name:
            if torch.isnan(param).any():
                num_nans = torch.isnan(param).sum().item()
                print(f"Found {num_nans} NaNs in {name}")
                
                if 'running_mean' in name:
                    # Replace NaN with 0 (neutral for mean)
                    param[torch.isnan(param)] = 0.0
                    print(f"  Replaced with 0.0")
                elif 'running_var' in name:
                    # Replace NaN with 1.0 (neutral for variance)
                    param[torch.isnan(param)] = 1.0
                    print(f"  Replaced with 1.0")
                
                fixed_count += num_nans
    
    return fixed_count

def main():
    parser = argparse.ArgumentParser(description='Fix NaN values in model BatchNorm layers')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup of original model')
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found")
        return
    
    # Create backup if requested
    if args.backup:
        backup_path = model_path.with_suffix('.pt.backup')
        print(f"Creating backup: {backup_path}")
        shutil.copy(model_path, backup_path)
    
    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Fix NaNs
    print("\nScanning for NaN values...")
    fixed_count = fix_batchnorm_nans(checkpoint)
    
    if fixed_count > 0:
        print(f"\nFixed {fixed_count} NaN value(s)")
        print(f"Saving corrected model to {model_path}")
        torch.save(checkpoint, model_path)
        print("Done!")
    else:
        print("\nNo NaN values found in BatchNorm layers")

if __name__ == '__main__':
    main()



