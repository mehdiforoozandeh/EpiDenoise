#!/usr/bin/env python3
"""
CANDI Prediction Module

This module provides functionality for loading trained CANDI models and running inference
on genomic data. It supports both merged and EIC datasets with configurable metadata
filling and prediction options.

Author: Refactored from old_eval.py
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn

# Import from current codebase
from data import CANDIDataHandler
from model import CANDI, CANDI_UNET
from _utils import NegativeBinomial, Gaussian, DataMasker


class CANDIPredictor:
    """
    CANDI model predictor for loading trained models and running inference.
    
    This class handles model loading from JSON config files and .pt checkpoints,
    data loading using CANDIDataHandler, and running predictions with optional
    latent representation extraction.
    """
    
    def __init__(self, model_dir: str, device: Optional[str] = None, DNA: bool = True):
        """
        Initialize CANDI predictor.
        
        Args:
            model_dir: Path to model directory containing config JSON and .pt checkpoint
            device: Device to use for inference (auto-detect if None)
            DNA: Whether to use DNA sequence input (must be True for current models)
        """
        self.model_dir = Path(model_dir)
        self.DNA = DNA
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize data handler (will be set up when needed)
        self.data_handler = None
        
        # Model and config
        self.model = None
        self.config = None
        
        # Token dictionary for masking
        self.token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}
        
        # Load model and config
        self._load_config()
        self._load_model()
        
        print(f"CANDI Predictor initialized on {self.device}")
        print(f"Model: {self.config.get('unet', False) and 'CANDI_UNET' or 'CANDI'}")
        print(f"Signal dim: {self.config.get('signal_dim', 'unknown')}")
    
    def _load_config(self):
        """Load model configuration from JSON file."""
        config_files = list(self.model_dir.glob("*_config.json"))
        if not config_files:
            raise FileNotFoundError(f"No config JSON file found in {self.model_dir}")
        
        # Use the first (and typically only) config file
        config_path = config_files[0]
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"Loaded config from {config_path}")
    
    def _load_model(self):
        """Load CANDI model from checkpoint."""
        # Find checkpoint file
        checkpoint_files = list(self.model_dir.glob("*.pt"))
        if not checkpoint_files:
            checkpoints_dir = self.model_dir / "checkpoints"
            if checkpoints_dir.exists():
                checkpoint_files = list(checkpoints_dir.glob("*.pt"))
            else:
                raise FileNotFoundError(f"No .pt checkpoint file found in {self.model_dir}")
        
        # Use the first checkpoint file (typically the final model)
        checkpoint_path = checkpoint_files[0]
        
        # Extract model parameters from config
        signal_dim = self.config.get('signal_dim', 35)
        metadata_embedding_dim = signal_dim * 4
        dropout = self.config.get('dropout', 0.1)
        nhead = self.config.get('nhead', 9)
        n_sab_layers = self.config.get('n-sab-layers', 4)
        n_cnn_layers = self.config.get('n-cnn-layers', 3)
        conv_kernel_size = self.config.get('conv-kernel-size', 3)
        pool_size = self.config.get('pool-size', 2)
        context_length = self.config.get('context-length', 1200)
        separate_decoders = self.config.get('separate-decoders', True)
        unet = self.config.get('unet', False)
        pos_enc = self.config.get('pos-enc', 'relative')
        expansion_factor = self.config.get('expansion-factor', 3)
        
        # Get metadata dimensions
        num_sequencing_platforms = self.config.get('num_sequencing_platforms', 10)
        num_runtypes = self.config.get('num_runtypes', 4)
        
        # Create model
        if unet:
            self.model = CANDI_UNET(
                signal_dim=signal_dim,
                metadata_embedding_dim=metadata_embedding_dim,
                conv_kernel_size=conv_kernel_size,
                n_cnn_layers=n_cnn_layers,
                nhead=nhead,
                n_sab_layers=n_sab_layers,
                pool_size=pool_size,
                dropout=dropout,
                context_length=context_length,
                pos_enc=pos_enc,
                expansion_factor=expansion_factor,
                separate_decoders=separate_decoders,
                num_sequencing_platforms=num_sequencing_platforms,
                num_runtypes=num_runtypes
            )
        else:
            self.model = CANDI(
                signal_dim=signal_dim,
                metadata_embedding_dim=metadata_embedding_dim,
                conv_kernel_size=conv_kernel_size,
                n_cnn_layers=n_cnn_layers,
                nhead=nhead,
                n_sab_layers=n_sab_layers,
                pool_size=pool_size,
                dropout=dropout,
                context_length=context_length,
                pos_enc=pos_enc,
                expansion_factor=expansion_factor,
                separate_decoders=separate_decoders,
                num_sequencing_platforms=num_sequencing_platforms,
                num_runtypes=num_runtypes
            )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data_handler(self, data_path: str, dataset_type: str = "merged", 
                          context_length: int = 1200, resolution: int = 25, split: str = "test"):
        """
        Setup data handler for loading genomic data.
        
        Args:
            data_path: Path to dataset directory
            dataset_type: Type of dataset ("merged" or "eic")
            context_length: Context length for genomic windows
            resolution: Genomic resolution in bp
        """
        self.data_handler = CANDIDataHandler(
            base_path=data_path,
            resolution=resolution,
            dataset_type=dataset_type,
            DNA=self.DNA
        )
        
        # Load required data files
        self.data_handler._load_files()
        
        print(f"Filtering navigation for split: {split}...")
        for bios in list(self.data_handler.navigation.keys()):
            if self.data_handler.split_dict[bios] != split:
                if dataset_type == "merged":
                    del self.data_handler.navigation[bios]
        
        print(f"Data handler setup for {dataset_type} dataset at {data_path}")
        print(f"Available experiments: {len(self.data_handler.aliases['experiment_aliases'])}")
    
    def load_data(self, bios_name: str, locus: List, dsf: int = 1, 
                  fill_y_prompt_spec: Optional[Dict] = None) -> Tuple:
        """
        Load data for a specific biosample and genomic locus.
        
        Args:
            bios_name: Name of the biosample
            locus: Genomic locus as [chrom, start, end]
            dsf: Downsampling factor
            fill_y_prompt_spec: Optional dictionary specifying custom metadata values
            
        Returns:
            Tuple of (X, Y, P, seq, mX, mY, avX, avY) for DNA models
            Tuple of (X, Y, P, mX, mY, avX, avY) for non-DNA models
        """
        if self.data_handler is None:
            raise RuntimeError("Data handler not setup. Call setup_data_handler() first.")
        
        print(f"Loading data for {bios_name} at {locus}")
        
        # Load count data
        temp_x, temp_mx = self.data_handler.load_bios_Counts(bios_name, locus, dsf)
        X, mX, avX = self.data_handler.make_bios_tensor_Counts(temp_x, temp_mx)
        del temp_x, temp_mx
        
        # Load target data
        temp_y, temp_my = self.data_handler.load_bios_Counts(bios_name, locus, 1)
        Y, mY, avY = self.data_handler.make_bios_tensor_Counts(temp_y, temp_my)
        del temp_y, temp_my
        
        # Fill in Y prompt metadata
        if fill_y_prompt_spec is not None:
            # Use custom metadata specification
            mY = self.data_handler.fill_in_prompt_manual(mY, fill_y_prompt_spec, overwrite=True)
        else:
            # Use median values (sample=False)
            mY = self.data_handler.fill_in_prompt(mY, missing_value=-1, sample=False)
        
        # Load p-value data
        temp_p = self.data_handler.load_bios_BW(bios_name, locus)
        P, avlP = self.data_handler.make_bios_tensor_BW(temp_p)
        del temp_p
        
        # Verify availability consistency
        assert (avlP == avY).all(), "Availability masks for P and Y do not match"
        
        # Load control data
        try:
            temp_control_data, temp_control_metadata = self.data_handler.load_bios_Control(bios_name, locus, dsf)
            control_data, control_meta, control_avail = self.data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
            del temp_control_data, temp_control_metadata
        except Exception as e:
            print(f"Warning: Failed to load control data for {bios_name}: {e}")
            print("Using missing values for control data")
            # Create control data with missing values
            L = X.shape[0]
            control_data = torch.full((L, 1), -1.0)  # missing_value
            control_meta = torch.full((4, 1), -1.0)  # missing_value
            control_avail = torch.zeros(1)  # not available
        
        # Concatenate control data to input data (same as in training)
        X = torch.cat([X, control_data], dim=1)      # (L, F+1)
        mX = torch.cat([mX, control_meta], dim=1)    # (4, F+1)
        avX = torch.cat([avX, control_avail], dim=0) # (F+1,)
        
        # Load DNA sequence if needed
        seq = None
        if self.DNA:
            seq = self.data_handler._dna_to_onehot(
                self.data_handler._get_DNA_sequence(locus[0], locus[1], locus[2])
            )
        
        # Reshape data to context windows
        context_length = self.config.get('context-length', 1200)
        num_rows = (X.shape[0] // context_length) * context_length
        X, Y, P = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :]
        
        if self.DNA:
            seq = seq[:num_rows * self.data_handler.resolution, :]
        
        # Reshape to context windows
        X = X.view(-1, context_length, X.shape[-1])
        Y = Y.view(-1, context_length, Y.shape[-1])
        P = P.view(-1, context_length, P.shape[-1])
        
        if self.DNA:
            seq = seq.view(-1, context_length * self.data_handler.resolution, seq.shape[-1])
        
        # Expand metadata and availability to match batch dimension
        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)
        
        if self.DNA:
            return X, Y, P, seq, mX, mY, avX, avY
        else:
            return X, Y, P, mX, mY, avX, avY
    
    def predict(self, X: torch.Tensor, mX: torch.Tensor, mY: torch.Tensor, 
                avail: torch.Tensor, seq: Optional[torch.Tensor] = None,
                imp_target: List[int] = []) -> Tuple[torch.Tensor, ...]:
        """
        Run inference on input data.
        
        Args:
            X: Input count data [B, L, F]
            mX: Input metadata [B, 4, F]
            mY: Target metadata [B, 4, F]
            avail: Availability mask [B, F]
            seq: DNA sequence [B, L*25, 4] (required if DNA=True)
            imp_target: List of feature indices to treat as imputation targets
            
        Returns:
            Tuple of (output_n, output_p, output_mu, output_var, output_peak)
        """
        batch_size = self.config.get('batch_size', 25)
        
        # Initialize output tensors - model outputs only for original features (without control)
        # Control is only used as input, not predicted as output
        original_feature_dim = X.shape[-1] - 1  # Subtract 1 for control
        n = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        p = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        mu = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        var = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        peak = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        
        # Process in batches
        for i in range(0, len(X), batch_size):
            # Get batch
            x_batch = X[i:i + batch_size]
            mX_batch = mX[i:i + batch_size]
            mY_batch = mY[i:i + batch_size]
            avail_batch = avail[i:i + batch_size]
            
            if self.DNA:
                seq_batch = seq[i:i + batch_size]
            
            with torch.no_grad():
                # Clone and prepare batch
                x_batch = x_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()
                avail_batch = avail_batch.clone()
                
                # Apply masking - use float tokens to match model expectations
                x_batch_missing = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing = (mX_batch == self.token_dict["missing_mask"])
                avail_batch_missing = (avail_batch == 0)
                
                x_batch[x_batch_missing] = float(self.token_dict["cloze_mask"])
                mX_batch[mX_batch_missing] = float(self.token_dict["cloze_mask"])
                
                # Apply imputation targets
                if len(imp_target) > 0:
                    x_batch[:, :, imp_target] = float(self.token_dict["cloze_mask"])
                    mX_batch[:, :, imp_target] = float(self.token_dict["cloze_mask"])
                    avail_batch[:, imp_target] = 0
                
                # Move to device first
                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)
                
                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    # Run model forward pass - convert to float for model input
                    outputs_p, outputs_n, outputs_mu, outputs_var, outputs_peak = self.model(
                        x_batch.float(), seq_batch, mX_batch.float(), mY_batch
                    )
                else:
                    outputs_p, outputs_n, outputs_mu, outputs_var, outputs_peak = self.model(
                        x_batch.float(), mX_batch.float(), mY_batch, avail_batch
                    )
            
            # Store predictions
            batch_end = min(i + batch_size, len(X))
            n[i:batch_end] = outputs_n.cpu()
            p[i:batch_end] = outputs_p.cpu()
            mu[i:batch_end] = outputs_mu.cpu()
            var[i:batch_end] = outputs_var.cpu()
            peak[i:batch_end] = outputs_peak.cpu()
            
            # Clean up
            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var, outputs_peak
            if self.DNA:
                del seq_batch
        
        return n, p, mu, var, peak
    
    def get_latent_z(self, X: torch.Tensor, mX: torch.Tensor, mY: torch.Tensor,
                     avail: torch.Tensor, seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract latent representations from the model encoder.
        
        Args:
            X: Input count data [B, L, F]
            mX: Input metadata [B, 4, F]
            mY: Target metadata [B, 4, F]
            avail: Availability mask [B, F]
            seq: DNA sequence [B, L*25, 4] (required if DNA=True)
            
        Returns:
            Latent representations Z [B, L, D]
        """
        batch_size = self.config.get('batch_size', 25)
        Z_all = []
        
        for i in range(0, len(X), batch_size):
            # Get batch
            x_batch = X[i:i + batch_size]
            mX_batch = mX[i:i + batch_size]
            mY_batch = mY[i:i + batch_size]
            avail_batch = avail[i:i + batch_size]
            
            if self.DNA:
                seq_batch = seq[i:i + batch_size]
            
            with torch.no_grad():
                # Clone and prepare batch
                x_batch = x_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()
                avail_batch = avail_batch.clone()
                
                # Apply masking
                x_batch_missing = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing = (mX_batch == self.token_dict["missing_mask"])
                
                x_batch[x_batch_missing] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing] = self.token_dict["cloze_mask"]
                
                # Move to device
                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)
                
                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    # Get latent representation
                    Z = self.model.encode(x_batch.float(), seq_batch, mX_batch)
                else:
                    Z = self.model.encode(x_batch.float(), mX_batch)
            
            Z_all.append(Z.cpu())
            
            # Clean up
            del x_batch, mX_batch, mY_batch, avail_batch, Z
            if self.DNA:
                del seq_batch
        
        return torch.cat(Z_all, dim=0)
    
    def predict_biosample(self, bios_name: str, x_dsf: int = 1, 
                         fill_y_prompt_spec: Optional[Dict] = None,
                         locus: Optional[List] = None,
                         get_latent_z: bool = False,
                         return_raw_predictions: bool = False) -> Dict[str, Any]:
        """
        High-level method to predict for an entire biosample.
        
        Args:
            bios_name: Name of the biosample
            x_dsf: Downsampling factor
            fill_y_prompt_spec: Optional custom metadata specification
            locus: Genomic locus (default: chr21)
            get_latent_z: Whether to extract latent representations
            return_raw_predictions: Whether to return raw prediction tensors
            
        Returns:
            Dictionary with organized predictions by biosample and experiment
        """
        if locus is None:
            # Default to chr21
            locus = ["chr21", 0, self.data_handler.chr_sizes["chr21"]]
        
        # Load data
        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_data(
                bios_name, locus, x_dsf, fill_y_prompt_spec
            )
        else:
            X, Y, P, mX, mY, avX, avY = self.load_data(
                bios_name, locus, x_dsf, fill_y_prompt_spec
            )
            seq = None
        
        print(f"Loaded data: {X.shape}, {Y.shape}, {P.shape}")
        
        # Get available experiments
        available_indices = torch.where(avX[0, :] == 1)[0].tolist()
        if 35 in available_indices: #remove control
            available_indices.remove(35)


        experiment_names = list(self.data_handler.aliases['experiment_aliases'].keys())
        
        # Initialize results structure
        results = {
            bios_name: {}
        }
        
        # Run leave-one-out predictions for imputation
        print(f"Running leave-one-out predictions for {len(available_indices)} experiments...")
        
        for leave_one_out in available_indices:
            exp_name = experiment_names[leave_one_out]
            print(f"  Predicting {exp_name} (index {leave_one_out})")
            
            # Run prediction
            if self.DNA:
                n, p, mu, var, peak = self.predict(X, mX, mY, avX, seq, [leave_one_out])
            else:
                n, p, mu, var, peak = self.predict(X, mX, mY, avX, None, [leave_one_out])
            
            p = p.view((p.shape[0] * p.shape[1]), p.shape[-1])
            n = n.view((n.shape[0] * n.shape[1]), n.shape[-1])
            mu = mu.view((mu.shape[0] * mu.shape[1]), mu.shape[-1])
            var = var.view((var.shape[0] * var.shape[1]), var.shape[-1])
            peak = peak.view((peak.shape[0] * peak.shape[1]), peak.shape[-1])
            
            # Create distributions
            count_dist = NegativeBinomial(p[:, leave_one_out], n[:, leave_one_out])
            pval_dist = Gaussian(mu[:, leave_one_out], var[:, leave_one_out])
            
            # Store results
            results[bios_name][exp_name] = {
                'type': 'imputed',
                'experiment_name': exp_name,
                'count_dist': count_dist,
                'count_params': {'p': p[:, leave_one_out], 'n': n[:, leave_one_out]},
                'pval_dist': pval_dist,
                'pval_params': {'mu': mu[:, leave_one_out], 'var': var[:, leave_one_out]},
                'peak_scores': peak[:, leave_one_out]
            }
            
            # Add raw predictions if requested
            if return_raw_predictions:
                results[bios_name][exp_name]['raw_predictions'] = {
                    'output_p': p[:, leave_one_out],
                    'output_n': n[:, leave_one_out],
                    'output_mu': mu[:, leave_one_out],
                    'output_var': var[:, leave_one_out],
                    'output_peak': peak[:, leave_one_out]
                }
        
        # Run upsampling predictions (predict all available experiments)
        print("Running upsampling predictions...")
        
        if self.DNA:
            n_ups, p_ups, mu_ups, var_ups, peak_ups = self.predict(X, mX, mY, avX, seq, [])
        else:
            n_ups, p_ups, mu_ups, var_ups, peak_ups = self.predict(X, mX, mY, avX, None, [])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])
        mu_ups = mu_ups.view((mu_ups.shape[0] * mu_ups.shape[1]), mu_ups.shape[-1])
        var_ups = var_ups.view((var_ups.shape[0] * var_ups.shape[1]), var_ups.shape[-1])
        peak_ups = peak_ups.view((peak_ups.shape[0] * peak_ups.shape[1]), peak_ups.shape[-1])
        
        # Store upsampling results for available experiments (denoised)
        for exp_idx in available_indices:
            exp_name = experiment_names[exp_idx]
            
            # Create distributions
            count_dist_ups = NegativeBinomial(p_ups[:, exp_idx], n_ups[:, exp_idx])
            pval_dist_ups = Gaussian(mu_ups[:, exp_idx], var_ups[:, exp_idx])
            
            # Add upsampling results
            results[bios_name][f"{exp_name}_upsampled"] = {
                'type': 'denoised',
                'experiment_name': exp_name,
                'count_dist': count_dist_ups,
                'count_params': {'p': p_ups[:, exp_idx], 'n': n_ups[:, exp_idx]},
                'pval_dist': pval_dist_ups,
                'pval_params': {'mu': mu_ups[:, exp_idx], 'var': var_ups[:, exp_idx]},
                'peak_scores': peak_ups[:, exp_idx]
            }
            
            # Add raw predictions if requested
            if return_raw_predictions:
                results[bios_name][f"{exp_name}_upsampled"]['raw_predictions'] = {
                    'output_p': p_ups[:, exp_idx],
                    'output_n': n_ups[:, exp_idx],
                    'output_mu': mu_ups[:, exp_idx],
                    'output_var': var_ups[:, exp_idx],
                    'output_peak': peak_ups[:, exp_idx]
                }
        
        # Optionally store predictions for non-available experiments (imputed from upsampling pass)
        all_experiment_indices = list(range(len(experiment_names)))
        non_available_indices = [idx for idx in all_experiment_indices if idx not in available_indices]
        
        if non_available_indices:
            print(f"Storing predictions for {len(non_available_indices)} non-available experiments...")
            for exp_idx in non_available_indices:
                exp_name = experiment_names[exp_idx]
                
                # Create distributions
                count_dist_imp = NegativeBinomial(p_ups[:, exp_idx], n_ups[:, exp_idx])
                pval_dist_imp = Gaussian(mu_ups[:, exp_idx], var_ups[:, exp_idx])
                
                # Add imputation results (from upsampling pass)
                results[bios_name][f"{exp_name}_imputed_from_upsampling"] = {
                    'type': 'imputed',
                    'experiment_name': exp_name,
                    'count_dist': count_dist_imp,
                    'count_params': {'p': p_ups[:, exp_idx], 'n': n_ups[:, exp_idx]},
                    'pval_dist': pval_dist_imp,
                    'pval_params': {'mu': mu_ups[:, exp_idx], 'var': var_ups[:, exp_idx]},
                    'peak_scores': peak_ups[:, exp_idx]
                }
                
                # Add raw predictions if requested
                if return_raw_predictions:
                    results[bios_name][f"{exp_name}_imputed_from_upsampling"]['raw_predictions'] = {
                        'output_p': p_ups[:, exp_idx],
                        'output_n': n_ups[:, exp_idx],
                        'output_mu': mu_ups[:, exp_idx],
                        'output_var': var_ups[:, exp_idx],
                        'output_peak': peak_ups[:, exp_idx]
                    }
        
        # Extract latent representations if requested
        if get_latent_z:
            print("Extracting latent representations...")
            if self.DNA:
                Z = self.get_latent_z(X, mX, mY, avX, seq)
            else:
                Z = self.get_latent_z(X, mX, mY, avX, None)
            
            results['latent_z'] = Z
        
        return results


def main():
    """CLI interface for CANDI prediction."""
    parser = argparse.ArgumentParser(
        description="CANDI Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction on chr21
  python pred.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                 --data-path /path/to/DATA_CANDI_MERGED \\
                 --bios-name GM12878

  # Prediction with custom metadata
  python pred.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                 --data-path /path/to/DATA_CANDI_MERGED \\
                 --bios-name GM12878 \\
                 --y-prompt-spec y_prompt.json \\
                 --output predictions.pkl

  # Extract latent representations
  python pred.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                 --data-path /path/to/DATA_CANDI_MERGED \\
                 --bios-name GM12878 \\
                 --get-latent-z \\
                 --output results.pkl
        """
    )
    
    # Required arguments
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to model directory containing config JSON and .pt checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--bios-name', type=str, required=True,
                       help='Name of biosample to predict')
    
    # Optional arguments
    parser.add_argument('--dataset', type=str, default='merged', choices=['merged', 'eic'],
                       help='Dataset type (default: merged)')
    parser.add_argument('--dsf', type=int, default=1,
                       help='Downsampling factor (default: 1)')
    parser.add_argument('--locus', type=str, nargs=3, default=['chr21', '0', '46709983'],
                       help='Genomic locus as chrom start end (default: chr21 0 46709983)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto-detect if not specified)')
    
    # Metadata specification
    parser.add_argument('--y-prompt-spec', type=str, default=None,
                       help='JSON file with custom metadata specification')
    
    # Output options
    parser.add_argument('--output', type=str, default='predictions.pkl',
                       help='Output file path (default: predictions.pkl)')
    parser.add_argument('--get-latent-z', action='store_true',
                       help='Extract latent representations')
    parser.add_argument('--return-raw-predictions', action='store_true',
                       help='Include raw prediction tensors in output')
    
    args = parser.parse_args()
    
    # Parse locus
    locus = [args.locus[0], int(args.locus[1]), int(args.locus[2])]
    
    # Load Y prompt specification if provided
    fill_y_prompt_spec = None
    if args.y_prompt_spec:
        with open(args.y_prompt_spec, 'r') as f:
            fill_y_prompt_spec = json.load(f)
        print(f"Loaded Y prompt specification from {args.y_prompt_spec}")
    
    try:
        # Initialize predictor
        predictor = CANDIPredictor(args.model_dir, args.device, DNA=True)
        
        # Setup data handler
        predictor.setup_data_handler(args.data_path, args.dataset)
        
        # Run prediction
        results = predictor.predict_biosample(
            bios_name=args.bios_name,
            x_dsf=args.dsf,
            fill_y_prompt_spec=fill_y_prompt_spec,
            locus=locus,
            get_latent_z=args.get_latent_z,
            return_raw_predictions=args.return_raw_predictions
        )
        
        # Save results
        with open(args.output, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Predictions saved to {args.output}")
        
        # Print summary
        bios_name = args.bios_name
        if bios_name in results:
            n_experiments = len([k for k in results[bios_name].keys() if not k.endswith('_upsampled')])
            n_upsampled = len([k for k in results[bios_name].keys() if k.endswith('_upsampled')])
            print(f"Results summary:")
            print(f"  Biosample: {bios_name}")
            print(f"  Imputed experiments: {n_experiments}")
            print(f"  Denoised experiments: {n_upsampled}")
            if 'latent_z' in results:
                print(f"  Latent representations: {results['latent_z'].shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
