#!/usr/bin/env python3
"""
Single Assay Prediction Module

A focused, practical CLI tool for single-assay predictions with impute/denoise tasks.
Positioned between eval.py's comprehensive evaluation and pred.py's full biosample prediction.
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from pred import CANDIPredictor
from data import CANDIDataHandler
from _utils import NegativeBinomial, Gaussian


class SingleAssayPredictor:
    """
    Single-assay predictor for focused imputation or denoising tasks.
    
    This class provides a simplified interface for predicting a single assay
    with either impute or denoise task, supporting both merged and eic datasets.
    """
    
    def __init__(self, model_dir: str, data_path: str, dataset_type: str = "merged",
                 context_length: int = 1200, resolution: int = 25, DNA: bool = True):
        """
        Initialize single-assay predictor.
        
        Args:
            model_dir: Path to model directory containing config JSON and checkpoint
            data_path: Path to dataset directory
            dataset_type: Type of dataset ("merged" or "eic")
            context_length: Context length for genomic windows
            resolution: Genomic resolution in bp
            DNA: Whether to use DNA sequence input
        """
        self.model_dir = model_dir
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.resolution = resolution
        self.DNA = DNA
        
        # Initialize predictor
        self.predictor = CANDIPredictor(model_dir, DNA=DNA)
        self.context_length = self.predictor.context_length
        
        # Setup data handler
        self.predictor.setup_data_handler(
            data_path, dataset_type, self.context_length, resolution, split="test"
        )
        
        # Get experiment names
        self.expnames = list(self.predictor.data_handler.aliases["experiment_aliases"].keys())
        
        # Load chromosome sizes
        self.chr_sizes = {}
        chr_sizes_file = "data/hg38.chrom.sizes"
        main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        if os.path.exists(chr_sizes_file):
            with open(chr_sizes_file, 'r') as f:
                for line in f:
                    chr_name, chr_size = line.strip().split('\t')
                    if chr_name in main_chrs:
                        self.chr_sizes[chr_name] = int(chr_size)
        else:
            self.chr_sizes = {"chr21": 46709983}
        
        print(f"SingleAssayPredictor initialized for {dataset_type} dataset")
        print(f"Available experiments: {len(self.expnames)}")
    
    def _check_assay_availability(self, assay_name: str, bios_name: str, 
                                  locus: List, dsf: int = 1) -> Tuple[bool, bool]:
        """
        Check if assay exists in input (X) and target (Y) data.
        
        Args:
            assay_name: Name of assay to check
            bios_name: Name of biosample
            locus: Genomic locus
            dsf: Downsampling factor
            
        Returns:
            Tuple of (available_in_input, available_in_target)
        """
        if assay_name not in self.expnames:
            return False, False
        
        assay_idx = self.expnames.index(assay_name)
        
        # Load input data
        temp_x, temp_mx = self.predictor.data_handler.load_bios_Counts(bios_name, locus, dsf)
        X, mX, avX = self.predictor.data_handler.make_bios_tensor_Counts(temp_x, temp_mx)
        
        # Load target data
        temp_y, temp_my = self.predictor.data_handler.load_bios_Counts(bios_name, locus, 1)
        Y, mY, avY = self.predictor.data_handler.make_bios_tensor_Counts(temp_y, temp_my)
        
        # Check availability
        # avX and avY are 1D tensors with shape (num_assays,) or (num_assays+1,) if control is added
        # Control is added at the end, so assay_idx should be valid if < num_assays
        available_in_input = False
        available_in_target = False
        
        # Check input availability (avX may have control appended)
        if avX.ndim == 1:
            if assay_idx < avX.shape[0]:
                available_in_input = avX[assay_idx].item() == 1
        else:
            # 2D case (batch dimension)
            if assay_idx < avX.shape[-1]:
                available_in_input = avX[0, assay_idx].item() == 1
        
        # Check target availability (avY doesn't have control)
        if avY.ndim == 1:
            if assay_idx < avY.shape[0]:
                available_in_target = avY[assay_idx].item() == 1
        else:
            # 2D case (batch dimension)
            if assay_idx < avY.shape[-1]:
                available_in_target = avY[0, assay_idx].item() == 1
        
        return available_in_input, available_in_target
    
    def _determine_imp_target(self, assay_name: str, bios_name: str, 
                              task: str, locus: List, dsf: int = 1) -> Tuple[List[int], Dict[str, Any]]:
        """
        Determine imp_target list based on task and dataset type.
        
        Args:
            assay_name: Name of assay to predict
            bios_name: Name of biosample
            task: Task type ("impute" or "denoise")
            locus: Genomic locus
            dsf: Downsampling factor
            
        Returns:
            Tuple of (imp_target list, info dict with availability details)
        """
        if assay_name not in self.expnames:
            raise ValueError(f"Assay '{assay_name}' not found in experiment aliases")
        
        assay_idx = self.expnames.index(assay_name)
        info = {"assay_idx": assay_idx, "assay_name": assay_name}
        
        if task == "denoise":
            # For denoise, check if assay is available in input
            available_in_input, _ = self._check_assay_availability(assay_name, bios_name, locus, dsf)
            if not available_in_input:
                raise ValueError(
                    f"Cannot denoise assay '{assay_name}': it is not available in input. "
                    f"Use 'impute' task to impute this assay."
                )
            info["available_in_input"] = True
            return [], info
        
        elif task == "impute":
            if self.dataset_type == "merged":
                # Check if assay is available
                available_in_input, available_in_target = self._check_assay_availability(
                    assay_name, bios_name, locus, dsf
                )
                info["available_in_input"] = available_in_input
                info["available_in_target"] = available_in_target
                
                if available_in_input:
                    # Mask it for imputation
                    return [assay_idx], info
                else:
                    # Not available, just predict
                    return [], info
            
            elif self.dataset_type == "eic":
                # For EIC, need to check T_* and B_* biosamples
                if bios_name.startswith("B_"):
                    T_biosname = bios_name.replace("B_", "T_")
                    B_biosname = bios_name
                elif bios_name.startswith("T_"):
                    T_biosname = bios_name
                    B_biosname = bios_name.replace("T_", "B_")
                else:
                    raise ValueError(f"Unexpected biosample name format for EIC: {bios_name}")
                
                # Check availability in T_* and B_*
                available_in_T, _ = self._check_assay_availability(T_biosname, T_biosname, locus, dsf)
                available_in_B, _ = self._check_assay_availability(B_biosname, B_biosname, locus, 1)
                
                info["available_in_T"] = available_in_T
                info["available_in_B"] = available_in_B
                
                if available_in_T:
                    # Mask it in T_* for imputation
                    return [assay_idx], info
                else:
                    # Not in T_*, just predict
                    return [], info
        
        raise ValueError(f"Unknown task: {task}")
    
    def predict_single_assay(self, bios_name: str, assay_name: str, task: str,
                            locus: List = None, dsf: int = 1,
                            fill_y_prompt_spec: Optional[Dict] = None,
                            fill_prompt_mode: str = "median") -> Dict[str, Any]:
        """
        Predict a single assay with impute or denoise task.
        
        Args:
            bios_name: Name of biosample
            assay_name: Name of assay to predict
            task: Task type ("impute" or "denoise")
            locus: Genomic locus (default: chr21)
            dsf: Downsampling factor
            fill_y_prompt_spec: Optional custom metadata specification. If provided,
                will directly overwrite mdY prompt metadata using fill_in_prompt_manual
                with overwrite=True, regardless of fill_prompt_mode.
            fill_prompt_mode: Mode for filling missing metadata (used only if
                fill_y_prompt_spec is None)
            
        Returns:
            Dictionary with predictions and metadata
        """
        if locus is None:
            locus = ["chr21", 0, self.chr_sizes.get("chr21", 46709983)]
        
        if assay_name not in self.expnames:
            raise ValueError(f"Assay '{assay_name}' not found in experiment aliases")
        
        assay_idx = self.expnames.index(assay_name)
        
        # Determine imp_target
        imp_target, info = self._determine_imp_target(assay_name, bios_name, task, locus, dsf)
        
        # Load data
        if self.dataset_type == "eic" and task == "impute":
            # For EIC impute, need to load T_* and B_* data
            if bios_name.startswith("B_"):
                T_biosname = bios_name.replace("B_", "T_")
                B_biosname = bios_name
            elif bios_name.startswith("T_"):
                T_biosname = bios_name
                B_biosname = bios_name.replace("T_", "B_")
            else:
                raise ValueError(f"Unexpected biosample name format for EIC: {bios_name}")
            
            # Load T_* data (input)
            # For EIC, we'll handle mY overwrite after loading B_* data
            # Load T_* - if we have custom spec, load with "none" mode and overwrite manually
            # Otherwise, use fill_prompt_mode
            load_fill_mode = "none" if fill_y_prompt_spec is not None else fill_prompt_mode
            
            if self.DNA:
                X, Y_T, P_T, seq, mX, mY_T, avX, avY_T = self.predictor.load_data(
                    T_biosname, locus, dsf, None, load_fill_mode
                )
            else:
                X, Y_T, P_T, mX, mY_T, avX, avY_T = self.predictor.load_data(
                    T_biosname, locus, dsf, None, load_fill_mode
                )
                seq = None
            
            # Load B_* data (target for some assays)
            temp_y_B, temp_my_B = self.predictor.data_handler.load_bios_Counts(B_biosname, locus, 1)
            Y_B, mY_B, avY_B = self.predictor.data_handler.make_bios_tensor_Counts(temp_y_B, temp_my_B)
            
            # Merge P-value and Peak data
            temp_p_B = self.predictor.data_handler.load_bios_BW(B_biosname, locus)
            temp_p_T = self.predictor.data_handler.load_bios_BW(T_biosname, locus)
            temp_p = {**temp_p_B, **temp_p_T}
            P, avlP = self.predictor.data_handler.make_bios_tensor_BW(temp_p)
            
            temp_peak_T = self.predictor.data_handler.load_bios_Peaks(T_biosname, locus)
            temp_peak_B = self.predictor.data_handler.load_bios_Peaks(B_biosname, locus)
            temp_peak = {**temp_peak_B, **temp_peak_T}
            Peak, avlPeak = self.predictor.data_handler.make_bios_tensor_Peaks(temp_peak)
            
            # Reshape to context windows
            num_rows = (X.shape[0] // self.context_length) * self.context_length
            Y_B = Y_B[:num_rows, :]
            P = P[:num_rows, :]
            Peak = Peak[:num_rows, :]
            
            Y_B = Y_B.view(-1, self.context_length, Y_B.shape[-1])
            P = P.view(-1, self.context_length, P.shape[-1])
            Peak = Peak.view(-1, self.context_length, Peak.shape[-1])
            
            # Use Y_B metadata (target side)
            mY = mY_B.expand(X.shape[0], -1, -1) if mY_B.ndim == 2 else mY_B
            avY = avY_B.expand(X.shape[0], -1) if avY_B.ndim == 1 else avY_B
            
            # If custom prompt spec provided, overwrite mY metadata
            # If no spec but fill_prompt_mode is not "none", apply fill_prompt_mode first
            # Then overwrite with spec if provided
            if fill_y_prompt_spec is None and fill_prompt_mode != "none":
                # Apply fill_prompt_mode first for all assays
                if mY.ndim == 3:
                    mY_2d = mY[0].clone()
                    batch_size = mY.shape[0]
                else:
                    mY_2d = mY.clone()
                    batch_size = 1
                
                if fill_prompt_mode == "sample":
                    mY_2d = self.predictor.data_handler.fill_in_prompt(mY_2d, missing_value=-1, sample=True)
                    print("Applied random sampling fill-in-prompt to all assays")
                elif fill_prompt_mode == "mode":
                    mY_2d = self.predictor.data_handler.fill_in_prompt(mY_2d, missing_value=-1, sample=False, use_mode=True)
                    print("Applied mode statistics fill-in-prompt to all assays")
                else:
                    mY_2d = self.predictor.data_handler.fill_in_prompt(mY_2d, missing_value=-1, sample=False, use_mode=False)
                    print("Applied median/mode statistics fill-in-prompt to all assays")
                
                if batch_size > 1:
                    mY = mY_2d.unsqueeze(0).repeat(batch_size, 1, 1)
                else:
                    mY = mY_2d
            
            if fill_y_prompt_spec is not None:
                # Reshape mY to [4, E] format for fill_in_prompt_manual
                if mY.ndim == 3:
                    # mY is [batch, 4, E], take first batch
                    mY_2d = mY[0].clone()
                    batch_size = mY.shape[0]
                else:
                    mY_2d = mY.clone()
                    batch_size = 1
                
                # Check if the predicted assay is in the spec
                if assay_name not in fill_y_prompt_spec:
                    print(f"Warning: Assay '{assay_name}' not found in y-prompt-spec. "
                          f"Only assays in spec will be overwritten: {list(fill_y_prompt_spec.keys())}")
                
                # Store original values for debugging
                assay_idx = self.expnames.index(assay_name)
                original_vals = mY_2d[:, assay_idx].clone()
                
                mY_2d = self.predictor.data_handler.fill_in_prompt_manual(
                    mY_2d, fill_y_prompt_spec, overwrite=True
                )
                
                # Debug: Show what changed for the predicted assay
                new_vals = mY_2d[:, assay_idx]
                if not torch.equal(original_vals, new_vals):
                    print(f"Metadata overwritten for '{assay_name}':")
                    print(f"  Before: depth={original_vals[0]:.2f}, platform={original_vals[1]:.0f}, "
                          f"read_len={original_vals[2]:.0f}, run_type={original_vals[3]:.0f}")
                    print(f"  After:  depth={new_vals[0]:.2f}, platform={new_vals[1]:.0f}, "
                          f"read_len={new_vals[2]:.0f}, run_type={new_vals[3]:.0f}")
                else:
                    print(f"Warning: Metadata for '{assay_name}' was NOT overwritten (not in spec or no change)")
                
                # Expand back to batch dimension using repeat() instead of expand() to ensure proper copy
                if batch_size > 1:
                    mY = mY_2d.unsqueeze(0).repeat(batch_size, 1, 1)
                else:
                    mY = mY_2d
                print(f"Overwrote mdY prompt metadata with custom specification (batch_size={batch_size})")
            
            # Determine ground truth: use T_* if assay was masked there, otherwise B_*
            if info.get("available_in_T", False):
                # Assay was in T_* and masked - use T_* as ground truth
                Y_ground_truth = Y_T
                # P_T is already loaded and reshaped by load_data
                P_ground_truth = P_T
            elif info.get("available_in_B", False):
                # Assay is in B_* - use B_* as ground truth
                Y_ground_truth = Y_B
                # For P, we need to extract B_* part from merged P
                # Since P is merged from both, we'll use the full merged P
                # (B_* assays should be in the merged P)
                P_ground_truth = P
            else:
                # Assay not available in either - can't evaluate
                Y_ground_truth = None
                P_ground_truth = None
            
        else:
            # Merged dataset or denoise task
            # Load data - if we have custom spec, load with "none" mode and overwrite manually
            # Otherwise, use fill_prompt_mode
            load_fill_mode = "none" if fill_y_prompt_spec is not None else fill_prompt_mode
            
            if self.DNA:
                X, Y, P, seq, mX, mY, avX, avY = self.predictor.load_data(
                    bios_name, locus, dsf, None, load_fill_mode
                )
            else:
                X, Y, P, mX, mY, avX, avY = self.predictor.load_data(
                    bios_name, locus, dsf, None, load_fill_mode
                )
                seq = None
            
            # If custom prompt spec provided, overwrite mY metadata directly
            # If no spec but fill_prompt_mode is not "none", apply fill_prompt_mode first
            # Then overwrite with spec if provided
            if fill_y_prompt_spec is None and fill_prompt_mode != "none":
                # Apply fill_prompt_mode first for all assays
                if mY.ndim == 3:
                    mY_2d = mY[0].clone()
                    batch_size = mY.shape[0]
                else:
                    mY_2d = mY.clone()
                    batch_size = 1
                
                if fill_prompt_mode == "sample":
                    mY_2d = self.predictor.data_handler.fill_in_prompt(mY_2d, missing_value=-1, sample=True)
                    print("Applied random sampling fill-in-prompt to all assays")
                elif fill_prompt_mode == "mode":
                    mY_2d = self.predictor.data_handler.fill_in_prompt(mY_2d, missing_value=-1, sample=False, use_mode=True)
                    print("Applied mode statistics fill-in-prompt to all assays")
                else:
                    mY_2d = self.predictor.data_handler.fill_in_prompt(mY_2d, missing_value=-1, sample=False, use_mode=False)
                    print("Applied median/mode statistics fill-in-prompt to all assays")
                
                if batch_size > 1:
                    mY = mY_2d.unsqueeze(0).repeat(batch_size, 1, 1)
                else:
                    mY = mY_2d
            
            if fill_y_prompt_spec is not None:
                # Reshape mY to [4, E] format for fill_in_prompt_manual
                if mY.ndim == 3:
                    # mY is [batch, 4, E], take first batch
                    mY_2d = mY[0].clone()
                    batch_size = mY.shape[0]
                else:
                    mY_2d = mY.clone()
                    batch_size = 1
                
                # Check if the predicted assay is in the spec
                if assay_name not in fill_y_prompt_spec:
                    print(f"Warning: Assay '{assay_name}' not found in y-prompt-spec. "
                          f"Only assays in spec will be overwritten: {list(fill_y_prompt_spec.keys())}")
                
                # Store original values for debugging
                assay_idx = self.expnames.index(assay_name)
                original_vals = mY_2d[:, assay_idx].clone()
                
                mY_2d = self.predictor.data_handler.fill_in_prompt_manual(
                    mY_2d, fill_y_prompt_spec, overwrite=True
                )
                
                # Debug: Show what changed for the predicted assay
                new_vals = mY_2d[:, assay_idx]
                if not torch.equal(original_vals, new_vals):
                    print(f"Metadata overwritten for '{assay_name}':")
                    print(f"  Before: depth={original_vals[0]:.2f}, platform={original_vals[1]:.0f}, "
                          f"read_len={original_vals[2]:.0f}, run_type={original_vals[3]:.0f}")
                    print(f"  After:  depth={new_vals[0]:.2f}, platform={new_vals[1]:.0f}, "
                          f"read_len={new_vals[2]:.0f}, run_type={new_vals[3]:.0f}")
                else:
                    print(f"Warning: Metadata for '{assay_name}' was NOT overwritten (not in spec or no change)")
                
                # Expand back to batch dimension using repeat() instead of expand() to ensure proper copy
                if batch_size > 1:
                    mY = mY_2d.unsqueeze(0).repeat(batch_size, 1, 1)
                else:
                    mY = mY_2d
                print(f"Overwrote mdY prompt metadata with custom specification (batch_size={batch_size})")
            
            Y_ground_truth = Y
            P_ground_truth = P
        
        # Run prediction
        if self.DNA:
            n, p, mu, var, peak = self.predictor.predict(X, mX, mY, avX, seq, imp_target)
        else:
            n, p, mu, var, peak = self.predictor.predict(X, mX, mY, avX, None, imp_target)
        
        # Flatten predictions
        n = n.view((n.shape[0] * n.shape[1]), n.shape[-1])
        p = p.view((p.shape[0] * p.shape[1]), p.shape[-1])
        mu = mu.view((mu.shape[0] * mu.shape[1]), mu.shape[-1])
        var = var.view((var.shape[0] * var.shape[1]), var.shape[-1])
        peak = peak.view((peak.shape[0] * peak.shape[1]), peak.shape[-1])
        
        # Extract predictions for this assay
        count_dist = NegativeBinomial(p[:, assay_idx], n[:, assay_idx])
        pval_dist = Gaussian(mu[:, assay_idx], var[:, assay_idx])
        
        count_mean = count_dist.mean().numpy()
        pval_mean = pval_dist.mean().numpy()
        peak_scores = peak[:, assay_idx].numpy()
        
        # Prepare ground truth data
        if self.dataset_type == "eic":
            # For EIC, ground truth depends on where assay is available
            if Y_ground_truth is not None:
                count_gt = Y_ground_truth.view(-1, Y_ground_truth.shape[-1])[:, assay_idx].numpy()
            else:
                count_gt = None
            
            if P_ground_truth is not None:
                # P_ground_truth is already reshaped to context windows
                pval_gt = P_ground_truth.view(-1, P_ground_truth.shape[-1])[:, assay_idx].numpy()
            else:
                pval_gt = None
            
            peak_gt = Peak.view(-1, Peak.shape[-1])[:, assay_idx].numpy()
        else:
            # Merged dataset
            count_gt = Y_ground_truth.view(-1, Y_ground_truth.shape[-1])[:, assay_idx].numpy()
            pval_gt = P_ground_truth.view(-1, P_ground_truth.shape[-1])[:, assay_idx].numpy()
            
            # Load peak data for merged
            temp_peak = self.predictor.data_handler.load_bios_Peaks(bios_name, locus)
            Peak_merged, _ = self.predictor.data_handler.make_bios_tensor_Peaks(temp_peak)
            num_rows = (Peak_merged.shape[0] // self.context_length) * self.context_length
            Peak_merged = Peak_merged[:num_rows, :]
            Peak_merged = Peak_merged.view(-1, self.context_length, Peak_merged.shape[-1])
            peak_gt = Peak_merged.view(-1, Peak_merged.shape[-1])[:, assay_idx].numpy()
        
        result = {
            "bios_name": bios_name,
            "assay_name": assay_name,
            "task": task,
            "locus": locus,
            "predictions": {
                "count": count_mean,
                "pval": pval_mean,
                "peak_scores": peak_scores,
                "count_params": {"p": p[:, assay_idx].numpy(), "n": n[:, assay_idx].numpy()},
                "pval_params": {"mu": mu[:, assay_idx].numpy(), "var": var[:, assay_idx].numpy()}
            },
            "ground_truth": {
                "count": count_gt,
                "pval": pval_gt,
                "peak": peak_gt
            },
            "info": info
        }
        
        return result
    
    def compute_metrics(self, result: Dict[str, Any], quick: bool = False) -> Dict[str, Any]:
        """
        Compute evaluation metrics for predictions.
        
        Args:
            result: Result dictionary from predict_single_assay()
            quick: Whether to compute quick metrics only
            
        Returns:
            Dictionary with metrics or error message
        """
        bios_name = result["bios_name"]
        assay_name = result["assay_name"]
        task = result["task"]
        info = result["info"]
        
        # Check if we can compute metrics
        if task == "impute":
            if self.dataset_type == "merged":
                if not info.get("available_in_target", False):
                    return {
                        "error": "Cannot evaluate: no observed values for this assay",
                        "reason": "Assay not available in biosample"
                    }
            elif self.dataset_type == "eic":
                if not info.get("available_in_B", False) and not info.get("available_in_T", False):
                    return {
                        "error": "Cannot evaluate: no observed values for this assay",
                        "reason": "Assay not available in B_* or T_* biosamples"
                    }
        
        # Get ground truth and predictions
        C_target = result["ground_truth"]["count"]
        P_target = result["ground_truth"]["pval"]
        Peak_target = result["ground_truth"]["peak"]
        
        # Check if ground truth is None (can't evaluate)
        if C_target is None or P_target is None:
            return {
                "error": "Cannot evaluate: no observed values for this assay",
                "reason": "Ground truth data not available"
            }
        
        count_pred = result["predictions"]["count"]
        pval_pred = result["predictions"]["pval"]
        peak_pred = result["predictions"]["peak_scores"]
        
        # Apply sinh transformation to p-values (inverse of arcsinh used during training)
        # Clip to prevent overflow in sinh (sinh overflows around 710)
        # Conservative clipping: sinh(20) ≈ 2.4e8, which is reasonable for p-values
        # P_target = np.clip(P_target, -20, 20)
        # pval_pred = np.clip(pval_pred, -20, 20)
        P_target = np.sinh(P_target)
        pval_pred = np.sinh(pval_pred)
        
        # Import metrics
        from _utils import METRICS
        metrics = METRICS()
        
        def safe_metric(fn, *args):
            try:
                return fn(*args)
            except Exception as e:
                return np.nan
        
        # Compute metrics
        metrics_dict = {
            "bios": bios_name,
            "assay": assay_name,
            "task": task,
            "comparison": task,
            
            # Count metrics
            "C_MSE-GW": safe_metric(metrics.mse, C_target, count_pred),
            "C_Pearson-GW": safe_metric(metrics.pearson, C_target, count_pred),
            "C_Spearman-GW": safe_metric(metrics.spearman, C_target, count_pred),
            
            # P-value metrics
            "P_MSE-GW": safe_metric(metrics.mse, P_target, pval_pred),
            "P_Pearson-GW": safe_metric(metrics.pearson, P_target, pval_pred),
            "P_Spearman-GW": safe_metric(metrics.spearman, P_target, pval_pred),
            
            # Peak metrics
            "Peak_AUCROC-GW": safe_metric(metrics.aucroc, Peak_target, peak_pred) if Peak_target is not None else np.nan,
        }
        
        if not quick:
            # Add more detailed metrics
            metrics_dict.update({
                "C_MSE-gene": safe_metric(metrics.mse_gene, C_target, count_pred),
                "C_Pearson_gene": safe_metric(metrics.pearson_gene, C_target, count_pred),
                "C_Spearman_gene": safe_metric(metrics.spearman_gene, C_target, count_pred),
                "C_MSE-prom": safe_metric(metrics.mse_prom, C_target, count_pred),
                "C_Pearson_prom": safe_metric(metrics.pearson_prom, C_target, count_pred),
                "C_Spearman_prom": safe_metric(metrics.spearman_prom, C_target, count_pred),
                
                "P_MSE-gene": safe_metric(metrics.mse_gene, P_target, pval_pred),
                "P_Pearson_gene": safe_metric(metrics.pearson_gene, P_target, pval_pred),
                "P_Spearman_gene": safe_metric(metrics.spearman_gene, P_target, pval_pred),
                "P_MSE-prom": safe_metric(metrics.mse_prom, P_target, pval_pred),
                "P_Pearson_prom": safe_metric(metrics.pearson_prom, P_target, pval_pred),
                "P_Spearman_prom": safe_metric(metrics.spearman_prom, P_target, pval_pred),
                
                "Peak_AUCROC-gene": safe_metric(metrics.aucroc_gene, Peak_target, peak_pred) if Peak_target is not None else np.nan,
                "Peak_AUCROC-prom": safe_metric(metrics.aucroc_prom, Peak_target, peak_pred) if Peak_target is not None else np.nan,
            })
        
        return metrics_dict
    
    def save_predictions(self, result: Dict[str, Any], output_dir: str):
        """
        Save predictions as npz files.
        
        Args:
            result: Result dictionary from predict_single_assay()
            output_dir: Directory to save npz files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        bios_name = result["bios_name"]
        assay_name = result["assay_name"]
        task = result["task"]
        locus = result["locus"]
        
        # Create filename
        locus_str = f"{locus[0]}_{locus[1]}_{locus[2]}"
        filename = f"{bios_name}_{assay_name}_{task}_{locus_str}.npz"
        filepath = os.path.join(output_dir, filename)
        
        # Extract arrays
        count = result["predictions"]["count"]
        pval = result["predictions"]["pval"]
        peak_scores = result["predictions"]["peak_scores"]
        
        # Save as npz
        np.savez_compressed(
            filepath,
            count=count,
            pval=pval,
            peak_scores=peak_scores,
            bios_name=bios_name,
            assay_name=assay_name,
            task=task,
            locus=locus
        )
        
        print(f"Saved predictions to {filepath}")
        return filepath


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary for pretty printing."""
    if "error" in metrics:
        return f"❌ {metrics['error']}: {metrics.get('reason', '')}"
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"Metrics for {metrics['bios']} / {metrics['assay']} ({metrics['task']})")
    lines.append("=" * 60)
    
    # Count metrics
    lines.append("\nCount Metrics:")
    lines.append(f"  MSE-GW:        {metrics.get('C_MSE-GW', 'N/A'):.6f}")
    lines.append(f"  Pearson-GW:    {metrics.get('C_Pearson-GW', 'N/A'):.6f}")
    lines.append(f"  Spearman-GW:   {metrics.get('C_Spearman-GW', 'N/A'):.6f}")
    
    if 'C_MSE-gene' in metrics:
        lines.append(f"  MSE-gene:      {metrics.get('C_MSE-gene', 'N/A'):.6f}")
        lines.append(f"  Pearson-gene: {metrics.get('C_Pearson_gene', 'N/A'):.6f}")
        lines.append(f"  Spearman-gene:{metrics.get('C_Spearman_gene', 'N/A'):.6f}")
        lines.append(f"  MSE-prom:      {metrics.get('C_MSE-prom', 'N/A'):.6f}")
        lines.append(f"  Pearson-prom: {metrics.get('C_Pearson_prom', 'N/A'):.6f}")
        lines.append(f"  Spearman-prom:{metrics.get('C_Spearman_prom', 'N/A'):.6f}")
    
    # P-value metrics
    lines.append("\nP-value Metrics:")
    lines.append(f"  MSE-GW:        {metrics.get('P_MSE-GW', 'N/A'):.6f}")
    lines.append(f"  Pearson-GW:    {metrics.get('P_Pearson-GW', 'N/A'):.6f}")
    lines.append(f"  Spearman-GW:   {metrics.get('P_Spearman-GW', 'N/A'):.6f}")
    
    if 'P_MSE-gene' in metrics:
        lines.append(f"  MSE-gene:      {metrics.get('P_MSE-gene', 'N/A'):.6f}")
        lines.append(f"  Pearson-gene: {metrics.get('P_Pearson_gene', 'N/A'):.6f}")
        lines.append(f"  Spearman-gene:{metrics.get('P_Spearman_gene', 'N/A'):.6f}")
        lines.append(f"  MSE-prom:      {metrics.get('P_MSE-prom', 'N/A'):.6f}")
        lines.append(f"  Pearson-prom: {metrics.get('P_Pearson_prom', 'N/A'):.6f}")
        lines.append(f"  Spearman-prom:{metrics.get('P_Spearman_prom', 'N/A'):.6f}")
    
    # Peak metrics
    if 'Peak_AUCROC-GW' in metrics and not np.isnan(metrics['Peak_AUCROC-GW']):
        lines.append("\nPeak Metrics:")
        lines.append(f"  AUCROC-GW:     {metrics.get('Peak_AUCROC-GW', 'N/A'):.6f}")
        if 'Peak_AUCROC-gene' in metrics:
            lines.append(f"  AUCROC-gene:   {metrics.get('Peak_AUCROC-gene', 'N/A'):.6f}")
            lines.append(f"  AUCROC-prom:   {metrics.get('Peak_AUCROC-prom', 'N/A'):.6f}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    """CLI interface for single-assay prediction."""
    parser = argparse.ArgumentParser(
        description="Single Assay Prediction Tool - Focused impute/denoise for one assay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Impute an assay (timing only)
  python predict_assay.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                          --data-path /path/to/DATA_CANDI_MERGED \\
                          --bios-name GM12878 \\
                          --assay-name H3K4me3 \\
                          --task impute \\
                          --dataset merged

  # Denoise an assay and print metrics
  python predict_assay.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                          --data-path /path/to/DATA_CANDI_MERGED \\
                          --bios-name GM12878 \\
                          --assay-name H3K4me3 \\
                          --task denoise \\
                          --dataset merged \\
                          --print-metrics

  # Impute and save as npz
  python predict_assay.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                          --data-path /path/to/DATA_CANDI_MERGED \\
                          --bios-name GM12878 \\
                          --assay-name H3K27ac \\
                          --task impute \\
                          --dataset merged \\
                          --save-npz \\
                          --output-dir predictions/

  # EIC dataset imputation
  python predict_assay.py --model-dir models/20251031_143320_CANDI_merged_ccre_5000loci_oct31 \\
                          --data-path /path/to/DATA_CANDI_EIC \\
                          --bios-name B_GM12878 \\
                          --assay-name H3K4me3 \\
                          --task impute \\
                          --dataset eic \\
                          --print-metrics

  # Impute with explicit metadata values for target assay
  python predict_assay.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                          --data-path /path/to/DATA_CANDI_MERGED \\
                          --bios-name GM12878 \\
                          --assay-name H3K4me3 \\
                          --task impute \\
                          --dataset merged \\
                          --depth 50000000 \\
                          --read-length 100 \\
                          --sequencing-platform "Illumina HiSeq 4000" \\
                          --run-type paired-ended \\
                          --print-metrics

  # Use JSON file for other assays, but override target assay with explicit values
  python predict_assay.py --model-dir models/20251110_232800_CANDI_merged_ccre_3000loci_Nov10_CrossAttn_BatchNorm \\
                          --data-path ../DATA_CANDI_MERGED \\
                          --bios-name motor_neuron_grp1_rep1 \\
                          --assay-name H3K27ac \\
                          --task impute \\
                          --dataset merged \\
                          --y-prompt-spec prompts/merged_mode.json \\
                          --depth 30000000 \\
                          --read-length 50 \\
                          --print-metrics
        """
    )
    
    # Required arguments
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to model directory containing config JSON and checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--bios-name', type=str, required=True,
                       help='Name of biosample')
    parser.add_argument('--assay-name', type=str, required=True,
                       help='Name of assay to predict')
    parser.add_argument('--task', type=str, required=True, choices=['impute', 'denoise'],
                       help='Task type: "impute" or "denoise"')
    parser.add_argument('--dataset', type=str, required=True, choices=['merged', 'eic'],
                       help='Dataset type: "merged" or "eic"')
    
    # Optional arguments
    parser.add_argument('--locus', type=str, nargs=3, default=['chr21', '0', '46709983'],
                       help='Genomic locus as chrom start end (default: chr21 0 46709983)')
    parser.add_argument('--y-prompt-spec', type=str, default=None,
                       help='JSON file with custom metadata specification for all assays. '
                            'When provided, will directly overwrite mdY prompt metadata with values from the JSON file. '
                            'Values for the target assay can be overridden with explicit CLI flags.')
    
    # Explicit metadata values for target assay (overrides JSON file for target assay)
    parser.add_argument('--depth', type=float, default=None,
                       help='Depth value for target assay (overrides JSON file if provided)')
    parser.add_argument('--read-length', type=float, default=None,
                       help='Read length for target assay (overrides JSON file if provided)')
    parser.add_argument('--sequencing-platform', type=str, default=None,
                       help='Sequencing platform for target assay (overrides JSON file if provided)')
    parser.add_argument('--run-type', type=str, choices=['single-ended', 'paired-ended'], default=None,
                       help='Run type for target assay: "single-ended" or "paired-ended" (overrides JSON file if provided)')
    
    parser.add_argument('--dsf', type=int, default=1,
                       help='Downsampling factor (default: 1)')
    
    # Output modes (mutually exclusive)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--timing-only', action='store_true',
                             help='Just measure execution time (default if no other mode specified)')
    output_group.add_argument('--save-npz', action='store_true',
                             help='Save predictions as npz files')
    output_group.add_argument('--print-metrics', action='store_true',
                             help='Compute and print evaluation metrics')
    
    # Required if save-npz
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save npz files (required if --save-npz)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.save_npz and args.output_dir is None:
        parser.error("--output-dir is required when using --save-npz")
    
    # Parse locus
    locus = [args.locus[0], int(args.locus[1]), int(args.locus[2])]
    
    # Load Y prompt specification if provided
    fill_y_prompt_spec = None
    fill_prompt_mode = "median"
    
    # Check if explicit metadata values are provided for target assay
    explicit_metadata = {}
    if args.depth is not None:
        explicit_metadata["depth"] = args.depth
    if args.read_length is not None:
        explicit_metadata["read_length"] = args.read_length
    if args.sequencing_platform is not None:
        explicit_metadata["sequencing_platform"] = args.sequencing_platform
    if args.run_type is not None:
        explicit_metadata["run_type"] = args.run_type
    
    # Load JSON file if provided
    if args.y_prompt_spec:
        with open(args.y_prompt_spec, 'r') as f:
            fill_y_prompt_spec = json.load(f)
        print(f"Loaded Y prompt specification from {args.y_prompt_spec}")
        
        # Override target assay values with explicit CLI values if provided
        if explicit_metadata:
            if args.assay_name not in fill_y_prompt_spec:
                fill_y_prompt_spec[args.assay_name] = {}
            fill_y_prompt_spec[args.assay_name].update(explicit_metadata)
            print(f"Overrode metadata for target assay '{args.assay_name}' with explicit CLI values")
            print(f"  Updated fields: {list(explicit_metadata.keys())}")
        else:
            print("Will overwrite mdY prompt metadata with custom specification")
    elif explicit_metadata:
        # No JSON file but explicit values provided - create minimal spec for target assay only
        fill_y_prompt_spec = {args.assay_name: explicit_metadata}
        print(f"Using explicit metadata values for target assay '{args.assay_name}'")
        print(f"  Specified fields: {list(explicit_metadata.keys())}")
        print("  Other assays will use default median/mode fill-in-prompt")
    else:
        # No JSON file and no explicit values - use default fill_prompt_mode
        print(f"Using default fill-in-prompt mode: {fill_prompt_mode}")
    
    try:
        # Initialize predictor
        predictor = SingleAssayPredictor(
            model_dir=args.model_dir,
            data_path=args.data_path,
            dataset_type=args.dataset
        )
        
        # Run prediction with timing
        start_time = time.time()
        result = predictor.predict_single_assay(
            bios_name=args.bios_name,
            assay_name=args.assay_name,
            task=args.task,
            locus=locus,
            dsf=args.dsf,
            fill_y_prompt_spec=fill_y_prompt_spec,
            fill_prompt_mode=fill_prompt_mode
        )
        elapsed_time = time.time() - start_time
        
        # Handle output modes
        if args.save_npz:
            predictor.save_predictions(result, args.output_dir)
            print(f"\nExecution time: {elapsed_time:.2f} seconds")
        
        elif args.print_metrics:
            metrics = predictor.compute_metrics(result, quick=False)
            print(format_metrics(metrics))
            print(f"\nExecution time: {elapsed_time:.2f} seconds")
        
        else:  # timing_only (default when no other mode specified)
            print(f"✅ Prediction completed successfully")
            print(f"   Biosample: {args.bios_name}")
            print(f"   Assay: {args.assay_name}")
            print(f"   Task: {args.task}")
            print(f"   Execution time: {elapsed_time:.2f} seconds")
            print(f"   Predictions shape: count={result['predictions']['count'].shape}, "
                  f"pval={result['predictions']['pval'].shape}, "
                  f"peak={result['predictions']['peak_scores'].shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

