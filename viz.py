#!/usr/bin/env python3
"""
CANDI Visualization Module

This module provides visualization functions for CANDI model predictions and evaluations.
It ports the VISUALS_CANDI class from old_eval.py with updates for the new data structures.

Author: Refactored from old_eval.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, nbinom
from typing import Dict, List, Any, Optional, Tuple

# Import from current codebase
from _utils import METRICS


class VISUALS_CANDI:
    """
    Visualization class for CANDI model predictions and evaluations.
    
    This class provides comprehensive visualization capabilities for:
    - Track visualizations (count and signal tracks)
    - Confidence plots and error analysis
    - Distribution plots and heatmaps
    - Scatter plots with marginals
    - Context-specific performance analysis
    - TSS/Gene body enrichment analysis
    - Metagene plots
    """
    
    def __init__(self, resolution: int = 25, savedir: str = "models/evals/"):
        """
        Initialize visualization class.
        
        Args:
            resolution: Genomic resolution in bp
            savedir: Directory to save plots
        """
        self.metrics = METRICS()
        self.resolution = resolution
        self.savedir = savedir
        
        # Create save directory if it doesn't exist
        os.makedirs(savedir, exist_ok=True)
    
    def clear_palette(self):
        """Clear matplotlib palette and close all figures."""
        sns.reset_orig()
        plt.close("all")
        plt.style.use('default')
        plt.clf()
    
    def _get_prediction_data(self, prediction_dict: Dict[str, Any], 
                           bios_name: str, experiment: str, 
                           prediction_type: str = "imputed") -> Dict[str, np.ndarray]:
        """
        Extract prediction data from new prediction dictionary structure.
        
        Args:
            prediction_dict: Dictionary from CANDIPredictor.predict_biosample()
            bios_name: Name of biosample
            experiment: Name of experiment
            prediction_type: Type of prediction ("imputed" or "denoised")
            
        Returns:
            Dictionary with prediction data arrays
        """
        # Handle both imputed and denoised cases
        if prediction_type == "denoised":
            exp_key = f"{experiment}_upsampled"
        else:
            exp_key = experiment
        
        if bios_name not in prediction_dict or exp_key not in prediction_dict[bios_name]:
            raise KeyError(f"Prediction not found for {bios_name}/{exp_key}")
        
        pred_data = prediction_dict[bios_name][exp_key]
        
        # Extract arrays from distributions and parameters
        result = {
            'pred_count': pred_data['count_dist'].mean().numpy(),
            'pred_pval': pred_data['pval_dist'].mean().numpy(),
            'pred_peak': pred_data['peak_scores'].numpy(),
            'pred_count_std': pred_data['count_dist'].std().numpy(),
            'pred_pval_std': pred_data['pval_dist'].std().numpy(),
            'pred_count_n': pred_data['count_params']['n'].numpy(),
            'pred_count_p': pred_data['count_params']['p'].numpy(),
            'pred_pval_mu': pred_data['pval_params']['mu'].numpy(),
            'pred_pval_var': pred_data['pval_params']['var'].numpy(),
        }
        
        # Add confidence intervals if available
        try:
            count_ci = pred_data['count_dist'].interval(confidence=0.95)
            pval_ci = pred_data['pval_dist'].interval(confidence=0.95)
            result.update({
                'pred_count_lower_95': count_ci[0].numpy(),
                'pred_count_upper_95': count_ci[1].numpy(),
                'pred_pval_lower_95': pval_ci[0].numpy(),
                'pred_pval_upper_95': pval_ci[1].numpy(),
            })
        except:
            # Confidence intervals not available
            pass
        
        return result
    
    def _get_ground_truth_data(self, data_handler, bios_name: str, 
                              experiment: str, locus: List) -> Dict[str, np.ndarray]:
        """
        Load ground truth data for comparison.
        
        Args:
            data_handler: CANDIDataHandler instance
            bios_name: Name of biosample
            experiment: Name of experiment
            locus: Genomic locus [chrom, start, end]
            
        Returns:
            Dictionary with ground truth data arrays
        """
        # Load count data
        temp_y, temp_my = data_handler.load_bios_Counts(bios_name, locus, 1)
        Y, mY, avY = data_handler.make_bios_tensor_Counts(temp_y, temp_my)
        
        # Load p-value data
        temp_p = data_handler.load_bios_BW(bios_name, locus, 1)
        P, avlP = data_handler.make_bios_tensor_BW(temp_p)
        
        # Get experiment index
        exp_names = list(data_handler.aliases['experiment_aliases'].keys())
        if experiment not in exp_names:
            raise KeyError(f"Experiment {experiment} not found in aliases")
        exp_idx = exp_names.index(experiment)
        
        # Extract data for this experiment
        result = {
            'obs_count': Y[:, exp_idx].numpy(),
            'obs_pval': P[:, exp_idx].numpy(),
        }
        
        return result
    
    def count_track(self, prediction_dict: Dict[str, Any], bios_name: str, 
                   experiment: str, data_handler=None, locus: List = None):
        """
        Generate count track visualization for a specific experiment.
        
        Args:
            prediction_dict: Dictionary from CANDIPredictor.predict_biosample()
            bios_name: Name of biosample
            experiment: Name of experiment
            data_handler: CANDIDataHandler instance for ground truth
            locus: Genomic locus (default: chr21)
        """
        if locus is None:
            locus = ["chr21", 0, 250000000]
        
        # Create save directory
        save_dir = f"{self.savedir}/{bios_name}_{experiment}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Example gene coordinates (same as original)
        example_gene_coord = (33481539//self.resolution, 33588914//self.resolution) # GART
        example_gene_coord2 = (25800151//self.resolution, 26235914//self.resolution) # APP
        example_gene_coord3 = (31589009//self.resolution, 31745788//self.resolution) # SOD1
        example_gene_coord4 = (39526359//self.resolution, 39802081//self.resolution) # B3GALT5
        example_gene_coord5 = (33577551//self.resolution, 33919338//self.resolution) # ITSN1
        
        example_gene_coords = [
            example_gene_coord, example_gene_coord2, example_gene_coord3,
            example_gene_coord4, example_gene_coord5
        ]
        
        # Get prediction data
        pred_imputed = self._get_prediction_data(prediction_dict, bios_name, experiment, "imputed")
        pred_denoised = self._get_prediction_data(prediction_dict, bios_name, experiment, "denoised")
        
        # Get ground truth data if available
        obs_data = None
        if data_handler is not None:
            try:
                obs_data = self._get_ground_truth_data(data_handler, bios_name, experiment, locus)
            except Exception as e:
                print(f"Warning: Could not load ground truth data: {e}")
        
        # Create figure
        fig, axes = plt.subplots(2, len(example_gene_coords), 
                                figsize=(6 * len(example_gene_coords), 4))
        if len(example_gene_coords) == 1:
            axes = axes.reshape(2, 1)
        
        # Plot imputed predictions
        for i, gene_coord in enumerate(example_gene_coords):
            ax = axes[0, i]
            x_values = range(gene_coord[0], gene_coord[1])
            
            # Plot observed data if available
            if obs_data is not None:
                observed_values = obs_data['obs_count'][gene_coord[0]:gene_coord[1]]
                ax.plot(x_values, observed_values, color="blue", alpha=0.7, 
                       label="Observed", linewidth=0.1)
                ax.fill_between(x_values, 0, observed_values, alpha=0.7, color="blue")
            
            # Plot imputed predictions
            imputed_values = pred_imputed['pred_count'][gene_coord[0]:gene_coord[1]]
            ax.plot(x_values, imputed_values, "--", color="red", alpha=0.5, 
                   label="Imputed", linewidth=0.1)
            ax.fill_between(x_values, 0, imputed_values, color="red", alpha=0.5)
            
            # Formatting
            start_coord = gene_coord[0] * self.resolution
            end_coord = gene_coord[1] * self.resolution
            ax.set_title(f"{experiment}_imputed")
            ax.set_ylabel("Count")
            ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
            ax.set_xticklabels([])
            
            custom_lines = [mlines.Line2D([], [], color='blue', label='Observed'),
                           mlines.Line2D([], [], color='red', label='Imputed')]
            ax.legend(handles=custom_lines)
        
        # Plot denoised predictions
        for i, gene_coord in enumerate(example_gene_coords):
            ax = axes[1, i]
            x_values = range(gene_coord[0], gene_coord[1])
            
            # Plot observed data if available
            if obs_data is not None:
                observed_values = obs_data['obs_count'][gene_coord[0]:gene_coord[1]]
                ax.plot(x_values, observed_values, color="blue", alpha=0.7, 
                       label="Observed", linewidth=0.1)
                ax.fill_between(x_values, 0, observed_values, alpha=0.7, color="blue")
            
            # Plot denoised predictions
            denoised_values = pred_denoised['pred_count'][gene_coord[0]:gene_coord[1]]
            ax.plot(x_values, denoised_values, "--", color="green", alpha=0.5, 
                   label="Denoised", linewidth=0.1)
            ax.fill_between(x_values, 0, denoised_values, color="green", alpha=0.5)
            
            # Formatting
            start_coord = gene_coord[0] * self.resolution
            end_coord = gene_coord[1] * self.resolution
            ax.set_title(f"{experiment}_denoised")
            ax.set_ylabel("Count")
            ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
            ax.set_xticklabels([])
            
            custom_lines = [mlines.Line2D([], [], color='blue', label='Observed'),
                           mlines.Line2D([], [], color='green', label='Denoised')]
            ax.legend(handles=custom_lines)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/count_tracks.png", dpi=300)
        plt.savefig(f"{save_dir}/count_tracks.svg", format="svg")
        plt.close()
    
    def signal_track(self, prediction_dict: Dict[str, Any], bios_name: str, 
                    experiment: str, data_handler=None, locus: List = None):
        """
        Generate signal (p-value) track visualization for a specific experiment.
        
        Args:
            prediction_dict: Dictionary from CANDIPredictor.predict_biosample()
            bios_name: Name of biosample
            experiment: Name of experiment
            data_handler: CANDIDataHandler instance for ground truth
            locus: Genomic locus (default: chr21)
        """
        if locus is None:
            locus = ["chr21", 0, 250000000]
        
        # Create save directory
        save_dir = f"{self.savedir}/{bios_name}_{experiment}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Example gene coordinates (same as original)
        example_gene_coord = (33481539//self.resolution, 33588914//self.resolution) # GART
        example_gene_coord2 = (25800151//self.resolution, 26235914//self.resolution) # APP
        example_gene_coord3 = (31589009//self.resolution, 31745788//self.resolution) # SOD1
        example_gene_coord4 = (39526359//self.resolution, 39802081//self.resolution) # B3GALT5
        example_gene_coord5 = (33577551//self.resolution, 33919338//self.resolution) # ITSN1
        
        example_gene_coords = [
            example_gene_coord, example_gene_coord2, example_gene_coord3,
            example_gene_coord4, example_gene_coord5
        ]
        
        # Get prediction data
        pred_imputed = self._get_prediction_data(prediction_dict, bios_name, experiment, "imputed")
        pred_denoised = self._get_prediction_data(prediction_dict, bios_name, experiment, "denoised")
        
        # Get ground truth data if available
        obs_data = None
        if data_handler is not None:
            try:
                obs_data = self._get_ground_truth_data(data_handler, bios_name, experiment, locus)
            except Exception as e:
                print(f"Warning: Could not load ground truth data: {e}")
        
        # Create figure
        fig, axes = plt.subplots(2, len(example_gene_coords), 
                                figsize=(6 * len(example_gene_coords), 4))
        if len(example_gene_coords) == 1:
            axes = axes.reshape(2, 1)
        
        # Plot imputed predictions
        for i, gene_coord in enumerate(example_gene_coords):
            ax = axes[0, i]
            x_values = range(gene_coord[0], gene_coord[1])
            
            # Plot observed data if available
            if obs_data is not None:
                observed_values = obs_data['obs_pval'][gene_coord[0]:gene_coord[1]]
                # Apply arcsinh transform
                observed_values = np.sinh(observed_values)
                ax.plot(x_values, observed_values, color="blue", alpha=0.7, 
                       label="Observed", linewidth=0.1)
                ax.fill_between(x_values, 0, observed_values, alpha=0.7, color="blue")
            
            # Plot imputed predictions
            imputed_values = pred_imputed['pred_pval'][gene_coord[0]:gene_coord[1]]
            # Apply arcsinh transform
            imputed_values = np.sinh(imputed_values)
            ax.plot(x_values, imputed_values, "--", color="red", alpha=0.5, 
                   label="Imputed", linewidth=0.1)
            ax.fill_between(x_values, 0, imputed_values, color="red", alpha=0.5)
            
            # Formatting
            start_coord = gene_coord[0] * self.resolution
            end_coord = gene_coord[1] * self.resolution
            ax.set_title(f"{experiment}_imputed")
            ax.set_ylabel("Signal")
            ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
            ax.set_xticklabels([])
            
            custom_lines = [mlines.Line2D([], [], color='blue', label='Observed'),
                           mlines.Line2D([], [], color='red', label='Imputed')]
            ax.legend(handles=custom_lines)
        
        # Plot denoised predictions
        for i, gene_coord in enumerate(example_gene_coords):
            ax = axes[1, i]
            x_values = range(gene_coord[0], gene_coord[1])
            
            # Plot observed data if available
            if obs_data is not None:
                observed_values = obs_data['obs_pval'][gene_coord[0]:gene_coord[1]]
                # Apply arcsinh transform
                observed_values = np.sinh(observed_values)
                ax.plot(x_values, observed_values, color="blue", alpha=0.7, 
                       label="Observed", linewidth=0.1)
                ax.fill_between(x_values, 0, observed_values, alpha=0.7, color="blue")
            
            # Plot denoised predictions
            denoised_values = pred_denoised['pred_pval'][gene_coord[0]:gene_coord[1]]
            # Apply arcsinh transform
            denoised_values = np.sinh(denoised_values)
            ax.plot(x_values, denoised_values, "--", color="green", alpha=0.5, 
                   label="Denoised", linewidth=0.1)
            ax.fill_between(x_values, 0, denoised_values, color="green", alpha=0.5)
            
            # Formatting
            start_coord = gene_coord[0] * self.resolution
            end_coord = gene_coord[1] * self.resolution
            ax.set_title(f"{experiment}_denoised")
            ax.set_ylabel("Signal")
            ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
            ax.set_xticklabels([])
            
            custom_lines = [mlines.Line2D([], [], color='blue', label='Observed'),
                           mlines.Line2D([], [], color='green', label='Denoised')]
            ax.legend(handles=custom_lines)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/signal_tracks.png", dpi=300)
        plt.savefig(f"{save_dir}/signal_tracks.svg", format="svg")
        plt.close()
    
    def count_confidence(self, prediction_dict: Dict[str, Any], bios_name: str, 
                        experiment: str, data_handler=None, locus: List = None):
        """
        Generate count confidence interval visualization.
        
        Args:
            prediction_dict: Dictionary from CANDIPredictor.predict_biosample()
            bios_name: Name of biosample
            experiment: Name of experiment
            data_handler: CANDIDataHandler instance for ground truth
            locus: Genomic locus (default: chr21)
        """
        if locus is None:
            locus = ["chr21", 0, 250000000]
        
        # Create save directory
        save_dir = f"{self.savedir}/{bios_name}_{experiment}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Get prediction data
        pred_imputed = self._get_prediction_data(prediction_dict, bios_name, experiment, "imputed")
        pred_denoised = self._get_prediction_data(prediction_dict, bios_name, experiment, "denoised")
        
        # Get ground truth data if available
        obs_data = None
        if data_handler is not None:
            try:
                obs_data = self._get_ground_truth_data(data_handler, bios_name, experiment, locus)
            except Exception as e:
                print(f"Warning: Could not load ground truth data: {e}")
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot imputed confidence
        ax = axes[0]
        x_values = range(len(pred_imputed['pred_count']))
        
        # Plot confidence intervals if available
        if 'pred_count_lower_95' in pred_imputed and 'pred_count_upper_95' in pred_imputed:
            ax.fill_between(x_values, pred_imputed['pred_count_lower_95'], 
                           pred_imputed['pred_count_upper_95'], 
                           alpha=0.3, color='red', label='95% CI')
        
        # Plot mean prediction
        ax.plot(x_values, pred_imputed['pred_count'], color='red', 
               label='Imputed Mean', linewidth=1)
        
        # Plot observed data if available
        if obs_data is not None:
            ax.plot(x_values, obs_data['obs_count'], color='blue', 
                   label='Observed', linewidth=1, alpha=0.7)
        
        ax.set_title(f"{experiment} - Imputed Count Confidence")
        ax.set_ylabel("Count")
        ax.set_xlabel("Genomic Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot denoised confidence
        ax = axes[1]
        
        # Plot confidence intervals if available
        if 'pred_count_lower_95' in pred_denoised and 'pred_count_upper_95' in pred_denoised:
            ax.fill_between(x_values, pred_denoised['pred_count_lower_95'], 
                           pred_denoised['pred_count_upper_95'], 
                           alpha=0.3, color='green', label='95% CI')
        
        # Plot mean prediction
        ax.plot(x_values, pred_denoised['pred_count'], color='green', 
               label='Denoised Mean', linewidth=1)
        
        # Plot observed data if available
        if obs_data is not None:
            ax.plot(x_values, obs_data['obs_count'], color='blue', 
                   label='Observed', linewidth=1, alpha=0.7)
        
        ax.set_title(f"{experiment} - Denoised Count Confidence")
        ax.set_ylabel("Count")
        ax.set_xlabel("Genomic Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/count_confidence.png", dpi=300)
        plt.savefig(f"{save_dir}/count_confidence.svg", format="svg")
        plt.close()
    
    def signal_confidence(self, prediction_dict: Dict[str, Any], bios_name: str, 
                         experiment: str, data_handler=None, locus: List = None):
        """
        Generate signal (p-value) confidence interval visualization.
        
        Args:
            prediction_dict: Dictionary from CANDIPredictor.predict_biosample()
            bios_name: Name of biosample
            experiment: Name of experiment
            data_handler: CANDIDataHandler instance for ground truth
            locus: Genomic locus (default: chr21)
        """
        if locus is None:
            locus = ["chr21", 0, 250000000]
        
        # Create save directory
        save_dir = f"{self.savedir}/{bios_name}_{experiment}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Get prediction data
        pred_imputed = self._get_prediction_data(prediction_dict, bios_name, experiment, "imputed")
        pred_denoised = self._get_prediction_data(prediction_dict, bios_name, experiment, "denoised")
        
        # Get ground truth data if available
        obs_data = None
        if data_handler is not None:
            try:
                obs_data = self._get_ground_truth_data(data_handler, bios_name, experiment, locus)
            except Exception as e:
                print(f"Warning: Could not load ground truth data: {e}")
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot imputed confidence
        ax = axes[0]
        x_values = range(len(pred_imputed['pred_pval']))
        
        # Plot confidence intervals if available
        if 'pred_pval_lower_95' in pred_imputed and 'pred_pval_upper_95' in pred_imputed:
            # Apply arcsinh transform to confidence intervals
            lower_ci = np.sinh(pred_imputed['pred_pval_lower_95'])
            upper_ci = np.sinh(pred_imputed['pred_pval_upper_95'])
            ax.fill_between(x_values, lower_ci, upper_ci, 
                           alpha=0.3, color='red', label='95% CI')
        
        # Plot mean prediction
        mean_pred = np.sinh(pred_imputed['pred_pval'])
        ax.plot(x_values, mean_pred, color='red', 
               label='Imputed Mean', linewidth=1)
        
        # Plot observed data if available
        if obs_data is not None:
            obs_signal = np.sinh(obs_data['obs_pval'])
            ax.plot(x_values, obs_signal, color='blue', 
                   label='Observed', linewidth=1, alpha=0.7)
        
        ax.set_title(f"{experiment} - Imputed Signal Confidence")
        ax.set_ylabel("Signal")
        ax.set_xlabel("Genomic Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot denoised confidence
        ax = axes[1]
        
        # Plot confidence intervals if available
        if 'pred_pval_lower_95' in pred_denoised and 'pred_pval_upper_95' in pred_denoised:
            # Apply arcsinh transform to confidence intervals
            lower_ci = np.sinh(pred_denoised['pred_pval_lower_95'])
            upper_ci = np.sinh(pred_denoised['pred_pval_upper_95'])
            ax.fill_between(x_values, lower_ci, upper_ci, 
                           alpha=0.3, color='green', label='95% CI')
        
        # Plot mean prediction
        mean_pred = np.sinh(pred_denoised['pred_pval'])
        ax.plot(x_values, mean_pred, color='green', 
               label='Denoised Mean', linewidth=1)
        
        # Plot observed data if available
        if obs_data is not None:
            obs_signal = np.sinh(obs_data['obs_pval'])
            ax.plot(x_values, obs_signal, color='blue', 
                   label='Observed', linewidth=1, alpha=0.7)
        
        ax.set_title(f"{experiment} - Denoised Signal Confidence")
        ax.set_ylabel("Signal")
        ax.set_xlabel("Genomic Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/signal_confidence.png", dpi=300)
        plt.savefig(f"{save_dir}/signal_confidence.svg", format="svg")
        plt.close()
    
    def count_scatter_with_marginals(self, prediction_dict: Dict[str, Any], 
                                   bios_name: str, experiment: str, 
                                   data_handler=None, locus: List = None,
                                   share_axes: bool = True, 
                                   percentile_cutoff: int = 99):
        """
        Generate count scatter plot with marginal distributions.
        
        Args:
            prediction_dict: Dictionary from CANDIPredictor.predict_biosample()
            bios_name: Name of biosample
            experiment: Name of experiment
            data_handler: CANDIDataHandler instance for ground truth
            locus: Genomic locus (default: chr21)
            share_axes: Whether to share axes between subplots
            percentile_cutoff: Percentile for outlier removal
        """
        if locus is None:
            locus = ["chr21", 0, 250000000]
        
        # Create save directory
        save_dir = f"{self.savedir}/{bios_name}_{experiment}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Get prediction data
        pred_imputed = self._get_prediction_data(prediction_dict, bios_name, experiment, "imputed")
        pred_denoised = self._get_prediction_data(prediction_dict, bios_name, experiment, "denoised")
        
        # Get ground truth data if available
        obs_data = None
        if data_handler is not None:
            try:
                obs_data = self._get_ground_truth_data(data_handler, bios_name, experiment, locus)
            except Exception as e:
                print(f"Warning: Could not load ground truth data: {e}")
        
        if obs_data is None:
            print("Warning: No ground truth data available for scatter plot")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 3, figure=fig)
        
        # Main scatter plot (imputed)
        ax_scatter_imp = fig.add_subplot(gs[1, 0])
        obs_count = obs_data['obs_count']
        pred_count_imp = pred_imputed['pred_count']
        
        # Remove outliers
        obs_percentile = np.percentile(obs_count, percentile_cutoff)
        pred_percentile_imp = np.percentile(pred_count_imp, percentile_cutoff)
        mask_imp = (obs_count <= obs_percentile) & (pred_count_imp <= pred_percentile_imp)
        
        ax_scatter_imp.scatter(obs_count[mask_imp], pred_count_imp[mask_imp], 
                              alpha=0.5, s=1, color='red')
        ax_scatter_imp.set_xlabel('Observed Count')
        ax_scatter_imp.set_ylabel('Predicted Count (Imputed)')
        ax_scatter_imp.set_title(f'{experiment} - Imputed')
        
        # Add diagonal line
        min_val = min(obs_count[mask_imp].min(), pred_count_imp[mask_imp].min())
        max_val = max(obs_count[mask_imp].max(), pred_count_imp[mask_imp].max())
        ax_scatter_imp.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Main scatter plot (denoised)
        ax_scatter_den = fig.add_subplot(gs[1, 1])
        pred_count_den = pred_denoised['pred_count']
        
        # Remove outliers
        pred_percentile_den = np.percentile(pred_count_den, percentile_cutoff)
        mask_den = (obs_count <= obs_percentile) & (pred_count_den <= pred_percentile_den)
        
        ax_scatter_den.scatter(obs_count[mask_den], pred_count_den[mask_den], 
                              alpha=0.5, s=1, color='green')
        ax_scatter_den.set_xlabel('Observed Count')
        ax_scatter_den.set_ylabel('Predicted Count (Denoised)')
        ax_scatter_den.set_title(f'{experiment} - Denoised')
        
        # Add diagonal line
        min_val = min(obs_count[mask_den].min(), pred_count_den[mask_den].min())
        max_val = max(obs_count[mask_den].max(), pred_count_den[mask_den].max())
        ax_scatter_den.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Marginal histograms
        ax_hist_obs = fig.add_subplot(gs[0, 0])
        ax_hist_obs.hist(obs_count[mask_imp], bins=50, alpha=0.7, color='blue', density=True)
        ax_hist_obs.set_ylabel('Density')
        ax_hist_obs.set_title('Observed')
        
        ax_hist_imp = fig.add_subplot(gs[2, 0])
        ax_hist_imp.hist(pred_count_imp[mask_imp], bins=50, alpha=0.7, color='red', density=True)
        ax_hist_imp.set_xlabel('Count')
        ax_hist_imp.set_ylabel('Density')
        
        ax_hist_den = fig.add_subplot(gs[2, 1])
        ax_hist_den.hist(pred_count_den[mask_den], bins=50, alpha=0.7, color='green', density=True)
        ax_hist_den.set_xlabel('Count')
        ax_hist_den.set_ylabel('Density')
        
        # Remove unused subplot
        fig.delaxes(fig.add_subplot(gs[0, 1]))
        fig.delaxes(fig.add_subplot(gs[0, 2]))
        fig.delaxes(fig.add_subplot(gs[1, 2]))
        fig.delaxes(fig.add_subplot(gs[2, 2]))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/count_scatter_marginals.png", dpi=300)
        plt.savefig(f"{save_dir}/count_scatter_marginals.svg", format="svg")
        plt.close()
    
    def signal_scatter_with_marginals(self, prediction_dict: Dict[str, Any], 
                                    bios_name: str, experiment: str, 
                                    data_handler=None, locus: List = None,
                                    share_axes: bool = True, 
                                    percentile_cutoff: int = 99):
        """
        Generate signal scatter plot with marginal distributions.
        
        Args:
            prediction_dict: Dictionary from CANDIPredictor.predict_biosample()
            bios_name: Name of biosample
            experiment: Name of experiment
            data_handler: CANDIDataHandler instance for ground truth
            locus: Genomic locus (default: chr21)
            share_axes: Whether to share axes between subplots
            percentile_cutoff: Percentile for outlier removal
        """
        if locus is None:
            locus = ["chr21", 0, 250000000]
        
        # Create save directory
        save_dir = f"{self.savedir}/{bios_name}_{experiment}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Get prediction data
        pred_imputed = self._get_prediction_data(prediction_dict, bios_name, experiment, "imputed")
        pred_denoised = self._get_prediction_data(prediction_dict, bios_name, experiment, "denoised")
        
        # Get ground truth data if available
        obs_data = None
        if data_handler is not None:
            try:
                obs_data = self._get_ground_truth_data(data_handler, bios_name, experiment, locus)
            except Exception as e:
                print(f"Warning: Could not load ground truth data: {e}")
        
        if obs_data is None:
            print("Warning: No ground truth data available for scatter plot")
            return
        
        # Apply arcsinh transform
        obs_signal = np.sinh(obs_data['obs_pval'])
        pred_signal_imp = np.sinh(pred_imputed['pred_pval'])
        pred_signal_den = np.sinh(pred_denoised['pred_pval'])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 3, figure=fig)
        
        # Main scatter plot (imputed)
        ax_scatter_imp = fig.add_subplot(gs[1, 0])
        
        # Remove outliers
        obs_percentile = np.percentile(obs_signal, percentile_cutoff)
        pred_percentile_imp = np.percentile(pred_signal_imp, percentile_cutoff)
        mask_imp = (obs_signal <= obs_percentile) & (pred_signal_imp <= pred_percentile_imp)
        
        ax_scatter_imp.scatter(obs_signal[mask_imp], pred_signal_imp[mask_imp], 
                              alpha=0.5, s=1, color='red')
        ax_scatter_imp.set_xlabel('Observed Signal')
        ax_scatter_imp.set_ylabel('Predicted Signal (Imputed)')
        ax_scatter_imp.set_title(f'{experiment} - Imputed')
        
        # Add diagonal line
        min_val = min(obs_signal[mask_imp].min(), pred_signal_imp[mask_imp].min())
        max_val = max(obs_signal[mask_imp].max(), pred_signal_imp[mask_imp].max())
        ax_scatter_imp.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Main scatter plot (denoised)
        ax_scatter_den = fig.add_subplot(gs[1, 1])
        
        # Remove outliers
        pred_percentile_den = np.percentile(pred_signal_den, percentile_cutoff)
        mask_den = (obs_signal <= obs_percentile) & (pred_signal_den <= pred_percentile_den)
        
        ax_scatter_den.scatter(obs_signal[mask_den], pred_signal_den[mask_den], 
                              alpha=0.5, s=1, color='green')
        ax_scatter_den.set_xlabel('Observed Signal')
        ax_scatter_den.set_ylabel('Predicted Signal (Denoised)')
        ax_scatter_den.set_title(f'{experiment} - Denoised')
        
        # Add diagonal line
        min_val = min(obs_signal[mask_den].min(), pred_signal_den[mask_den].min())
        max_val = max(obs_signal[mask_den].max(), pred_signal_den[mask_den].max())
        ax_scatter_den.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Marginal histograms
        ax_hist_obs = fig.add_subplot(gs[0, 0])
        ax_hist_obs.hist(obs_signal[mask_imp], bins=50, alpha=0.7, color='blue', density=True)
        ax_hist_obs.set_ylabel('Density')
        ax_hist_obs.set_title('Observed')
        
        ax_hist_imp = fig.add_subplot(gs[2, 0])
        ax_hist_imp.hist(pred_signal_imp[mask_imp], bins=50, alpha=0.7, color='red', density=True)
        ax_hist_imp.set_xlabel('Signal')
        ax_hist_imp.set_ylabel('Density')
        
        ax_hist_den = fig.add_subplot(gs[2, 1])
        ax_hist_den.hist(pred_signal_den[mask_den], bins=50, alpha=0.7, color='green', density=True)
        ax_hist_den.set_xlabel('Signal')
        ax_hist_den.set_ylabel('Density')
        
        # Remove unused subplot
        fig.delaxes(fig.add_subplot(gs[0, 1]))
        fig.delaxes(fig.add_subplot(gs[0, 2]))
        fig.delaxes(fig.add_subplot(gs[1, 2]))
        fig.delaxes(fig.add_subplot(gs[2, 2]))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/signal_scatter_marginals.png", dpi=300)
        plt.savefig(f"{save_dir}/signal_scatter_marginals.svg", format="svg")
        plt.close()
    
    def generate_all_plots(self, prediction_dict: Dict[str, Any], bios_name: str, 
                          experiment: str, data_handler=None, locus: List = None):
        """
        Generate all available plots for a specific experiment.
        
        Args:
            prediction_dict: Dictionary from CANDIPredictor.predict_biosample()
            bios_name: Name of biosample
            experiment: Name of experiment
            data_handler: CANDIDataHandler instance for ground truth
            locus: Genomic locus (default: chr21)
        """
        print(f"Generating all plots for {bios_name}/{experiment}")
        
        # Generate track plots
        self.count_track(prediction_dict, bios_name, experiment, data_handler, locus)
        self.signal_track(prediction_dict, bios_name, experiment, data_handler, locus)
        
        # Generate confidence plots
        self.count_confidence(prediction_dict, bios_name, experiment, data_handler, locus)
        self.signal_confidence(prediction_dict, bios_name, experiment, data_handler, locus)
        
        # Generate scatter plots
        self.count_scatter_with_marginals(prediction_dict, bios_name, experiment, data_handler, locus)
        self.signal_scatter_with_marginals(prediction_dict, bios_name, experiment, data_handler, locus)
        
        print(f"All plots saved to {self.savedir}/{bios_name}_{experiment}/")


def main():
    """CLI interface for CANDI visualization."""
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(
        description="CANDI Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots for a specific experiment
  python viz.py --predictions predictions.pkl \\
                --bios-name GM12878 \\
                --experiment H3K4me3 \\
                --data-path /path/to/DATA_CANDI_MERGED

  # Generate specific plot types
  python viz.py --predictions predictions.pkl \\
                --bios-name GM12878 \\
                --experiment H3K4me3 \\
                --plot-types count_track signal_track \\
                --output-dir plots/
        """
    )
    
    # Required arguments
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions pickle file from pred.py')
    parser.add_argument('--bios-name', type=str, required=True,
                       help='Name of biosample')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Name of experiment')
    
    # Optional arguments
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to dataset directory for ground truth data')
    parser.add_argument('--dataset', type=str, default='merged', choices=['merged', 'eic'],
                       help='Dataset type (default: merged)')
    parser.add_argument('--locus', type=str, nargs=3, default=['chr21', '0', '250000000'],
                       help='Genomic locus as chrom start end (default: chr21 0 250000000)')
    parser.add_argument('--plot-types', type=str, nargs='+', 
                       default=['all'],
                       choices=['all', 'count_track', 'signal_track', 'count_confidence', 
                               'signal_confidence', 'count_scatter', 'signal_scatter'],
                       help='Types of plots to generate')
    parser.add_argument('--output-dir', type=str, default='models/evals/',
                       help='Output directory for plots')
    parser.add_argument('--resolution', type=int, default=25,
                       help='Genomic resolution in bp')
    
    args = parser.parse_args()
    
    # Parse locus
    locus = [args.locus[0], int(args.locus[1]), int(args.locus[2])]
    
    try:
        # Load predictions
        with open(args.predictions, 'rb') as f:
            prediction_dict = pickle.load(f)
        
        # Setup data handler if data path provided
        data_handler = None
        if args.data_path:
            from data import CANDIDataHandler
            data_handler = CANDIDataHandler(
                base_path=args.data_path,
                resolution=args.resolution,
                dataset_type=args.dataset,
                DNA=True
            )
            data_handler._load_files()
            print(f"Loaded data handler for {args.dataset} dataset")
        
        # Initialize visualizer
        visualizer = VISUALS_CANDI(resolution=args.resolution, savedir=args.output_dir)
        
        # Generate plots
        if 'all' in args.plot_types:
            visualizer.generate_all_plots(prediction_dict, args.bios_name, 
                                        args.experiment, data_handler, locus)
        else:
            for plot_type in args.plot_types:
                if plot_type == 'count_track':
                    visualizer.count_track(prediction_dict, args.bios_name, 
                                         args.experiment, data_handler, locus)
                elif plot_type == 'signal_track':
                    visualizer.signal_track(prediction_dict, args.bios_name, 
                                          args.experiment, data_handler, locus)
                elif plot_type == 'count_confidence':
                    visualizer.count_confidence(prediction_dict, args.bios_name, 
                                              args.experiment, data_handler, locus)
                elif plot_type == 'signal_confidence':
                    visualizer.signal_confidence(prediction_dict, args.bios_name, 
                                               args.experiment, data_handler, locus)
                elif plot_type == 'count_scatter':
                    visualizer.count_scatter_with_marginals(prediction_dict, args.bios_name, 
                                                          args.experiment, data_handler, locus)
                elif plot_type == 'signal_scatter':
                    visualizer.signal_scatter_with_marginals(prediction_dict, args.bios_name, 
                                                           args.experiment, data_handler, locus)
        
        print(f"Visualization complete. Plots saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
