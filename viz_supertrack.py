#!/usr/bin/env python3
"""
Supertrack Visualization Module

This module performs single-parameter sweeps over metadata prompts and visualizes
how performance metrics change as a function of prompt parameters.

It generates three types of visualizations:
1. Multi-panel metric summary plots (bar/line plots)
2. Genomic track stacks comparing predictions across sweep values
3. Metrics CSV export for further analysis

Author: EpiDenoise Team
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import seaborn as sns

from predict_assay import SingleAssayPredictor


class SupertrackVisualizer:
    """
    Parameter sweep visualizer for analyzing model sensitivity to metadata prompts.
    
    This class extends SingleAssayPredictor to perform parameter sweeps and
    generate comprehensive visualizations showing how predictions change as
    metadata prompts vary.
    """
    
    def __init__(self, model_dir: str, data_path: str, dataset_type: str,
                 bios_name: str, assay_name: str, task: str,
                 sweep_param: str, sweep_values: List,
                 baseline_spec_path: str, uniform_prompt: bool = False,
                 locus: List = None, dsf: int = 1, output_dir: str = None,
                 context_length: int = 1200, resolution: int = 25, DNA: bool = True):
        """
        Initialize supertrack visualizer.
        
        Args:
            model_dir: Path to model directory
            data_path: Path to dataset directory
            dataset_type: Type of dataset ("merged" or "eic")
            bios_name: Name of biosample
            assay_name: Name of assay to predict
            task: Task type ("impute" or "denoise")
            sweep_param: Metadata parameter to sweep
            sweep_values: List of values to test
            baseline_spec_path: Path to baseline metadata JSON
            uniform_prompt: Apply swept value to all assays (extreme test)
            locus: Genomic locus (default: chr21)
            dsf: Downsampling factor
            output_dir: Output directory for plots and metrics
            context_length: Context length for genomic windows
            resolution: Genomic resolution in bp
            DNA: Whether to use DNA sequence input
        """
        self.model_dir = model_dir
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.bios_name = bios_name
        self.assay_name = assay_name
        self.task = task
        self.sweep_param = sweep_param
        self.sweep_values = sweep_values
        self.uniform_prompt = uniform_prompt
        self.locus = locus if locus is not None else ["chr21", 0, 46709983]
        self.dsf = dsf
        # Create output directory structure: bios-name/assay-name/task_sweep-param
        base_output_dir = output_dir if output_dir is not None else model_dir
        self.output_dir = os.path.join(
            base_output_dir,
            bios_name,
            assay_name,
            f"{task}_{sweep_param}"
        )
        self.resolution = resolution
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize predictor
        print("Initializing SingleAssayPredictor...")
        self.predictor = SingleAssayPredictor(
            model_dir=model_dir,
            data_path=data_path,
            dataset_type=dataset_type,
            context_length=context_length,
            resolution=resolution,
            DNA=DNA
        )
        
        # Load baseline metadata specification
        self.baseline_spec = self._load_baseline_spec(baseline_spec_path)
        
        # Storage for results
        self.results_df = None
        self.predictions_cache = {}
        
        # Define example gene coordinates (from viz.py lines 174-183)
        self.example_gene_coords = [
            (33481539 // resolution, 33588914 // resolution),  # GART
            (25800151 // resolution, 26235914 // resolution),  # APP
            (31589009 // resolution, 31745788 // resolution),  # SOD1
            (39526359 // resolution, 39802081 // resolution),  # B3GALT5
            (33577551 // resolution, 33919338 // resolution),  # ITSN1
        ]
        self.gene_names = ['GART', 'APP', 'SOD1', 'B3GALT5', 'ITSN1']
        
        print(f"SupertrackVisualizer initialized")
        print(f"  Sweep parameter: {sweep_param}")
        print(f"  Sweep values: {sweep_values}")
        print(f"  Uniform prompt mode: {uniform_prompt}")
        print(f"  Output directory: {self.output_dir}")
    
    def _load_baseline_spec(self, spec_path: str) -> Dict:
        """Load baseline metadata specification from JSON file."""
        with open(spec_path, 'r') as f:
            spec = json.load(f)
        print(f"Loaded baseline metadata from {spec_path}")
        print(f"  Contains {len(spec)} assays")
        return spec
    
    def _create_sweep_spec(self, sweep_value) -> Dict:
        """
        Create metadata specification for a specific sweep value.
        
        Args:
            sweep_value: Value to use for swept parameter
            
        Returns:
            Dictionary with metadata specification for all assays
        """
        # Start with deep copy of baseline
        spec = copy.deepcopy(self.baseline_spec)
        
        if not self.uniform_prompt:
            # Only update target assay
            if self.assay_name in spec:
                spec[self.assay_name][self.sweep_param] = sweep_value
            else:
                # Create entry if not exists
                spec[self.assay_name] = {self.sweep_param: sweep_value}
        else:
            # Extreme test: apply to all assays
            for assay in spec:
                spec[assay][self.sweep_param] = sweep_value
        
        return spec
    
    def run_sweep(self):
        """Execute parameter sweep across all sweep values."""
        results_list = []
        
        print("\n" + "=" * 80)
        print(f"Running parameter sweep: {self.sweep_param}")
        print("=" * 80)
        
        for i, sweep_value in enumerate(self.sweep_values):
            print(f"\n[{i+1}/{len(self.sweep_values)}] Testing {self.sweep_param} = {sweep_value}")
            print("-" * 80)
            
            # Create metadata spec for this iteration
            sweep_spec = self._create_sweep_spec(sweep_value)
            
            # Run prediction
            start_time = time.time()
            result = self.predictor.predict_single_assay(
                bios_name=self.bios_name,
                assay_name=self.assay_name,
                task=self.task,
                locus=self.locus,
                dsf=self.dsf,
                fill_y_prompt_spec=sweep_spec,
                fill_prompt_mode="none"  # Always use spec, not fill
            )
            elapsed = time.time() - start_time
            print(f"  Prediction completed in {elapsed:.2f}s")
            
            # Compute metrics
            metrics = self.predictor.compute_metrics(result, quick=False)
            
            # Check if metrics computation failed
            if "error" in metrics:
                print(f"  ⚠️  Warning: {metrics['error']}")
                print(f"     Reason: {metrics.get('reason', 'Unknown')}")
                # Store with NaN values
                metrics = {
                    'bios': self.bios_name,
                    'assay': self.assay_name,
                    'task': self.task,
                    'comparison': self.task,
                    'C_MSE-GW': np.nan,
                    'C_Pearson-GW': np.nan,
                    'C_Spearman-GW': np.nan,
                    'P_MSE-GW': np.nan,
                    'P_Pearson-GW': np.nan,
                    'P_Spearman-GW': np.nan,
                    'Peak_AUCROC-GW': np.nan,
                }
            else:
                # Print key metrics
                print(f"  C_Pearson-GW:  {metrics.get('C_Pearson-GW', np.nan):.4f}")
                print(f"  P_Pearson-GW:  {metrics.get('P_Pearson-GW', np.nan):.4f}")
                print(f"  Peak_AUCROC-GW: {metrics.get('Peak_AUCROC-GW', np.nan):.4f}")
            
            # Add sweep information
            metrics['sweep_param'] = self.sweep_param
            metrics['sweep_value'] = sweep_value
            results_list.append(metrics)
            
            # Cache predictions for track visualization
            self.predictions_cache[sweep_value] = result
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results_list)
        
        print("\n" + "=" * 80)
        print("Parameter sweep completed!")
        print("=" * 80)
    
    def save_results(self, filename: str = None):
        """
        Save metrics DataFrame to CSV file.
        
        Args:
            filename: Output filename (default: auto-generated)
        """
        if self.results_df is None:
            print("Warning: No results to save. Run sweep first.")
            return
        
        if filename is None:
            filename = f"sweep_{self.sweep_param}_{self.bios_name}_{self.assay_name}_{self.task}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        self.results_df.to_csv(filepath, index=False)
        print(f"Saved metrics to {filepath}")
    
    def plot_metrics_summary_categorical(self):
        """Generate multi-panel bar plots for categorical parameter sweeps."""
        if self.results_df is None:
            print("Warning: No results to plot. Run sweep first.")
            return
        
        metrics_to_plot = [
            'C_Pearson-GW', 'C_Spearman-GW', 'C_MSE-GW',
            'P_Pearson-GW', 'P_Spearman-GW', 'P_MSE-GW',
            'Peak_AUCROC-GW'
        ]
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Filter out NaN values for cleaner plotting and sort by sweep value
            plot_data = self.results_df[['sweep_value', metric]].dropna()
            plot_data = plot_data.sort_values(by='sweep_value')
            
            if len(plot_data) > 0:
                # Bar plot
                x_pos = np.arange(len(plot_data))
                ax.bar(x_pos, plot_data[metric], alpha=0.7, color='steelblue')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(plot_data['sweep_value'], rotation=45, ha='right')
                ax.set_xlabel(self.sweep_param, fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(f'{metric} vs {self.sweep_param}', fontsize=14, fontweight='bold')
                
                # Set fixed y-axis limits for certain metrics
                if 'Pearson' in metric or 'Spearman' in metric:
                    ax.set_ylim([-1, 1])
                elif 'AUCROC' in metric:
                    ax.set_ylim([0, 1])
                
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric} vs {self.sweep_param}', fontsize=14, fontweight='bold')
        
        # Hide extra subplot
        if n_metrics < len(axes):
            axes[-1].axis('off')
        
        plt.suptitle(f'Metrics Summary: {self.bios_name} / {self.assay_name} ({self.task})',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save
        filename = f"metrics_summary_barplot_{self.sweep_param}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved bar plot to {filepath}")
        plt.close()
    
    def plot_metrics_summary_continuous(self):
        """Generate multi-panel line plots for continuous parameter sweeps."""
        if self.results_df is None:
            print("Warning: No results to plot. Run sweep first.")
            return
        
        metrics_to_plot = [
            'C_Pearson-GW', 'C_Spearman-GW', 'C_MSE-GW',
            'P_Pearson-GW', 'P_Spearman-GW', 'P_MSE-GW',
            'Peak_AUCROC-GW'
        ]
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Filter out NaN values and sort by sweep value
            plot_data = self.results_df[['sweep_value', metric]].dropna()
            plot_data = plot_data.sort_values(by='sweep_value')
            
            if len(plot_data) > 0:
                # Line plot
                ax.plot(plot_data['sweep_value'], plot_data[metric], 
                       marker='o', linewidth=2, markersize=8, color='steelblue')
                
                # Special handling for depth: use log2 x-axis
                if self.sweep_param == 'depth':
                    ax.set_xscale('log', base=2)
                
                ax.set_xlabel(self.sweep_param, fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(f'{metric} vs {self.sweep_param}', fontsize=14, fontweight='bold')
                
                # Set fixed y-axis limits for certain metrics
                if 'Pearson' in metric or 'Spearman' in metric:
                    ax.set_ylim([-1, 1])
                elif 'AUCROC' in metric:
                    ax.set_ylim([0, 1])
                
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric} vs {self.sweep_param}', fontsize=14, fontweight='bold')
        
        # Hide extra subplot
        if n_metrics < len(axes):
            axes[-1].axis('off')
        
        plt.suptitle(f'Metrics Summary: {self.bios_name} / {self.assay_name} ({self.task})',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save
        filename = f"metrics_summary_lineplot_{self.sweep_param}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved line plot to {filepath}")
        plt.close()
    
    def plot_genomic_tracks(self):
        """
        Generate genomic track visualizations across all genes and sweep values.
        
        Creates a multi-column figure where:
        - Columns = different genes (5 genes)
        - Rows = stacked tracks organized in 3 groups (count, pval, peak)
        - Each group: 1 observed + N predicted (one per sweep value)
        """
        if len(self.predictions_cache) == 0:
            print("Warning: No predictions cached. Run sweep first.")
            return
        
        n_genes = len(self.example_gene_coords)
        n_sweeps = len(self.sweep_values)
        
        # Layout: 5 columns (genes) × multiple rows
        # Rows organized in 3 groups (count, pval, peak)
        # Each group: 1 observed + n_sweeps predicted tracks
        n_rows_per_group = 1 + n_sweeps
        n_rows_total = 3 * n_rows_per_group
        n_cols = n_genes
        
        fig = plt.figure(figsize=(5 * n_cols, 2 * n_rows_total))
        gs = GridSpec(n_rows_total, n_cols, figure=fig, hspace=0.3, wspace=0.3)
        
        # Colors
        color_observed = 'blue'
        color_predicted = 'red'
        
        # GROUP 1: Count tracks
        print("Plotting count tracks...")
        group_start_row = 0
        for col_idx, gene_coord in enumerate(self.example_gene_coords):
            x_values = range(gene_coord[0], gene_coord[1])
            row_idx = group_start_row
            
            # Row: Observed count
            ax = fig.add_subplot(gs[row_idx, col_idx])
            obs_count = self.predictions_cache[self.sweep_values[0]]['ground_truth']['count']
            if obs_count is not None:
                obs_count_region = obs_count[gene_coord[0]:gene_coord[1]]
                ax.fill_between(x_values, 0, obs_count_region, color=color_observed, alpha=0.7)
            ax.set_ylabel('Obs Count', fontsize=8)
            ax.set_xticklabels([])
            ax.tick_params(labelsize=6)
            if col_idx == 0:
                ax.text(-0.15, 0.5, 'COUNT', transform=ax.transAxes,
                       fontsize=10, fontweight='bold', rotation=90, va='center')
            if row_idx == 0:  # Add gene name title
                ax.set_title(self.gene_names[col_idx], fontsize=12, fontweight='bold')
            row_idx += 1
            
            # Rows: Predicted counts for each sweep value
            for sweep_val in self.sweep_values:
                ax = fig.add_subplot(gs[row_idx, col_idx])
                pred_count = self.predictions_cache[sweep_val]['predictions']['count']
                pred_count_region = pred_count[gene_coord[0]:gene_coord[1]]
                ax.fill_between(x_values, 0, pred_count_region, color=color_predicted, alpha=0.7)
                # Format sweep value for label
                if isinstance(sweep_val, float):
                    label = f'{self.sweep_param}={sweep_val:.1e}'
                else:
                    label = f'{self.sweep_param}={sweep_val}'
                ax.set_ylabel(label, fontsize=7)
                ax.set_xticklabels([])
                ax.tick_params(labelsize=6)
                row_idx += 1
        
        # GROUP 2: P-value tracks
        print("Plotting p-value tracks...")
        group_start_row = n_rows_per_group
        for col_idx, gene_coord in enumerate(self.example_gene_coords):
            x_values = range(gene_coord[0], gene_coord[1])
            row_idx = group_start_row
            
            # Row: Observed p-value
            ax = fig.add_subplot(gs[row_idx, col_idx])
            obs_pval = self.predictions_cache[self.sweep_values[0]]['ground_truth']['pval']
            if obs_pval is not None:
                obs_pval_region = np.sinh(obs_pval[gene_coord[0]:gene_coord[1]])  # arcsinh transform
                ax.fill_between(x_values, 0, obs_pval_region, color=color_observed, alpha=0.7)
            ax.set_ylabel('Obs P-val', fontsize=8)
            ax.set_xticklabels([])
            ax.tick_params(labelsize=6)
            if col_idx == 0:
                ax.text(-0.15, 0.5, 'P-VALUE', transform=ax.transAxes,
                       fontsize=10, fontweight='bold', rotation=90, va='center')
            row_idx += 1
            
            # Rows: Predicted p-values for each sweep value
            for sweep_val in self.sweep_values:
                ax = fig.add_subplot(gs[row_idx, col_idx])
                pred_pval = self.predictions_cache[sweep_val]['predictions']['pval']
                pred_pval_region = np.sinh(pred_pval[gene_coord[0]:gene_coord[1]])
                ax.fill_between(x_values, 0, pred_pval_region, color=color_predicted, alpha=0.7)
                # Format sweep value for label
                if isinstance(sweep_val, float):
                    label = f'{self.sweep_param}={sweep_val:.1e}'
                else:
                    label = f'{self.sweep_param}={sweep_val}'
                ax.set_ylabel(label, fontsize=7)
                ax.set_xticklabels([])
                ax.tick_params(labelsize=6)
                row_idx += 1
        
        # GROUP 3: Peak tracks
        print("Plotting peak tracks...")
        group_start_row = 2 * n_rows_per_group
        for col_idx, gene_coord in enumerate(self.example_gene_coords):
            x_values = range(gene_coord[0], gene_coord[1])
            row_idx = group_start_row
            
            # Row: Observed peak
            ax = fig.add_subplot(gs[row_idx, col_idx])
            obs_peak = self.predictions_cache[self.sweep_values[0]]['ground_truth']['peak']
            if obs_peak is not None:
                obs_peak_region = obs_peak[gene_coord[0]:gene_coord[1]]
                ax.fill_between(x_values, 0, obs_peak_region, color=color_observed, alpha=0.7)
            ax.set_ylabel('Obs Peak', fontsize=8)
            ax.set_xticklabels([])
            ax.tick_params(labelsize=6)
            if col_idx == 0:
                ax.text(-0.15, 0.5, 'PEAK', transform=ax.transAxes,
                       fontsize=10, fontweight='bold', rotation=90, va='center')
            row_idx += 1
            
            # Rows: Predicted peaks for each sweep value
            for i, sweep_val in enumerate(self.sweep_values):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                pred_peak = self.predictions_cache[sweep_val]['predictions']['peak_scores']
                pred_peak_region = pred_peak[gene_coord[0]:gene_coord[1]]
                ax.fill_between(x_values, 0, pred_peak_region, color=color_predicted, alpha=0.7)
                # Format sweep value for label
                if isinstance(sweep_val, float):
                    label = f'{self.sweep_param}={sweep_val:.1e}'
                else:
                    label = f'{self.sweep_param}={sweep_val}'
                ax.set_ylabel(label, fontsize=7)
                ax.tick_params(labelsize=6)
                
                # Add x-label on last row
                if i == len(self.sweep_values) - 1:
                    ax.set_xlabel(f'chr21:{gene_coord[0]*self.resolution}-{gene_coord[1]*self.resolution}', fontsize=7)
                else:
                    ax.set_xticklabels([])
                row_idx += 1
        
        plt.suptitle(f'Genomic Tracks: {self.bios_name} / {self.assay_name} ({self.task})\n'
                     f'Sweep: {self.sweep_param}',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        filename = f"genomic_tracks_all_genes_{self.sweep_param}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved genomic tracks to {filepath}")
        
        # Also save SVG
        filepath_svg = os.path.join(self.output_dir, f"genomic_tracks_all_genes_{self.sweep_param}.svg")
        plt.savefig(filepath_svg, format='svg', bbox_inches='tight')
        print(f"Saved genomic tracks (SVG) to {filepath_svg}")
        
        plt.close()


def main():
    """CLI interface for supertrack visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize performance metrics vs metadata prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sweep over depth values
  python viz_supertrack.py --model-dir models/my_model \\
                           --data-path /path/to/DATA_CANDI_MERGED \\
                           --bios-name GM12878 \\
                           --assay-name H3K4me3 \\
                           --task impute \\
                           --dataset merged \\
                           --sweep-param depth \\
                           --sweep-values "10000000,30000000,50000000,100000000"

  # Sweep over sequencing platforms
  python viz_supertrack.py --model-dir models/my_model \\
                           --data-path /path/to/DATA_CANDI_MERGED \\
                           --bios-name GM12878 \\
                           --assay-name H3K27ac \\
                           --task impute \\
                           --dataset merged \\
                           --sweep-param sequencing_platform \\
                           --sweep-values "Illumina HiSeq 2000,Illumina HiSeq 4000,Illumina NovaSeq 6000"

  # Uniform prompt test (apply to all assays)
  python viz_supertrack.py --model-dir models/my_model \\
                           --data-path /path/to/DATA_CANDI_MERGED \\
                           --bios-name GM12878 \\
                           --assay-name H3K4me3 \\
                           --task impute \\
                           --dataset merged \\
                           --sweep-param read_length \\
                           --sweep-values "36,50,75,100" \\
                           --uniform-prompt
        """
    )
    
    # Core arguments (same as predict_assay.py)
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
    
    # Sweep-specific arguments
    parser.add_argument('--sweep-param', type=str, required=True,
                       choices=['depth', 'read_length', 'sequencing_platform', 'run_type'],
                       help='Metadata parameter to sweep')
    parser.add_argument('--sweep-values', type=str, required=True,
                       help='Comma-separated values to test (e.g., "10000000,30000000,50000000")')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: model-dir)')
    parser.add_argument('--prompt-spec', type=str, default='prompts/merged_mode.json',
                       help='JSON file with baseline metadata (default: prompts/merged_mode.json)')
    parser.add_argument('--uniform-prompt', action='store_true',
                       help='Apply swept value to all 35 assays (extreme test)')
    parser.add_argument('--locus', type=str, nargs=3, default=['chr21', '0', '46709983'],
                       help='Genomic locus as chrom start end (default: chr21 0 46709983)')
    parser.add_argument('--dsf', type=int, default=1,
                       help='Downsampling factor (default: 1)')
    
    args = parser.parse_args()
    
    # Parse locus
    locus = [args.locus[0], int(args.locus[1]), int(args.locus[2])]
    
    # Parse sweep values based on parameter type
    if args.sweep_param in ['depth', 'read_length']:
        sweep_values = [float(x.strip()) for x in args.sweep_values.split(',')]
    else:  # categorical
        sweep_values = [x.strip() for x in args.sweep_values.split(',')]
    
    # Default output_dir to model_dir
    if args.output_dir is None:
        args.output_dir = args.model_dir
    
    try:
        # Initialize visualizer
        print("=" * 80)
        print("SUPERTRACK VISUALIZER")
        print("=" * 80)
        visualizer = SupertrackVisualizer(
            model_dir=args.model_dir,
            data_path=args.data_path,
            dataset_type=args.dataset,
            bios_name=args.bios_name,
            assay_name=args.assay_name,
            task=args.task,
            sweep_param=args.sweep_param,
            sweep_values=sweep_values,
            baseline_spec_path=args.prompt_spec,
            uniform_prompt=args.uniform_prompt,
            locus=locus,
            dsf=args.dsf,
            output_dir=args.output_dir
        )
        
        # Run parameter sweep
        start_time = time.time()
        visualizer.run_sweep()
        sweep_time = time.time() - start_time
        
        # Save metrics CSV
        print("\nSaving results...")
        visualizer.save_results()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        if args.sweep_param in ['depth', 'read_length']:
            visualizer.plot_metrics_summary_continuous()
        else:
            visualizer.plot_metrics_summary_categorical()
        
        visualizer.plot_genomic_tracks()
        
        print("\n" + "=" * 80)
        print("✅ All tasks completed successfully!")
        print(f"   Total execution time: {sweep_time:.2f} seconds")
        print(f"   Outputs saved to: {args.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

