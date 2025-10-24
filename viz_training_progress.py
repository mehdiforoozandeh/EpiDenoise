#!/usr/bin/env python3
"""
Training Progress Visualization Script

This script reads training progress CSV files and generates a comprehensive
multipanel figure visualizing training metrics with scatter plots sized by
num_mask and smooth trend lines.

Usage:
    python viz_training_progress.py <csv_file_path>

The script will save a PNG file with the same name as the input CSV file
(but with .png extension instead of .csv).
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path
import sys


def load_and_prepare_data(csv_path):
    """
    Load CSV data and prepare it for visualization.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Prepared dataframe with iteration numbers
    """
    # Load CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from {csv_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # Check required columns
    required_columns = [
        'epoch', 'batch_idx', 'learning_rate', 'gradient_norm', 'num_mask',
        'imp_count_loss', 'obs_count_loss', 'imp_pval_loss', 'obs_pval_loss',
        'imp_peak_loss', 'obs_peak_loss', 'total_loss',
        'imp_count_r2_median', 'imp_count_spearman_median', 'imp_count_pearson_median',
        'imp_pval_r2_median', 'imp_pval_spearman_median', 'imp_pval_pearson_median',
        'imp_peak_auc_median', 'obs_count_r2_median', 'obs_count_spearman_median',
        'obs_count_pearson_median', 'obs_pval_r2_median', 'obs_pval_spearman_median',
        'obs_pval_pearson_median', 'obs_peak_auc_median'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Calculate iteration number
    # Estimate batches per epoch from the data
    max_batch_per_epoch = df.groupby('epoch')['batch_idx'].max()
    estimated_batches_per_epoch = int(max_batch_per_epoch.max()) + 1
    
    # Calculate iteration number: epoch * estimated_batches_per_epoch + batch_idx
    df['iteration'] = df['epoch'] * estimated_batches_per_epoch + df['batch_idx']
    
    print(f"Estimated batches per epoch: {estimated_batches_per_epoch}")
    print(f"Iteration range: {df['iteration'].min()} to {df['iteration'].max()}")
    
    return df


def create_exponential_moving_average(x, y, alpha=0.05):
    """
    Create exponential moving average trend line, skipping NaN and inf values.
    
    Args:
        x (array): X values
        y (array): Y values
        alpha (float): Smoothing factor (0 < alpha <= 1)
        
    Returns:
        tuple: (x_sorted, y_ema)
    """
    # Remove NaN and inf values
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return x_clean, y_clean
    
    # Sort by x values
    sort_idx = np.argsort(x_clean)
    x_sorted = x_clean[sort_idx]
    y_sorted = y_clean[sort_idx]
    
    # Calculate exponential moving average, skipping inf
    y_ema = np.zeros_like(y_sorted)
    prev_ema = None
    for i in range(len(y_sorted)):
        if np.isinf(y_sorted[i]) or np.isnan(y_sorted[i]):
            y_ema[i] = np.nan if prev_ema is None else prev_ema
        elif prev_ema is None:
            y_ema[i] = y_sorted[i]
            prev_ema = y_ema[i]
        else:
            y_ema[i] = alpha * y_sorted[i] + (1 - alpha) * prev_ema
            prev_ema = y_ema[i]
    
    return x_sorted, y_ema


def plot_metric_subplot(ax, df, metric_name, title, color='blue'):
    """
    Plot a single metric subplot with scatter points and trend line.
    
    Args:
        ax: Matplotlib axis object
        df: DataFrame with data
        metric_name (str): Name of the metric column
        title (str): Title for the subplot
        color (str): Color for the plot (not used, kept for compatibility)
    """
    # Get data
    x = df['iteration'].values
    y = df[metric_name].values
    sizes = df['num_mask'].values
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(sizes))
    x_clean = x[mask]
    y_clean = y[mask]
    sizes_clean = sizes[mask]
    
    if len(x_clean) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Normalize sizes for scatter plot (range 10-100)
    if len(sizes_clean) > 0:
        size_min, size_max = sizes_clean.min(), sizes_clean.max()
        if size_max > size_min:
            sizes_normalized = 10 + 90 * (sizes_clean - size_min) / (size_max - size_min)
        else:
            sizes_normalized = np.full_like(sizes_clean, 50)
    else:
        sizes_normalized = np.full_like(x_clean, 50)
    
    # Plot scatter points with very transparent black
    # ax.scatter(x_clean, y_clean, s=sizes_normalized, alpha=0.01, c='black', edgecolors='none')
    
    # Create and plot exponential moving average trend line with color based on metric_name
    x_trend, y_trend = create_exponential_moving_average(x_clean, y_clean, alpha=0.005)
    if len(x_trend) > 0:
        if 'imp' in metric_name:
            trend_color = 'skyblue'
        elif 'obs' in metric_name:
            trend_color = 'salmon'
        else:
            trend_color = 'forestgreen'
        ax.plot(x_trend, y_trend, color=trend_color, linewidth=3, alpha=0.9)
    # Set x-axis limits to show full iteration range
    ax.set_xlim(df['iteration'].min(), df['iteration'].max())
    
    # Set y-axis limits based on metric type
    if 'r2' in metric_name.lower():
        ax.set_ylim(-1, 1)
    elif any(corr in metric_name.lower() for corr in ['spearman', 'pearson']):
        ax.set_ylim(0, 1)
    elif 'pval_loss' in metric_name.lower() or 'count_loss' in metric_name.lower():
        ax.set_ylim(-1, 3)
    elif 'gradient_norm' in metric_name.lower():
        # Use log scale for gradient norm
        ax.set_yscale('log')
    
    # Formatting
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def create_visualization(df, output_path):
    """
    Create the complete multipanel visualization.
    
    Args:
        df: DataFrame with training data
        output_path (str): Path to save the PNG file
    """
    # Create figure with subplots (10 rows, 3 columns)
    fig, axes = plt.subplots(10, 3, figsize=(20, 33), dpi=150)
    # fig.suptitle('Training Progress Visualization', fontsize=16, fontweight='bold')
    
    # Define the layout of metrics (8 rows × 3 columns, removing num_mask)
    metrics_layout = [
        # Row 1: Training Dynamics (removed num_mask)
        [('learning_rate', 'Learning Rate', 'blue'),
         ('gradient_norm', 'Gradient Norm', 'green'),
         (None, None, None)],  # Empty
        
        # Row 2: Count Loss (Imputation vs Observed)
        [('imp_count_loss', 'Imputation Count Loss', 'purple'),
         ('obs_count_loss', 'Observed Count Loss', 'orange'),
         (None, None, None)],  # Empty
        
        # Row 3: P-value Loss
        [('imp_pval_loss', 'Imputation P-value Loss', 'brown'),
         ('obs_pval_loss', 'Observed P-value Loss', 'pink'),
         (None, None, None)],  # Empty
        
        # Row 4: Peak Loss
        [('imp_peak_loss', 'Imputation Peak Loss', 'gray'),
         ('obs_peak_loss', 'Observed Peak Loss', 'olive'),
         ('total_loss', 'Total Loss', 'navy')],
        
        # Row 5: Count R2/Correlation (Imputation)
        [('imp_count_r2_median', 'Imp Count R²', 'blue'),
         ('imp_count_spearman_median', 'Imp Count Spearman', 'green'),
         ('imp_count_pearson_median', 'Imp Count Pearson', 'red')],
        
        # Row 6: P-value R2/Correlation (Imputation)
        [('imp_pval_r2_median', 'Imp P-value R²', 'purple'),
         ('imp_pval_spearman_median', 'Imp P-value Spearman', 'orange'),
         ('imp_pval_pearson_median', 'Imp P-value Pearson', 'brown')],
        
        # Row 7: Peak AUC (Imputation)
        [('imp_peak_auc_median', 'Imp Peak AUC', 'pink'),
         (None, None, None),  # Empty
         (None, None, None)],  # Empty
        
        # Row 8: Count R2/Correlation (Observed)
        [('obs_count_r2_median', 'Obs Count R²', 'gray'),
         ('obs_count_spearman_median', 'Obs Count Spearman', 'olive'),
         ('obs_count_pearson_median', 'Obs Count Pearson', 'navy')],
        
        # Row 9: P-value R2/Correlation (Observed) + Peak AUC
        [('obs_pval_r2_median', 'Obs P-value R²', 'blue'),
         ('obs_pval_spearman_median', 'Obs P-value Spearman', 'green'),
         ('obs_pval_pearson_median', 'Obs P-value Pearson', 'red')],
        
        # Row 10: Peak AUC (Observed)
        [('obs_peak_auc_median', 'Obs Peak AUC', 'purple'),
         (None, None, None),  # Empty
         (None, None, None)],  # Empty
    ]
    
    # Plot each subplot
    for row in range(10):
        for col in range(3):
            ax = axes[row, col]
            
            metric_info = metrics_layout[row][col]
            if metric_info[0] is not None:  # If there's a metric to plot
                metric_name, title, color = metric_info
                plot_metric_subplot(ax, df, metric_name, title, color)
            else:
                # Hide empty subplots
                ax.set_visible(False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show the plot (optional - comment out for headless environments)
    # plt.show()
    
    plt.close()


def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(
        description="Generate training progress visualization from CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python viz_training_progress.py training_progress.csv
    python viz_training_progress.py /path/to/training_progress_20251019_152903.csv
        """
    )
    
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the training progress CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output PNG file path (default: same as input but with .png extension)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = csv_path.with_suffix('.png')
    
    print(f"Input CSV: {csv_path}")
    print(f"Output PNG: {output_path}")
    
    # Load and prepare data
    df = load_and_prepare_data(csv_path)
    
    # Create visualization
    create_visualization(df, output_path)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
