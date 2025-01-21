import random
import torch
import pickle
import os, time, gc, psutil
from CANDI import *
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.decomposition import PCA
import umap, scipy
from difflib import SequenceMatcher

################################################################################
metrics_class = METRICS()
################################################################################

def viz_feature_importance(df, savedir="models/output/"):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    def plot_metric_heatmap(df, metric, title, min_avail=6):
        accessibility_assays = ["ATAC-seq", "DNase-seq"]
        histone_mods = ["H3K4me3", "H3K4me1", "H3K27ac", "H3K27me3", "H3K9me3", "H3K36me3"]

        # Calculate mean and standard deviation
        mean_pivot = df.pivot_table(values=metric, index='input', columns='output', aggfunc='mean')
        std_pivot = df.pivot_table(values=metric, index='input', columns='output', aggfunc='std')

        # Count number of available entries for each row and column
        available_counts_rows = mean_pivot.notna().sum(axis=1)
        available_counts_cols = mean_pivot.notna().sum(axis=0)
        
        # Filter rows and columns based on minimum availability
        mean_pivot = mean_pivot.loc[available_counts_rows >= min_avail, 
                                available_counts_cols >= min_avail]
        std_pivot = std_pivot.loc[available_counts_rows >= min_avail, 
                                available_counts_cols >= min_avail]

        # Create figure and axis
        plt.figure(figsize=(12, 8))
        
        # Create a normalized colormap for each column
        normalized_data = mean_pivot.copy()
        for col in mean_pivot.columns:
            col_data = mean_pivot[col]
            vmin, vmax = col_data.min(), col_data.max()
            
            # Normalize the column
            normalized_data[col] = (col_data - vmin) / (vmax - vmin)
        
        mask = normalized_data.isna()
        
        # Create heatmap using normalized data
        ax = sns.heatmap(normalized_data, annot=False, cmap='viridis', vmin=0, vmax=1)
        
        # Add annotations with both mean and std
        for i, row in enumerate(mean_pivot.index):
            for j, col in enumerate(mean_pivot.columns):
                if bool(
                    row == "accessibility" and col in accessibility_assays) or bool(
                        row == "histone_mods" and col in histone_mods) or row == col:
                    # Add X mark for matching input/output
                    ax.plot([j, j+1], [i, i+1], 'k-', linewidth=1.5)  # First line of X
                    ax.plot([j, j+1], [i+1, i], 'k-', linewidth=1.5)  # Second line of X

                elif mask.loc[row, col]:
                    # Add grey rectangle for NaN values
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='#F5DEB3'))

                mean_val = mean_pivot.loc[row, col]
                try:
                    std_val = std_pivot.loc[row, col]
                except:
                    std_val = 0

                if not np.isnan(mean_val) and np.isnan(std_val):  # Check if the value exists
                    plt.text(j + 0.5, i + 0.5, f'{mean_val:.2f}',
                            ha='center', va='center', fontsize=6,
                            color='white' if normalized_data.iloc[i, j] < 0.5 else 'black')

                elif not np.isnan(mean_val) and not np.isnan(std_val):  # Check if the value exists
                    plt.text(j + 0.5, i + 0.5, f'{mean_val:.2f}\n±{std_val:.2f}',
                            ha='center', va='center', fontsize=6,
                            color='white' if normalized_data.iloc[i, j] < 0.5 else 'black')
        
        plt.title(f'{title} - {metric}\n(mean ± std)')
        plt.tight_layout()
        plt.savefig(f'{savedir}/heatmap_{metric}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{savedir}/heatmap_{metric}.svg', dpi=300, bbox_inches='tight', format='svg')
        plt.close()

    def plot_metric_correlations(df, metrics):

        sns.pairplot(df[metrics], diag_kind='kde', size=1.5)
        plt.savefig(f'{savedir}/metric_correlations.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{savedir}/metric_correlations.svg', dpi=300, bbox_inches='tight', format='svg')
        plt.close()

    def plot_metric_clustermap(df, metric, title, min_avail=6):
        accessibility_assays = ["ATAC-seq", "DNase-seq"]
        histone_mods = ["H3K4me3", "H3K4me1", "H3K27ac", "H3K27me3", "H3K9me3", "H3K36me3"]

        # Calculate mean and standard deviation
        mean_pivot = df.pivot_table(values=metric, index='input', columns='output', aggfunc='mean')
        std_pivot = df.pivot_table(values=metric, index='input', columns='output', aggfunc='std')

        # Count number of available entries for each row and column
        available_counts_rows = mean_pivot.notna().sum(axis=1)
        available_counts_cols = mean_pivot.notna().sum(axis=0)
        
        # Filter rows and columns based on minimum availability
        mean_pivot = mean_pivot.loc[available_counts_rows >= min_avail, 
                            available_counts_cols >= min_avail]
        std_pivot = std_pivot.loc[available_counts_rows >= min_avail, 
                            available_counts_cols >= min_avail]

        # Create a normalized colormap for each column (for display only)
        normalized_data = mean_pivot.copy()
        for col in mean_pivot.columns:
            col_data = mean_pivot[col]
            vmin, vmax = col_data.min(), col_data.max()
            if pd.isna(vmin) or pd.isna(vmax):  # Handle columns with all NaN
                normalized_data[col] = pd.NA
            else:
                normalized_data[col] = (col_data - vmin) / (vmax - vmin)
        
        # Create mask for NaN values
        mask = normalized_data.isna()
        
        # Fill NaN values with the mean for clustering purposes
        # Use raw data for clustering instead of normalized data
        clustering_data = mean_pivot.fillna(mean_pivot.mean().mean())
        
        # Create clustered heatmap using raw data for clustering but normalized data for display
        g = sns.clustermap(
            data=clustering_data,  # Use raw data for clustering
            row_colors=None,
            col_colors=None,
            cmap='viridis',
            vmin=0,
            vmax=1,
            mask=mask,
            figsize=(14, 10),
            dendrogram_ratio=(.1, .2),
            xticklabels=True,
            yticklabels=True,
        )
        
        # After clustering, update the heatmap data with normalized values
        g.ax_heatmap.collections[0].set_array(normalized_data.values[np.array(g.dendrogram_row.reordered_ind)][:, np.array(g.dendrogram_col.reordered_ind)].ravel())
        
        ax = g.ax_heatmap
        
        # Get the reordered index and columns after clustering
        row_order = g.dendrogram_row.reordered_ind
        col_order = g.dendrogram_col.reordered_ind
        reordered_rows = [mean_pivot.index[i] for i in row_order]
        reordered_cols = [mean_pivot.columns[i] for i in col_order]
        
        # Add X marks and grey cells
        for i, row in enumerate(reordered_rows):
            for j, col in enumerate(reordered_cols):
                if bool(
                    row == "accessibility" and col in accessibility_assays) or bool(
                        row == "histone_mods" and col in histone_mods) or row == col:
                    # Add X mark for matching input/output
                    ax.plot([j, j+1], [i, i+1], 'k-', linewidth=1.5)
                    ax.plot([j, j+1], [i+1, i], 'k-', linewidth=1.5)
                elif mask.loc[row, col]:
                    # Add grey rectangle for NaN values
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='#F5DEB3'))
        
        # Add annotations
        for i, row in enumerate(reordered_rows):
            for j, col in enumerate(reordered_cols):
                if row != col:  # Skip annotations for matching input/output
                    mean_val = mean_pivot.loc[row, col]
                    try:
                        std_val = std_pivot.loc[row, col]
                    except:
                        std_val = 0

                    if not pd.isna(mean_val):  # Only add text for non-NaN values
                        if pd.isna(std_val):
                            ax.text(j + 0.5, i + 0.5, f'{mean_val:.2f}',
                                    ha='center', va='center', fontsize=7,
                                    color='white' if normalized_data.loc[row, col] < 0.5 else 'black')
                        else:
                            ax.text(j + 0.5, i + 0.5, f'{mean_val:.2f}\n±{std_val:.2f}',
                                    ha='center', va='center', fontsize=7,
                                    color='white' if normalized_data.loc[row, col] < 0.5 else 'black')
        
        # Adjust layout and labels
        plt.suptitle(f'{title} - {metric}\n(mean ± std)', y=1.02)
        
        # Save figures
        plt.savefig(f'{savedir}/clustermap_{metric}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{savedir}/clustermap_{metric}.svg', dpi=300, bbox_inches='tight', format='svg')
        plt.close()

    metrics_to_plot = [
        'gw_pearson_count', 'gw_pearson_pval', 
        # 'gw_pp_count', 'gw_pp_pval', 
        'gw_spearman_count', 'gw_spearman_pval',
        # 'gene_pearson_count', 'gene_pearson_pval', 
        # 'gene_spearman_count', 'gene_spearman_pval',
        # 'prom_pearson_count', 'prom_pearson_pval', 
        # 'prom_spearman_count', 'prom_spearman_pval',
        # 'one_obs_pearson_count', 'one_obs_pearson_pval',  
        # 'one_obs_spearman_count', 'one_obs_spearman_pval',
        # 'one_imp_pearson_count', 'one_imp_pearson_pval', 
        # 'one_imp_spearman_count', 'one_imp_spearman_pval',
        'peak_overlap_count', 'peak_overlap_pval'
        ]
        
    for metric in metrics_to_plot:
        plot_metric_clustermap(df, metric, 'Assay Prediction Performance')
        # plot_metric_heatmap(df, metric, 'Assay Prediction Performance')
    
    # plot_metric_correlations(df, metrics_to_plot)

def viz_eic_metrics(data, savedir="models/output/"):
    # Split the data into IMP and UPS DataFrames based on the `comparison` column
    imp_data = data[data['comparison'] == 'imputed']
    ups_data = data[data['comparison'] == 'upsampled']
    metrics = [col for col in imp_data.columns if col not in ['comparison', 'experiment', 'bios_name']]
    
    def create_individual_boxplot(data, metrics, category):
        for metric in metrics:
            # Skip specific metrics
            if "pp" in metric.lower() or "r2" in metric.lower():
                continue
            plt.figure(figsize=(8, 6))
            sns.boxplot(
                data=data,
                x='experiment',
                y=metric,
                palette='viridis'  # Customize color palette if needed
            )
            plt.title(f'{category} Metric: {metric}', fontsize=14)
            plt.ylabel(metric, fontsize=12)
            plt.xlabel('Experiment', fontsize=12)
            plt.xticks(rotation=90)  # Rotate x-axis labels for readability
            plt.tight_layout()
            # Use log scale for y-axis if metric is MSE
            if "mse" in metric.lower():
                plt.yscale('log')
            # Save the plot to the specified directory
            save_path = f"{savedir}/{category}_{metric}.png"
            plt.savefig(save_path)
            plt.show()
            plt.close()

    # Generate boxplots for IMP metrics
    if not imp_data.empty:
        create_individual_boxplot(imp_data, metrics, category="IMP")
    
    # Generate boxplots for UPS metrics
    if not ups_data.empty:
        create_individual_boxplot(ups_data, metrics, category="UPS")

def viz_full_metrics(data, savedir="models/output/"):
    # Separate metrics into two categories: imp_metrics and ups_metrics
    imp_metrics = [col for col in data.columns if 'imp' in col.lower()]
    ups_metrics = [col for col in data.columns if 'ups' in col.lower()]
    
    def create_individual_boxplot(data, metrics, category):
        for metric in metrics:
            if "pp" in metric.lower() or "r2" in metric.lower():
                continue
            plt.figure(figsize=(8, 6))
            sns.boxplot(
                data=data,
                x='experiment',
                y=metric,
                # hue='bios_name',  # Color points based on celltype
                # alpha=0.7,
                # palette='viridis'
            )
            plt.title(f'{category} Metric: {metric}', fontsize=14)
            plt.ylabel(metric, fontsize=12)
            plt.xlabel('Experiment', fontsize=12)
            plt.xticks(rotation=90)  # Rotate x-axis labels 90 degrees
            # plt.legend(title='Cell Type', loc='upper right', fontsize=10)
            plt.tight_layout()
            if "mse" in metric.lower():
                plt.yscale('log')
            # Save the plot to the specified directory
            plt.savefig(f"{savedir}/{category}_{metric}.png", dpi=300)
            plt.savefig(f"{savedir}/{category}_{metric}.svg", format="svg")
            # plt.show()
            # plt.close()

    # Create individual plots for imp_metrics
    if imp_metrics:
        create_individual_boxplot(data, imp_metrics, category="IMP")
    
    # Create individual plots for ups_metrics
    if ups_metrics:
        create_individual_boxplot(data, ups_metrics, category="UPS")

def viz_calibration(
    imp_pval_dist, ups_pval_dist, imp_count_dist, ups_count_dist,
    pval_true, count_true, title, savedir="models/output/"):

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # calibration curve
    imp_pval_calibration = confidence_calibration(imp_pval_dist, pval_true)
    ups_pval_calibration = confidence_calibration(ups_pval_dist, pval_true)
    imp_count_calibration = confidence_calibration(imp_count_dist, count_true)
    ups_count_calibration = confidence_calibration(ups_count_dist, count_true)

    fig = plot_calibration_grid(
        [imp_pval_calibration, ups_pval_calibration, imp_count_calibration, ups_count_calibration],
        ["Imputed signal", "Upsampled signal", "Imputed count", "Upsampled count"])
    fig.savefig(f"{savedir}/calibration_curve_{title}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{savedir}/calibration_curve_{title}.svg", format="svg", bbox_inches='tight')

    plt.close(fig)

def viz_calibration_eic(pval_dist, count_dist, pval_true, count_true, comparison, title, savedir="models/output/"):

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # calibration curve
    pval_calibration = confidence_calibration(pval_dist, pval_true)
    count_calibration = confidence_calibration(count_dist, count_true)

    fig = plot_calibration_grid(
        [pval_calibration, count_calibration],
        [f"{comparison} signal", f"{comparison} count"])
    fig.savefig(f"{savedir}/calibration_curve_{title}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{savedir}/calibration_curve_{title}.svg", format="svg", bbox_inches='tight')

    plt.close(fig)

def viz_eic_paper_comparison(res_dir="models/DEC18_RESULTS/"):
    team_id = {
        'Blind': -1,
        'Hongyang Li and Yuanfang Guan v1': 3393417,
        'Average': 100,
        'Hongyang Li and Yuanfang Guan': 3330254,
        'Hongyang Li and Yuanfang Guan v2': 3393418,
        'Song Lab 3': 3393580,
        'CostaLab': 3379312,
        'BrokenNodes_v3': 3393817,
        'imp1': 3393756,
        'Lavawizard': 3393574,
        'Guacamole': 3393847,
        'CUWA': 3393458,
        'CUImpute1': 3393860,
        'ICU': 3393861,
        'imp': 3393457,
        'LiPingChun': 3388185,
        'BrokenNodes': 3379072,
        'Aug2019Imputation': 3393128,
        'Avocado': 0,
        'BrokenNodes_v2': 3393606,
        'UIOWA Michaelson Lab': 3391272,
        'NittanyLions': 3344979,
        'KKT-ENCODE-Impute': 3386902,
        'Song Lab 2': 3393579,
        'Song Lab': 3389318,
        'NittanyLions2': 3393851
    }

    assay_id = {
        'M01': 'ATAC-seq',
        'M02': 'DNase-seq',
        'M03': 'H2AFZ',
        'M04': 'H2AK5ac',
        'M05': 'H2AK9ac',
        'M06': 'H2BK120ac',
        'M07': 'H2BK12ac',
        'M08': 'H2BK15ac',
        'M09': 'H2BK20ac',
        'M10': 'H2BK5ac',
        'M11': 'H3F3A',
        'M12': 'H3K14ac',
        'M13': 'H3K18ac',
        'M14': 'H3K23ac',
        'M15': 'H3K23me2',
        'M16': 'H3K27ac',
        'M17': 'H3K27me3',
        'M18': 'H3K36me3',
        'M19': 'H3K4ac',
        'M20': 'H3K4me1',
        'M21': 'H3K4me2',
        'M22': 'H3K4me3',
        'M23': 'H3K56ac',
        'M24': 'H3K79me1',
        'M25': 'H3K79me2',
        'M26': 'H3K9ac',
        'M27': 'H3K9me1',
        'M28': 'H3K9me2',
        'M29': 'H3K9me3',
        'M30': 'H3T11ph',
        'M31': 'H4K12ac',
        'M32': 'H4K20me1',
        'M33': 'H4K5ac',
        'M34': 'H4K8ac',
        'M35': 'H4K91ac'
    }

    cell_type_id = {
        'C01': 'adipose_tissue',
        'C02': 'adrenal_gland',
        'C03': 'adrenalglandembryonic',
        'C04': 'amnion',
        'C05': 'BE2C',
        'C06': 'brainmicrovascularendothelial_cell',
        'C07': 'Caco-2',
        'C08': 'cardiac_fibroblast',
        'C09': 'CD4-positivealpha-betamemoryTcell',
        'C10': 'chorion',
        'C11': 'dermismicrovascularlymphaticvesselendothelial_cell',
        'C12': 'DND-41',
        'C13': 'endocrine_pancreas',
        'C14': 'ES-I3',
        'C15': 'G401',
        'C16': 'GM06990',
        'C17': 'H1',
        'C18': 'H9',
        'C19': 'HAP-1',
        'C20': 'heartleftventricle',
        'C21': 'hematopoieticmultipotentprogenitor_cell',
        'C22': 'HL-60',
        'C23': 'IMR-90',
        'C24': 'K562',
        'C25': 'KMS-11',
        'C26': 'lowerlegskin',
        'C27': 'mesenchymalstemcell',
        'C28': 'MG63',
        'C29': 'myoepithelialcellofmammarygland',
        'C30': 'NCI-H460',
        'C31': 'NCI-H929',
        'C32': 'neuralstemprogenitor_cell',
        'C33': 'occipital_lobe',
        'C34': 'OCI-LY7',
        'C35': 'omentalfatpad',
        'C36': 'peripheralbloodmononuclear_cell',
        'C37': 'prostate',
        'C38': 'RWPE2',
        'C39': 'SJCRH30',
        'C40': 'SJSA1',
        'C41': 'SK-MEL-5',
        'C42': 'skin_fibroblast',
        'C43': 'skinofbody',
        'C44': 'T47D',
        'C45': 'testis',
        'C46': 'trophoblast_cell',
        'C47': 'upperlobeofleftlung',
        'C48': 'urinary_bladder',
        'C49': 'uterus',
        'C50': 'vagina',
        'C51': 'WERI-Rb-1'
    }
    def comparison_stripplot(merged_df, name="val", savedir=res_dir):
        """
        Create separate stripplots for each metric comparing teams across assays,
        with jittered x positions to prevent overlapping points.
        """
        # Get unique teams automatically
        teams_to_visualize = ['CANDI'] + [team for team in merged_df['team'].unique() if team != 'CANDI']
        num_teams = len(teams_to_visualize)
        
        # Generate color map automatically using a colorblind-friendly palette
        # Using 'Dark2' colormap which is colorblind-friendly and supports up to 8 colors
        # For more teams, fall back to 'tab20' which supports up to 20 colors
        if num_teams <= 8:
            color_palette = plt.get_cmap('Dark2')
            colors = [color_palette(i) for i in np.linspace(0, 1, 8)]
        else:
            color_palette = plt.get_cmap('tab20')
            colors = [color_palette(i) for i in np.linspace(0, 1, 20)]
        
        # Create dictionaries mapping teams to colors and markers
        custom_colors = {team: colors[i % len(colors)] for i, team in enumerate(teams_to_visualize)}
        
        # List of distinct markers available in matplotlib
        marker_list = ['*', 's', '^', 'D', 'v', 'o', 'p', 'h', '8', '+', 'x', 'P']
        custom_markers = {team: marker_list[i % len(marker_list)] for i, team in enumerate(teams_to_visualize)}
        
        # Define marker sizes - use larger size for first team (assumed to be the main model)
        marker_sizes = {team: 150 if team == "CANDI" else 30 for i, team in enumerate(teams_to_visualize)}
        
        # Setup for plotting
        unique_assays = merged_df['assay'].unique()
        assay_positions = {assay: i for i, assay in enumerate(unique_assays)}
        jitter_range = 0.3
        
        metrics = [
            ('gwcorr', 'Genome-wide Correlation'),
            ('gwspear', 'Genome-wide Spearman'),
            ('mse', 'MSE')
        ]
        
        for metric, title in metrics:
            plt.figure(figsize=(14, 7))
            
            for team in teams_to_visualize:
                team_data = merged_df[merged_df['team'] == team]
                
                x_positions = [assay_positions[assay] + np.random.uniform(-jitter_range, jitter_range) 
                            for assay in team_data['assay']]
                
                # Plot scatter points with team-specific marker size
                plt.scatter(
                    x_positions, 
                    team_data[metric],
                    label=team,
                    alpha=0.8,
                    color=custom_colors[team],
                    marker=custom_markers[team],
                    s=marker_sizes[team],
                    edgecolor='none'
                )
            
            # Customize plot
            plt.xlabel('Assay', fontsize=14)  # Increased fontsize by 50%
            plt.ylabel(title, fontsize=14)  # Increased fontsize by 50%
            plt.xticks(list(assay_positions.values()), list(assay_positions.keys()), rotation=90, fontsize=12)  # Increased fontsize by 50%
            for i in range(len(unique_assays) - 1):
                plt.axvline(x=i + 0.5, color='k', linestyle='--', linewidth=0.5)

            # plt.title(name)
            
            if metric == 'mse':
                plt.yscale('log')
                plt.ylabel('MSE', fontsize=14)  # Increased fontsize by 50%
            
            # Custom legend with different marker sizes
            handles = [
                plt.Line2D(
                    [0], [0],
                    marker=custom_markers[team],
                    color='w',
                    label=team,
                    markerfacecolor=custom_colors[team],
                    markersize=15 if team == "CANDI" else 10 
                ) for i, team in enumerate(teams_to_visualize)
            ]
            
            # Adjust legend position and columns based on number of teams
            ncols = min(8, num_teams)  # Maximum 4 columns
            plt.legend(
                handles=handles,
                loc='upper center',
                ncol= ncols,
                bbox_to_anchor=(0.5, 1.08),
                frameon=False,
                # handlelength=2.5  # Increase the length of the legend handle
            )
            
            plt.tight_layout()
            plt.savefig(f"{savedir}/eic_paper_comparison_{name}_{metric}.png", dpi=300)
            plt.savefig(f"{savedir}/eic_paper_comparison_{name}_{metric}.svg", format="svg")
            plt.clf()
            plt.close()
            plt.cla()
            # plt.show()

    def comparison_boxplot(merged_df, name="val", savedir=res_dir):
        """
        Create separate boxplots for each metric comparing teams across assays.
        """
        # Get unique teams automatically
        teams_to_visualize = ['CANDI'] + [team for team in merged_df['team'].unique() if team != 'CANDI']
        num_teams = len(teams_to_visualize)
        
        # Generate color map automatically using a colorblind-friendly palette
        if num_teams <= 8:
            color_palette = plt.get_cmap('Dark2')
            colors = [color_palette(i) for i in np.linspace(0, 1, 8)]
        else:
            color_palette = plt.get_cmap('tab20')
            colors = [color_palette(i) for i in np.linspace(0, 1, 20)]
        
        # Create dictionary mapping teams to colors
        custom_colors = {team: colors[i % len(colors)] for i, team in enumerate(teams_to_visualize)}
        
        # Setup for plotting
        unique_assays = merged_df['assay'].unique()
        assay_positions = {assay: i for i, assay in enumerate(unique_assays)}
        
        metrics = [
            ('gwcorr', 'Genome-wide Correlation'),
            ('gwspear', 'Genome-wide Spearman'),
            ('mse', 'MSE')
        ]
        
        for metric, title in metrics:
            plt.figure(figsize=(14, 7))
            
            for i, assay in enumerate(unique_assays):
                assay_data = merged_df[merged_df['assay'] == assay]
                
                # Create boxplot data for each team in the current assay
                boxplot_data = [
                    assay_data[assay_data['team'] == team][metric].dropna()
                    for team in teams_to_visualize
                ]
                
                # Boxplot for the current assay
                box = plt.boxplot(
                    boxplot_data,
                    positions=np.arange(len(teams_to_visualize)) + i * (len(teams_to_visualize) + 1),
                    patch_artist=True,
                    widths=0.6,
                    showcaps=False,
                    boxprops=dict(linewidth=1.2),
                    medianprops=dict(color='black'),
                    whiskerprops=dict(linewidth=1.2)
                )
                
                # Set boxplot colors
                for patch, team in zip(box['boxes'], teams_to_visualize):
                    patch.set_facecolor(custom_colors[team])
            
            # Customize plot
            plt.xlabel('Assay', fontsize=14)  # Increased fontsize by 50%
            plt.ylabel(title, fontsize=14)  # Increased fontsize by 50%
            xticks_positions = [
                i * (len(teams_to_visualize) + 1) + len(teams_to_visualize) / 2 - 0.5
                for i in range(len(unique_assays))
            ]
            plt.xticks(xticks_positions, unique_assays, rotation=90, fontsize=12)  # Increased fontsize by 50%
            
            for i in range(len(unique_assays) - 1):
                plt.axvline(x=(i + 1) * (len(teams_to_visualize) + 1) - 1, color='k', linestyle='--', linewidth=0.5)
            
            if metric == 'mse':
                plt.yscale('log')
                plt.ylabel('MSE', fontsize=14)  # Increased fontsize by 50%
            
            # Custom legend
            handles = [
                plt.Line2D([0], [0], color=custom_colors[team], lw=4, label=team)
                for team in teams_to_visualize
            ]
            ncols = min(8, num_teams)
            plt.legend(
                handles=handles,
                loc='upper center',
                ncol=ncols,
                bbox_to_anchor=(0.5, 1.08),
                frameon=False,
                fontsize=12  # Increased fontsize by 50%
            )
            
            plt.tight_layout()
            plt.savefig(f"{savedir}/eic_paper_comparison_{name}_{metric}_boxplot.png", dpi=300)
            plt.savefig(f"{savedir}/eic_paper_comparison_{name}_{metric}_boxplot.svg", format="svg")
            plt.clf()
            plt.close()
            plt.cla()

    # Reverse the team_id dictionary to map team names to their IDs
    reversed_team_id = {v: k for k, v in team_id.items()}
    reversed_assay_id = {v: k for k, v in assay_id.items()}
    reversed_cell_type_id = {v: k for k, v in cell_type_id.items()}

    ####################################################################################
    # compare eic paper val res with candieic val res
    ####################################################################################

    val_res_raw = pd.read_csv(f"{res_dir}/eic_paper/13059_2023_2915_MOESM6_ESM.csv")
    val_res_raw = val_res_raw[val_res_raw["bootstraip_id"] == 1].reset_index(drop=True)

    val_res_raw = val_res_raw[val_res_raw["team"].isin([
        "Avocado_p0", "Average", "BrokenNodes", "LiPingChun",
        "HongyangLiandYuanfangGuan", "KKT-ENCODE-Impute-model_1"])].reset_index(drop=True)
    candieic_val_res = pd.read_csv(f"{res_dir}/eic_val_metrics.csv")
    candieic_val_res = candieic_val_res[candieic_val_res["comparison"] == "imputed"].reset_index(drop=True)

    df1 = val_res_raw[['team', 'assay', 'cell', "mse", "gwcorr", "gwspear"]]
    df2 = candieic_val_res[["bios_name", "experiment", "pval_ups_gw_mse", "pval_ups_gw_pearson", "pval_ups_gw_spearman"]]

    df1['cell'] = df1['cell'].apply(lambda x: x.replace(" ", "_"))
    df1['cell'] = df1['cell'].str.replace('H1-hESC', 'H1')
    df2['cell'] = df2['bios_name'].apply(lambda x: x.replace("V_", ""))
    df2.drop(columns=['bios_name'], inplace=True)

    df2.rename(columns={
            'experiment': "assay",
            'pval_ups_gw_mse': 'mse',
            'pval_ups_gw_pearson': 'gwcorr',
            'pval_ups_gw_spearman': 'gwspear'}, inplace=True)

    df2['team'] = 'CANDI'
    merged_df = pd.concat([df1, df2], ignore_index=True)

    print(merged_df)
    comparison_stripplot(merged_df)
    comparison_boxplot(merged_df)

    ####################################################################################
    ####################################################################################

    candieic_blind_res = pd.read_csv(f"{res_dir}/eic_test_metrics.csv")
    candieic_blind_res = candieic_blind_res[candieic_blind_res["comparison"] == "imputed"].reset_index(drop=True)

    df1 = candieic_blind_res[["bios_name", "experiment", "pval_ups_gw_mse", "pval_ups_gw_pearson", "pval_ups_gw_spearman"]]
    df1['cell'] = df1['bios_name'].apply(lambda x: x.replace("B_", ""))
    df1.drop(columns=['bios_name'], inplace=True)
    df1.rename(columns={
            'experiment': "assay",
            'pval_ups_gw_mse': 'mse',
            'pval_ups_gw_pearson': 'gwcorr',
            'pval_ups_gw_spearman': 'gwspear'}, inplace=True)
    df1['team'] = 'CANDI'

    ####################################################################################
    # compare eic paper blind res with candieic blind res raw
    ####################################################################################
    blind_res_raw = pd.read_csv(f"{res_dir}/eic_paper/13059_2023_2915_MOESM2_ESM.csv")
    blind_res_raw = blind_res_raw[blind_res_raw["bootstrap_id"] == 1].reset_index(drop=True)

    df2 = blind_res_raw[['team_id', 'assay', 'cell', "mse", "gwcorr", "gwspear"]]
    df2['team'] = df2['team_id'].apply(lambda x: reversed_team_id[x])
    df2 = df2[df2["team"].isin([
        "Avocado", "Average", "Hongyang Li and Yuanfang Guan v1", "Lavawizard",
        "Guacamole", "imp"])].reset_index(drop=True)

    df2.drop(columns=['team_id'], inplace=True)
    df2['cell'] = df2['cell'].apply(lambda x: cell_type_id[x])
    df2['assay'] = df2['assay'].apply(lambda x: assay_id[x])

    merged_df = pd.concat([df1, df2], ignore_index=True)

    # print(merged_df)
    comparison_stripplot(merged_df, "raw_blind")
    comparison_boxplot(merged_df, "raw_blind")
    
    ####################################################################################
    # compare eic paper blind res with candieic blind res after qnorm
    ####################################################################################
    blind_res_after_qnorm = pd.read_csv(f"{res_dir}/eic_paper/13059_2023_2915_MOESM3_ESM.csv")
    blind_res_after_qnorm = blind_res_after_qnorm[blind_res_after_qnorm["bootstrap_id"] == 1].reset_index(drop=True)

    df2 = blind_res_after_qnorm[['team_id', 'assay', 'cell', "mse", "gwcorr", "gwspear"]]
    df2['team'] = df2['team_id'].apply(lambda x: reversed_team_id[x])
    df2 = df2[df2["team"].isin([
        "Avocado", "Average", "Hongyang Li and Yuanfang Guan v1", "Lavawizard",
        "Guacamole", "imp"])].reset_index(drop=True)

    df2.drop(columns=['team_id'], inplace=True)
    df2['cell'] = df2['cell'].apply(lambda x: cell_type_id[x])
    df2['assay'] = df2['assay'].apply(lambda x: assay_id[x])

    merged_df = pd.concat([df1, df2], ignore_index=True)

    # print(merged_df)
    comparison_stripplot(merged_df, "qnorm_blind")
    comparison_boxplot(merged_df, "qnorm_blind")

    ####################################################################################
    # compare eic paper blind res with candieic blind res after qnorm reprocessed  
    ####################################################################################
    blind_res_after_qnorm_reprocessed = pd.read_csv(f"{res_dir}/eic_paper/13059_2023_2915_MOESM4_ESM.csv")
    blind_res_after_qnorm_reprocessed = blind_res_after_qnorm_reprocessed[blind_res_after_qnorm_reprocessed["bootstrap_id"] == 1].reset_index(drop=True)

    df2 = blind_res_after_qnorm_reprocessed[['team_id', 'assay', 'cell', "mse", "gwcorr", "gwspear"]]
    df2['team'] = df2['team_id'].apply(lambda x: reversed_team_id[x])
    df2 = df2[df2["team"].isin([
        "Avocado", "Average", "Hongyang Li and Yuanfang Guan v1", "Lavawizard",
        "Guacamole", "imp"])].reset_index(drop=True)

    df2.drop(columns=['team_id'], inplace=True)
    df2['cell'] = df2['cell'].apply(lambda x: cell_type_id[x])
    df2['assay'] = df2['assay'].apply(lambda x: assay_id[x])

    merged_df = pd.concat([df1, df2], ignore_index=True)

    # print(merged_df)
    comparison_stripplot(merged_df, "qnorm_reprocessed_blind")
    comparison_boxplot(merged_df, "qnorm_reprocessed_blind")

################################################################################

def perplexity(probabilities):
    N = len(probabilities)
    epsilon = 1e-6  # Small constant to prevent log(0)
    log_prob_sum = torch.sum(torch.log(probabilities + epsilon))
    perplexity = torch.exp(-log_prob_sum / N)
    return perplexity

def fraction_within_ci(dist, x, c=0.95):
    lower, upper = dist.interval(c)

    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(lower):
        lower = lower.numpy()

    if torch.is_tensor(upper):
        upper = upper.numpy()

    if torch.is_tensor(x):
        x = x.numpy()

    # Add small epsilon to avoid divide by zero
    # eps = np.finfo(float).eps
    # x = np.asarray(x) + eps
    return np.mean((x >= lower) & (x <= upper))

def confidence_calibration(dist, true, n_bins=20):
    # Generate confidence levels from 0 to 1
    confidence_levels = np.linspace(0, 1, n_bins)
    
    # Calculate empirical coverage for each confidence level
    calibration = []
    for c in confidence_levels:
        empirical = fraction_within_ci(dist, true, c)
        calibration.append([c, empirical])
    
    return calibration

def plot_calibration_grid(calibrations, titles):
    """
    Visualize calibration curves in a grid layout based on the number of calibrations.
    
    Parameters:
    - calibrations: list of calibration outputs, where each calibration output
                   is a list of [c, empirical] pairs
    - titles: list of strings for subplot titles
    - figsize: tuple specifying figure size (width, height)
    
    Returns:
    - fig: matplotlib figure object
    """
    
    # Determine the number of subplots needed
    num_calibrations = len(calibrations)
    
    # Create figure and subplots based on the number of calibrations
    if num_calibrations == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
    elif num_calibrations == 2:
        fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    else:
        raise ValueError("Number of calibrations must be 2 or 4.")
    
    # Reference line points (perfect calibration)
    ref_line = np.linspace(0, 1, 100)
    
    # Plot each calibration curve
    for idx, (cal, title, ax) in enumerate(zip(calibrations, titles, axes)):
        # Convert calibration data to numpy arrays for easier plotting
        cal_array = np.array(cal)
        c_values = cal_array[:, 0]
        empirical_values = cal_array[:, 1]
        
        # Plot reference line (perfect calibration)
        ax.plot(ref_line, ref_line, '--', color='orange', alpha=0.8, label='Perfect calibration')
        
        # Plot empirical calibration
        ax.plot(c_values, empirical_values, '-', color='grey', linewidth=2, label='Empirical calibration')
        
        # Customize plot
        ax.set_xlabel('c')
        ax.set_ylabel('Fraction within c% confidence interval')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add legend
        if idx == 0:  # Only add legend to first subplot
            ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

################################################################################

def get_metrics(prob_pval, prob_count, pval_true, pval_pred, count_true, count_pred):
    # Calculate perplexity
    gw_pp_pval = perplexity(prob_pval).item()
    gw_pp_count = perplexity(prob_count).item()

    prom_pp_pval =  perplexity(metrics_class.get_signals(prob_pval, df=metrics_class.prom_df)).item()
    prom_pp_count = perplexity(metrics_class.get_signals(prob_count, df=metrics_class.prom_df)).item()

    gene_pp_pval =  perplexity(metrics_class.get_signals(prob_pval, df=metrics_class.gene_df)).item()
    gene_pp_count = perplexity(metrics_class.get_signals(prob_count, df=metrics_class.gene_df)).item()

    ### global metrics
    gw_pearson_pval = metrics_class.pearson(pval_true, pval_pred)
    gw_spearman_pval = metrics_class.spearman(pval_true, pval_pred)

    gw_pearson_count = metrics_class.pearson(count_true, count_pred)
    gw_spearman_count = metrics_class.spearman(count_true, count_pred)   

    ### gene metrics
    gene_pearson_pval = metrics_class.pearson_gene(pval_true, pval_pred)
    gene_spearman_pval = metrics_class.spearman_gene(pval_true, pval_pred)

    gene_pearson_count = metrics_class.pearson_gene(count_true, count_pred)
    gene_spearman_count = metrics_class.spearman_gene(count_true, count_pred)

    ### promoter metrics
    prom_pearson_pval = metrics_class.pearson_prom(pval_true, pval_pred)
    prom_spearman_pval = metrics_class.spearman_prom(pval_true, pval_pred)

    prom_pearson_count = metrics_class.pearson_prom(count_true, count_pred)
    prom_spearman_count = metrics_class.spearman_prom(count_true, count_pred)

    ### one observation metrics
    one_obs_pearson_pval = metrics_class.pearson1_obs(pval_true, pval_pred)
    one_obs_spearman_pval = metrics_class.spearman1_obs(pval_true, pval_pred)

    one_obs_pearson_count = metrics_class.pearson1_obs(count_true, count_pred)
    one_obs_spearman_count = metrics_class.spearman1_obs(count_true, count_pred)

    ### one imputation metrics
    one_imp_pearson_pval = metrics_class.pearson1_imp(pval_true, pval_pred)
    one_imp_spearman_pval = metrics_class.spearman1_imp(pval_true, pval_pred)

    one_imp_pearson_count = metrics_class.pearson1_imp(count_true, count_pred)
    one_imp_spearman_count = metrics_class.spearman1_imp(count_true, count_pred)

    ### peak overlap metrics
    peak_overlap_pval = metrics_class.peak_overlap(pval_true, pval_pred)
    peak_overlap_count = metrics_class.peak_overlap(count_true, count_pred)

    metr = {"gw_pp_pval": gw_pp_pval, "gw_pp_count": gw_pp_count,
            "prom_pp_pval": prom_pp_pval, "prom_pp_count": prom_pp_count,
            "gene_pp_pval": gene_pp_pval, "gene_pp_count": gene_pp_count,
            "gw_pearson_pval": gw_pearson_pval, "gw_spearman_pval": gw_spearman_pval,
            "gw_pearson_count": gw_pearson_count, "gw_spearman_count": gw_spearman_count,
            "gene_pearson_pval": gene_pearson_pval, "gene_spearman_pval": gene_spearman_pval,
            "gene_pearson_count": gene_pearson_count, "gene_spearman_count": gene_spearman_count,
            "prom_pearson_pval": prom_pearson_pval, "prom_spearman_pval": prom_spearman_pval,
            "prom_pearson_count": prom_pearson_count, "prom_spearman_count": prom_spearman_count,
            "one_obs_pearson_pval": one_obs_pearson_pval, "one_obs_spearman_pval": one_obs_spearman_pval,
            "one_obs_pearson_count": one_obs_pearson_count, "one_obs_spearman_count": one_obs_spearman_count,
            "one_imp_pearson_pval": one_imp_pearson_pval, "one_imp_spearman_pval": one_imp_spearman_pval,
            "one_imp_pearson_count": one_imp_pearson_count, "one_imp_spearman_count": one_imp_spearman_count,
            "peak_overlap_pval": peak_overlap_pval, "peak_overlap_count": peak_overlap_count}

    return metr

class CANDIPredictor:
    def __init__(self, model, hyper_parameters_path, 
        split="test", DNA=False, eic=True, chr="chr21", resolution=25, context_length=1600,
        savedir="models/output/", data_path="/project/compbio-lab/encode_data/"):

        self.model = model
        self.chr = chr
        self.resolution = resolution
        self.savedir = savedir
        self.DNA = DNA
        self.context_length = context_length
        self.data_path = data_path
        self.eic = eic
        self.split = split

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.dataset.init_eval(
            self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=5, eic=eic, merge_ct=True)

        if isinstance(self.model, str):
            with open(hyper_parameters_path, 'rb') as f:
                self.hyper_parameters = pickle.load(f)
                self.hyper_parameters["signal_dim"] = self.dataset.signal_dim
                self.hyper_parameters["metadata_embedding_dim"] = self.dataset.signal_dim
                self.context_length = self.hyper_parameters["context_length"]

            loader = CANDI_LOADER(model, self.hyper_parameters, DNA=self.DNA)
            self.model = loader.load_CANDI()

        self.model = self.model.to(self.device)
        self.model.eval()

        self.chr_sizes = {}
        with open("data/hg38.chrom.sizes", 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                # if chr_name == self.chr:
                self.chr_sizes[chr_name] = int(chr_size)
                    # break

        self.context_length = self.hyper_parameters["context_length"]
        self.batch_size = self.hyper_parameters["batch_size"]
        self.token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
            }

    def load_encoder_input_bios(self, bios_name, x_dsf=1, chr=None, y_dsf=1):
        print("loading encoder inputs for biosample: ", bios_name)
        if chr == None:
            chr = self.chr

        if self.eic:
            if self.split == "test":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [chr, 0, self.chr_sizes[chr]], x_dsf)
            elif self.split == "val":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), [chr, 0, self.chr_sizes[chr]], x_dsf)

        else:
            temp_x, temp_mx = self.dataset.load_bios(bios_name, [chr, 0, self.chr_sizes[chr]], x_dsf)

        # print(temp_x.keys(), temp_mx.keys())
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx

        if self.DNA:
            seq = dna_to_onehot(get_DNA_sequence(self.chr, 0, self.chr_sizes[self.chr]))

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X = X[:num_rows, :]

        if self.DNA:
            seq = seq[:num_rows*self.resolution, :]
            
        X = X.view(-1, self.context_length, X.shape[-1])

        if self.DNA:
            seq = seq.view(-1, self.context_length*self.resolution, seq.shape[-1])

        mX= mX.expand(X.shape[0], -1, -1)
        avX = avX.expand(X.shape[0], -1)

        if self.DNA:
            return X, seq, mX
        else:
            return X, mX

    def load_bios(self, bios_name, x_dsf, y_dsf=1, fill_in_y_prompt=False, chr=None, start=None, end=None):
        # Load biosample data
        if self.eic:
            # Initialize lists to store tensors from each chromosome
            all_X, all_mX, all_avX = [], [], []
            all_Y, all_mY, all_avY = [], [], []
            all_P, all_avlP = [], []

            # Iterate through all chromosomes
            for chr_name, chr_size in self.chr_sizes.items():
                print(f"Loading chromosome: {chr_name}. Available memory: {psutil.virtual_memory().available / (1024 ** 2):.2f} MB")
                if self.split == "test":
                    temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [chr_name, 0, chr_size], x_dsf)
                elif self.split == "val":
                    temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), [chr_name, 0, chr_size], x_dsf)
                
                X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
                all_X.append(X)
                all_mX.append(mX)
                all_avX.append(avX)
                del temp_x, temp_mx
                
                temp_y, temp_my = self.dataset.load_bios(bios_name, [chr_name, 0, chr_size], y_dsf)
                Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
                if fill_in_y_prompt:
                    mY = self.dataset.fill_in_y_prompt(mY)
                all_Y.append(Y)
                all_mY.append(mY)
                all_avY.append(avY)
                del temp_y, temp_my

                temp_py = self.dataset.load_bios_BW(bios_name, [chr_name, 0, chr_size], y_dsf)
                if self.split == "test":
                    temp_px = self.dataset.load_bios_BW(bios_name.replace("B_", "T_"), [chr_name, 0, chr_size], x_dsf)
                elif self.split == "val":
                    temp_px = self.dataset.load_bios_BW(bios_name.replace("V_", "T_"), [chr_name, 0, chr_size], x_dsf)

                temp_p = {**temp_py, **temp_px}
                P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
                all_P.append(P)
                all_avlP.append(avlP)
                del temp_py, temp_px, temp_p

            # Concatenate tensors from all chromosomes
            X = torch.cat(all_X, dim=0)
            mX = all_mX[0]  # Metadata tensors should be identical for all chromosomes
            avX = all_avX[0]
            Y = torch.cat(all_Y, dim=0)
            mY = all_mY[0]
            avY = all_avY[0]
            P = torch.cat(all_P, dim=0)
            avlP = all_avlP[0]
            
            # if self.split == "test":
            #     temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            # elif self.split == "val":
            #     temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            
            # # print(temp_x.keys(), temp_mx.keys())
            # X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            # del temp_x, temp_mx
            
            # temp_y, temp_my = self.dataset.load_bios(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            # Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            # if fill_in_y_prompt:
            #     mY = self.dataset.fill_in_y_prompt(mY)
            # del temp_y, temp_my

            # temp_py = self.dataset.load_bios_BW(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            # if self.split == "test":
            #     temp_px = self.dataset.load_bios_BW(bios_name.replace("B_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            # elif self.split == "val":
            #     temp_px = self.dataset.load_bios_BW(bios_name.replace("V_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)

            # temp_p = {**temp_py, **temp_px}
            # P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            # del temp_py, temp_px, temp_p

        else:
            temp_x, temp_mx = self.dataset.load_bios(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
            temp_y, temp_my = self.dataset.load_bios(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            if fill_in_y_prompt:
                mY = self.dataset.fill_in_y_prompt(mY)
            del temp_y, temp_my

            temp_p = self.dataset.load_bios_BW(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            assert (avlP == avY).all(), "avlP and avY do not match"
            del temp_p

        if self.DNA:
            seq = dna_to_onehot(get_DNA_sequence(self.chr, 0, self.chr_sizes[self.chr]))

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y, P = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :]

        if self.DNA:
            seq = seq[:num_rows*self.resolution, :]
            
        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])
        P = P.view(-1, self.context_length, P.shape[-1])

        if self.DNA:
            seq = seq.view(-1, self.context_length*self.resolution, seq.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        if self.DNA:
            return X, Y, P, seq, mX, mY, avX, avY
        else:
            return X, Y, P, mX, mY, avX, avY

    def pred(self, X, mX, mY, avail, imp_target=[], seq=None):
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        mu = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        var = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        Z = torch.empty((X.shape[0], self.model.l2, self.model.latent_dim), device="cpu", dtype=torch.float32)

        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]
            mX_batch = mX[i:i+ self.batch_size]
            mY_batch = mY[i:i+ self.batch_size]
            avail_batch = avail[i:i+ self.batch_size]

            if self.DNA:
                seq_batch = seq[i:i + self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.clone()
                avail_batch = avail_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()

                x_batch_missing_vals = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing_vals = (mX_batch == self.token_dict["missing_mask"])
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    outputs_p, outputs_n, outputs_mu, outputs_var, latent = self.model(x_batch.float(), seq_batch, mX_batch, mY_batch, avail_batch, return_z=True)
                else:
                    outputs_p, outputs_n, outputs_mu, outputs_var, latent = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch, return_z=True)

            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()
            mu[i:i+outputs_mu.shape[0], :, :] = outputs_mu.cpu()
            var[i:i+outputs_var.shape[0], :, :] = outputs_var.cpu()
            Z[i:i+latent.shape[0], :, :] = latent.cpu()

            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var, latent
            torch.cuda.empty_cache()
            
        n = n.view(n.shape[0] * n.shape[1], n.shape[-1])
        p = p.view(p.shape[0] * p.shape[1], p.shape[-1])
        mu = mu.view(mu.shape[0] * mu.shape[1], mu.shape[-1])
        var = var.view(var.shape[0] * var.shape[1], var.shape[-1])
        Z = Z.view(Z.shape[0] * Z.shape[1], Z.shape[-1])
        return n, p, mu, var, Z

    def get_latent_representations_cropped(self, X, mX, imp_target=[], seq=None, crop_percent=0.1):
        # Calculate dimensions
        crop_size = int(self.context_length * crop_percent)
        stride = self.context_length - (crop_size * 2)
        num_windows = X.shape[0]
        total_length = num_windows * self.context_length

        Z_crop_size = int(crop_size * (self.model.l2 / self.model.l1))
        
        # Flatten input tensors
        X_flat = X.view(-1, X.shape[-1])
        if self.DNA:
            seq_flat = seq.view(-1, seq.shape[-1])
        
        # Initialize output tensors
        Z = torch.zeros((num_windows * self.model.l2, self.model.latent_dim), device="cpu", dtype=torch.float32)
        z_coverage_mask = torch.zeros(num_windows * self.model.l2, dtype=torch.bool, device="cpu")  # New mask for Z
        
        # Collect all windows and their metadata
        window_data = []
        target_regions = []
        
        # Process sliding windows
        for i in range(0, total_length, stride):
            if i + self.context_length >= total_length:
                i = total_length - self.context_length
                
            window_end = i + self.context_length
            x_window = X_flat[i:window_end].unsqueeze(0)
            
            # Use first row of metadata tensors (verified identical)
            mx_window = mX[0].unsqueeze(0)
            
            if self.DNA:
                seq_start = i * self.resolution
                seq_end = window_end * self.resolution
                seq_window = seq_flat[seq_start:seq_end].unsqueeze(0)
            
            # Determine prediction regions
            if i == 0:  # First window
                start_idx = 0
                end_idx = self.context_length - crop_size
            elif i + self.context_length >= total_length:  # Last window
                start_idx = crop_size
                end_idx = self.context_length
            else:  # Middle windows
                start_idx = crop_size
                end_idx = self.context_length - crop_size
                
            target_start = i + start_idx
            target_end = i + end_idx
            
            # Store window data and target regions
            window_info = {
                'x': x_window,
                'mx': mx_window,
                'seq': seq_window if self.DNA else None
            }
            target_info = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'target_start': target_start,
                'target_end': target_end
            }
            
            window_data.append(window_info)
            target_regions.append(target_info)
        
        # Process windows in batches
        for i in range(0, len(window_data), self.batch_size):
            batch_windows = window_data[i:i + self.batch_size]
            batch_targets = target_regions[i:i + self.batch_size]
            
            # Prepare batch tensors
            x_batch = torch.cat([w['x'] for w in batch_windows])
            mx_batch = torch.cat([w['mx'] for w in batch_windows])
            
            if self.DNA:
                seq_batch = torch.cat([w['seq'] for w in batch_windows])
            
            # Apply imp_target if specified
            if len(imp_target) > 0:
                x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                mx_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
            
            # Get predictions
            with torch.no_grad():
                if self.DNA:
                    outputs = self.model.encode(
                        x_batch.float().to(self.device),
                        seq_batch.to(self.device),
                        mx_batch.to(self.device)
                    )
                else:
                    outputs = self.model.encode(
                        x_batch.float().to(self.device),
                        mx_batch.to(self.device)
                    )
                
                outputs_Z = outputs
            
            # Update predictions for each window in batch
            for j, (out_Z, target) in enumerate(zip(outputs_Z, batch_targets)):
                start_idx = target['start_idx']
                end_idx = target['end_idx']
                target_start = target['target_start']
                target_end = target['target_end']

                i = target_start - start_idx

                i_z = i * (self.model.l2 / self.model.l1)
                if start_idx == 0:
                    start_z_idx = 0
                elif start_idx == crop_size:
                    start_z_idx = Z_crop_size

                if end_idx == self.context_length - crop_size:
                    end_z_idx = self.model.l2 - Z_crop_size
                elif end_idx == self.context_length:
                    end_z_idx = self.model.l2

                target_z_start = int(i_z + start_z_idx)
                target_z_end = int(i_z + end_z_idx)
                
                Z[target_z_start:target_z_end, :] = out_Z[start_z_idx:end_z_idx, :].cpu()
                
                z_coverage_mask[target_z_start:target_z_end] = True  # Track Z coverage
            
            del outputs
            torch.cuda.empty_cache()

            
        if not z_coverage_mask.all():
            print(f"Missing Z predictions for positions: {torch.where(~z_coverage_mask)[0]}")
            raise ValueError("Missing Z predictions")
        
        return Z

    def pred_cropped(self, X, mX, mY, avail, imp_target=[], seq=None, crop_percent=0.1):
        # Calculate dimensions
        crop_size = int(self.context_length * crop_percent)
        stride = self.context_length - (crop_size * 2)
        num_windows = X.shape[0]
        total_length = num_windows * self.context_length

        Z_crop_size = int(crop_size * (self.model.l2 / self.model.l1))
        
        # Flatten input tensors
        X_flat = X.view(-1, X.shape[-1])
        if self.DNA:
            seq_flat = seq.view(-1, seq.shape[-1])
        
        # Initialize output tensors
        n = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        p = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        mu = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        var = torch.zeros_like(X_flat, dtype=torch.float32, device="cpu")
        Z = torch.zeros((num_windows * self.model.l2, self.model.latent_dim), device="cpu", dtype=torch.float32)
        coverage_mask = torch.zeros(total_length, dtype=torch.bool, device="cpu")
        z_coverage_mask = torch.zeros(num_windows * self.model.l2, dtype=torch.bool, device="cpu")  # New mask for Z
        
        # Collect all windows and their metadata
        window_data = []
        target_regions = []
        
        # Process sliding windows
        for i in range(0, total_length, stride):
            if i + self.context_length >= total_length:
                i = total_length - self.context_length
                
            window_end = i + self.context_length
            x_window = X_flat[i:window_end].unsqueeze(0)
            
            # Use first row of metadata tensors (verified identical)
            mx_window = mX[0].unsqueeze(0)
            my_window = mY[0].unsqueeze(0)
            avail_window = avail[0].unsqueeze(0)
            
            if self.DNA:
                seq_start = i * self.resolution
                seq_end = window_end * self.resolution
                seq_window = seq_flat[seq_start:seq_end].unsqueeze(0)
            
            # Determine prediction regions
            if i == 0:  # First window
                start_idx = 0
                end_idx = self.context_length - crop_size
            elif i + self.context_length >= total_length:  # Last window
                start_idx = crop_size
                end_idx = self.context_length
            else:  # Middle windows
                start_idx = crop_size
                end_idx = self.context_length - crop_size
                
            target_start = i + start_idx
            target_end = i + end_idx
            
            # Store window data and target regions
            window_info = {
                'x': x_window,
                'mx': mx_window,
                'my': my_window,
                'avail': avail_window,
                'seq': seq_window if self.DNA else None
            }
            target_info = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'target_start': target_start,
                'target_end': target_end
            }
            
            window_data.append(window_info)
            target_regions.append(target_info)
        
        # Process windows in batches
        for i in range(0, len(window_data), self.batch_size):
            batch_windows = window_data[i:i + self.batch_size]
            batch_targets = target_regions[i:i + self.batch_size]
            
            # Prepare batch tensors
            x_batch = torch.cat([w['x'] for w in batch_windows])
            mx_batch = torch.cat([w['mx'] for w in batch_windows])
            my_batch = torch.cat([w['my'] for w in batch_windows])
            avail_batch = torch.cat([w['avail'] for w in batch_windows])
            
            if self.DNA:
                seq_batch = torch.cat([w['seq'] for w in batch_windows])
            
            # Apply imp_target if specified
            if len(imp_target) > 0:
                x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                mx_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                avail_batch[:, imp_target] = 0
            
            # Get predictions
            with torch.no_grad():
                if self.DNA:
                    outputs = self.model(
                        x_batch.float().to(self.device),
                        seq_batch.to(self.device),
                        mx_batch.to(self.device),
                        my_batch.to(self.device),
                        avail_batch.to(self.device),
                        return_z=True
                    )
                else:
                    outputs = self.model(
                        x_batch.float().to(self.device),
                        mx_batch.to(self.device),
                        my_batch.to(self.device),
                        avail_batch.to(self.device),
                        return_z=True
                    )
                
                outputs_p, outputs_n, outputs_mu, outputs_var, outputs_Z = outputs
            
            # Update predictions for each window in batch
            for j, (window_pred, target) in enumerate(zip(zip(outputs_n, outputs_p, outputs_mu, outputs_var, outputs_Z), batch_targets)):
                out_n, out_p, out_mu, out_var, out_Z = window_pred
                start_idx = target['start_idx']
                end_idx = target['end_idx']
                target_start = target['target_start']
                target_end = target['target_end']

                i = target_start - start_idx

                i_z = i * (self.model.l2 / self.model.l1)
                if start_idx == 0:
                    start_z_idx = 0
                elif start_idx == crop_size:
                    start_z_idx = Z_crop_size

                if end_idx == self.context_length - crop_size:
                    end_z_idx = self.model.l2 - Z_crop_size
                elif end_idx == self.context_length:
                    end_z_idx = self.model.l2

                target_z_start = int(i_z + start_z_idx)
                target_z_end = int(i_z + end_z_idx)
                
                n[target_start:target_end, :] = out_n[start_idx:end_idx, :].cpu()
                p[target_start:target_end, :] = out_p[start_idx:end_idx, :].cpu()
                mu[target_start:target_end, :] = out_mu[start_idx:end_idx, :].cpu()
                var[target_start:target_end, :] = out_var[start_idx:end_idx, :].cpu()
                Z[target_z_start:target_z_end, :] = out_Z[start_z_idx:end_z_idx, :].cpu()
                
                coverage_mask[target_start:target_end] = True
                z_coverage_mask[target_z_start:target_z_end] = True  # Track Z coverage
            
            del outputs
            torch.cuda.empty_cache()
        
        # Verify complete coverage for both signal and Z
        if not coverage_mask.all():
            print(f"Missing predictions for positions: {torch.where(~coverage_mask)[0]}")
            raise ValueError("Missing signal predictions")
            
        if not z_coverage_mask.all():
            print(f"Missing Z predictions for positions: {torch.where(~z_coverage_mask)[0]}")
            raise ValueError("Missing Z predictions")
        
        return n, p, mu, var, Z

    def get_latent_representations(self, X, mX, mY, avX, seq=None):
        if self.DNA:
            _, _, _, _, Z = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            _, _, _, _, Z = self.pred(X, mX, mY, avX, seq=None, imp_target=[])
        return Z

    def get_decoded_signal(self, X, mX, mY, avX, seq=None):
        if self.DNA:
            p, n, mu, var, _ = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            p, n, mu, var, _ = self.pred(X, mX, mY, avX, seq=None, imp_target=[])

        count_dist = NegativeBinomial(p, n)
        pval_dist = Gaussian(mu, var)
        
        return count_dist.expect(), pval_dist.expect()

    def evaluate_leave_one_out(self, X, mX, mY, avX, Y, P, seq=None, crop_edges=True, return_preds=False):
        available_indices = torch.where(avX[0, :] == 1)[0]
        expnames = list(self.dataset.aliases["experiment_aliases"].keys())
        
        # Initialize tensors for imputation predictions
        n_imp = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        p_imp = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        mu_imp = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        var_imp = torch.empty((X.shape[0]*X.shape[1], X.shape[2]), device="cpu", dtype=torch.float32)
        
        if crop_edges:
            # Get upsampling predictions 
            n_ups, p_ups, mu_ups, var_ups, _ = self.pred_cropped(X, mX, mY, avX, imp_target=[], seq=seq)
        else:
            # Get upsampling predictions 
            n_ups, p_ups, mu_ups, var_ups, _ = self.pred(X, mX, mY, avX, imp_target=[], seq=seq)
        
        # Perform leave-one-out predictions
        for ii, leave_one_out in enumerate(available_indices):
            if crop_edges:
                n, p, mu, var, _ = self.pred_cropped(X, mX, mY, avX, imp_target=[leave_one_out], seq=seq)
            else:
                n, p, mu, var, _ = self.pred(X, mX, mY, avX, imp_target=[leave_one_out], seq=seq)
            n_imp[:, leave_one_out] = n[:, leave_one_out]
            p_imp[:, leave_one_out] = p[:, leave_one_out]
            mu_imp[:, leave_one_out] = mu[:, leave_one_out]
            var_imp[:, leave_one_out] = var[:, leave_one_out]
            print(f"Completed feature {ii+1}/{len(available_indices)}")
        
        # Create distributions and get means
        Y = Y.view(-1, Y.shape[-1])
        P = P.view(-1, P.shape[-1])
        
        imp_count_dist = NegativeBinomial(p_imp, n_imp)
        ups_count_dist = NegativeBinomial(p_ups, n_ups)
        imp_count_mean = imp_count_dist.expect()
        ups_count_mean = ups_count_dist.expect()
        
        imp_pval_dist = Gaussian(mu_imp, var_imp)
        ups_pval_dist = Gaussian(mu_ups, var_ups)
        imp_pval_mean = imp_pval_dist.expect()
        ups_pval_mean = ups_pval_dist.expect()

        prob_imp_pval = imp_pval_dist.pdf(P)
        prob_imp_count = imp_count_dist.pmf(Y)
        prob_ups_pval = ups_pval_dist.pdf(P)
        prob_ups_count = ups_count_dist.pmf(Y)
        
        if return_preds:
            return imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist
        
        # Calculate metrics for each feature
        metrics = {}
        for idx in available_indices:
            # true 
            count_true = Y[:, idx].numpy()
            pval_true = P[:, idx].numpy()

            # pred
            imp_count = imp_count_mean[:, idx].numpy()
            ups_count = ups_count_mean[:, idx].numpy()
            
            # P-value (apply sinh transformation)
            imp_pval = np.sinh(imp_pval_mean[:, idx].numpy())
            ups_pval = np.sinh(ups_pval_mean[:, idx].numpy())
            pval_true = np.sinh(pval_true)

            imp_metr = get_metrics(prob_imp_pval[:, idx], prob_imp_count[:, idx], pval_true, imp_pval, count_true, imp_count)
            ups_metr = get_metrics(prob_ups_pval[:, idx], prob_ups_count[:, idx], pval_true, ups_pval, count_true, ups_count)
            
            metrics[idx.item()] = {
                'count_metrics': {
                    'imp_gw_mse': np.mean((count_true - imp_count) ** 2),
                    'imp_gw_r2': 1 - (np.sum((count_true - imp_count) ** 2) / 
                                np.sum((count_true - np.mean(count_true)) ** 2)),
                    'ups_gw_mse': np.mean((count_true - ups_count) ** 2),
                    'ups_gw_r2': 1 - (np.sum((count_true - ups_count) ** 2) / 
                                np.sum((count_true - np.mean(count_true)) ** 2)),
                    'imp_gw_pp': imp_metr['gw_pp_count'], 'ups_gw_pp': ups_metr['gw_pp_count'],
                    'imp_prom_pp': imp_metr['prom_pp_count'], 'ups_prom_pp': ups_metr['prom_pp_count'],
                    'imp_gene_pp': imp_metr['gene_pp_count'], 'ups_gene_pp': ups_metr['gene_pp_count'], 
                    'imp_gw_pearson': imp_metr['gw_pearson_count'], 'ups_gw_pearson': ups_metr['gw_pearson_count'],
                    'imp_gw_spearman': imp_metr['gw_spearman_count'], 'ups_gw_spearman': ups_metr['gw_spearman_count'],
                    'imp_gene_pearson': imp_metr['gene_pearson_count'], 'ups_gene_pearson': ups_metr['gene_pearson_count'],
                    'imp_gene_spearman': imp_metr['gene_spearman_count'], 'ups_gene_spearman': ups_metr['gene_spearman_count'],
                    'imp_prom_pearson': imp_metr['prom_pearson_count'], 'ups_prom_pearson': ups_metr['prom_pearson_count'],
                    'imp_prom_spearman': imp_metr['prom_spearman_count'], 'ups_prom_spearman': ups_metr['prom_spearman_count'],
                    'imp_one_obs_pearson': imp_metr['one_obs_pearson_count'], 'ups_one_obs_pearson': ups_metr['one_obs_pearson_count'],
                    'imp_one_obs_spearman': imp_metr['one_obs_spearman_count'], 'ups_one_obs_spearman': ups_metr['one_obs_spearman_count'],
                    'imp_one_imp_pearson': imp_metr['one_imp_pearson_count'], 'ups_one_imp_pearson': ups_metr['one_imp_pearson_count'],
                    'imp_one_imp_spearman': imp_metr['one_imp_spearman_count'], 'ups_one_imp_spearman': ups_metr['one_imp_spearman_count'],
                    'imp_peak_overlap': imp_metr['peak_overlap_count'], 'ups_peak_overlap': ups_metr['peak_overlap_count'],
                },
                'pval_metrics': {
                    'imp_gw_mse': np.mean((pval_true - imp_pval) ** 2),
                    'imp_gw_r2': 1 - (np.sum((pval_true - imp_pval) ** 2) / 
                                np.sum((pval_true - np.mean(pval_true)) ** 2)),
                    'ups_gw_mse': np.mean((pval_true - ups_pval) ** 2),
                    'ups_gw_r2': 1 - (np.sum((pval_true - ups_pval) ** 2) / np.sum((pval_true - np.mean(pval_true)) ** 2)),
                    'imp_gw_pp': imp_metr['gw_pp_pval'], 'ups_gw_pp': ups_metr['gw_pp_pval'],
                    'imp_prom_pp': imp_metr['prom_pp_pval'], 'ups_prom_pp': ups_metr['prom_pp_pval'],
                    'imp_gene_pp': imp_metr['gene_pp_pval'], 'ups_gene_pp': ups_metr['gene_pp_pval'], 
                    'imp_gw_pearson': imp_metr['gw_pearson_pval'], 'ups_gw_pearson': ups_metr['gw_pearson_pval'],
                    'imp_gw_spearman': imp_metr['gw_spearman_pval'], 'ups_gw_spearman': ups_metr['gw_spearman_pval'],
                    'imp_gene_pearson': imp_metr['gene_pearson_pval'], 'ups_gene_pearson': ups_metr['gene_pearson_pval'],
                    'imp_gene_spearman': imp_metr['gene_spearman_pval'], 'ups_gene_spearman': ups_metr['gene_spearman_pval'],
                    'imp_prom_pearson': imp_metr['prom_pearson_pval'], 'ups_prom_pearson': ups_metr['prom_pearson_pval'],
                    'imp_prom_spearman': imp_metr['prom_spearman_pval'], 'ups_prom_spearman': ups_metr['prom_spearman_pval'],
                    'imp_one_obs_pearson': imp_metr['one_obs_pearson_pval'], 'ups_one_obs_pearson': ups_metr['one_obs_pearson_pval'],
                    'imp_one_obs_spearman': imp_metr['one_obs_spearman_pval'], 'ups_one_obs_spearman': ups_metr['one_obs_spearman_pval'],
                    'imp_one_imp_pearson': imp_metr['one_imp_pearson_pval'], 'ups_one_imp_pearson': ups_metr['one_imp_pearson_pval'],
                    'imp_one_imp_spearman': imp_metr['one_imp_spearman_pval'], 'ups_one_imp_spearman': ups_metr['one_imp_spearman_pval'],
                    'imp_peak_overlap': imp_metr['peak_overlap_pval'], 'ups_peak_overlap': ups_metr['peak_overlap_pval'],
                }
            }

        # Print summary
        print("\nEvaluation Results:")
        
        def print_metric_table(metrics_type, available_indices, metrics, expnames):
            # Define column headers
            headers = [
                "MSE", 
                "GW_Pearson", "1obs_Pearson", "1imp_Pearson",
                "GW_Spearman", "1obs_Spearman", "1imp_Spearman",
                "GW_PP", "Prom_PP", "Gene_PP",
                "Peak_Overlap"
            ]
            
            # Print header
            print(f"\n{metrics_type} Metrics:")
            print("Feature | Type      |", end=" ")
            for header in headers:
                print(f"{header:12s}", end=" ")
            print("\n" + "-" * (20 + 13 * len(headers)))
            
            for idx in available_indices:
                m = metrics[idx.item()][f'{metrics_type.lower()}_metrics']
                
                # Print imputed metrics
                print(f"{expnames[idx]:10s} | Imputed   |", end=" ")
                print(f"{m['imp_gw_mse']:12.4f}", end=" ")  # MSE
                print(f"{m['imp_gw_pearson']:12.4f}", end=" ")  # GW_Pearson
                print(f"{m['imp_one_obs_pearson']:12.4f}", end=" ")  # 1obs_Pearson
                print(f"{m['imp_one_imp_pearson']:12.4f}", end=" ")  # 1imp_Pearson
                print(f"{m['imp_gw_spearman']:12.4f}", end=" ")  # GW_Spearman
                print(f"{m['imp_one_obs_spearman']:12.4f}", end=" ")  # 1obs_Spearman
                print(f"{m['imp_one_imp_spearman']:12.4f}", end=" ")  # 1imp_Spearman
                print(f"{m['imp_gw_pp']:12.4f}", end=" ")  # GW_PP
                print(f"{m['imp_prom_pp']:12.4f}", end=" ")  # Prom_PP
                print(f"{m['imp_gene_pp']:12.4f}", end=" ")  # Gene_PP
                print(f"{m['imp_peak_overlap']:12.4f}")  # Peak_Overlap
                
                # Print upsampled metrics
                print(f"{' '*10} | Upsampled |", end=" ")
                print(f"{m['ups_gw_mse']:12.4f}", end=" ")  # MSE
                print(f"{m['ups_gw_pearson']:12.4f}", end=" ")  # GW_Pearson
                print(f"{m['ups_one_obs_pearson']:12.4f}", end=" ")  # 1obs_Pearson
                print(f"{m['ups_one_imp_pearson']:12.4f}", end=" ")  # 1imp_Pearson
                print(f"{m['ups_gw_spearman']:12.4f}", end=" ")  # GW_Spearman
                print(f"{m['ups_one_obs_spearman']:12.4f}", end=" ")  # 1obs_Spearman
                print(f"{m['ups_one_imp_spearman']:12.4f}", end=" ")  # 1imp_Spearman
                print(f"{m['ups_gw_pp']:12.4f}", end=" ")  # GW_PP
                print(f"{m['ups_prom_pp']:12.4f}", end=" ")  # Prom_PP
                print(f"{m['ups_gene_pp']:12.4f}", end=" ")  # Gene_PP
                print(f"{m['ups_peak_overlap']:12.4f}")  # Peak_Overlap
                
                print("-" * (20 + 13 * len(headers)))

        # Print tables for both count and p-value metrics
        print_metric_table("Count", available_indices, metrics, expnames)
        print_metric_table("Pval", available_indices, metrics, expnames)
        return metrics

    def evaluate_leave_one_out_eic(self, X, mX, mY, avX, Y, P, avY, seq=None, crop_edges=True, return_preds=False):

        available_X_indices = torch.where(avX[0, :] == 1)[0]
        available_Y_indices = torch.where(avY[0, :] == 1)[0]

        expnames = list(self.dataset.aliases["experiment_aliases"].keys())

        if crop_edges:
            # Get upsampling predictions 
            n, p, mu, var, Z = self.pred_cropped(X, mX, mY, avX, imp_target=[], seq=seq)
        else:
            # Get upsampling predictions 
            n, p, mu, var, Z = self.pred(X, mX, mY, avX, imp_target=[], seq=seq)

        # Create distributions and get means
        Y = Y.view(-1, Y.shape[-1])
        P = P.view(-1, P.shape[-1])
        X = X.view(-1, X.shape[-1])

        ups_count_dist = NegativeBinomial(p, n)
        ups_count_mean = ups_count_dist.expect()
        
        ups_pval_dist = Gaussian(mu, var)
        ups_pval_mean = ups_pval_dist.expect()

        prob_ups_pval = ups_pval_dist.pdf(P)
        prob_ups_count = ups_count_dist.pmf(Y)

        if return_preds:
            return ups_count_dist, ups_pval_dist

        metrics = {}
        for j in range(Y.shape[1]):
            pred_count = ups_count_mean[:, j].numpy()
            pred_pval = np.sinh(ups_pval_mean[:, j].numpy())
            pval_true = np.sinh(P[:, j].numpy())

            if j in list(available_X_indices):
                comparison = "upsampled"
                count_true = X[:, j].numpy()

            elif j in list(available_Y_indices):
                comparison = "imputed"
                count_true = Y[:, j].numpy()

            else:
                continue
        
            ups_metr = get_metrics(prob_ups_pval[:, j], prob_ups_count[:, j], pval_true, pred_pval, count_true, pred_count)

            metrics[j] = {
                'comparison': comparison,
                'count_metrics': {
                    'ups_gw_mse': np.mean((count_true - pred_count) ** 2),
                    'ups_gw_r2': 1 - (np.sum((count_true - pred_count) ** 2) / 
                                np.sum((count_true - np.mean(count_true)) ** 2)),
                    'gw_pp': ups_metr['gw_pp_count'],
                    'prom_pp': ups_metr['prom_pp_count'],
                    'gene_pp': ups_metr['gene_pp_count'], 
                    'gw_pearson': ups_metr['gw_pearson_count'],
                    'gw_spearman': ups_metr['gw_spearman_count'],
                    'gene_pearson': ups_metr['gene_pearson_count'],
                    'gene_spearman': ups_metr['gene_spearman_count'],
                    'prom_pearson': ups_metr['prom_pearson_count'],
                    'prom_spearman': ups_metr['prom_spearman_count'],
                    'one_obs_pearson': ups_metr['one_obs_pearson_count'],
                    'one_obs_spearman': ups_metr['one_obs_spearman_count'],
                    'one_imp_pearson': ups_metr['one_imp_pearson_count'],
                    'one_imp_spearman': ups_metr['one_imp_spearman_count'],
                    'peak_overlap': ups_metr['peak_overlap_count'],
                },
                'pval_metrics': {
                    'ups_gw_mse': np.mean((pval_true - pred_pval) ** 2),
                    'ups_gw_r2': 1 - (np.sum((pval_true - pred_pval) ** 2) / np.sum((pval_true - np.mean(pval_true)) ** 2)),
                    'ups_gw_pp': ups_metr['gw_pp_pval'],
                    'ups_prom_pp': ups_metr['prom_pp_pval'],
                    'ups_gene_pp': ups_metr['gene_pp_pval'], 
                    'ups_gw_pearson': ups_metr['gw_pearson_pval'],
                    'ups_gw_spearman': ups_metr['gw_spearman_pval'],
                    'ups_gene_pearson': ups_metr['gene_pearson_pval'],
                    'ups_gene_spearman': ups_metr['gene_spearman_pval'],
                    'ups_prom_pearson': ups_metr['prom_pearson_pval'],
                    'ups_prom_spearman': ups_metr['prom_spearman_pval'],
                    'ups_one_obs_pearson': ups_metr['one_obs_pearson_pval'],
                    'ups_one_obs_spearman': ups_metr['one_obs_spearman_pval'],
                    'ups_one_imp_pearson': ups_metr['one_imp_pearson_pval'],
                    'ups_one_imp_spearman': ups_metr['one_imp_spearman_pval'],
                    'ups_peak_overlap': ups_metr['peak_overlap_pval'],

                }
            }

        # Print summary with updated headers and format
        print("\nEvaluation Results:")
        print("\nCount Metrics:")
        print("Feature | Type      | GW_Pearson | GW_Spearman | MSE    | R2     | GW_PP")
        print("-" * 75)
        
        for idx, m in metrics.items():
            feature_name = expnames[idx]
            comp_type = m['comparison']
            count_m = m['count_metrics']
            print(f"{feature_name:10s} | {comp_type:9s} | {count_m['gw_pearson']:10.4f} | {count_m['gw_spearman']:11.4f} | "
                  f"{count_m['ups_gw_mse']:6.4f} | {count_m['ups_gw_r2']:6.4f} | {count_m['gw_pp']:6.4f}")
        
        print("\nP-value Metrics:")
        print("Feature | Type      | GW_Pearson | GW_Spearman | MSE    | R2     | GW_PP")
        print("-" * 75)
        
        for idx, m in metrics.items():
            feature_name = expnames[idx]
            comp_type = m['comparison']
            pval_m = m['pval_metrics']
            print(f"{feature_name:10s} | {comp_type:9s} | {pval_m['ups_gw_pearson']:10.4f} | {pval_m['ups_gw_spearman']:11.4f} | "
                  f"{pval_m['ups_gw_mse']:6.4f} | {pval_m['ups_gw_r2']:6.4f} | {pval_m['ups_gw_pp']:6.4f}")

        return metrics

    def evaluate(self, bios_name):
        X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf=1)
        if self.eic:
            metrics = self.evaluate_leave_one_out_eic(X, mX, mY, avX, Y, P, avY, seq=seq, crop_edges=True, return_preds=False)
        else:
            metrics = self.evaluate_leave_one_out(X, mX, mY, avX, Y, P, seq=seq, crop_edges=True, return_preds=False)
        return metrics

################################################################################

def latent_reproducibility(
    model_path, hyper_parameters_path, 
    repr1_bios, repr2_bios, 
    chr="chr21", dataset_path="/project/compbio-lab/encode_data/"):

    candi = CANDIPredictor(model_path, hyper_parameters_path, data_path=dataset_path, DNA=True, eic=True)
    candi.chr = chr

    # Load latent representations
    X1, seq1, mX1 = candi.load_encoder_input_bios(repr1_bios)
    X2, seq2, mX2 = candi.load_encoder_input_bios(repr2_bios)

    latent_repr1 = candi.get_latent_representations_cropped(X1, mX1, seq=seq1)
    latent_repr2 = candi.get_latent_representations_cropped(X2, mX2, seq=seq2)

    del X1, X2, seq1, seq2, mX1, mX2
    
    # Assume latent_repr1 and latent_repr2 are tensors of shape (L, d)
    assert latent_repr1.shape == latent_repr2.shape, "latent_repr1 and latent_repr2 must have the same shape"

    # Convert cosine similarity to cosine distance
    cosine_distances = 1 - F.cosine_similarity(latent_repr1, latent_repr2, dim=-1)  # Shape: (L,)
    euclidean_distances = torch.sqrt(torch.sum((latent_repr1 - latent_repr2)**2, dim=1))

    # Scale cosine distances by 1/2
    cosine_distances_scaled = cosine_distances / 2

    # Calculate summary statistics
    stats = {
        'euclidean': {
            'mean': euclidean_distances.mean().item(),
            'std': euclidean_distances.std().item(),
            'median': euclidean_distances.median().item(),
            'min': euclidean_distances.min().item(),
            'max': euclidean_distances.max().item()
        },
        'cosine': {
            'mean': cosine_distances_scaled.mean().item(),
            'std': cosine_distances_scaled.std().item(),
            'median': cosine_distances_scaled.median().item(),
            'min': cosine_distances_scaled.min().item(),
            'max': cosine_distances_scaled.max().item()
        }
    }
    
    # Function to plot CDF and calculate AUC
    def plot_cdf(ax, data, title, xlabel, color='blue', is_cosine=False):
        sorted_data = np.sort(data.cpu().numpy())
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Extend the CDF to cover the entire x-axis range
        sorted_data = np.concatenate([[0], sorted_data, [1 if is_cosine else sorted_data[-1]]])
        cumulative = np.concatenate([[0], cumulative, [1]])
        
        # Plot CDF
        ax.plot(sorted_data, cumulative, color=color)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Fraction of bins')
        ax.grid(True, alpha=0.3)
        
        # Calculate AUC using trapezoidal rule
        auc = integrate.trapz(cumulative, sorted_data)
        
        # Annotate AUC on the plot
        if is_cosine:
            ax.text(0.8, 0.8, f"AUC: {auc:.4f}", transform=ax.transAxes, color=color, fontsize=10)
        
        return auc
    
    fig = plt.figure(figsize=(12, 8))
    
    # Plot CDFs (top row)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    
    # Plot for cosine distance (scaled)
    stats['cosine']['auc'] = plot_cdf(
        ax1, 
        cosine_distances_scaled,
        'Cosine Distance CDF',
        'Cosine Distance / 2',
        'blue',
        is_cosine=True
    )
    
    # Plot for euclidean distance
    stats['euclidean']['auc'] = plot_cdf(
        ax2,
        euclidean_distances,
        'Euclidean Distance CDF',
        'Euclidean Distance',
        'green',
        is_cosine=False
    )
    
    # Compute PCA
    pca = PCA(n_components=2)
    pca1 = pca.fit_transform(latent_repr1.cpu().numpy())
    pca2 = pca.fit_transform(latent_repr2.cpu().numpy())
    
    # Plot PCA (bottom row)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    # PCA plots
    ax3.scatter(pca1[:, 0], pca1[:, 1], alpha=0.5, s=1)
    ax3.set_title(f'PCA of {repr1_bios}')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    
    ax4.scatter(pca2[:, 0], pca2[:, 1], alpha=0.5, s=1)
    ax4.set_title(f'PCA of {repr2_bios}')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'latent_space_comparison_{repr1_bios}_{repr2_bios}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Latent space comparison statistics:")
    print(f"Mean cosine distance: {stats['cosine']['mean']:.4f}")
    print(f"Mean euclidean distance: {stats['euclidean']['mean']:.4f}")
    print(f"AUC cosine: {stats['cosine']['auc']:.4f}")
    return stats, euclidean_distances, cosine_distances_scaled

class ChromatinStateProbe(nn.Module):
    def __init__(self, input_dim, output_dim=18):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.class_to_index = None  # Placeholder for the class-to-index mapping
    
    def forward(self, x, normalize=False):
        if normalize:
            x = F.normalize(x, p=2, dim=1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)  # Apply log_softmax to get log probabilities
        return x

    def encode_class_indices(self, class_names):
        """
        Convert a list of class names to class indices.
        """
        if self.class_to_index is None:
            unique_classes = sorted(set(class_names))
            self.class_to_index = {name: idx for idx, name in enumerate(unique_classes)}
            self.index_to_class = {idx: name for name, idx in self.class_to_index.items()}
        return [self.class_to_index[name] for name in class_names]

    def decode_class_indices(self, class_indices):
        """
        Convert class indices back to class names.
        """
        if self.class_to_index is None:
            raise ValueError("class_to_index mapping is not defined.")
        return [self.index_to_class[idx] for idx in class_indices]

    def train_batch(self, X, y, optimizer, criterion):
        optimizer.zero_grad()
        output = self(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def validate(self, X, y):
        self.eval()
        with torch.no_grad():
            output = self(X)
            criterion = nn.NLLLoss()
            val_loss = criterion(output, y)
            
            # Convert log probabilities to probabilities
            probabilities = torch.exp(output)
            
            # Get predicted classes
            _, predicted = torch.max(probabilities, 1)
            
            # Convert tensors to numpy arrays
            y_true = y.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            
            # Get class names
            class_names = [self.index_to_class[idx] for idx in range(len(self.index_to_class))]
            
            # Compute classification report
            report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
            
            # Calculate overall accuracy
            total = y.size(0)
            correct = (predicted == y).sum().item()
            overall_accuracy = 100 * correct / total
            
            # Print metrics
            print(f'\nOverall Validation Loss: {val_loss:.4f}, Accuracy: {overall_accuracy:.2f}%')
            print('\nClassification Report:')
            print(report)
            
        self.train()  # Set the model back to training mode
        return val_loss.item(), overall_accuracy

    def fit(self, X_train, y_train, X_val, y_val, num_epochs=10, learning_rate=0.001, batch_size=200):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        # Encode class names to indices
        y_train = torch.tensor(self.encode_class_indices(y_train), dtype=torch.long)
        y_val = torch.tensor(self.encode_class_indices(y_val), dtype=torch.long)

        # Convert inputs to tensors if they aren't already
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)

        n_batches = (len(X_train) + batch_size - 1) // batch_size
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            total_loss = 0

            # Shuffle training data
            indices = torch.randperm(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Train in batches
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                loss = self.train_batch(batch_X, batch_y, optimizer, criterion)
                total_loss += loss

            avg_loss = total_loss / n_batches
            print(f'Epoch {epoch + 1}/{num_epochs}: Training Loss: {avg_loss:.4f}')

            if epoch % 20 == 0:
                # Validate every epoch
                val_loss, val_acc = self.validate(X_val, y_val)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Optionally save model weights here
                # torch.save(self.state_dict(), 'best_model.pt')
        
        return val_loss, val_acc

################################################################################

def chromatin_state_dataset_eic_train_test_val_split(solar_data_path="/project/compbio-lab/encode_data/"):
    bios_names = [t for t in os.listdir(solar_data_path) if t.startswith("T_")]
    # print(bios_names)

    cs_names = [t for t in os.listdir(os.path.join(solar_data_path, "chromatin_state_annotations"))]

    # Remove 'T_' prefix from biosample names for comparison
    bios_names_cleaned = [name.replace("T_", "") for name in bios_names]
    
    def similar(a, b, threshold=0.70):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

    # Find exact and similar matches
    shared_names = set()
    similar_matches = {}  # Store similar but not exact matches
    
    for bios_name in bios_names_cleaned:
        if bios_name in cs_names:
            shared_names.add(bios_name)
        else:
            # Look for similar names
            for cs_name in cs_names:
                if similar(bios_name, cs_name):
                    similar_matches[bios_name] = cs_name
                    shared_names.add(cs_name)  # Add the CS name as it's the reference

    print(f"\nNumber of shared cell types (including similar matches): {len(shared_names)}")

    # Add 'T_' prefix back to shared names for comparison with original bios_names
    shared_names_with_prefix = [f"T_{name}" for name in shared_names]
    
    # Find unshared biosamples
    unshared_bios = [name for name in bios_names if name not in shared_names_with_prefix]
    
    print("\nBiosamples without matching chromatin states:")
    for name in unshared_bios:
        print(name)
    
    print("\nShared cell types between biosamples and chromatin states:")
    for name in shared_names:
        print(name)
        
    print("\nSimilar name matches found:")
    print(f"Biosample: {bios_name} -> Chromatin State: {cs_name}")


    print("\nAll paired biosamples and chromatin states:")
    print("Format: Biosample -> Chromatin State")
    print("-" * 50)
    
    # Print exact matches (where biosample name without T_ prefix matches CS name)
    for name in shared_names:
        if name in bios_names_cleaned:  # It's an exact match
            print(f"T_{name} -> {name}")
    
    # Print similar matches
    for bios_name, cs_name in similar_matches.items():
        print(f"T_{bios_name} -> {cs_name}")

    # Create a list of all valid pairs
    paired_data = []
    
    # Add exact matches
    for name in shared_names:
        if name in bios_names_cleaned:  # It's an exact match
            paired_data.append({
                'biosample': f"T_{name}",
                'chromatin_state': name
            })
    
    # Add similar matches
    for bios_name, cs_name in similar_matches.items():
        paired_data.append({
            'biosample': f"T_{bios_name}",
            'chromatin_state': cs_name
        })

    # Shuffle the pairs randomly
    random.seed(7)  # For reproducibility
    random.shuffle(paired_data)

    # Calculate split sizes
    total_samples = len(paired_data)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    # test_size will be the remainder

    # Split the data
    train_pairs = paired_data[:train_size]
    val_pairs = paired_data[train_size:train_size + val_size]
    test_pairs = paired_data[train_size + val_size:]

    # Print the splits
    print(f"\nTotal number of paired samples: {total_samples}")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    print(f"Test samples: {len(test_pairs)}")

    print("\nTrain Split:")
    print("-" * 50)
    for pair in train_pairs:
        print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    print("\nValidation Split:")
    print("-" * 50)
    for pair in val_pairs:
        print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    print("\nTest Split:")
    print("-" * 50)
    for pair in test_pairs:
        print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    # Optionally, save the splits to files
    import json
    
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    return splits

def chromatin_state_dataset_merged_train_test_val_split(solar_data_path="/project/compbio-lab/encode_data/"):
    merged_navigation = os.path.join(solar_data_path, "merged_navigation.json")
    
    import json
    with open(merged_navigation, "r") as f:
        navigation = json.load(f)

    # Get original biosample names
    original_bios_names = [t for t in navigation.keys()]

    # Clean biosample names and create mapping
    clean_to_original = {}
    for name in original_bios_names:
        # Remove _grp\d+_rep\d+ pattern
        cleaned_name = '_'.join([part for part in name.split('_') 
                               if not ('grp' in part or 'rep' in part or 'nonrep' in part)])
        if cleaned_name not in clean_to_original:
            clean_to_original[cleaned_name] = []
        clean_to_original[cleaned_name].append(name)

    # Get unique cleaned names
    unique_cleaned_names = list(clean_to_original.keys())
    
    # Get chromatin state names
    cs_names = [t for t in os.listdir(os.path.join(solar_data_path, "chromatin_state_annotations"))]

    # Find intersection between cleaned names and chromatin states
    shared_names = set(unique_cleaned_names) & set(cs_names)
    
    print(f"\nNumber of shared cell types: {len(shared_names)}")
    
    # Convert to list and shuffle
    shared_names = list(shared_names)
    random.seed(7)  # For reproducibility
    random.shuffle(shared_names)

    # Calculate split sizes
    total_samples = len(shared_names)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)

    # Split the cleaned names
    train_names = shared_names[:train_size]
    val_names = shared_names[train_size:train_size + val_size]
    test_names = shared_names[train_size + val_size:]

    # Create final splits with original biosample names
    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    # Helper function to create pairs
    def create_pairs(clean_names, split_name):
        pairs = []
        for clean_name in clean_names:
            # Get all original biosample names for this cleaned name
            original_names = clean_to_original[clean_name]
            # Create pairs with corresponding chromatin state
            for orig_name in original_names:
                pairs.append({
                    'biosample': orig_name,
                    'chromatin_state': clean_name  # clean_name is same as cs_name
                })
        return pairs

    # Create final splits
    splits['train'] = create_pairs(train_names, 'train')
    splits['val'] = create_pairs(val_names, 'val')
    splits['test'] = create_pairs(test_names, 'test')

    # Print statistics and results
    print("\nSplit Statistics:")
    print(f"Total unique cell types: {total_samples}")
    print(f"Train cell types: {len(train_names)}")
    print(f"Validation cell types: {len(val_names)}")
    print(f"Test cell types: {len(test_names)}")
    
    print(f"\nTotal biosamples: {len(original_bios_names)}")
    print(f"Train biosamples: {len(splits['train'])}")
    print(f"Validation biosamples: {len(splits['val'])}")
    print(f"Test biosamples: {len(splits['test'])}")

    # print("\nTrain Split:")
    # print("-" * 50)
    # for pair in splits['train']:
    #     print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    # print("\nValidation Split:")
    # print("-" * 50)
    # for pair in splits['val']:
    #     print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    # print("\nTest Split:")
    # print("-" * 50)
    # for pair in splits['test']:
    #     print(f"{pair['biosample']} -> {pair['chromatin_state']}")

    return splits

def train_chromatin_state_probe(
    model_path, hyper_parameters_path, 
    num_train_regions=10000, num_val_regions=3000, num_test_regions=30, 
    train_chrs=["chr19"], val_chrs=["chr21"], test_chrs=["chr21"],
    dataset_path="/project/compbio-lab/encode_data/", resolution=200, eic=True, stratified=True):

    candi = CANDIPredictor(model_path, hyper_parameters_path, data_path=dataset_path, DNA=True, eic=eic, split="all")

    probe = ChromatinStateProbe(candi.model.d_model, output_dim=18)

    if eic:
        splits = chromatin_state_dataset_eic_train_test_val_split(dataset_path)
    else:
        splits = chromatin_state_dataset_merged_train_test_val_split(dataset_path)

    splits["train"] = splits["train"][:50]
    splits["val"] = splits["val"][:15]

    def prepare_data(split, chrs, start_idx, end_idx):
        chromatin_state_data = {}
        # Process each chromosome
        for chr in chrs:  
            candi.chr = chr
            chromatin_state_data[chr] = {}

            # Load chromatin state data for each cell type in training split
            for pair in splits[split][start_idx:end_idx]:
                try:
                    bios_name = pair['biosample']
                    cs_name = pair['chromatin_state']
                    cs_dir = os.path.join(dataset_path, "chromatin_state_annotations", cs_name)
                    parsed_dirs = [d for d in os.listdir(cs_dir) if d.startswith(f'parsed{resolution}_')]

                    X, seq, mX = candi.load_encoder_input_bios(bios_name, x_dsf=1)
                    Z = candi.get_latent_representations_cropped(X, mX, seq=seq)
                    del X, seq, mX
                    Z = Z.cpu()
                except:
                    continue

                chromatin_state_data[chr][cs_name] = (Z, [])
                for idx, parsed_cs in enumerate(parsed_dirs):
                    annot = load_region_chromatin_states(os.path.join(cs_dir, parsed_cs), chr) 
                    context_len = (candi.model.l1 * 25) // resolution
                    target_len = ((len(annot) // context_len) * context_len)
                    annot = annot[:target_len]

                    chromatin_state_data[chr][cs_name][1].append(annot)
                
                del Z 
                gc.collect()

        return chromatin_state_data
    
    def stratify_batch(Z_batch, Y_batch):
        """Helper function to stratify a single batch of data"""
        # Check for empty batch
        if len(Z_batch) == 0 or len(Y_batch) == 0:
            return np.array([]), np.array([])
            
        # Convert lists to numpy arrays if needed
        Z_batch = torch.stack([z for z in Z_batch if z is not None]).numpy()  # Stack non-None tensors
        Y_batch = np.array([y for y in Y_batch if y is not None])
        
        # Get class distribution
        unique_labels, counts = np.unique(Y_batch, return_counts=True)
        min_count = min(counts)
        
        # Stratify the batch
        stratified_indices = []
        for label in unique_labels:
            label_indices = np.where(Y_batch == label)[0]
            # If we have fewer samples than min_count, use all of them
            n_samples = min(min_count, len(label_indices))
            selected_indices = np.random.choice(label_indices, n_samples, replace=False)
            stratified_indices.extend(selected_indices)
        
        # Shuffle the stratified indices
        np.random.shuffle(stratified_indices)
        
        return Z_batch[stratified_indices], Y_batch[stratified_indices]

    Z_train = []
    Y_train = []
    batch_size = len(splits["train"])//10
    for i in range(0, len(splits["train"]), batch_size):
        train_chromatin_state_data = prepare_data("train", train_chrs, i, i+batch_size)
        
        # Collect data for current batch
        Z_batch = []
        Y_batch = []
        
        for chr in train_chromatin_state_data.keys():
            for ct in train_chromatin_state_data[chr].keys():
                z, annots = train_chromatin_state_data[chr][ct]
                for annot in annots:
                    assert len(annot) == len(z), f"annot and Z are not the same length for {ct} on {chr}"
                    for bin in range(len(annot)):
                        label = annot[bin]
                        latent_vector = z[bin]
                        
                        if label is not None:
                            Z_batch.append(latent_vector)
                            Y_batch.append(label)
        
        if stratified:
            # Print batch distribution before stratification
            unique_labels, counts = np.unique(Y_batch, return_counts=True)
            total_samples = len(Y_batch)
            
            print(f"\nBatch {i//batch_size + 1} Distribution Before Stratification:")
            print("Label | Count | Percentage")
            print("-" * 30)
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                print(f"{label:10s} | {count:5d} | {percentage:6.2f}%")
            
            # Stratify the current batch
            Z_batch_stratified, Y_batch_stratified = stratify_batch(Z_batch, Y_batch)
            
            # Print batch distribution after stratification
            unique_labels, counts = np.unique(Y_batch_stratified, return_counts=True)
            total_samples = len(Y_batch_stratified)
            
            print(f"\nBatch {i//batch_size + 1} Distribution After Stratification:")
            print("Label | Count | Percentage")
            print("-" * 30)
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                print(f"{label:10s} | {count:5d} | {percentage:6.2f}%")
            
            # Add stratified batch to training data
            Z_train.extend(Z_batch_stratified)
            Y_train.extend(Y_batch_stratified)
        else:
            # Add unstratified batch to training data
            Z_train.extend(Z_batch)
            Y_train.extend(Y_batch)
        
        del train_chromatin_state_data, Z_batch, Y_batch
        gc.collect()

    # Convert lists to numpy arrays
    Z_train = np.stack(Z_train)
    Y_train = np.array(Y_train)

    # Print final training set statistics
    print("\nFinal Training Dataset Analysis:")
    print(f"Z_train shape: {Z_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")

    unique_labels, counts = np.unique(Y_train, return_counts=True)
    total_samples = len(Y_train)

    print("\nFinal Class Distribution:")
    print("Label | Count | Percentage")
    print("-" * 30)
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100
        print(f"{label:10s} | {count:5d} | {percentage:6.2f}%")

    #  Analysis and stratification of training data
    print("\nTraining Dataset Analysis:")
    print(f"Z_train shape: {Z_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")

    # Analyze class distribution
    unique_labels, counts = np.unique(Y_train, return_counts=True)
    total_samples = len(Y_train)

    print("\nClass Distribution:")
    print("Label | Count | Percentage")
    print("-" * 30)
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100
        print(f"{label:10s} | {count:5d} | {percentage:6.2f}%")  # Changed :5d to :5s for label

    Z_val = [] 
    Y_val = []
    batch_size = len(splits["val"])//10
    for i in range(0, len(splits["val"]), batch_size):
        val_chromatin_state_data = prepare_data("val", val_chrs, i, i+batch_size)
        
        for chr in val_chromatin_state_data.keys():
            for ct in val_chromatin_state_data[chr].keys():
                z, annots = val_chromatin_state_data[chr][ct]
                for annot in annots:

                    assert len(annot) == len(z), f"annot and Z are not the same length for {ct} on {chr}"
                    for bin in range(len(annot)):
                        label = annot[bin]
                        latent_vector = z[bin]

                        if label is not None:
                            Z_val.append(latent_vector)
                            Y_val.append(label)
    
        del val_chromatin_state_data
        gc.collect()

    # Convert lists to tensors first since Z contains torch tensors
    Z_val = np.stack(Z_val)
    Y_val = np.array(Y_val)

    # # Use stratified training data for model training
    probe.fit(Z_train, Y_train, Z_val, Y_val, 
        num_epochs=800, learning_rate=0.001, batch_size=100)

################################################################################

def assay_importance(candi, bios_name, crop_edges=True):
    """
    we want to evaluate predictability of different assays as a function of input assays
    we want to see which input assays are most important for predicting the which output assay

    different tested settings:
        - just one input assay
        - top 6 histone mods
        - accessibility
        - accessibility + top 6 histone mods
    """

    X, Y, P, seq, mX, mY, avX, avY = candi.load_bios(bios_name, x_dsf=1)
    available_indices = torch.where(avX[0, :] == 1)[0]
    expnames = list(candi.dataset.aliases["experiment_aliases"].keys())

    available_assays = list(candi.dataset.navigation[bios_name].keys())
    print("available assays: ", available_assays)

    # Create distributions and get means
    Y = Y.view(-1, Y.shape[-1])
    P = P.view(-1, P.shape[-1])

    # keys: list of inputs, values: metrics per output assay | metrics: PP, Pearson, Spearman
    results = {}

    # # pred based on just one input assay
    for ii, keep_only in enumerate(available_indices):
        # Create mask where everything is masked except the current assay
        imp_target = [idx for idx in available_indices if idx != keep_only]
        print(f"single input: {expnames[keep_only]}")

        results[expnames[keep_only]] = {}
        
        if crop_edges:
            n, p, mu, var, _ = candi.pred_cropped(X, mX, mY, avX, imp_target=imp_target, seq=seq)
        else:
            n, p, mu, var, _ = candi.pred(X, mX, mY, avX, imp_target=imp_target, seq=seq)

        pval_dist = Gaussian(mu, var)
        count_dist = NegativeBinomial(p, n)

        pval_mean = pval_dist.expect()
        count_mean = count_dist.expect()

        prob_pval = pval_dist.pdf(P)
        prob_count = count_dist.pmf(Y)

        for jj in imp_target:
            # Calculate metrics for assay jj
            count_true = Y[:, jj].numpy()
            pval_true = P[:, jj].numpy()

            prob_pval_jj = prob_pval[:, jj]
            prob_count_jj = prob_count[:, jj]
            
            # Get predictions
            count_pred = count_mean[:, jj].numpy()
            pval_pred = pval_mean[:, jj].numpy()

            pval_pred = np.sinh(pval_pred)
            pval_true = np.sinh(pval_true)
            
            metr = get_metrics(prob_pval_jj, prob_count_jj, pval_true, pval_pred, count_true, count_pred)
            results[expnames[keep_only]][expnames[jj]] = metr

    accessibility_assays = ["ATAC-seq", "DNase-seq"]
    has_accessibility = all(assay in available_assays for assay in accessibility_assays)
    if has_accessibility:
        print(f"accessibility inputs: {accessibility_assays}")
        # Create mask where everything is masked except the current assay
        imp_target = [idx for idx in available_indices if expnames[idx] not in accessibility_assays]
        results["accessibility"] = {}
        
        if crop_edges:
            n, p, mu, var, _ = candi.pred_cropped(X, mX, mY, avX, imp_target=imp_target, seq=seq)
        else:
            n, p, mu, var, _ = candi.pred(X, mX, mY, avX, imp_target=imp_target, seq=seq)

        pval_dist = Gaussian(mu, var)
        count_dist = NegativeBinomial(p, n)

        pval_mean = pval_dist.expect()
        count_mean = count_dist.expect()

        prob_pval = pval_dist.pdf(P)
        prob_count = count_dist.pmf(Y)

        for jj in imp_target:
            # Calculate metrics for assay jj
            count_true = Y[:, jj].numpy()
            pval_true = P[:, jj].numpy()

            prob_pval_jj = prob_pval[:, jj]
            prob_count_jj = prob_count[:, jj]
            
            # Get predictions
            count_pred = count_mean[:, jj].numpy()
            pval_pred = pval_mean[:, jj].numpy()

            pval_pred = np.sinh(pval_pred)
            pval_true = np.sinh(pval_true)
            
            metr = get_metrics(prob_pval_jj, prob_count_jj, pval_true, pval_pred, count_true, count_pred)
            results["accessibility"][expnames[jj]] = metr

    histone_mods = ["H3K4me3", "H3K4me1", "H3K27ac", "H3K27me3", "H3K9me3", "H3K36me3"]
    has_histone_mods = all(assay in available_assays for assay in histone_mods)
    if has_histone_mods:
        print(f"6 histone mods inputs: {histone_mods}")
        # Create mask where everything is masked except the current assay
        imp_target = [idx for idx in available_indices if expnames[idx] not in histone_mods]
        results["histone_mods"] = {}
        
        if crop_edges:
            n, p, mu, var, _ = candi.pred_cropped(X, mX, mY, avX, imp_target=imp_target, seq=seq)
        else:
            n, p, mu, var, _ = candi.pred(X, mX, mY, avX, imp_target=imp_target, seq=seq)

        pval_dist = Gaussian(mu, var)
        count_dist = NegativeBinomial(p, n)

        pval_mean = pval_dist.expect()
        count_mean = count_dist.expect()

        prob_pval = pval_dist.pdf(P)
        prob_count = count_dist.pmf(Y)

        for jj in imp_target:
            # Calculate metrics for assay jj
            count_true = Y[:, jj].numpy()
            pval_true = P[:, jj].numpy()

            prob_pval_jj = prob_pval[:, jj]
            prob_count_jj = prob_count[:, jj]
            
            # Get predictions
            count_pred = count_mean[:, jj].numpy()
            pval_pred = pval_mean[:, jj].numpy()
            
            pval_pred = np.sinh(pval_pred)
            pval_true = np.sinh(pval_true)

            metr = get_metrics(prob_pval_jj, prob_count_jj, pval_true, pval_pred, count_true, count_pred)
            results["histone_mods"][expnames[jj]] = metr

    if has_accessibility and has_histone_mods and len(available_assays) > len(accessibility_assays) + len(histone_mods):
        print(f"6 histone mods + accessibility inputs: {histone_mods + accessibility_assays}")
        # Create mask where everything is masked except the current assay
        imp_target = [idx for idx in available_indices if expnames[idx] not in histone_mods + accessibility_assays]
        results["histone_mods_accessibility"] = {}
        
        if crop_edges:
            n, p, mu, var, _ = candi.pred_cropped(X, mX, mY, avX, imp_target=imp_target, seq=seq)
        else:
            n, p, mu, var, _ = candi.pred(X, mX, mY, avX, imp_target=imp_target, seq=seq)

        pval_dist = Gaussian(mu, var)
        count_dist = NegativeBinomial(p, n)

        pval_mean = pval_dist.expect()
        count_mean = count_dist.expect()

        prob_pval = pval_dist.pdf(P)
        prob_count = count_dist.pmf(Y)

        for jj in imp_target:
            # Calculate metrics for assay jj
            count_true = Y[:, jj].numpy()
            pval_true = P[:, jj].numpy()
            
            # Get predictions
            count_pred = count_mean[:, jj].numpy()
            pval_pred = pval_mean[:, jj].numpy()

            pval_pred = np.sinh(pval_pred)
            pval_true = np.sinh(pval_true)
            
            metr = get_metrics(prob_pval, prob_count, pval_true, pval_pred, count_true, count_pred)
            results["histone_mods_accessibility"][expnames[jj]] = metr

    return results  

def calibration_curve(candi, bios_name, crop_edges=True, eic=False, savedir=f"models/output/"):
    expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
    X, Y, P, seq, mX, mY, avX, avY = candi.load_bios(bios_name, x_dsf=1)

    if not eic:
        imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist = candi.evaluate_leave_one_out(
            X, mX, mY, avX, Y, P, seq=seq, return_preds=True, crop_edges=crop_edges)
        
        Y = Y.view(-1, Y.shape[-1])
        P = P.view(-1, P.shape[-1])

        available_indices = torch.where(avX[0, :] == 1)[0]
        for jj in available_indices:
            pval_true = P[:, jj].numpy()
            count_true = Y[:, jj].numpy()

            imp_pval_dist_idx = Gaussian(imp_pval_dist.mu[:, jj], imp_pval_dist.var[:, jj])
            imp_count_dist_idx = NegativeBinomial(imp_count_dist.p[:, jj], imp_count_dist.n[:, jj])

            ups_pval_dist_idx = Gaussian(ups_pval_dist.mu[:, jj], ups_pval_dist.var[:, jj])
            ups_count_dist_idx = NegativeBinomial(ups_count_dist.p[:, jj], ups_count_dist.n[:, jj])

            viz_calibration(
                imp_pval_dist_idx, ups_pval_dist_idx, imp_count_dist_idx, ups_count_dist_idx,
                pval_true, count_true, f"{expnames[jj]}", savedir=f"{savedir}/calibration_{bios_name}/")

    else:
        ups_count_dist, ups_pval_dist = candi.evaluate_leave_one_out_eic(
            X, mX, mY, avX, Y, P, avY, seq=seq, return_preds=True, crop_edges=crop_edges)

        Y = Y.view(-1, Y.shape[-1])
        P = P.view(-1, P.shape[-1])
        X = X.view(-1, X.shape[-1])

        available_X_indices = torch.where(avX[0, :] == 1)[0]
        available_Y_indices = torch.where(avY[0, :] == 1)[0]

        for jj in range(Y.shape[1]):
            
            pval_dist = Gaussian(ups_pval_dist.mu[:, jj], ups_pval_dist.var[:, jj])
            count_dist = NegativeBinomial(ups_count_dist.p[:, jj], ups_count_dist.n[:, jj])
            pval_true = P[:, jj].numpy()
            count_true = Y[:, jj].numpy()

            if jj in available_Y_indices:
                print(f"imputed assay: {expnames[jj]}")
                viz_calibration_eic(
                    pval_dist, count_dist, pval_true, count_true, "imputed", f"{expnames[jj]}", savedir=f"{savedir}/calibration_{bios_name}/")

            elif jj in available_X_indices:
                print(f"upsampled assay: {expnames[jj]}")
                viz_calibration_eic(
                    pval_dist, count_dist, pval_true, count_true, "upsampled", f"{expnames[jj]}", savedir=f"{savedir}/calibration_{bios_name}/")
                
            else:
                continue

    # exit()
    # print(bios_name)

################################################################################
if __name__ == "__main__":
    if sys.argv[1] == "cs_probe":
        model_path = "models/CANDIfull_DNA_random_mask_Dec8_model_checkpoint_epoch0.pth"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec8_20241208194100_params45093285.pkl"
        eic = False
        train_chromatin_state_probe(model_path, hyper_parameters_path, dataset_path="/project/compbio-lab/encode_data/", eic=eic)

    elif sys.argv[1] == "latent_repr":
        # model_path = "models/CANDIeic_DNA_random_mask_Nov28_model_checkpoint_epoch3.pth"
        hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov28_20241128164234_params45093285.pkl"
        eic = True

        ct0_repr1 = "ENCBS706NOO"
        ct0_repr2 = "ENCBS314QQU"
        latent_reproducibility(model_path, hyper_parameters_path, ct0_repr1, ct0_repr2, dataset_path="/project/compbio-lab/encode_data/")

        ct1_repr1 = "ENCBS674MPN"
        ct1_repr2 = "ENCBS639AAA"
        latent_reproducibility(model_path, hyper_parameters_path, ct1_repr1, ct1_repr2, dataset_path="/project/compbio-lab/encode_data/")

        ct2_repr1 = "ENCBS967MVZ"
        ct2_repr2 = "ENCBS789UPK"
        latent_reproducibility(model_path, hyper_parameters_path, ct2_repr1, ct2_repr2, dataset_path="/project/compbio-lab/encode_data/") 

        ct3_repr1 = "ENCBS715VCP"
        ct3_repr2 = "ENCBS830CIQ"
        latent_reproducibility(model_path, hyper_parameters_path, ct3_repr1, ct3_repr2, dataset_path="/project/compbio-lab/encode_data/")  
        
        ct4_repr1 = "ENCBS865RXK"
        ct4_repr2 = "ENCBS188BKX"
        latent_reproducibility(model_path, hyper_parameters_path, ct4_repr1, ct4_repr2, dataset_path="/project/compbio-lab/encode_data/")   

        ct5_repr1 = "ENCBS655ARO"
        ct5_repr2 = "ENCBS075PNA"
        latent_reproducibility(model_path, hyper_parameters_path, ct5_repr1, ct5_repr2, dataset_path="/project/compbio-lab/encode_data/")   

        # Random pairs from different cell types to test cross-cell-type reproducibility
        print("\nTesting cross-cell-type reproducibility:")
        # CT1 vs CT2
        latent_reproducibility(model_path, hyper_parameters_path, ct1_repr1, ct2_repr1, dataset_path="/project/compbio-lab/encode_data/")
        
        # CT2 vs CT3 
        latent_reproducibility(model_path, hyper_parameters_path, ct2_repr2, ct3_repr1, dataset_path="/project/compbio-lab/encode_data/")
        
        # CT3 vs CT4
        latent_reproducibility(model_path, hyper_parameters_path, ct3_repr2, ct4_repr1, dataset_path="/project/compbio-lab/encode_data/")
        
        # CT4 vs CT5
        latent_reproducibility(model_path, hyper_parameters_path, ct4_repr2, ct5_repr1, dataset_path="/project/compbio-lab/encode_data/")
        
        # CT5 vs CT1 
        latent_reproducibility(model_path, hyper_parameters_path, ct5_repr2, ct1_repr2, dataset_path="/project/compbio-lab/encode_data/")

        # CT0 vs CT3
        print("\nTesting CT0 vs CT3 reproducibility:")
        latent_reproducibility(model_path, hyper_parameters_path, ct0_repr1, ct3_repr1, dataset_path="/project/compbio-lab/encode_data/")

    elif sys.argv[1] == "perplexity":
        model_path = "models/CANDIfull_DNA_random_mask_Dec8_model_checkpoint_epoch0.pth"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec8_20241208194100_params45093285.pkl"
        eic = False

        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=True)
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"
        bios_name = sys.argv[2]

        # Load latent representations
        X, Y, P, seq, mX, mY, avX, avY = candi.load_bios(bios_name, x_dsf=1)

        n, p, mu, var, Z = candi.pred_cropped(X, mX, mY, avX, seq=seq, crop_percent=0.3)
        
        Y = Y.view(-1, Y.shape[-1])
        P = P.view(-1, P.shape[-1])

        count_dist = NegativeBinomial(p, n)
        pval_dist = Gaussian(mu, var)

        count_probabilities = count_dist.pmf(Y)
        pval_probabilities = pval_dist.pdf(P)

        for i in range(Y.shape[1]):
            # print(Y[:, i].mean())
            if avY[0, i] == 1:
                print(
                f"Assay: {expnames[i]}, PP_count: {perplexity(count_probabilities[:, i]):.3f}, PP_pval: {perplexity(pval_probabilities[:, i]):.3f}")
        
        # position_PP_count = []
        # position_PP_pval = []
        # for i in range(Y.shape[0]):
        #     # Get probabilities for available assays at each position
        #     p_count = count_probabilities[i, avY[0]==1]
        #     p_pval = pval_probabilities[i, avY[0]==1]
        #     position_PP_count.append(perplexity(p_count))
        #     position_PP_pval.append(perplexity(p_pval))

        # Create mask for available assays
        available_mask = (avY[0] == 1)

        # Calculate perplexity for all positions at once using broadcasting
        # Shape: (n_positions, n_available_assays)
        count_probs_available = count_probabilities[:, available_mask]
        pval_probs_available = pval_probabilities[:, available_mask]

        # Calculate perplexity using vectorized operations
        # Add small epsilon to prevent log(0)
        epsilon = 1e-10
        position_PP_count = torch.exp(-torch.sum(torch.log(count_probs_available + epsilon), dim=1) / count_probs_available.shape[1])
        position_PP_pval = torch.exp(-torch.sum(torch.log(pval_probs_available + epsilon), dim=1) / pval_probs_available.shape[1])

        # Convert to numpy for statistics (if needed)
        position_PP_count = position_PP_count.cpu().numpy()
        position_PP_pval = position_PP_pval.cpu().numpy()

        # Print statistics
        print(f"Position PP_count: {np.mean(position_PP_count):.3f}, Position PP_pval: {np.mean(position_PP_pval):.3f}")
        print(f"Position PP_count std: {np.std(position_PP_count):.3f}, Position PP_pval std: {np.std(position_PP_pval):.3f}")
        print(f"Position PP_count 95% CI: {np.percentile(position_PP_count, 2.5):.3f} - {np.percentile(position_PP_count, 97.5):.3f}")
        print(f"Position PP_pval 95% CI: {np.percentile(position_PP_pval, 2.5):.3f} - {np.percentile(position_PP_pval, 97.5):.3f}")
        
        # Reduce resolution of perplexity scores by averaging every 8 values
        def reduce_resolution(arr, factor=8):
            # Ensure the array length is divisible by factor
            pad_length = (factor - (len(arr) % factor)) % factor
            if pad_length > 0:
                arr = np.pad(arr, (0, pad_length), mode='edge')
            
            # Reshape and average
            arr_reshaped = arr.reshape(-1, factor)
            return np.mean(arr_reshaped, axis=1)

        # Reduce resolution of perplexity scores
        position_PP_count_reduced = reduce_resolution(position_PP_count)
        position_PP_pval_reduced = reduce_resolution(position_PP_pval)

        # Get 99th percentile values to clip outliers
        pp_count_99th = np.percentile(position_PP_count_reduced, 99)
        pp_pval_99th = np.percentile(position_PP_pval_reduced, 99)

        # Clip values above 99th percentile
        position_PP_count_reduced = np.clip(position_PP_count_reduced, None, pp_count_99th)
        position_PP_pval_reduced = np.clip(position_PP_pval_reduced, None, pp_pval_99th)

        # Print statistics after reduction
        print("\nAfter resolution reduction:")
        print(f"Position PP_count: {np.mean(position_PP_count_reduced):.3f}, Position PP_pval: {np.mean(position_PP_pval_reduced):.3f}")
        print(f"Position PP_count std: {np.std(position_PP_count_reduced):.3f}, Position PP_pval std: {np.std(position_PP_pval_reduced):.3f}")
        print(f"Position PP_count 99% CI: {np.percentile(position_PP_count_reduced, 0.5):.3f} - {np.percentile(position_PP_count_reduced, 99.5):.3f}")
        print(f"Position PP_pval 99% CI: {np.percentile(position_PP_pval_reduced, 0.5):.3f} - {np.percentile(position_PP_pval_reduced, 99.5):.3f}")
        # Print min and max values
        print("\nMin/Max values:")
        print(f"Position PP_count min: {np.min(position_PP_count_reduced):.3f}, max: {np.max(position_PP_count_reduced):.3f}")
        print(f"Position PP_pval min: {np.min(position_PP_pval_reduced):.3f}, max: {np.max(position_PP_pval_reduced):.3f}")

        # Calculate and print correlations between count and p-value perplexity scores
        pearson_corr, pearson_p = scipy.stats.pearsonr(position_PP_count_reduced, position_PP_pval_reduced)
        spearman_corr, spearman_p = scipy.stats.spearmanr(position_PP_count_reduced, position_PP_pval_reduced)

        print("\nCorrelations between count and p-value perplexity:")
        print(f"Pearson correlation: {pearson_corr:.3f} (p={pearson_p:.3e})")
        print(f"Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3e})")
        # exit()

        # Verify lengths match
        assert len(position_PP_count_reduced) == len(Z), f"Length mismatch: {len(position_PP_count_reduced)} vs {len(Z)}"
        
        # Create visualization plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Compute PCA
        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(Z.cpu().numpy())
        
        # Compute UMAP
        reducer = umap.UMAP(random_state=42)
        Z_umap = reducer.fit_transform(Z.cpu().numpy())
        
        # Plot PCA colored by count perplexity
        scatter1 = axes[0,0].scatter(Z_pca[:, 0], Z_pca[:, 1], 
                                    c=position_PP_count_reduced, 
                                    cmap='viridis', 
                                    alpha=0.5, 
                                    s=1)
        axes[0,0].set_title('PCA - Colored by Count Perplexity')
        axes[0,0].set_xlabel('PC1')
        axes[0,0].set_ylabel('PC2')
        plt.colorbar(scatter1, ax=axes[0,0])
        
        # Plot PCA colored by p-value perplexity
        scatter2 = axes[0,1].scatter(Z_pca[:, 0], Z_pca[:, 1], 
                                    c=position_PP_pval_reduced, 
                                    cmap='viridis', 
                                    alpha=0.5, 
                                    s=1)
        axes[0,1].set_title('PCA - Colored by P-value Perplexity')
        axes[0,1].set_xlabel('PC1')
        axes[0,1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[0,1])
        
        # Plot UMAP colored by count perplexity
        scatter3 = axes[1,0].scatter(Z_umap[:, 0], Z_umap[:, 1], 
                                    c=position_PP_count_reduced, 
                                    cmap='viridis', 
                                    alpha=0.5, 
                                    s=1)
        axes[1,0].set_title('UMAP - Colored by Count Perplexity')
        axes[1,0].set_xlabel('UMAP1')
        axes[1,0].set_ylabel('UMAP2')
        plt.colorbar(scatter3, ax=axes[1,0])
        
        # Plot UMAP colored by p-value perplexity
        scatter4 = axes[1,1].scatter(Z_umap[:, 0], Z_umap[:, 1], 
                                    c=position_PP_pval_reduced, 
                                    cmap='viridis', 
                                    alpha=0.5, 
                                    s=1)
        axes[1,1].set_title('UMAP - Colored by P-value Perplexity')
        axes[1,1].set_xlabel('UMAP1')
        axes[1,1].set_ylabel('UMAP2')
        plt.colorbar(scatter4, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(f'latent_space_perplexity_{bios_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    elif sys.argv[1] == "eval_full":
        model_path = "models/CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pt"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pkl"
        eic = False

        splits = ["test", "val"]  

        for split in splits:
            # Load latent representations
            candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic, split=split)
            expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
            candi.chr = "chr21"
            metrics = {}

            # for bios_name in random.sample(list(candi.dataset.navigation.keys()), 3):
            for bios_name in random.sample(list(candi.dataset.navigation.keys()), len(candi.dataset.navigation)):
                try:
                    print(bios_name)
                    start_time = time.time()
                    metrics[bios_name] = candi.evaluate(bios_name)
                    end_time = time.time()
                    print(f"Evaluation took {end_time - start_time:.2f} seconds")
                    print("\n\n")

                except Exception as e:
                    print(f"Error processing {bios_name}: {e}")
                    continue
            
            results = []
            for bios_name in metrics.keys():
                for exp in metrics[bios_name].keys():
                    results.append({
                        "bios_name": bios_name,
                        "experiment": expnames[exp],
                        **{"count_" + k: v for k, v in metrics[bios_name][exp]["count_metrics"].items()},
                        **{"pval_" + k: v for k, v in metrics[bios_name][exp]["pval_metrics"].items()},
                    })

            df = pd.DataFrame(results)
            df.to_csv(f"models/output/full_{split}_metrics.csv", index=False)
            print(df)

    elif sys.argv[1] == "eval_eic":
        model_path = "models/CANDIeic_DNA_random_mask_Jan15_20250114215927_params45093285.pt"
        hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Jan15_20250114215927_params45093285.pkl"

        # model_path = "models/CANDIfull_DNA_random_mask_Dec8_model_checkpoint_epoch0.pth"
        # hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec8_20241208194100_params45093285.pkl"
        eic = True

        splits = ["test", "val"]  

        for split in splits:
            # Load latent representations
            candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic, split=split)
            expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
            candi.chr = "chr21"
            metrics = {}

            for bios_name in random.sample(list(candi.dataset.navigation.keys()), len(candi.dataset.navigation)):
                try:
                    print(bios_name)
                    start_time = time.time()
                    metrics[bios_name] = candi.evaluate(bios_name)
                    end_time = time.time()
                    print(f"Evaluation took {end_time - start_time:.2f} seconds")
                    print("\n\n")

                except Exception as e:
                    print(f"Error processing {bios_name}: {e}")
                    continue
            
            results = []
            for bios_name in metrics.keys():
                for exp in metrics[bios_name].keys():
                    results.append({
                        "bios_name": bios_name,
                        "experiment": expnames[exp],
                        "comparison": metrics[bios_name][exp]["comparison"],
                        **{"count_" + k: v for k, v in metrics[bios_name][exp]["count_metrics"].items()},
                        **{"pval_" + k: v for k, v in metrics[bios_name][exp]["pval_metrics"].items()},
                    })

            df = pd.DataFrame(results)
            df.to_csv(f"models/output/eic_{split}_metrics.csv", index=False)
            print(df)
    
    elif sys.argv[1] == "eval_full_bios":
    
        model_path = "models/CANDIfull_DNA_random_mask_Dec9_20241209114510_params45093285.pt"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pkl"
        eic = False


        # model_path = "models/CANDIeic_DNA_random_mask_Nov28_model_checkpoint_epoch3.pth"
        # hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov28_20241128164234_params45093285.pkl"

        # Load latent representations
        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic, split="test")
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"

        if sys.argv[2] == "show_test_bios":
            print(candi.dataset.navigation.keys())
            exit()
        else:
            bios_name = sys.argv[2]

        try:
            print(bios_name)
            metrics = candi.evaluate(bios_name)
            print("\n\n")
            
        except Exception as e:
            print(f"Error processing {bios_name}: {e}")
    
    elif sys.argv[1] == "eval_eic_bios":

        # model_path = "models/CANDIfull_DNA_random_mask_Dec8_model_checkpoint_epoch0.pth"
        # hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec8_20241208194100_params45093285.pkl"

        model_path = "models/CANDIeic_DNA_random_mask_Nov28_model_checkpoint_epoch3.pth"
        hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov28_20241128164234_params45093285.pkl"
        eic = True

        # Load latent representations
        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic, split="val")
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"

        if sys.argv[2] == "show_test_bios":
            print(candi.dataset.navigation.keys())
            exit()
        else:
            bios_name = sys.argv[2]

        try:
            print(bios_name)
            metrics = candi.evaluate(bios_name)
            print("\n\n")
            
        except Exception as e:
            print(f"Error processing {bios_name}: {e}")

    elif sys.argv[1] == "assay_importance":
        model_path = "models/CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pt"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pkl"
        eic = False

        # Load latent representations
        candi = CANDIPredictor(
            model_path, hyper_parameters_path, 
            data_path="/project/compbio-lab/encode_data/", 
            DNA=True, eic=eic, split="test")
        expnames = list(candi.dataset.aliases["experiment_aliases"].keys())
        candi.chr = "chr21"
        
        metrics = {}
        bios_names = list(candi.dataset.navigation.keys())
        for bios_name in bios_names:
            try:
                print(bios_name)
                metrics[bios_name] = assay_importance(candi, bios_name)
            except Exception as e:
                print(f"Error processing {bios_name}: {e}")
                continue

        results = []
        for bios_name in metrics.keys():
            for input in metrics[bios_name]:
                for output in metrics[bios_name][input]:
                    results.append({
                        "bios_name": bios_name,
                        "input": input,
                        "output": output,
                        "gw_pp_pval": metrics[bios_name][input][output]["gw_pp_pval"],
                        "gw_pp_count": metrics[bios_name][input][output]["gw_pp_count"],
                        "gene_pp_pval": metrics[bios_name][input][output]["gene_pp_pval"],
                        "gene_pp_count": metrics[bios_name][input][output]["gene_pp_count"],
                        "prom_pp_pval": metrics[bios_name][input][output]["prom_pp_pval"],
                        "prom_pp_count": metrics[bios_name][input][output]["prom_pp_count"],
                        "gw_pearson_pval": metrics[bios_name][input][output]["gw_pearson_pval"],
                        "gw_spearman_pval": metrics[bios_name][input][output]["gw_spearman_pval"],
                        "gw_pearson_count": metrics[bios_name][input][output]["gw_pearson_count"],
                        "gw_spearman_count": metrics[bios_name][input][output]["gw_spearman_count"],
                        "gene_pearson_pval": metrics[bios_name][input][output]["gene_pearson_pval"],
                        "gene_spearman_pval": metrics[bios_name][input][output]["gene_spearman_pval"],
                        "gene_pearson_count": metrics[bios_name][input][output]["gene_pearson_count"],
                        "gene_spearman_count": metrics[bios_name][input][output]["gene_spearman_count"],
                        "prom_pearson_pval": metrics[bios_name][input][output]["prom_pearson_pval"],
                        "prom_spearman_pval": metrics[bios_name][input][output]["prom_spearman_pval"],
                        "prom_pearson_count": metrics[bios_name][input][output]["prom_pearson_count"],
                        "prom_spearman_count": metrics[bios_name][input][output]["prom_spearman_count"],
                        "one_obs_pearson_pval": metrics[bios_name][input][output]["one_obs_pearson_pval"],
                        "one_obs_spearman_pval": metrics[bios_name][input][output]["one_obs_spearman_pval"],
                        "one_obs_pearson_count": metrics[bios_name][input][output]["one_obs_pearson_count"],
                        "one_obs_spearman_count": metrics[bios_name][input][output]["one_obs_spearman_count"],
                        "one_imp_pearson_pval": metrics[bios_name][input][output]["one_imp_pearson_pval"],
                        "one_imp_spearman_pval": metrics[bios_name][input][output]["one_imp_spearman_pval"],
                        "one_imp_pearson_count": metrics[bios_name][input][output]["one_imp_pearson_count"],
                        "one_imp_spearman_count": metrics[bios_name][input][output]["one_imp_spearman_count"],
                        "peak_overlap_pval": metrics[bios_name][input][output]["peak_overlap_pval"],
                        "peak_overlap_count": metrics[bios_name][input][output]["peak_overlap_count"]
                    })

        df = pd.DataFrame(results)
        print(df)

        df.to_csv("models/output/assay_importance.csv", index=False)

    elif sys.argv[1] == "viz_calibration":
        model_path = "models/CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pt"
        hyper_parameters_path = "models/hyper_parameters_CANDIfull_DNA_random_mask_Dec12_20241212134626_params45093285.pkl"
        eic = False

        # model_path = "models/CANDIeic_DNA_random_mask_Nov28_model_checkpoint_epoch3.pth"
        # hyper_parameters_path = "models/hyper_parameters_CANDIeic_DNA_random_mask_Nov28_20241128164234_params45093285.pkl"
        # eic = True

        candi = CANDIPredictor(model_path, hyper_parameters_path, data_path="/project/compbio-lab/encode_data/", DNA=True, eic=eic, split="test")
        candi.chr = "chr21"

        if sys.argv[2] == "all":
            bios_names = list(candi.dataset.navigation.keys())
            for bios_name in bios_names:
                try:
                    calibration_curve(candi, bios_name, eic=eic)
                except Exception as e:
                    print(f"Error processing {bios_name}: {e}")
                    continue
        else:
            bios_name = sys.argv[2]
            calibration_curve(candi, bios_name, eic=eic)

    elif sys.argv[1] == "viz":
        if os.path.exists("models/DEC18_RESULTS/"):
            viz_eic_paper_comparison(res_dir="models/DEC18_RESULTS/")
        else:
            print("EIC test metrics not computed")

        # if os.path.exists("models/DEC18_RESULTS/assay_importance.csv"):
        #     df = pd.read_csv("models/DEC18_RESULTS/assay_importance.csv")
        #     viz_feature_importance(df, savedir="models/DEC18_RESULTS/")
        # else:
        #     print("Assay importance not computed")
            
        exit()

        ###################################################### 
        
        # if os.path.exists("models/output/eic_test_metrics.csv"):
        #     df = pd.read_csv("models/output/eic_test_metrics.csv")
        #     viz_eic_metrics(df, savedir="models/output/")
        # else:
        #     print("EIC test metrics not computed")
        
        ######################################################

        if os.path.exists("models/DEC18_RESULTS/full_test_metrics.csv"):
            df = pd.read_csv("models/DEC18_RESULTS/full_test_metrics.csv")
            viz_full_metrics(df, savedir="models/DEC18_RESULTS/")
        else:
            print("Full test metrics not computed")  

        ######################################################

        # if os.path.exists("models/output/eic_val_metrics.csv"):
        #     df = pd.read_csv("models/output/eic_val_metrics.csv")
        #     viz_eic_metrics(df, savedir="models/output/")
        # else:
        #     print("EIC val metrics not computed")   

        ######################################################

        # if os.path.exists("models/output/full_val_metrics.csv"):
        #     df = pd.read_csv("models/output/full_val_metrics.csv")
        #     viz_full_metrics(df, savedir="models/output/")
        # else:
        #     print("Full val metrics not computed")  
