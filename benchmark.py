import pyBigWig, os
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def comparison_boxplot(merged_df, name="test", savedir="models/DEC18_RESULTS/"):
    """
    Create separate boxplots for each metric comparing teams across assays.
    """

    merged_df = merged_df[merged_df['assay'] != 'ATAC-seq'].reset_index(drop=True)
    
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
    
    # Sort assays based on the average of different teams on that assay
    assay_means = merged_df.groupby('assay')[['gwcorr', 'gwspear', 'mse']].mean().mean(axis=1)
    sorted_assays = assay_means.sort_values().index.tolist()
    
    metrics = [
        ('gwcorr', 'Genome-wide Correlation'),
        ('gwspear', 'Genome-wide Spearman'),
        ('mse', 'MSE')
    ]
    
    for metric, title in metrics:
        plt.figure(figsize=(14, 7))
        
        for i, assay in enumerate(sorted_assays):
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
            for i in range(len(sorted_assays))
        ]
        plt.xticks(xticks_positions, sorted_assays, rotation=90, fontsize=12)  # Increased fontsize by 50%
        
        for i in range(len(sorted_assays) - 1):
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
        plt.savefig(f"{savedir}/eic_paper_comparison_{metric}_boxplot.png", dpi=300)
        plt.savefig(f"{savedir}/eic_paper_comparison_{metric}_boxplot.svg", format="svg")
        plt.clf()
        plt.close()
        plt.cla()

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

candieic_blind_res = pd.read_csv(f"models/DEC18_RESULTS/eic_test_metrics.csv")
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

# Reverse the team_id dictionary to map team names to their IDs
reversed_assay_id = {v: k for k, v in assay_id.items()}
reversed_cell_type_id = {v: k for k, v in cell_type_id.items()}

chr21_data = pd.read_csv("models/DEC18_RESULTS/eic_paper/benchmark.csv")

chr21_data['cell_code'] = chr21_data['file'].str[:3]
chr21_data['assay_code'] = chr21_data['file'].str[3:6]
chr21_data['cell'] = chr21_data['cell_code'].map(cell_type_id)
chr21_data['assay'] = chr21_data['assay_code'].map(assay_id)
chr21_data.drop(columns=['cell_code', 'assay_code'], inplace=True)

print("\nChromosome 21 Benchmark Data with cell types and assays:")
print(chr21_data.head())

print("\nHead of df1:")
print(df1.head())

# Rename columns in chr21_data to match df1's format
chr21_data = chr21_data.rename(columns={
    'pearson': 'gwcorr',
    'spearman': 'gwspear',
    'model': 'team'
})

# Combine the dataframes
merged_df = pd.concat([chr21_data, df1], ignore_index=True)

# Optional: Sort by assay and team for better organization
merged_df = merged_df.sort_values(['assay', 'team']).reset_index(drop=True)

print("\nMerged DataFrame:")
print(merged_df.head())

comparison_boxplot(merged_df, name="test_chr21")

# exit()
# def get_binned_vals(bigwig_file, chr, resolution=25):
#     with pyBigWig.open(bigwig_file) as bw:
#         if chr not in bw.chroms():
#             raise ValueError(f"{chr} not found in the BigWig file.")
#         chr_length = bw.chroms()[chr]
#         start, end = 0, chr_length
#         vals = np.array(bw.values(chr, start, end, numpy=True))
#         vals = np.nan_to_num(vals, nan=0.0)
#         vals = vals[:end - (end % resolution)]
#         vals = vals.reshape(-1, resolution)
#         bin_means = np.mean(vals, axis=1)
#         return bin_means

# chr = "chr21"
# true_data = "/project/compbio-lab/EIC/blind_data/"
# eic_data = [
#     "/project/compbio-lab/EIC/blind_Hongyang_Li_and_Yuanfang_Guan_v1/",
#     "/project/compbio-lab/EIC/blind_lavawizard/",
#     "/project/compbio-lab/EIC/blind_avg/",
#     "/project/compbio-lab/EIC/blind_guacamole/",
#     "/project/compbio-lab/EIC/blind_avocado/",
#     "/project/compbio-lab/EIC/blind_imp/",
# ]

# results = []

# for bw in os.listdir(true_data):
#     if ".bigwig" in bw:
#         true_bw = true_data + bw
#         true_chr21 = get_binned_vals(true_bw, chr, resolution=25)

#         for eic in eic_data:
#             eic_bw = eic + bw
#             eic_chr21 = get_binned_vals(eic_bw, chr, resolution=25)

#             mse = mean_squared_error(true_chr21, eic_chr21)
#             pearson_corr, _ = pearsonr(true_chr21, eic_chr21)
#             spearman_corr, _ = spearmanr(true_chr21, eic_chr21)

#             metrics = {
#                 'file': bw,
#                 'model': eic_bw.split("/")[-2].replace("blind_", ""),
#                 'mse': mse,
#                 'pearson': pearson_corr,
#                 'spearman': spearman_corr
#             }
#             results.append(metrics)
#             print(metrics)

# # Convert results to a DataFrame
# results_df = pd.DataFrame(results)

# # Save the DataFrame to a CSV file
# results_df.to_csv('benchmark.csv', index=False)
