import matplotlib.pyplot as plt
import seaborn as sns

mark_dict = {
    "M01": "ATAC-seq",
    "M02": "DNase-seq",
    "M03": "H2AFZ",
    "M04": "H2AK5ac",
    "M05": "H2AK9ac",
    "M06": "H2BK120ac",
    "M07": "H2BK12ac",
    "M08": "H2BK15ac",
    "M09": "H2BK20ac",
    "M10": "H2BK5ac",
    "M11": "H3F3A",
    "M12": "H3K14ac",
    "M13": "H3K18ac",
    "M14": "H3K23ac",
    "M15": "H3K23me2",
    "M16": "H3K27ac",
    "M17": "H3K27me3",
    "M18": "H3K36me3",
    "M19": "H3K4ac",
    "M20": "H3K4me1",
    "M21": "H3K4me2",
    "M22": "H3K4me3",
    "M23": "H3K56ac",
    "M24": "H3K79me1",
    "M25": "H3K79me2",
    "M26": "H3K9ac",
    "M27": "H3K9me1",
    "M28": "H3K9me2",
    "M29": "H3K9me3",
    "M30": "H3T11ph",
    "M31": "H4K12ac",
    "M32": "H4K20me1",
    "M33": "H4K5ac",
    "M34": "H4K8ac",
    "M35": "H4K91ac",
}


import numpy as np

def create_scatter(df, metric):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Scatter plots for Imputed and Denoised')
    # Plot for Imputed
    imputed_df = df[df['comparison'] == 'imputed']
    x_imputed = imputed_df['available train assays']
    y_imputed = imputed_df[metric]
    if "MSE" in metric:
        y_imputed = np.log(y_imputed)
    sns.regplot(x=x_imputed, y=y_imputed, scatter=True, line_kws={"color": "red"}, scatter_kws={"color": "red"}, ax=ax, label='Imputed')
    # Plot for Denoised
    denoised_df = df[df['comparison'] == 'denoised']
    x_denoised = denoised_df['available train assays']
    y_denoised = denoised_df[metric]
    if "MSE" in metric:
        y_denoised = np.log(y_denoised)
    sns.regplot(x=x_denoised, y=y_denoised, scatter=True, line_kws={"color": "green"}, scatter_kws={"color": "green"}, ax=ax, label='Denoised')
    ax.set(xlabel='Number of Available Train Assays', ylabel='log('+metric+')' if "MSE" in metric else metric)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"regplot_{metric}.png", dpi=300)


def create_boxplots(df, metric, mark_dict):
    # Convert feature codes to actual names using mark_dict
    df = df.copy()
    df['feature'] = df['feature'].map(mark_dict)
    # Sort the dataframe by 'feature'
    df.sort_values('feature', inplace=True)
    fig, axs = plt.subplots(2, figsize=(10, 6))
    fig.suptitle('Boxplots for Imputed and Denoised')
    # Boxplot for Imputed
    imputed_df = df[df['comparison'] == 'imputed']
    if "MSE" in metric:
        imputed_df[metric] = np.log(imputed_df[metric])
    sns.boxplot(x='feature', y=metric, data=imputed_df, ax=axs[0], color="grey")
    axs[0].set_title('Imputed')
    axs[0].set(xlabel='Assay', ylabel='log('+metric+')' if "MSE" in metric else metric)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)  # Rotate x-axis labels
    # Boxplot for Denoised
    denoised_df = df[df['comparison'] == 'denoised']
    if "MSE" in metric:
        denoised_df[metric] = np.log(denoised_df[metric])
    sns.boxplot(x='feature', y=metric, data=denoised_df, ax=axs[1], color="grey")
    axs[1].set_title('Denoised')
    axs[1].set(xlabel='Assay', ylabel='log('+metric+')' if "MSE" in metric else metric)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)  # Rotate x-axis labels
    plt.tight_layout()
    plt.savefig(f"boxplot_{metric}.png", dpi=300)



def create_boxplot_context(A, B, metric):
    # Prepare data
    A['dataset'] = 'context length 10kb'
    B['dataset'] = 'context length 20kb'
    df = pd.concat([A, B])
    # Create boxplot
    plt.figure(figsize=(10, 6))
    box_plot = sns.boxplot(x='comparison', y=metric, hue='dataset', data=df, palette=sns.color_palette("husl", 2))
    # Set labels
    box_plot.set_ylabel('log('+metric+')' if "MSE" in metric else metric)
    plt.tight_layout()
    plt.savefig(f"boxplot_context_{metric}.png", dpi=300)

