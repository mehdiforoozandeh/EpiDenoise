import pyBigWig, os
import pandas as pd 
import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr


def get_binned_vals(bigwig_file, chr, resolution=25):
    with pyBigWig.open(bigwig_file) as bw:
        if chr not in bw.chroms():
            raise ValueError(f"{chr} not found in the BigWig file.")
        chr_length = bw.chroms()[chr]
        start, end = 0, chr_length
        vals = np.array(bw.values(chr, start, end, numpy=True))
        vals = np.nan_to_num(vals, nan=0.0)
        vals = vals[:end - (end % resolution)]
        vals = vals.reshape(-1, resolution)
        bin_means = np.mean(vals, axis=1)
        return bin_means

chr = "chr21"
true_data = "/project/compbio-lab/EIC/blind_data/"
eic_data = [
    "/project/compbio-lab/EIC/blind_Hongyang_Li_and_Yuanfang_Guan_v1/",
    "/project/compbio-lab/EIC/blind_lavawizard/",
    "/project/compbio-lab/EIC/blind_avg/",
    "/project/compbio-lab/EIC/blind_guacamole/",
    "/project/compbio-lab/EIC/blind_avocado/",
    "/project/compbio-lab/EIC/blind_imp/",
]

results = []

for bw in os.listdir(true_data):
    if ".bigwig" in bw:
        true_bw = true_data + bw
        true_chr21 = get_binned_vals(true_bw, chr, resolution=25)

        for eic in eic_data:
            eic_bw = eic + bw
            eic_chr21 = get_binned_vals(eic_bw, chr, resolution=25)

            mse = mean_squared_error(true_chr21, eic_chr21)
            pearson_corr, _ = pearsonr(true_chr21, eic_chr21)
            spearman_corr, _ = spearmanr(true_chr21, eic_chr21)

            metrics = {
                'file': bw,
                'model': eic_bw.split("/")[-2].replace("blind_", ""),
                'mse': mse,
                'pearson': pearson_corr,
                'spearman': spearman_corr
            }
            results.append(metrics)
            print(metrics)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
results_df.to_csv('benchmark.csv', index=False)
