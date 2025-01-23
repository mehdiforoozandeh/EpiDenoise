import pyBigWig, os
import pandas as pd 
import numpy as np

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
    "/project/compbio-lab/EIC/blind_imp",
]

for bw in os.listdir(true_data):
    if ".bigwig" in bw:
        true_bw = true_data + bw
        true_chr21 = get_binned_vals(true_bw, chr, resolution=25)

        for eic in eic_data:
            eic_bw = eic + bw
            eic_chr21 = get_binned_vals(eic_bw, chr, resolution=25)

            print(len(true_chr21), len(eic_chr21))