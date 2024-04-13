import os, pyBigWig, pybedtools, random, datetime, gzip, pickle
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import multiprocessing as mp
import torch

random.seed(73)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reshape_tensor(tensor, context_length_factor):
    # Get the original size of the tensor
    samples, seq_length, features = tensor.size()

    # Calculate the new sequence length and number of samples
    new_seq_length = int(seq_length * context_length_factor)
    new_samples = int(samples / context_length_factor)

    # Check if the new sequence length is valid
    if seq_length % new_seq_length != 0:
        raise ValueError("The context_length_factor does not evenly divide the sequence length")

    # Reshape the tensor
    reshaped_tensor = tensor.view(new_samples, new_seq_length, features)

    return reshaped_tensor

def mask_data(data, mask_value=-1, chunk=False, n_chunks=1, mask_percentage=0.15): # used for epidenoise 1.0
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    seq_len = data.size(1)

    if chunk:
        # Calculate the size of each chunk
        chunk_size = int(mask_percentage * seq_len / n_chunks)
    else: 
        chunk_size = 1
        n_chunks =  int(mask_percentage * seq_len)

    # Initialize an empty list to store the start indices
    start_indices = []
    while len(start_indices) < n_chunks:
        # Generate a random start index
        start = torch.randint(0, seq_len - chunk_size, (1,))
        # Check if the chunk overlaps with any existing chunks
        if not any(start <= idx + chunk_size and start + chunk_size >= idx for idx in start_indices):
            # If not, add the start index to the list
            start_indices.append(start.item())

    # Loop over the start indices
    for start in start_indices:
        # Calculate the end index for the current chunk
        end = start + chunk_size
        # Set the mask values for the current chunk to True
        mask[:, start:end, :] = True

    # Create a copy of the data tensor
    masked_data = data.clone()
    # Set the masked data values to the mask_value
    masked_data[mask] = mask_value
    # Return the masked data and the mask

    return masked_data, mask

def mask_data15(data, mask_value=-1, chunk=False, n_chunks=1, mask_percentage=0.15): # used for epidenoise 1.5
    """
    in this version, we added special tokens and made sure not to mask them
    similar to BERT, using 3 different maskings:
        1. mask
        2. replace with random data
        3. do nothing
    """
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    seq_len = data.size(1)
    seglength = (seq_len - 3)/2

    cls_sep_indices = [0, seglength+1, 2*seglength + 2]
    
    if chunk:
        # Calculate the size of each chunk
        chunk_size = int(mask_percentage * seq_len / n_chunks)
    else: 
        chunk_size = 1
        n_chunks =  int(mask_percentage * seq_len)

    # Initialize an empty list to store the start indices
    start_indices = []
    while len(start_indices) < n_chunks:
        # Generate a random start index
        start = torch.randint(0, seq_len - chunk_size, (1,))
        # Check if the chunk overlaps with any existing chunks
        if not any(start <= idx + chunk_size and start + chunk_size >= idx for idx in start_indices + cls_sep_indices):
            # If not, add the start index to the list
            start_indices.append(start.item())

    # Create a copy of the data tensor
    masked_data = data.clone()

    # Loop over the start indices
    for start in start_indices:
        # Calculate the end index for the current chunk
        end = start + chunk_size
        # Set the mask values for the current chunk to True
        mask[:, start:end, :] = True

        # For each position in the chunk, decide how to mask it
        for pos in range(start, end):
            rand_num = random.random()
            if rand_num < 0.8:
                # 80% of the time, replace with mask_value
                masked_data[:, pos, :] = mask_value
            elif rand_num < 0.9:
                # 10% of the time, replace with a random value in the range of the data
                data_min = 0
                data_max = torch.max(data)
                random_value = data_min + torch.rand(1) * (data_max - data_min)
                masked_data[:, pos, :] = random_value

    # Return the masked data and the mask
    return masked_data, mask

def mask_data16(data, available_features, mask_value=-1, chunk_size=6, mask_percentage=0.15): # used for epidenoise 1.6 and 1.7
    """
    dimensions of the data: (batch_size, context_length, features)
    in this version, we make the following changes
    find available features -> for unavailable features, are corresponding values are -1. 
    num_all_signals = context * num_available_features
    num_mask_start = (num_all_signals * mask_percentage) / chunk_size
    randomly select mask_start coordinates 
        length: axis2 (start + chunk_size -> no overlap with special tokens)
        feature: random.choice(available_features)
    """
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    if mask_percentage == 0:
        return data, mask

    seq_len = data.size(1)
    seglength = (seq_len - 3)/2

    special_tokens = [0, seglength+1, (2*seglength)+2]

    # Calculate total number of signals and number of chunks to be masked
    num_all_signals = data.size(1) * len(available_features)
    num_mask_start = int((num_all_signals * mask_percentage) / chunk_size)

    # Loop over the number of chunks to be masked
    for _ in range(num_mask_start):
        while True:
            # Randomly select start coordinates for the chunk
            length_start = torch.randint(0, seq_len - chunk_size, (1,))
            feature_start = available_features[torch.randint(0, len(available_features), (1,))]

            # Check if the chunk overlaps with any special tokens
            if not any(length_start <= idx < length_start+chunk_size for idx in special_tokens):
                break

        # Apply the masking to the selected chunk
        mask[:, length_start:length_start+chunk_size, feature_start] = True
        data[mask] = mask_value

    return data, mask

def mask_data18(data, available_features, mask_value=-1, mask_percentage=0.15):
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)

    if len(available_features) == 1:
        mask_percentage = 0

    if mask_percentage == 0:
        return data, mask

    seq_len = data.size(1)
    num_mask_features = int(len(available_features) * mask_percentage)
    
    if num_mask_features == 0:
        num_mask_features += 1

    selected_indices = []
    while len(selected_indices) < num_mask_features:
        randomF = random.choice(available_features)
        if randomF not in selected_indices:
            selected_indices.append(randomF)

    # Loop over the selected indices
    for mask_f in selected_indices:

        # Apply the masking to the selected chunk
        mask[:, :, mask_f] = True
        
    data[mask] = mask_value

    return data, mask
 
class DataMasker:
    def __init__(self, mask_value, mask_percentage, chunk_size=6):
        self.mask_value = mask_value
        self.mask_percentage = mask_percentage
        self.chunk_size = chunk_size

    def mask_chunks(self, data):
        data = data.clone()
        N, L, F = data.size()
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        num_masks = int((L * self.mask_percentage) / self.chunk_size)
        for _ in range(num_masks):
            start = random.randint(0, L - self.chunk_size)
            mask_indicator[:, start:start+self.chunk_size, :] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mask_features(self, data, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)

        if len(available_features) == 1:
            return data, mask_indicator
        else:
            num_features_to_mask = int(len(self.available_features) * self.mask_percentage) + 1

        features_to_mask = random.sample(self.available_features, num_features_to_mask)

        for feature in features_to_mask:
            mask_indicator[:, :, feature] = True

        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mask_chunk_features(self, data, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        num_all_signals = L * len(self.available_features)
        num_masks = int((num_all_signals * self.mask_percentage) / self.chunk_size)
        for _ in range(num_masks):
            length_start = random.randint(0, L - self.chunk_size)
            feature_start = random.choice(self.available_features)
            mask_indicator[:, length_start:length_start+self.chunk_size, feature_start] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mid_slice_mask(self, data, available_features):
        data = data.clone()
        N, L, F = data.size()
        slice_length = int(L * self.mask_percentage)
        start = L // 2 - slice_length // 2
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        mask_indicator[:, start:start+slice_length, available_features] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mid_slice_mask_features(self, data, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()
        slice_length = int(L * self.mask_percentage)
        num_features_to_mask = int(len(self.available_features) * self.mask_percentage)
        features_to_mask = random.sample(self.available_features, num_features_to_mask)
        start = L // 2 - slice_length // 2
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        for feature in features_to_mask:
            mask_indicator[:, start:start+slice_length, feature] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mid_slice_focused_full_feature_mask(self, data, missing_mask_value, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()

        num_features_to_mask = int(len(self.available_features) * self.mask_percentage)
        if num_features_to_mask == 0:
            features_to_mask = available_features
        else:
            features_to_mask = random.sample(self.available_features, num_features_to_mask)
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)

        # Mask features completely
        for feature in features_to_mask:
            data[:, :, feature] = missing_mask_value

        # Mark only the middle part of those masked features
        slice_length = int(L * self.mask_percentage)
        start = L // 2 - slice_length // 2
        mask_indicator[:, start:start+slice_length, features_to_mask] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

def sequence_pad(data, max_length, pad_value=-1):
    # Get the original dimensions of the data
    original_size = data.size()
    
    # Create a tensor filled with the pad value with the desired size
    padded_data = torch.full((original_size[0], max_length, original_size[2]), pad_value)
    
    # Copy the original data into the padded data tensor
    padded_data[:, :original_size[1], :] = data
    
    # Create a boolean mask indicating whether each value is padded or not
    pad_mask = padded_data == pad_value
    
    return padded_data, pad_mask

def get_bin_value(input_dict):
    if input_dict["bw_obj"] == False:
        input_dict["bw"] = pyBigWig.open(input_dict["bw"])

    bw, chr, start, end, resolution = input_dict["bw"], input_dict["chr"], input_dict["start"], input_dict["end"], input_dict["resolution"]
    bin_value = bw.stats(chr, start, end, type="mean", nBins=(end - start) // resolution)

    if input_dict["bw_obj"] == False:
        bw.close()

    return bin_value

def get_bin_value_dict(input_dict):
    if input_dict["bw_obj"] == False:
        input_dict["bw"] = pyBigWig.open(input_dict["bw"])

    bw, chr, start, end, resolution = input_dict["bw"], input_dict["chr"], input_dict["start"], input_dict["end"], input_dict["resolution"]
    bin_value = bw.stats(chr, start, end, type="mean", nBins=(end - start) // resolution)

    input_dict["signals"] = bin_value

    if input_dict["bw_obj"] == False:
        bw.close()
        del input_dict["bw"]
        

    return input_dict

def add_noise(data, noise_factor):
    noise = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=data.shape))
    noisy_data = data + noise_factor * noise
    noisy_data = torch.clamp(noisy_data, min=0)
    return noisy_data.to(torch.float32)

def peak_overlap(y_true, y_pred, p=0.01):
    top_p_percent = int(p * len(y_true))
    
    # Get the indices of the top p percent of the observed (true) values
    top_p_percent_obs_i = np.argsort(y_true)[-top_p_percent:]
    
    # Get the indices of the top p percent of the predicted values
    top_p_percent_pred_i = np.argsort(y_pred)[-top_p_percent:]

    # Calculate the overlap
    overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))

    # Calculate the percentage of overlap
    overlap_percent = overlap / top_p_percent 

    return overlap_percent
        
class COORD(object):
    def __init__(self, Meuleman_file="data/Meuleman.tsv", cCRE_file="data/GRCh38-cCREs.bed", 
                resolution=1000, chr_sizes_file="data/hg38.chrom.sizes", outdir="data/"):
        
        self.resolution = resolution
        self.cCRE_file = cCRE_file
        self.Meuleman_file = Meuleman_file
        self.outdir = outdir
        self.chr_sizes_file = chr_sizes_file    

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        main_chrs.remove("chr21") # reserved for validation
        self.chr_sizes = {}

        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)    

    def init_bins(self):
        if os.path.exists(f"{self.outdir}/bins_{self.resolution}bp.csv"):
            self.bins = pd.read_csv(f"{self.outdir}/bins_{self.resolution}bp.csv").drop("Unnamed: 0", axis=1)
        else:

            # Create bins
            self.bins = []
            for chr, size in self.chr_sizes.items():
                start_coords = range(0, size, self.resolution)
                end_coords = range(self.resolution, size + self.resolution, self.resolution)
                self.bins.extend([[chr, start, end] for start, end in zip(start_coords, end_coords)][:-1])

            self.bins = pd.DataFrame(self.bins, columns =["chrom", "start", "end"])
            self.bins = self.bins.sort_values(["chrom", "start"]).reset_index(drop=True)
        self.bins.to_csv(f"{self.outdir}/bins_{self.resolution}bp.csv")

    def get_foreground(self):
        if os.path.exists(f'{self.outdir}/foreground_nobin.csv'):
            self.foreground = pd.read_csv(f'{self.outdir}/foreground_nobin.csv').drop("Unnamed: 0", axis=1)
        else:
            ccre = pybedtools.BedTool(self.cCRE_file)
            if self.Meuleman_file == "_":
                self.foreground = ccre.to_dataframe()

            else:
                Meuleman = pd.read_csv(self.Meuleman_file, sep="\t")
                Meuleman.columns = ["chr", "start", "end", "identifier", "mean_signal", "numsamples", "summit", "core_start", "core_end", "component"]
                Meuleman = pybedtools.BedTool.from_dataframe(Meuleman)

                # get the union of ccre and Meuleman
                self.foreground = ccre.cat(Meuleman, postmerge=False)
                self.foreground = self.foreground.to_dataframe()

            self.foreground = self.foreground[["chrom", "start", "end"]]
            self.foreground = self.foreground.sort_values(["chrom", "start"]).reset_index(drop=True)
            self.foreground.to_csv(f'{self.outdir}/foreground_nobin.csv')

    def bin_fg_bg(self):
        self.bins = pybedtools.BedTool.from_dataframe(self.bins)
        self.foreground = pybedtools.BedTool.from_dataframe(self.foreground)

        if os.path.exists(f"{self.outdir}/foreground_bins_{self.resolution}bp.csv") == False:
            # Get the subset of bins that overlap with the foreground
            self.fg_bins = self.bins.intersect(self.foreground, u=True)
            self.fg_bins = self.fg_bins.to_dataframe()
            self.fg_bins.to_csv(f"{self.outdir}/foreground_bins_{self.resolution}bp.csv")
        else:
            self.fg_bins = pd.read_csv(f"{self.outdir}/foreground_bins_{self.resolution}bp.csv").drop("Unnamed: 0", axis=1)

        if os.path.exists(f"{self.outdir}/background_bins_{self.resolution}bp.csv") == False:
            # Get the subset of bins that do not overlap with the foreground
            self.bg_bins = self.bins.intersect(self.foreground, v=True)
            self.bg_bins = self.bg_bins.to_dataframe()
            self.bg_bins.to_csv(f"{self.outdir}/background_bins_{self.resolution}bp.csv")
        else:
            self.bg_bins = pd.read_csv(f"{self.outdir}/background_bins_{self.resolution}bp.csv").drop("Unnamed: 0", axis=1)

        print(f"number of foreground bins: {len(self.fg_bins)} | number of background bins: {len(self.bg_bins)}")

    # def random_genome_subset(self, max_seq_len, stratified=True)

class PROCESS_EIC_DATA(object):
    def __init__(self, path, max_len=8000, resolution=25, stratified=False):
        self.path = path
        self.stratified = stratified
        self.resolution = resolution
        self.max_len = max_len * self.resolution #converts max_len from #bins to #bp
        self.util = COORD(resolution=self.resolution, Meuleman_file="_", outdir=self.path)
        self.genomesize = sum(list(self.util.chr_sizes.values()))
        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]

        self.biosamples = {}
        for f in os.listdir(self.path):
            if ".bigwig" in f: 
                if f[:3] not in self.biosamples.keys():
                    self.biosamples[f[:3]] = {}
                    
                self.biosamples[f[:3]][f[3:6]] = pyBigWig.open(self.path + "/" + f)

    def pkl_generate_m_samples(self, m, multi_p=True, n_p=20): # m per biosample           
        if self.stratified:
            self.util.get_foreground()
            df = self.util.foreground
            df = df[df["chrom"].isin(self.util.chr_sizes.keys())]
            m_regions = []
            used_regions = {chr: [] for chr in df['chrom'].unique()}

            # Sort the DataFrame by chromosome and start position
            df = df.sort_values(['chrom', 'start'])

            # Select m/2 regions from the DataFrame
            for _ in range(m // 2):
                while True:
                    # Select a random row from the DataFrame
                    row = df.sample(1).iloc[0]

                    # Generate a start position that is divisible by self.resolution and within the region
                    rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len


                    # Check if the region overlaps with any existing region in the same chromosome
                    if rand_start >= 0 and rand_end <= self.util.chr_sizes[row['chrom']]:
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                            m_regions.append([row['chrom'], rand_start, rand_end])
                            used_regions[row['chrom']].append((rand_start, rand_end))
                            break
                        
            # Select m/2 regions that are not necessarily in the DataFrame 
            for chr, size in self.util.chr_sizes.items():
                m_c = int((m // 2) * (size / self.genomesize))  # Calculate the number of instances from each chromosome proportional to its size
                for _ in range(m_c):
                    while True:
                        # Generate a random start position that is divisible by self.resolution
                        rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                        rand_end = rand_start + self.max_len

                        # Check if the region overlaps with any existing region
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                            m_regions.append([chr, rand_start, rand_end])
                            used_regions[chr].append((rand_start, rand_end))
                            break

        else:
            m_regions = []
            used_regions = {chr: [] for chr in self.util.chr_sizes.keys()}

            for chr, size in self.util.chr_sizes.items():
                m_c = int(m * (size / self.genomesize))

                for _ in range(m_c):
                    while True:
                        # Generate a random start position that is divisible by self.resolution
                        rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                        rand_end = rand_start + self.max_len

                        # Check if the region overlaps with any existing region in the same chromosome
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                            m_regions.append([chr, rand_start, rand_end])
                            used_regions[chr].append((rand_start, rand_end))
                            break

        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in self.biosamples[bios].keys():
                bios_data[assay] = []

                bw = self.biosamples[bios][assay]
                bw_query_dicts = []
                for i in range(len(m_regions)):
                    r = m_regions[i]
                    bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})

                if multi_p:
                    with mp.Pool(n_p) as p:
                        m_signals = p.map(get_bin_value, bw_query_dicts)
                    
                    for i in range(len(m_signals)):
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], m_signals[i]))
                else:
                    for i in range(len(bw_query_dicts)):
                        signals = get_bin_value(bw_query_dicts[i])
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], signals))
                    
            file_path = f"{self.path}/{bios}_m{m}_{self.resolution}bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")

    def generate_m_samples(self, m, n_datasets=50, multi_p=True, n_p=10):
        if self.stratified:
            self.util.get_foreground()
            df = self.util.foreground
            df = df[df["chrom"].isin(self.util.chr_sizes.keys())]
            m_regions = []
            used_regions = {chr: [] for chr in df['chrom'].unique()}

            # Sort the DataFrame by chromosome and start position
            df = df.sort_values(['chrom', 'start'])

            # Select m/2 regions from the DataFrame
            while len(m_regions) < (m // 2):
                while True:
                    # Select a random row from the DataFrame
                    row = df.sample(1).iloc[0]

                    # Generate a start position that is divisible by self.resolution and within the region
                    rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region in the same chromosome
                    if rand_start >= 0 and rand_end <= self.util.chr_sizes[row['chrom']]:
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                            m_regions.append([row['chrom'], rand_start, rand_end])
                            used_regions[row['chrom']].append((rand_start, rand_end))
                            break
                        
            # Select m/2 regions that are not necessarily in the DataFrame 
            for chr, size in self.util.chr_sizes.items():
                m_c = int((m // 2) * (size / self.genomesize))  # Calculate the number of instances from each chromosome proportional to its size
                mii = 0
                while mii < m_c:
                    # Generate a random start position that is divisible by self.resolution
                    rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                        m_regions.append([chr, rand_start, rand_end])
                        used_regions[chr].append((rand_start, rand_end))
                        mii += 1 
                        break

        else:
            m_regions = []
            used_regions = {chr: [] for chr in self.util.chr_sizes.keys()}

            for chr, size in self.util.chr_sizes.items():
                m_c = int(m * (size / self.genomesize))
                mii = 0

                while mii < m_c:
                    # Generate a random start position that is divisible by self.resolution
                    rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region in the same chromosome
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                        m_regions.append([chr, rand_start, rand_end])
                        used_regions[chr].append((rand_start, rand_end))
                        mii += 1 

        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        ds_number = 0  
        print("m2:   ", len(m_regions))
        samples_per_ds = len(m_regions) // n_datasets
        for ds_i in range(0, len(m_regions), samples_per_ds):
            ds_number += 1

            ds_i_regions = m_regions[ds_i : (ds_i + samples_per_ds)]
            ds_i_regions.sort(key=lambda x: x[1]) # sorted based on start coord
            
            all_samples_tensor = []

            for bios in self.biosamples.keys():
                print("     ct:   ", bios)
                bios_data = {}

                for assay in self.all_assays:
                    bios_data[assay] = []

                    if assay in self.biosamples[bios].keys(): # if available
                        print("         assay:   ", assay)
                        bw = self.biosamples[bios][assay]
                        bw_query_dicts = []

                        for r in ds_i_regions:
                            bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})
                        
                        if multi_p:
                            with mp.Pool(n_p) as p:
                                outs = p.map(get_bin_value_dict, bw_query_dicts)
                        else:
                            outs = []
                            for ii in range(len(bw_query_dicts)):
                                outs.append(get_bin_value_dict(bw_query_dicts[ii]))

                        outs.sort(key=lambda x: x['start']) # assert is sorted based on start coord
                        m_signals = [o["signals"] for o in outs]
                        
                        for sample in m_signals:
                            bios_data[assay].append(sample)

                    else: # if missing
                        for r in ds_i_regions:
                            bios_data[assay].append([-1 for _ in range(self.max_len // self.resolution)])
                
                # Convert bios_data to a numpy array
                bios_data_array = np.array([bios_data[assay] for assay in self.all_assays], dtype=np.float32)

                # Add bios_data_array to all_samples
                all_samples_tensor.append(bios_data_array)

            # Convert all_samples to a numpy array
            all_samples_tensor = np.array(all_samples_tensor)

            # Convert all_samples_array to a PyTorch tensor
            all_samples_tensor = torch.from_numpy(all_samples_tensor)

            # Ensure the tensor is of type float
            all_samples_tensor = all_samples_tensor.float()

            all_samples_tensor = torch.permute(all_samples_tensor, (2, 0, 3, 1))
            # Get the shape of the current tensor
            shape = all_samples_tensor.shape

            # Calculate the new dimensions
            new_shape = [shape[0]*shape[1]] + list(shape[2:])

            # Reshape the tensor
            all_samples_tensor = all_samples_tensor.reshape(new_shape)
            
            file_path = f"{self.path}/mixed_dataset{ds_number}_{m//n_datasets}samples_{self.resolution}bp.pt"
            torch.save(all_samples_tensor, file_path)
            print(f"saved DS # {ds_number}, with shape {all_samples_tensor.shape}")

    def load_m_regions(self, file_path):
        # Open the gzip file
        with gzip.open(file_path, 'rb') as f:
            # Load the data using pickle
            bios_data = pickle.load(f)

        # Initialize an empty list to store the m_regions
        m_regions = []

        # Iterate over each biosample and assay
        for sample in bios_data[list(bios_data.keys())[0]]:

            # Append the regions to the m_regions list
            if sample[0] not in m_regions:
                m_regions.append(sample[0])
            
        return m_regions
    
    def generate_m_samples_from_predefined_regions(self, m_regions, multi_p=True, n_p=100):
        m = len(m_regions)
        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in self.biosamples[bios].keys():
                bios_data[assay] = []

                bw = self.biosamples[bios][assay]
                bw_query_dicts = []
                for i in range(len(m_regions)):
                    r = m_regions[i]
                    bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})

                if multi_p:
                    with mp.Pool(n_p) as p:
                        m_signals = p.map(get_bin_value, bw_query_dicts)
                    
                    for i in range(len(m_signals)):
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], m_signals[i]))
                else:
                    for i in range(len(bw_query_dicts)):
                        signals = get_bin_value(bw_query_dicts[i])
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], signals))
                    
            file_path = f"{self.path}/{bios}_m{m}_{self.resolution}bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")
                
    def generate_wg_samples(self):
        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in biosamples[bios].keys():
                bios_data[assay] = {}

                bw = biosamples[bios][assay]
                for chr, size in self.util.chr_sizes.items():
                    signals = get_bin_value(bw, chr, 0, size, self.resolution)
                    bios_data[assay][chr] = signals
            
            file_path = f"{self.path}/{bios}_WG_25bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")

if __name__ == "__main__":
    # solar_path = "/project/compbio-lab/EIC/"
    # sample = "/project/compbio-lab/EIC/training_data/C01_m2000_25bp.pkl.gz"
    # traineic = PROCESS_EIC_DATA(solar_path+"training_data/", stratified=True)
    # m_regions = traineic.load_m_regions(sample)

    # valeic = PROCESS_EIC_DATA(solar_path+"validation_data/", stratified=True)
    # blindeic = PROCESS_EIC_DATA(solar_path+"blind_data/", stratified=True)

    # valeic.generate_m_samples_from_predefined_regions(m_regions=m_regions, multi_p=True, n_p=20)
    # blindeic.generate_m_samples_from_predefined_regions(m_regions=m_regions, multi_p=True, n_p=20)

    # exit()
    # solar_path = "/project/compbio-lab/EIC/training_data/"
    # # solar_path = "data/test/"
    # eic = PROCESS_EIC_DATA(solar_path, stratified=True)
    # t0 = datetime.datetime.now()
    # # m_regions = eic.load_m_regions("data/test/C02_m500_25bp.pkl.gz")
    # # print(len(m_regions))
    # eic.generate_m_samples(m=2000, multi_p=True)
    # t1 = datetime.datetime.now()
    # # print("generated training datasets in :", t1-t0)
    # exit()



    solar_path = "/project/compbio-lab/EIC/training_data/"
    eic = PROCESS_EIC_DATA(solar_path, stratified=False, resolution=25, max_len=8000)
    t0 = datetime.datetime.now()
    eic.generate_m_samples(m=6000, n_datasets=150, multi_p=True, n_p=2)
    t1 = datetime.datetime.now()
    print("generated training datasets in :", t1-t0)

    # solar_path = "/project/compbio-lab/EIC/training_data/"
    # eic = PROCESS_EIC_DATA(solar_path, stratified=True, resolution=128, max_len=8000)
    # t0 = datetime.datetime.now()
    # eic.generate_m_samples(m=2000, n_datasets=30, multi_p=True, n_p=10)
    # t1 = datetime.datetime.now()
    # print("generated training datasets in :", t1-t0)



    
    # solar_path = "/project/compbio-lab/EIC/validation_data/"
    # eic = PROCESS_EIC_DATA(solar_path, stratified=True)
    # t0 = datetime.datetime.now()
    # eic.generate_m_samples(m=2000, multi_p=True)
    # t1 = datetime.datetime.now()
    # print("generated validation datasets in :", t1-t0)

    # solar_path = "/project/compbio-lab/EIC/blind_data/"
    # eic = PROCESS_EIC_DATA(solar_path, stratified=True)
    # t0 = datetime.datetime.now()
    # eic.generate_m_samples(m=2000, multi_p=True)
    # t1 = datetime.datetime.now()
    # print("generated blind datasets in :", t1-t0)
    exit()
    # c = COORD(resolution=25, Meuleman_file="_")
    # c.init_bins()
    # c.get_foreground()
    # c.bin_fg_bg()

    # c.fg_bins["len"]= c.fg_bins["end"] - c.fg_bins["start"] 
    # c.bg_bins["len"]= c.bg_bins["end"] - c.bg_bins["start"] 

    # print(c.fg_bins["len"].mean())
    # print(c.bg_bins["len"].mean())

    # plt.hist(c.foreground["len"], bins=100)

    # plt.show()

    

    # exit()

    # b = BIOSAMPLE("data/test", "C02", chr_sizes_file="data/hg38.chrom.sizes", resolution=25)
    # print(b.tracks)