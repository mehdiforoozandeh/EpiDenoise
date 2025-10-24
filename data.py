import pandas as pd
import numpy as np
import torch
import os
import json
import random
import math
import time
import glob
from intervaltree import IntervalTree
import pysam
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import multiprocessing

import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import islice
from collections import defaultdict

def set_global_seed(seed):
    """
    Set the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ========= CANDI Data Handler =========
class CANDIDataHandler:
    def __init__(self, base_path, resolution=25, dataset_type="merged", DNA=True, 
                 bios_batchsize=8, loci_batchsize=16, dsf_list=[1, 2, 4]):
        """
        load files like navigation, aliases, split, etc.
        if any of the files don't exist, generate them
        set up the data handler

        aliases, navigations, and splits are stored in the data/ directory and are different for merged and eic datasets
        
        Args:
            base_path: Path to the dataset directory
            resolution: Genomic resolution in base pairs
            dataset_type: Type of dataset ("merged" or "eic")
            DNA: Whether to include DNA sequence data
            bios_batchsize: Number of biosamples to process in each batch (default: 8)
            loci_batchsize: Number of loci to process in each batch (default: 16)
            dsf_list: List of downsampling factors to use (default: [1, 2, 4])
        """
        self.base_path = base_path
        self.resolution = resolution
        self.dataset_type = dataset_type
        self.DNA = DNA
        self.max_thread_workers = 4
        
        # Hierarchical batching parameters
        self.bios_batchsize = bios_batchsize
        self.loci_batchsize = loci_batchsize
        self.dsf_list = dsf_list

        self.includes=[
            'ATAC-seq', 'DNase-seq', 'H2AFZ', 'H2AK5ac', 'H2AK9ac', 'H2BK120ac', 'H2BK12ac', 'H2BK15ac', 
            'H2BK20ac', 'H2BK5ac', 'H3F3A', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K23me2', 'H3K27ac', 'H3K27me3', 
            'H3K36me3', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me1', 'H3K79me2', 'H3K9ac', 
            'H3K9me1', 'H3K9me2', 'H3K9me3', 'H3T11ph', 'H4K12ac', 'H4K20me1', 'H4K5ac', 'H4K8ac', 'H4K91ac']
        
        # Defaults used by loaders below
        if dataset_type == "merged":
            self.merge_ct = True
            self.eic = False
            self.metadata_path = os.path.join("data/", "merged_metadata.csv")
            self.split_path = os.path.join('data/', f"train_va_test_split_merged.json")
        elif dataset_type == "eic":
            self.merge_ct = False
            self.eic = True
            self.metadata_path = os.path.join("data/", "eic_metadata.csv")
            self.split_path = os.path.join('data/', f"train_va_test_split_eic.json")
        
        self.chr_sizes_file = os.path.join("data/", "hg38.chrom.sizes")
        self.blacklist_file = os.path.join("data/", "hg38_blacklist_v2.bed") 
        
        self.alias_path = os.path.join(self.base_path, f"aliases.json")
        self.navigation_path = os.path.join(self.base_path, f"navigation.json")
        
        self.fasta_file = os.path.join("data/", "hg38.fa")
        self.ccre_filename = os.path.join("data/", "GRCh38-cCREs.bed")

        # DNA sequence cache: locus tuple -> one-hot tensor
        self.dna_cache = {}
        self.stat_lookup = None

        # self._load_fasta()
        self._load_blacklist()
        self._load_files()
        self._load_genomic_coords()
        
    def _load_files(self):
        if not os.path.exists(self.alias_path):
            self._make_alias()
            
        if not os.path.exists(self.navigation_path):
            self._make_navigation()
            
        self._load_alias()
        self._load_navigation()
        self._load_metadata()
        self._load_split()
    
    # ========= Dataset Info Loading =========
    def _load_alias(self):
        with open(self.alias_path, 'r') as aliasfile:
            self.aliases = json.load(aliasfile)
    
    def _load_metadata(self):
        with open(self.metadata_path, 'r') as metadatafile:
            self.metadata = pd.read_csv(metadatafile)

        self.unique_assays = self.metadata['assay_name'].unique()
        # Build stable, reproducible platform mapping (sorted), with an 'unknown' bucket at 0
        platforms = sorted([str(x) for x in self.metadata['sequencing_platform'].dropna().unique()])
        self.sequencing_platform_to_id = {"unknown": 0}
        for i, p in enumerate(platforms, start=1):
            self.sequencing_platform_to_id[p] = i
        self.id_to_sequencing_platform = {v: k for k, v in self.sequencing_platform_to_id.items()}
        self.num_sequencing_platforms = len(self.sequencing_platform_to_id)

        self.unique_sequencing_platforms = platforms
        self.unique_read_lengths = self.metadata['read_length'].unique()
        self.unique_run_types = self.metadata['run_type'].unique()
        self.unique_labs = self.metadata['lab'].unique()
        self.unique_biosample_names = self.metadata['biosample_name'].unique()

    def _load_navigation(self):
        with open(self.navigation_path, 'r') as navigationfile:
            self.navigation = json.load(navigationfile)
    
    def _load_fasta(self):
        fasta_path = os.fspath(self.fasta_file)  # ensure str/Path-like -> str
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")
        if not os.path.exists(fasta_path + ".fai"):
            pysam.faidx(fasta_path)               # build index if needed
        self.fasta = pysam.FastaFile(fasta_path)  # pass path, not file handle
    
    def _load_blacklist(self):
        """Load blacklist regions from a BED file into IntervalTrees and assign to self.blacklist."""
        blacklist = {}
        with open(self.blacklist_file, 'r') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                if chrom not in blacklist:
                    blacklist[chrom] = IntervalTree()
                blacklist[chrom].addi(start, end)
        self.blacklist = blacklist
    
    def _load_split(self):
        if not os.path.exists(self.split_path):
            self._make_split()

        with open(self.split_path, 'r') as splitfile:
            self.split_dict = json.load(splitfile)
    
    def _load_genomic_coords(self, mode="train"):
        main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        if mode == "train":
            main_chrs.remove("chr21") # reserved for test
        self.chr_sizes = {}

        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)    
        
        self.genomesize = sum(list(self.chr_sizes.values()))
        self.main_chrs = main_chrs

    # ========= Making Files =========
    def _make_alias(self):
        data_matrix = {}
        for bios in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, bios)):
                data_matrix[bios] = {}
                for exp in os.listdir(os.path.join(self.base_path, bios)):
                    exp_path = os.path.join(self.base_path, bios, exp)
                    if os.path.isdir(exp_path):
                        data_matrix[bios][exp] = 1
        
        data_matrix = pd.DataFrame(data_matrix).T
        # data_matrix = data_matrix[[exp for exp in self.includes if exp in data_matrix.columns]]

        # Alias for experiments
        experiment_counts = data_matrix.count().sort_values(ascending=False)
        # experiment_counts = experiment_counts[self.includes]
        num_experiments = len(experiment_counts)
        experiment_alias = {
            experiment: f"M{str(index+1).zfill(len(str(num_experiments)))}" for index, experiment in enumerate(
                experiment_counts.index)}

        # Avoid changing dict size during iteration by iterating over a list of keys
        for exp in list(experiment_alias.keys()):
            if exp not in self.includes:
                experiment_alias.pop(exp)

        for exp in self.includes:
            if exp not in experiment_alias:
                experiment_alias[exp] = f"M{str(len(experiment_alias)+1).zfill(len(str(num_experiments)))}"

        self.aliases = {"experiment_aliases": experiment_alias}
    
        with open(self.alias_path, 'w') as aliasfile:
            json.dump(self.aliases, aliasfile)
        
    def _make_navigation(self):
        """
        Generate navigation for the dataset
        """
        self.navigation = {}
        for bios in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, bios)):
                self.navigation[bios] = {}
                for exp in os.listdir(os.path.join(self.base_path, bios)):
                    exp_path = os.path.join(self.base_path, bios, exp)
                    if os.path.isdir(exp_path):
                        self.navigation[bios][exp] = [os.path.join(exp_path, f) for f in os.listdir(exp_path)]

        with open(self.navigation_path, 'w') as navigationfile:
            json.dump(self.navigation, navigationfile, indent=4)
        
    def _make_split(self):
        self.split_dict = {}
        
        if self.merge_ct:
            """
            Generate stratified train-test split for merged dataset
            - 70-30 split ratio
            - Biosample_names with RNA-seq go to test
            - No leakage: entire biosample_term groups stay together
            - Stratified by sequencing_platform, lab, and number of assays
            """
            import re
            from collections import defaultdict
            
            # Load metadata
            df = pd.read_csv(self.metadata_path)
            
            # Extract biosample term from biosample_name
            def extract_biosample_term(name):
                match = re.match(r'^(.+?)(?:_grp\d+_rep\d+|_nonrep)$', name)
                return match.group(1) if match else name
            
            # Compute features for each biosample_name
            biosample_info = {}
            for name in df['biosample_name'].unique():
                name_df = df[df['biosample_name'] == name]
                
                # Get mode (most common) sequencing_platform
                platform_mode = name_df['sequencing_platform'].mode()
                platform = platform_mode.iloc[0] if len(platform_mode) > 0 else name_df['sequencing_platform'].iloc[0]
                
                # Get mode lab
                lab_mode = name_df['lab'].mode()
                lab = lab_mode.iloc[0] if len(lab_mode) > 0 else name_df['lab'].iloc[0]
                
                # Get biosample term
                term = extract_biosample_term(name)
                
                # Count assays
                num_assays = len(name_df)
                
                # Check if has RNA-seq
                has_rna_seq = 'RNA-seq' in name_df['assay_name'].values
                
                biosample_info[name] = {
                    'biosample_term': term,
                    'sequencing_platform': platform,
                    'lab': lab,
                    'num_assays': num_assays,
                    'has_rna_seq': has_rna_seq
                }
            
            # Group biosample_names by biosample_term
            term_to_names = defaultdict(list)
            term_has_rna_seq = {}
            for name, info in biosample_info.items():
                term = info['biosample_term']
                term_to_names[term].append(name)
                if term not in term_has_rna_seq:
                    term_has_rna_seq[term] = False
                if info['has_rna_seq']:
                    term_has_rna_seq[term] = True
            
            # Separate RNA-seq terms (must go to test) from other terms
            rna_seq_terms = [term for term, has_rna in term_has_rna_seq.items() if has_rna]
            non_rna_seq_terms = [term for term, has_rna in term_has_rna_seq.items() if not has_rna]
            
            print(f"Total biosample_terms: {len(term_to_names)}")
            print(f"Terms with RNA-seq (going to test): {len(rna_seq_terms)}")
            print(f"Terms without RNA-seq (for stratified split): {len(non_rna_seq_terms)}")
            
            # Count how many biosample_names from RNA-seq terms
            rna_seq_biosample_count = sum(len(term_to_names[term]) for term in rna_seq_terms)
            total_biosample_count = sum(len(names) for names in term_to_names.values())
            
            print(f"Biosample_names with RNA-seq: {rna_seq_biosample_count} / {total_biosample_count}")
            
            # For non-RNA-seq terms, prepare stratification features
            term_features = []
            for term in non_rna_seq_terms:
                names = term_to_names[term]
                
                # Aggregate features across all biosample_names in this term
                platforms = [biosample_info[name]['sequencing_platform'] for name in names]
                labs = [biosample_info[name]['lab'] for name in names]
                assay_counts = [biosample_info[name]['num_assays'] for name in names]
                
                # Use mode (most common) values
                from collections import Counter
                platform_mode = Counter(platforms).most_common(1)[0][0]
                lab_mode = Counter(labs).most_common(1)[0][0]
                avg_assay_count = np.mean(assay_counts)
                
                term_features.append({
                    'term': term,
                    'platform': platform_mode,
                    'lab': lab_mode,
                    'avg_assay_count': avg_assay_count,
                    'num_biosample_names': len(names)
                })
            
            term_features_df = pd.DataFrame(term_features)
            
            # Create stratification key
            # Bin assay counts
            term_features_df['assay_bin'] = pd.cut(
                term_features_df['avg_assay_count'],
                bins=[0, 6, 9, 12, 100],
                labels=['4-6', '7-9', '10-12', '13+']
            )
            
            # Group less common platforms/labs as "Other"
            top_platforms = term_features_df['platform'].value_counts().head(6).index
            term_features_df['platform_grouped'] = term_features_df['platform'].apply(
                lambda x: x if x in top_platforms else 'Other'
            )
            
            top_labs = term_features_df['lab'].value_counts().head(6).index
            term_features_df['lab_grouped'] = term_features_df['lab'].apply(
                lambda x: x if x in top_labs else 'Other'
            )
            
            # Create combined stratification key
            term_features_df['strat_key'] = (
                term_features_df['assay_bin'].astype(str) + '_' +
                term_features_df['platform_grouped'].astype(str) + '_' +
                term_features_df['lab_grouped'].astype(str)
            )
            
            # Calculate target test size (30% of total, minus RNA-seq biosamples already in test)
            target_test_biosample_count = int(0.3 * total_biosample_count)
            remaining_test_count = target_test_biosample_count - rna_seq_biosample_count
            
            if remaining_test_count < 0:
                print(f"Warning: RNA-seq biosamples ({rna_seq_biosample_count}) exceed 30% target. Using all RNA-seq in test.")
                remaining_test_count = 0
            
            print(f"Target test set size: {target_test_biosample_count} biosample_names")
            print(f"RNA-seq already in test: {rna_seq_biosample_count}")
            print(f"Need to select {remaining_test_count} more biosample_names for test from non-RNA-seq terms")
            
            # Stratified split on non-RNA-seq terms
            from sklearn.model_selection import train_test_split
            
            # We need to split terms (not biosample_names) to avoid leakage
            # But we want approximately the right number of biosample_names
            # Use iterative stratified sampling
            
            test_terms = []
            train_terms = []
            
            # Sort by stratification key for reproducibility
            term_features_df = term_features_df.sort_values(['strat_key', 'term'])
            
            # Group by stratification key
            current_test_count = 0
            for strat_key, group in term_features_df.groupby('strat_key'):
                group_terms = group['term'].tolist()
                group_biosample_counts = group['num_biosample_names'].tolist()
                
                # Calculate how many from this group should go to test
                group_total = sum(group_biosample_counts)
                non_rna_total = total_biosample_count - rna_seq_biosample_count
                group_target_test = int(remaining_test_count * group_total / non_rna_total)
                
                # Select terms for test until we reach target
                group_test_count = 0
                for i, (term, count) in enumerate(zip(group_terms, group_biosample_counts)):
                    if group_test_count < group_target_test and current_test_count < remaining_test_count:
                        test_terms.append(term)
                        group_test_count += count
                        current_test_count += count
                    else:
                        train_terms.append(term)
            
            print(f"Selected {len(test_terms)} non-RNA-seq terms for test ({current_test_count} biosample_names)")
            print(f"Selected {len(train_terms)} non-RNA-seq terms for train")
            
            # Build split dictionary
            # RNA-seq terms -> test
            for term in rna_seq_terms:
                for name in term_to_names[term]:
                    self.split_dict[name] = "test"
            
            # Stratified test terms -> test
            for term in test_terms:
                for name in term_to_names[term]:
                    self.split_dict[name] = "test"
            
            # Remaining terms -> train
            for term in train_terms:
                for name in term_to_names[term]:
                    self.split_dict[name] = "train"
            
            # Validation
            train_count = sum(1 for v in self.split_dict.values() if v == "train")
            test_count = sum(1 for v in self.split_dict.values() if v == "test")
            
            print("\n" + "="*60)
            print("Train-Test Split Summary:")
            print("="*60)
            print(f"Total biosample_names: {len(self.split_dict)}")
            print(f"Train: {train_count} ({train_count/len(self.split_dict)*100:.1f}%)")
            print(f"Test: {test_count} ({test_count/len(self.split_dict)*100:.1f}%)")
            
            # Check for leakage
            train_terms_set = set()
            test_terms_set = set()
            for name, split in self.split_dict.items():
                term = biosample_info[name]['biosample_term']
                if split == "train":
                    train_terms_set.add(term)
                else:
                    test_terms_set.add(term)
            
            overlap = train_terms_set & test_terms_set
            if overlap:
                print(f"WARNING: Found leakage! {len(overlap)} terms appear in both train and test: {overlap}")
            else:
                print("✓ No leakage: All biosample_terms are exclusively in train or test")
            
            # RNA-seq validation
            rna_seq_in_test = sum(1 for name, split in self.split_dict.items() 
                                  if split == "test" and biosample_info[name]['has_rna_seq'])
            print(f"✓ RNA-seq biosample_names in test: {rna_seq_in_test} / {rna_seq_biosample_count}")
            
            # Assay count distribution
            train_assays = [biosample_info[name]['num_assays'] for name, split in self.split_dict.items() if split == "train"]
            test_assays = [biosample_info[name]['num_assays'] for name, split in self.split_dict.items() if split == "test"]
            
            print(f"\nAssay count distribution:")
            print(f"  Train: min={min(train_assays)}, max={max(train_assays)}, mean={np.mean(train_assays):.1f}")
            print(f"  Test:  min={min(test_assays)}, max={max(test_assays)}, mean={np.mean(test_assays):.1f}")
            print("="*60)

        elif self.eic:
            for bios in os.listdir(self.base_path):
                if os.path.isdir(os.path.join(self.base_path, bios)) and bios[0] in ["V", "T", "B"]:
                    if bios[0] == "V":
                        self.split_dict[bios] = "valid"
                    elif bios[0] == "T":
                        self.split_dict[bios] = "train"
                    elif bios[0] == "B":
                        self.split_dict[bios] = "test"

        with open(self.split_path, 'w') as splitfile:
            json.dump(self.split_dict, splitfile)

    # ========= Generating Genomic Loci =========
    def _generate_genomic_loci(self, m, context_length, strategy="random"):
        
        """
        Generate genomic loci according to the requested strategy.

        Parameters
        ----------
        m : int
            Number of regions to generate (used by 'random' and 'ccre'; ignored by 'full_chr' and 'gw').
        context_length : int
            Window length (bp) for each region.
        strategy : str
            One of {'random', 'ccre', 'full_chr', 'gw'}:
            - 'random': sample windows genome-wide proportional to chromosome sizes without overlap.
            - 'ccre'  : sample windows centered within randomly chosen cCREs (requires a BED in data/).
            - 'full_chr': tile specified chromosomes into back-to-back windows of context_length.
            - 'gw'    : like 'full_chr' but for (chr1..chr22, chrX), excluding chr21 to match prior logic.

        Notes
        -----
        Expects the following attributes on `self`:
        - self.chr_sizes: dict mapping 'chrN' -> chromosome length (int bp)
        - self.genomesize: int, sum of chromosome sizes used for proportional sampling
        - self.resolution: int, grid to snap window starts to (e.g., 128, 256, 512 bp)
        - self.is_region_allowed(chr, start, end): callable returning True if window is valid
        """

        self.context_length = context_length
        # Initialize the container for resulting regions (each is [chrom, start, end]).
        self.m_regions = []

        # Helper: checks overlap between a candidate [start, end] and a list of (start, end) tuples.
        def _overlaps(existing, s, e):
            # Overlap if any interval satisfies: start <= e and end >= s.
            return any(es <= e and ee >= s for es, ee in existing)

        # -----------------------------
        # Strategy: RANDOM
        # -----------------------------
        if strategy == "random":
            # Chromosomes to consider (exclude chr21 by default to mirror prior behavior).
            # Track used intervals per chromosome to prevent overlaps.
            used_regions = {c: [] for c in self.chr_sizes.keys()}
            # Generate a proportional target count per chromosome based on its size.
            # We also track how many we actually place to later top-up to a total of m.
            target_per_chr = {}
            placed_per_chr = {c: 0 for c in used_regions.keys()}
            for c in used_regions.keys():
                # Proportional count; +1 ensures each chromosome gets at least an attempt.
                target_per_chr[c] = int(m * (self.chr_sizes[c] / self.genomesize)) + 1

            # First pass: place up to target_per_chr[c] non-overlapping windows per chromosome.
            for c in used_regions.keys():
                # Local variables for readability.
                size = self.chr_sizes[c]
                # Place windows until we meet the per-chromosome target (or exhaust attempts).
                while placed_per_chr[c] < target_per_chr[c] and len(self.m_regions) < m:
                    # Sample a start index snapped to resolution within valid range.
                    max_start_bins = max(0, (size - context_length) // self.resolution)
                    rand_start = random.randint(0, max_start_bins) * self.resolution
                    rand_end = rand_start + context_length
                    # Enforce non-overlap and genome-specific constraints.
                    if not _overlaps(used_regions[c], rand_start, rand_end):
                        if self._is_region_allowed(c, rand_start, rand_end):
                            # Record and update tracking structures.
                            self.m_regions.append([c, rand_start, rand_end])
                            used_regions[c].append((rand_start, rand_end))
                            placed_per_chr[c] += 1

            # Second pass: if we undershot m (due to overlaps/filters), keep sampling globally to fill.
            while len(self.m_regions) < m and used_regions:
                # Randomly pick a chromosome among those we’re using.
                c = random.choice(list(used_regions.keys()))
                size = self.chr_sizes[c]
                # Draw a candidate window aligned to resolution.
                max_start_bins = max(0, (size - context_length) // self.resolution)
                rand_start = random.randint(0, max_start_bins) * self.resolution
                rand_end = rand_start + context_length
                # Accept if it doesn’t overlap and passes region filter.
                if not _overlaps(used_regions[c], rand_start, rand_end):
                    if self._is_region_allowed(c, rand_start, rand_end):
                        self.m_regions.append([c, rand_start, rand_end])
                        used_regions[c].append((rand_start, rand_end))

            # Done with 'random' strategy; return for clarity.
            return

        # -----------------------------
        # Strategy: CCRE
        # -----------------------------
        if strategy == "ccre":
            # Read BED into DataFrame.
            ccres = pd.read_csv(self.ccre_filename, sep="\t", header=None)
            # Assign column names for readability.
            ccres.columns = ["chrom", "start", "end", "id1", "id2", "desc"]
            # Keep chromosomes present in chr_sizes (consistent genome build) and not excluded.
            ccres = ccres[ccres["chrom"].isin(self.chr_sizes.keys())]
            # Sort for stable sampling (optional but nice).
            ccres = ccres.sort_values(["chrom", "start"]).reset_index(drop=True)

            # Track used regions per chromosome to enforce non-overlap.
            used_regions = {c: [] for c in ccres["chrom"].unique()}

            # Sample until we have m valid windows (or exhaust reasonable attempts).
            # Guard against pathological cases by capping total attempts.
            attempts, max_attempts = 0, m * 10
            while len(self.m_regions) < m and attempts < max_attempts and len(ccres) > 0:
                attempts += 1
                # Randomly pick a cCRE row.
                row = ccres.sample(n=1).iloc[0]
                # Compute the valid start bin range inside this cCRE, snapped to self.resolution.
                start_bin = row["start"] // self.resolution
                end_bin = row["end"] // self.resolution
                # If the cCRE is too small (no bin), skip.
                if end_bin <= start_bin:
                    continue
                # Draw a random start within cCRE, on-resolution.
                rand_start = random.randint(start_bin, end_bin-1) * self.resolution
                # Define the window end.
                rand_end = rand_start + context_length
                # Ensure window stays within the chromosome bounds.
                if rand_start < 0 or rand_end > self.chr_sizes[row["chrom"]]:
                    continue
                # Enforce non-overlap and region-level constraints.
                if self._is_region_allowed(row["chrom"], rand_start, rand_end):
                    if not _overlaps(used_regions[row["chrom"]], rand_start, rand_end):
                        # Record the accepted window.
                        self.m_regions.append([row["chrom"], rand_start, rand_end])
                        used_regions[row["chrom"]].append((rand_start, rand_end))

            # Done with 'ccre' strategy; return for clarity.
            return

        # -----------------------------
        # Strategy: FULL_CHR (explicit list) and GW (genome-wide tiling)
        # -----------------------------
        if strategy in ("full_chr", "gw"):
            # Determine which chromosomes to tile.
            if strategy == "gw":
                # Genome-wide tiling for chr1..chr22 and chrX (excluding chr21 as in prior code).
                chrs = self.main_chrs
            else:
                chrs = ["chr19"]

            # For each chromosome in the list, tile end-to-end windows of fixed context_length.
            for c in chrs:
                # Ensure chromosome exists in the size map.
                if c not in self.chr_sizes:
                    continue
                # Compute the largest multiple of context_length that fits in the chromosome.
                tiling_size = (self.chr_sizes[c] // context_length) * context_length
                # Iterate from 0 to tiling_size (exclusive) in steps of context_length.
                for s in range(0, tiling_size, context_length):
                    # Window end coordinate.
                    e = s + context_length
                    # Apply user-provided region filter.
                    if self._is_region_allowed(c, s, e):
                        # Append this window.
                        self.m_regions.append([c, s, e])

            # Done with 'full_chr'/'gw' strategies; return for clarity.
            return

        # -----------------------------
        # Unknown strategy: raise a helpful error
        # -----------------------------
        raise ValueError(f"Unknown strategy '{strategy}'. Expected one of: 'random', 'ccre', 'full_chr', 'gw'.")

    # ========= Helper Functions =========
    def _get_DNA_sequence(self, chrom, start, end):
        """
        Retrieve the sequence for a given chromosome and coordinate range from a fasta file.

        :param fasta_file: Path to the fasta file.
        :param chrom: Chromosome name (e.g., 'chr1').
        :param start: Start position (0-based).
        :param end: End position (1-based, exclusive).
        :return: Sequence string.
        """
        if not hasattr(self, 'fasta') or self.fasta is None:
            self._load_fasta()
        try:
            # Ensure coordinates are within the valid range
            if start < 0 or end <= start:
                raise ValueError("Invalid start or end position")
            
            # Retrieve the sequence
            sequence = self.fasta.fetch(chrom, start, end)
            
            return sequence

        except Exception as e:
            print(f"Error retrieving sequence {chrom}:{start}-{end}: {e}")
            return None

    def _dna_to_onehot(self, sequence):
        # Create a mapping from nucleotide to index
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4}
        
        # Convert the sequence to indices
        indices = torch.tensor([mapping[nuc.upper()] for nuc in sequence], dtype=torch.long)
        
        # Create one-hot encoding
        one_hot = torch.nn.functional.one_hot(indices, num_classes=5)

        # Remove the fifth column which corresponds to 'N'
        one_hot = one_hot[:, :4]
        
        return one_hot

    def _is_region_allowed(self, chrom, start, end):
        """Check if a region overlaps with blacklist regions using IntervalTree."""
        if chrom not in self.blacklist:
            return True
        tree = self.blacklist[chrom]
        overlapping = tree.overlap(start, end)
        return len(overlapping) == 0

    def _onehot_for_locus(self, locus):
        """
        Helper to fetch DNA and convert to one-hot for a given locus [chrom, start, end].
        Returns a tensor [context_length_bp, 4].
        """
        chrom, start, end = locus[0], int(locus[1]), int(locus[2])
        seq = self._get_DNA_sequence(chrom, start, end)
        if seq is None:
            return None
        return self._dna_to_onehot(seq)

    def _get_cached_onehot_for_locus(self, locus):
        """
        Get cached one-hot DNA sequence for a locus, computing if not cached.
        Returns a tensor [context_length_bp, 4].
        """
        # Use tuple as cache key for hashability
        locus_key = tuple(locus)
        
        if locus_key not in self.dna_cache:
            # Compute and cache the one-hot sequence
            self.dna_cache[locus_key] = self._onehot_for_locus(locus)
        
        return self.dna_cache[locus_key]
  
    def _load_npz(self, file_name):
        with np.load(file_name, allow_pickle=True) as data:
        # with np.load(file_name, allow_pickle=True, mmap_mode='r') as data:
            return {file_name.split("/")[-3]: data[data.files[0]]}
    
    def _filter_navigation(self, include=[], exclude=[]):
        """
        filter based on a list of assays to include
        """
        for bios in list(self.navigation.keys()):
            if self.eic:
                if bios[0] not in ["T", "V", "B"]:
                    del self.navigation[bios]

            elif self.merge_ct:
                if bios[0] in ["T", "V", "B"]:
                    del self.navigation[bios]

            else:
                bios_exps = list(self.navigation[bios].keys())
                if self.must_have_chr_access: 
                    if ("ATAC-seq" not in bios_exps) and ("DNase-seq" not in bios_exps):
                        del self.navigation[bios]

        if len(include) == 0 and len(exclude) == 0:
            return

        elif len(exclude) == 0 and len(include) != 0:
            for bios in list(self.navigation.keys()):
                for exp in list(self.navigation[bios].keys()):
                    if exp not in include:
                        del self.navigation[bios][exp]

            for exp in list(self.aliases["experiment_aliases"].keys()):
                if exp not in include:
                    del self.aliases["experiment_aliases"][exp]

        elif len(include) == 0 and len(exclude) != 0:
            for bios in list(self.navigation.keys()):
                for exp in list(self.navigation[bios].keys()):
                    if exp in exclude:
                        del self.navigation[bios][exp]
                        
            for exp in exclude:
                if exp in self.aliases["experiment_aliases"].keys():
                    del self.aliases["experiment_aliases"][exp]
        
        else:
            return

    def _select_region_from_loaded_data(self, loaded_data, locus):
        region = {}
        start_bin = int(locus[1]) // self.resolution
        end_bin = int(locus[2]) // self.resolution
        for exp, data in loaded_data.items():
            region[exp] = data[start_bin:end_bin]
        
        return region

    # ========= Data Loading =========
    def load_bios_Counts(self, bios_name, locus, DSF=1, f_format="npz"): # count data 
        exps = list(self.navigation[bios_name].keys())

        if "RNA-seq" in exps:
            exps.remove("RNA-seq")

        loaded_data = {}
        loaded_metadata = {}

        npz_files = []
        for e in exps:
            if self.merge_ct and self.eic==False:
                l =    os.path.join("/".join(self.navigation[bios_name][e][0].split("/")[:-1]), f"signal_DSF{DSF}_res{self.resolution}", f"{locus[0]}.{f_format}")
                jsn1 = os.path.join("/".join(self.navigation[bios_name][e][0].split("/")[:-1]), f"signal_DSF{DSF}_res{self.resolution}", "metadata.json")
                jsn2 = os.path.join("/".join(self.navigation[bios_name][e][0].split("/")[:-1]), "file_metadata.json")

            else:
                l = os.path.join(self.base_path, bios_name, e, f"signal_DSF{DSF}_res{self.resolution}", f"{locus[0]}.{f_format}")
                jsn1 = os.path.join(self.base_path, bios_name, e, f"signal_DSF{DSF}_res{self.resolution}", "metadata.json")
                jsn2 = os.path.join(self.base_path, bios_name, e, "file_metadata.json")

            npz_files.append(l)
            with open(jsn1, 'r') as jsnfile:
                md1 = json.load(jsnfile)

            with open(jsn2, 'r') as jsnfile:
                md2 = json.load(jsnfile)


            md = {
                "depth":md1["depth"], "sequencing_platform": md2["sequencing_platform"], 
                "read_length":md2["read_length"], "run_type":md2["run_type"] 
            }
            loaded_metadata[e] = md

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_workers) as executor:
            for result in executor.map(self._load_npz, npz_files):
                if result is not None:
                    for exp, data in result.items():
                        if len(locus) == 1:
                            loaded_data[exp] = data.astype(np.int16)
                                
                        else:
                            start_bin = int(locus[1]) // self.resolution
                            end_bin = int(locus[2]) // self.resolution
                            loaded_data[exp] = data[start_bin:end_bin]
            
        return loaded_data, loaded_metadata

    def load_bios_BW(self, bios_name, locus, f_format="npz", arcsinh=True): # signal data 
        exps = list(self.navigation[bios_name].keys())

        if "RNA-seq" in exps:
            exps.remove("RNA-seq")

        if "chipseq-control" in exps:
            exps.remove("chipseq-control")

        loaded_data = {}
        npz_files = []
        for e in exps:
            if self.merge_ct and self.eic==False:
                l = os.path.join("/".join(self.navigation[bios_name][e][0].split("/")[:-1]), f"signal_BW_res{self.resolution}", f"{locus[0]}.{f_format}")
            else:
                l = os.path.join(self.base_path, bios_name, e, f"signal_BW_res{self.resolution}", f"{locus[0]}.{f_format}")
            npz_files.append(l)

        # Load files in parallel
        with ThreadPoolExecutor(max_workers=self.max_thread_workers) as executor:
            loaded = list(executor.map(self._load_npz, npz_files))
        
        if len(locus) == 1:
            for l in loaded:
                for exp, data in l.items():
                    if arcsinh:
                        loaded_data[exp] = np.arcsinh(data).astype(np.float16)
                    else:
                        loaded_data[exp] = data.astype(np.float16)
            return loaded_data

        else:
            start_bin = int(locus[1]) // self.resolution
            end_bin = int(locus[2]) // self.resolution
            for l in loaded:
                for exp, data in l.items():
                    if arcsinh:
                        loaded_data[exp] = np.arcsinh(data[start_bin:end_bin])
                    else:
                        loaded_data[exp] = data[start_bin:end_bin]
            return loaded_data

    def load_bios_Peaks(self, bios_name, locus, f_format="npz"):
        """
        Load peak indicator/scores per assay from peaks_res{resolution}/<chr>.npz.
        Returns a dict exp_name -> np.array for chrom or sliced locus.
        """
        exps = list(self.navigation[bios_name].keys())

        if "RNA-seq" in exps:
            exps.remove("RNA-seq")
            exps.remove("chipseq-control")

        loaded_data = {}
        npz_files = []
        for e in exps:
            if self.merge_ct and self.eic==False:
                l = os.path.join("/".join(self.navigation[bios_name][e][0].split("/")[:-1]), f"peaks_res{self.resolution}", f"{locus[0]}.{f_format}")
            else:
                l = os.path.join(self.base_path, bios_name, e, f"peaks_res{self.resolution}", f"{locus[0]}.{f_format}")
            npz_files.append(l)

        with ThreadPoolExecutor(max_workers=self.max_thread_workers) as executor:
            loaded = list(executor.map(self._load_npz, npz_files))

        if len(locus) == 1:
            for l in loaded:
                for exp, data in l.items():
                    loaded_data[exp] = data.astype(np.float16)
            return loaded_data

        else:
            start_bin = int(locus[1]) // self.resolution
            end_bin = int(locus[2]) // self.resolution
            for l in loaded:
                for exp, data in l.items():
                    loaded_data[exp] = data[start_bin:end_bin]
            return loaded_data
    
    def load_bios_Control(self, bios_name, locus, DSF=1, f_format="npz"):
        """Load chipseq-control signal for a biosample."""
        control_path = os.path.join(self.base_path, bios_name, "chipseq-control")
        if not os.path.exists(control_path):
            return {}, {}
        
        loaded_data = {}
        loaded_metadata = {}
        
        signal_path = os.path.join(control_path, f"signal_DSF{DSF}_res{self.resolution}", f"{locus[0]}.{f_format}")
        metadata_path_1 = os.path.join(control_path, f"signal_DSF{DSF}_res{self.resolution}", "metadata.json")
        metadata_path_2 = os.path.join(control_path, "file_metadata.json")
        
        if not os.path.exists(signal_path):
            return {}, {}
        
        try:
            result = self._load_npz(signal_path)
            if result is None:
                return {}, {}
            
            for key, data in result.items():
                if len(locus) == 1:
                    loaded_data["chipseq-control"] = data.astype(np.int16)
                else:
                    start_bin = int(locus[1]) // self.resolution
                    end_bin = int(locus[2]) // self.resolution
                    loaded_data["chipseq-control"] = data[start_bin:end_bin]
            
            with open(metadata_path_1, 'r') as f:
                md1 = json.load(f)
            with open(metadata_path_2, 'r') as f:
                md2 = json.load(f)
            
            loaded_metadata["chipseq-control"] = {
                "depth": md1["depth"],
                "sequencing_platform": md2.get("sequencing_platform", {"2": "unknown"}).get("2", "unknown"),
                "read_length": md2.get("read_length", {"2": None}).get("2", None),
                "run_type": md2.get("run_type", {"2": "single-ended"}).get("2", "single-ended")
            }
            
            return loaded_data, loaded_metadata
        
        except Exception as e:
            print(f"Error loading control for {bios_name}: {e}")
            return {}, {}

    # ========= Making BiosampleTensors =========
    def make_bios_tensor_Counts(self, loaded_data, loaded_metadata, missing_value=-1): # count data 
        dtensor = []
        mdtensor = []
        availability = []

        L = len(loaded_data[list(loaded_data.keys())[0]])
        i = 0
        for assay, alias in self.aliases["experiment_aliases"].items():
            
            if assay in loaded_data.keys():
                dtensor.append(loaded_data[assay])
                availability.append(1)

                if "single" in loaded_metadata[assay]['run_type'][list(loaded_metadata[assay]['run_type'].keys())[0]]:
                    runt = 0
                elif "pair" in loaded_metadata[assay]['run_type'][list(loaded_metadata[assay]['run_type'].keys())[0]]:
                    runt = 1

                readl = loaded_metadata[assay]['read_length'][list(loaded_metadata[assay]['read_length'].keys())[0]]

                sequencing_platform = loaded_metadata[assay]['sequencing_platform'][list(loaded_metadata[assay]['sequencing_platform'].keys())[0]]
                # Encode platform to stable int id (unknown->0)
                platform_id = self.sequencing_platform_to_id.get(str(sequencing_platform), 0)

                mdtensor.append([
                    np.log2(loaded_metadata[assay]['depth']), platform_id,
                    readl, runt])

            else:
                dtensor.append([missing_value for _ in range(L)])
                availability.append(0)
                mdtensor.append([missing_value, missing_value, missing_value, missing_value])

            i += 1
        
        dtensor = torch.tensor(np.array(dtensor)).permute(1, 0)
        mdtensor = torch.tensor(np.array(mdtensor)).permute(1, 0)
        availability = torch.tensor(np.array(availability))
        return dtensor, mdtensor, availability

    def make_bios_tensor_BW(self, loaded_data, missing_value=-1): # signal data
        dtensor = []
        availability = []

        L = len(loaded_data[list(loaded_data.keys())[0]])
        i = 0
        for assay, alias in self.aliases["experiment_aliases"].items():
            if assay in loaded_data.keys():
                dtensor.append(loaded_data[assay])
                availability.append(1)
            else:
                dtensor.append([missing_value for _ in range(L)])
                availability.append(0)
            i += 1

        dtensor = torch.tensor(np.array(dtensor)).permute(1, 0)
        availability = torch.tensor(np.array(availability))
        return dtensor, availability

    def make_bios_tensor_Peaks(self, loaded_data, missing_value=-1):
        dtensor = []
        availability = []

        L = len(loaded_data[list(loaded_data.keys())[0]])
        i = 0
        for assay, alias in self.aliases["experiment_aliases"].items():
            if assay in loaded_data.keys():
                dtensor.append(loaded_data[assay])
                availability.append(1)
            else:
                dtensor.append([missing_value for _ in range(L)])
                availability.append(0)
            i += 1

        dtensor = torch.tensor(np.array(dtensor)).permute(1, 0)
        availability = torch.tensor(np.array(availability))
        return dtensor, availability

    def make_bios_tensor_Control(self, loaded_data, loaded_metadata, missing_value=-1):
        """Format control data for ONE biosample into tensors."""
        if loaded_data and "chipseq-control" in loaded_data:
            L = len(loaded_data["chipseq-control"])
            
            dtensor = loaded_data["chipseq-control"].reshape(-1, 1)  # (L, 1)
            
            meta = loaded_metadata["chipseq-control"]
            run_type_str = str(meta['run_type']).lower()
            runt = 0 if "single" in run_type_str else (1 if "pair" in run_type_str else 0)
            readl = meta['read_length'] if meta['read_length'] is not None else 50
            platform_id = self.sequencing_platform_to_id.get(str(meta['sequencing_platform']), 0)
            
            mdtensor = np.array([[np.log2(meta['depth'])], [platform_id], [readl], [runt]])  # (4, 1)
            availability = np.array([1])  # (1,)
            
        else:
            L = self.context_length // self.resolution 
            dtensor = np.full((L, 1), missing_value)
            mdtensor = np.full((4, 1), missing_value)
            availability = np.array([0])
        
        dtensor = torch.tensor(dtensor).float()
        mdtensor = torch.tensor(mdtensor).float()
        availability = torch.tensor(availability).float()
        
        return dtensor, mdtensor, availability

    # ========= Making RegionTensors =========
    def make_region_tensor_Counts(self, loaded_data, loaded_metadata): 
        data, metadata, availability = [], [], []
        for i in range(len(loaded_data)):
            d, md, avl = self.make_bios_tensor_Counts(loaded_data[i], loaded_metadata[i])
            data.append(d)
            metadata.append(md)
            availability.append(avl)
        
        data, metadata, availability = torch.stack(data), torch.stack(metadata), torch.stack(availability)
        return data, metadata, availability

    def make_region_tensor_BW(self, loaded_data): # signal data
        data, availability = [], []
        for i in range(len(loaded_data)):
            d, avl = self.make_bios_tensor_BW(loaded_data[i])
            data.append(d)
            availability.append(avl)

        data, availability = torch.stack(data), torch.stack(availability)
        return data, availability
   
    def make_region_tensor_Peaks(self, loaded_data):
        data, availability = [], []
        for i in range(len(loaded_data)):
            d, avl = self.make_bios_tensor_Peaks(loaded_data[i])
            data.append(d)
            availability.append(avl)

        data, availability = torch.stack(data), torch.stack(availability)
        return data, availability

    def make_region_tensor_Control(self, loaded_data_list, loaded_metadata_list):
        """Stack control tensors from MULTIPLE biosamples into a batch."""
        data, metadata, availability = [], [], []
        
        for i in range(len(loaded_data_list)):
            d, md, avl = self.make_bios_tensor_Control(loaded_data_list[i], loaded_metadata_list[i])
            data.append(d)
            metadata.append(md)
            availability.append(avl)
        
        data = torch.stack(data)          # (B, L, 1)
        metadata = torch.stack(metadata)  # (B, 4, 1)
        availability = torch.stack(availability)  # (B, 1)
        
        return data, metadata, availability

    def init_stat_lookup(self):
        # Build per-assay lookup only once per call
        self.stat_lookup = {}
        for assay in self.aliases["experiment_aliases"].keys():
            assay_df = self.metadata[self.metadata['assay_name'] == assay]
            if assay_df.empty:
                self.stat_lookup[assay] = None
                continue
            # Platform id mode (most frequent) mapped through sequencing_platform_to_id
            platforms = [self.sequencing_platform_to_id.get(str(x), 0) for x in assay_df['sequencing_platform'].dropna().values]
            platform_mode = int(pd.Series(platforms).mode().iloc[0]) if len(platforms) else 0
            # Run type id: 0 for single, 1 for pair (mode)
            run_types = assay_df['run_type'].dropna().astype(str).values
            run_ids = [1 if ('pair' in r.lower()) else 0 for r in run_types]
            run_mode = int(pd.Series(run_ids).mode().iloc[0]) if len(run_ids) else 1
            self.stat_lookup[assay] = {
                "depth_log2_median": float(np.nanmedian(np.log2(assay_df['depth'].astype(float)))) if 'depth' in assay_df else 0.0,
                "platform_mode": platform_mode,
                "read_length_median": float(np.nanmedian(assay_df['read_length'].astype(float))) if 'read_length' in assay_df else 50.0,
                "run_type_mode": run_mode,
            }

        return self.stat_lookup

    # ========= Filling in Prompt =========
    def fill_in_prompt(self, md, missing_value=-1, sample=True):
        """
        Fill missing assay metadata columns (marked by missing_value) either by sampling from
        dataset metadata per assay (sample=True) or by using median/mode statistics (sample=False).

        Expected md shape: [4, E] where rows are [depth_log2, platform_id, read_length, run_type_id].
        """
        
        if self.stat_lookup is None:
            self.init_stat_lookup()

        # md is [4, E]
        filled = md.clone().squeeze(0)
        num_assays = filled.shape[1]
        for i, (assay, alias) in enumerate(self.aliases["experiment_aliases"].items()):
            if i >= num_assays:
                break
            col = filled[:, i]
            if torch.all(col == missing_value):
                if sample:
                    assay_df = self.metadata[self.metadata['assay_name'] == assay]
                    if len(assay_df) > 0:
                        # Sample per field with fallbacks
                        depth_vals = np.log2(assay_df['depth'].dropna().astype(float).values) if 'depth' in assay_df else [0.0]
                        platform_vals = [self.sequencing_platform_to_id.get(str(x), 0) for x in assay_df['sequencing_platform'].dropna().values] or [0]
                        readlen_vals = assay_df['read_length'].dropna().astype(float).values if 'read_length' in assay_df else [50.0]
                        runt_vals = [1 if ('pair' in str(x).lower()) else 0 for x in assay_df['run_type'].dropna().values] or [1]
                        filled[0, i] = float(random.choice(depth_vals))
                        filled[1, i] = float(random.choice(platform_vals))
                        filled[2, i] = float(random.choice(readlen_vals))
                        filled[3, i] = float(random.choice(runt_vals))
                else:
                    stats = self.stat_lookup.get(assay)
                    if stats is not None:
                        filled[0, i] = stats["depth_log2_median"]
                        filled[1, i] = stats["platform_mode"]
                        filled[2, i] = stats["read_length_median"]
                        filled[3, i] = stats["run_type_mode"]

        filled = filled.unsqueeze(0)
        return filled
 
    def fill_in_prompt_manual(self, md, manual_spec, missing_value=-1, overwrite=True):
        """
        Manually set metadata per assay using an explicit dict spec.
        manual_spec format: { assay_name: { metadata_name: value, ... }, ... }
        Accepted metadata_name keys (case-insensitive):
        - depth or depth_log2
        - sequencing_platform or platform_id
        - read_length
        - run_type or run_type_id (single->0, pair->1)
        If overwrite=False, only fills entries equal to missing_value.
        md shape: [4, E] with rows [depth_log2, platform_id, read_length, run_type_id].
        """
        if md is None:
            return md
        filled = md.clone()
        # Map assay -> column index
        assay_to_idx = {assay: i for i, (assay, alias) in enumerate(self.aliases["experiment_aliases"].items())}

        def to_platform_id(v):
            if isinstance(v, (int, float)):
                return int(v)
            return int(self.sequencing_platform_to_id.get(str(v), 0))

        def to_run_type_id(v):
            if isinstance(v, (int, float)):
                return int(v)
            return 1 if ("pair" in str(v).lower()) else 0

        for assay, fields in manual_spec.items():
            if assay not in assay_to_idx:
                continue
            i = assay_to_idx[assay]
            # Current column values
            cur = filled[:, i]
            # depth/depth_log2
            for k in fields.keys():
                key = str(k).lower()
                if key in ("depth", "depth_log2", "sequencing_platform", "platform_id", "read_length", "run_type", "run_type_id"):
                    # decide overwrite condition per row after computing value
                    pass
            if "depth" in fields or "depth_log2" in fields:
                if "depth_log2" in fields:
                    val = float(fields["depth_log2"])
                else:
                    val = float(np.log2(float(fields["depth"])) )
                if overwrite or cur[0].item() == missing_value:
                    filled[0, i] = val
            if "sequencing_platform" in fields or "platform_id" in fields:
                src = fields.get("platform_id", fields.get("sequencing_platform"))
                pid = float(to_platform_id(src))
                if overwrite or cur[1].item() == missing_value:
                    filled[1, i] = pid
            if "read_length" in fields:
                rl = float(fields["read_length"])
                if overwrite or cur[2].item() == missing_value:
                    filled[2, i] = rl
            if "run_type" in fields or "run_type_id" in fields:
                src = fields.get("run_type_id", fields.get("run_type"))
                rt = float(to_run_type_id(src))
                if overwrite or cur[3].item() == missing_value:
                    filled[3, i] = rt
        return filled
           
    # ========= RNA-seq Data Loading =========
    def has_rnaseq(self, bios_name):
        if self.merge_ct:
            if "RNA-seq" in self.navigation[bios_name].keys():
                return True
            else:
                return False

    def has_chr_access(self, bios_name):
        if "ATAC-seq" in self.navigation[bios_name].keys() or "DNase-seq" in self.navigation[bios_name].keys():
            return True
        else:
            return False

    def load_rna_seq_data(self, bios_name, gene_coord):
        if self.merge_ct:
            directory = os.path.dirname(self.navigation[bios_name]["RNA-seq"][0])
        else:
            directory = os.path.join(self.base_path, bios_name, "RNA-seq/")
            
        tsv_files = glob.glob(os.path.join(directory, '*.tsv'))

        file = os.path.join(directory, tsv_files[0])
        trn_data = pd.read_csv(file, sep="\t")

        for j in range(len(trn_data)):
            trn_data.at[j, "gene_id"] = trn_data["gene_id"][j].split(".")[0]
        
        for i in range(len(gene_coord)):
            gene_coord.at[i, "gene_id"] = gene_coord["gene_id"][i].split(".")[0]

        mapped_trn_data = []
        for i in range(len(gene_coord)):
            geneID = gene_coord["gene_id"][i]
            subset = trn_data.loc[trn_data["gene_id"] == geneID, :].reset_index(drop=True)

            if len(subset) > 0:
                mapped_trn_data.append([
                    gene_coord["chr"][i], gene_coord["start"][i], gene_coord["end"][i], gene_coord["strand"][i], geneID, subset["length"][0], subset["TPM"][0], subset["FPKM"][0]
                ])

        mapped_trn_data = pd.DataFrame(mapped_trn_data, columns=["chr", "start", "end", "strand", "geneID", "length", "TPM", "FPKM"])
        return mapped_trn_data

    # ========= Integrated Data Looping and Batching =========
    def setup_datalooper(self, m, context_length, bios_batchsize, loci_batchsize,
                        loci_gen_strategy="random", split="train", bios_min_exp_avail_threshold=3,
                        shuffle_bios=True, dsf_list=[1, 2, 4], includes=None, excludes=[], must_have_chr_access=False):
        """
        Configures the data handler for iterating through epochs of data.

        Args:
            m (int): Number of genomic loci to generate.
            context_length (int): The length of each genomic locus in base pairs.
            bios_batchsize (int): Number of biosamples per batch.
            loci_batchsize (int): Number of loci per batch.
            loci_gen_strategy (str): Strategy for generating loci ('random', 'ccre', 'gw', 'chrN').
            split (str): The dataset split to use ('train', 'val', 'test').
            bios_min_exp_avail_threshold (int): Minimum number of available experiments for a biosample to be included.
            shuffle_bios (bool): Whether to shuffle the order of biosamples.
            dsf_list (list): List of downsampling factors to use.
            includes (list): List of experiment types to include. Defaults to all standard types.
            excludes (list): List of experiment types to exclude.
            must_have_chr_access (bool): Whether to require chromosome access for all experiments.
        """
        print(f"--- Setting up data looper for split: {split} ---")
        if includes is None: includes = self.includes
        self.must_have_chr_access = must_have_chr_access
        self._load_genomic_coords(mode=split)
        
        print(f"Generating {m} loci using '{loci_gen_strategy}' strategy...")
        self._generate_genomic_loci(m, context_length, strategy=loci_gen_strategy)
        print(f"Generated {len(self.m_regions)} regions.")
        
        # Reload and filter navigation for the specific split and criteria
        self._load_navigation()
        self._filter_navigation(includes, excludes)

        print(f"Filtering navigation for split: {split}...")
        for bios in list(self.navigation.keys()):
            if self.split_dict[bios] != split:
                del self.navigation[bios]

            elif len(self.navigation[bios]) < bios_min_exp_avail_threshold:
                del self.navigation[bios]

            elif must_have_chr_access and self.has_chr_access(bios) == False:
                del self.navigation[bios]
        
        if shuffle_bios:
            keys = list(self.navigation.keys())
            random.shuffle(keys)
            self.navigation = {key: self.navigation[key] for key in keys}

        self.signal_dim = len(self.aliases["experiment_aliases"])
        self.num_regions = len(self.m_regions)
        self.num_bios = len(self.navigation)
        print(f"Using {self.num_bios} biosamples with signal dimension {self.signal_dim}.")

        self.bios_batchsize = bios_batchsize
        self.loci_batchsize = loci_batchsize
        self.dsf_list = dsf_list
        self.num_batches = math.ceil(self.num_bios / self.bios_batchsize)

        self.loci = {}
        for r in self.m_regions:
            self.loci.setdefault(r[0], []).append(r)
        
        print("--- Datalooper setup complete. ---")

    def new_epoch(self, shuffle_chr=True):
        """Resets pointers and pre-loads data for the start of a new epoch."""
        self.chr_pointer = 0
        self.bios_pointer = 0
        self.dsf_pointer = 0
        self.chr_loci_pointer = 0

        if shuffle_chr:
            keys = list(self.loci.keys())
            random.shuffle(keys)
            self.loci = {key: self.loci[key] for key in keys}

        batch_bios_list = list(self.navigation.keys())[self.bios_pointer: self.bios_pointer + self.bios_batchsize]
        current_chr = list(self.loci.keys())[self.chr_pointer]

        self.loaded_data, self.loaded_metadata, self.loaded_control, self.loaded_control_metadata = [], [], [], []
        # Pre-loading data for biosamples
        for bios in batch_bios_list:
            d, md = self.load_bios_Counts(bios, [current_chr], self.dsf_list[self.dsf_pointer])
            self.loaded_data.append(d)
            self.loaded_metadata.append(md)

            c, cm = self.load_bios_Control(bios, [current_chr], self.dsf_list[self.dsf_pointer])
            self.loaded_control.append(c)
            self.loaded_control_metadata.append(cm)
        
        self.Y_loaded_data, self.Y_loaded_metadata = [], []
        self.Y_loaded_data = self.loaded_data.copy()
        self.Y_loaded_metadata = self.loaded_metadata.copy()

        self.Y_loaded_pval = []
        for bios in batch_bios_list:
            pval_d = self.load_bios_BW(bios, [current_chr])
            self.Y_loaded_pval.append(pval_d)

        self.Y_loaded_peaks = []
        for bios in batch_bios_list:
            peaks_d = self.load_bios_Peaks(bios, [current_chr])
            self.Y_loaded_peaks.append(peaks_d)

    def _update_batch_pointers(self):
        """Updates pointers to move to the next batch. Returns True if epoch is finished."""
        self.chr_loci_pointer += self.loci_batchsize
        
        if self.chr_loci_pointer >= len(self.loci[list(self.loci.keys())[self.chr_pointer]]):
            self.chr_loci_pointer = 0
            self.dsf_pointer += 1

            if self.dsf_pointer >= len(self.dsf_list):
                self.dsf_pointer = 0
                self.bios_pointer += self.bios_batchsize

                if self.bios_pointer >= self.num_bios:
                    self.bios_pointer = 0
                    self.chr_pointer += 1

                    if self.chr_pointer >= len(self.loci.keys()):
                        return True  # End of epoch
                
            # Pre-load data for the new state (new bios, chr, or dsf)
            batch_bios_list = list(self.navigation.keys())[self.bios_pointer: self.bios_pointer + self.bios_batchsize]
            current_chr = list(self.loci.keys())[self.chr_pointer]
            current_dsf = self.dsf_list[self.dsf_pointer]
            
            # Updating pointers and pre-loading data
            self.loaded_data, self.loaded_metadata, self.loaded_control, self.loaded_control_metadata = [], [], [], []
            for bios in batch_bios_list:
                d, md = self.load_bios_Counts(bios, [current_chr], current_dsf)
                self.loaded_data.append(d)
                self.loaded_metadata.append(md)

                c, cm = self.load_bios_Control(bios, [current_chr], current_dsf)
                self.loaded_control.append(c)
                self.loaded_control_metadata.append(cm)
            
            if self.dsf_pointer == 0:
                self.Y_loaded_data, self.Y_loaded_metadata = [], []
                self.Y_loaded_data = self.loaded_data.copy()
                self.Y_loaded_metadata = self.loaded_metadata.copy()

                self.Y_loaded_pval = []
                for bios in batch_bios_list:
                    self.Y_loaded_pval.append(
                        self.load_bios_BW(bios, [current_chr]))

                self.Y_loaded_peaks = []
                for bios in batch_bios_list:
                    self.Y_loaded_peaks.append(
                        self.load_bios_Peaks(bios, [current_chr]))

        return False # Not end of epoch

    def get_batch(self, side="x", y_prompt=False):
        """
        Constructs and returns a batch of data, always including one-hot encoded DNA.
        If side='y', it also includes p-value signal data.
        """
        current_chr = list(self.loci.keys())[self.chr_pointer]
        batch_loci_list = self.loci[current_chr][self.chr_loci_pointer : self.chr_loci_pointer + self.loci_batchsize]
        if not batch_loci_list: return None

        one_hot_sequences = torch.stack([self._get_cached_onehot_for_locus(l) for l in batch_loci_list])

        data_source = self.Y_loaded_data if side == "y" else self.loaded_data
        metadata_source = self.Y_loaded_metadata if side == "y" else self.loaded_metadata

        locus_tensors = []
        for locus in batch_loci_list:
            loc_data_list = [self._select_region_from_loaded_data(d, locus) for d in data_source]
            locus_tensors.append(self.make_region_tensor_Counts(loc_data_list, metadata_source))

        batch_data = torch.cat([t[0] for t in locus_tensors], dim=0)
        batch_metadata = torch.cat([t[1] for t in locus_tensors], dim=0)
        batch_availability = torch.cat([t[2] for t in locus_tensors], dim=0)

        # Repeat DNA sequence tensor to match batch size
        num_biosamples_in_batch = batch_data.shape[0] // len(batch_loci_list)
        one_hot_sequences = one_hot_sequences.repeat_interleave(num_biosamples_in_batch, dim=0)

        if side == 'y':
            if y_prompt:
                batch_metadata = self.fill_in_prompt(batch_metadata, sample=True)
            
            # Process p-value data for the y-side
            pval_tensors = []
            for locus in batch_loci_list:
                loc_p_list = [self._select_region_from_loaded_data(p, locus) for p in self.Y_loaded_pval]
                pval_tensors.append(self.make_region_tensor_BW(loc_p_list))
            
            batch_pval_data = torch.cat([t[0] for t in pval_tensors], dim=0)
            batch_pval_avail = torch.cat([t[1] for t in pval_tensors], dim=0)
            assert (batch_availability == batch_pval_avail).all(), "Availability mismatch between counts and signal"
            
            # Process peaks data for the y-side
            peaks_tensors = []
            for locus in batch_loci_list:
                loc_peaks_list = [self._select_region_from_loaded_data(p, locus) for p in self.Y_loaded_peaks]
                peaks_tensors.append(self.make_region_tensor_Peaks(loc_peaks_list))
            
            batch_peaks_data = torch.cat([t[0] for t in peaks_tensors], dim=0)
            batch_peaks_avail = torch.cat([t[1] for t in peaks_tensors], dim=0)
            assert (batch_availability == batch_peaks_avail).all(), "Availability mismatch between counts and peaks"
            
            return batch_data, batch_metadata, batch_availability, batch_pval_data, batch_peaks_data, one_hot_sequences
            
        else: # side == 'x'
            control_source = self.loaded_control
            control_metadata_source = self.loaded_control_metadata

            control_locus_tensors = []
            for locus in batch_loci_list:
                loc_data_list = [self._select_region_from_loaded_data(d, locus) for d in control_source]
                control_locus_tensors.append(self.make_region_tensor_Control(loc_data_list, control_metadata_source))

            batch_control_data = torch.cat([t[0] for t in control_locus_tensors], dim=0)
            batch_control_metadata = torch.cat([t[1] for t in control_locus_tensors], dim=0)
            batch_control_availability = torch.cat([t[2] for t in control_locus_tensors], dim=0)

            return batch_data, batch_metadata, batch_availability, one_hot_sequences, batch_control_data, batch_control_metadata, batch_control_availability

# ========= CANDIIterableDataset =========
class CANDIIterableDataset(CANDIDataHandler, torch.utils.data.IterableDataset):
    """
    A PyTorch IterableDataset wrapper for the CANDIDataHandler.

    This class turns the stateful CANDIDataHandler into a Python iterator that can be
    used with a PyTorch DataLoader. It correctly handles data sharding for both
    multi-processing (num_workers > 0) and Distributed Data Parallel (DDP) training,
    ensuring each worker and each GPU gets a unique, non-overlapping subset of the data.
    """
    def __init__(self, **kwargs):
        """
        Initializes the dataset.

        Args:
            **kwargs: All arguments required by both CANDIDataHandler's __init__
                      and setup_dataloper methods.
        """
        super().__init__(
            base_path=kwargs.get("base_path"),
            resolution=kwargs.get("resolution", 25),
            dataset_type=kwargs.get("dataset_type", "merged"),
            DNA=kwargs.get("DNA", True),
            bios_batchsize=1,  # Force to 1
            loci_batchsize=1,  # Force to 1
            dsf_list=kwargs.get("dsf_list", [1, 2])
        )

        print(f"CANDIIterableDataset initialized with bios_batchsize={self.bios_batchsize}, loci_batchsize={self.loci_batchsize}, dsf_list={self.dsf_list}")
        self.kwargs = kwargs

    def __iter__(self):
        """
        Yields a single sample of data, one at a time.
        """
        # --- 1. Setup the Data Looper ---
        self.setup_datalooper(
            m=self.kwargs.get("m"),
            context_length=self.kwargs.get("context_length"),
            bios_batchsize=1,
            loci_batchsize=1,
            loci_gen_strategy=self.kwargs.get("loci_gen_strategy", "random"),
            split=self.kwargs.get("split", "train"),
            shuffle_bios=self.kwargs.get("shuffle_bios", True),
            dsf_list=self.kwargs.get("dsf_list", [1, 2, 4]),
            includes=self.kwargs.get("includes"),
            excludes=self.kwargs.get("excludes", []),
            must_have_chr_access=self.kwargs.get("must_have_chr_access", False),
            bios_min_exp_avail_threshold=self.kwargs.get("bios_min_exp_avail_threshold", 0)
        )

        # --- 2. Shard the Data for Parallel Loading ---
        biosample_keys = list(self.navigation.keys())
        world_size = 1
        rank = 0
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1
        worker_id = 0
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        
        total_consumers = num_workers * world_size
        global_rank = rank * num_workers + worker_id

        my_biosample_keys = list(islice(biosample_keys, global_rank, None, total_consumers))
        self.navigation = {key: self.navigation[key] for key in my_biosample_keys}
        self.num_bios = len(self.navigation)

        if self.num_bios == 0:
            return

        # --- 3. Iterate and Yield Samples ---
        self.new_epoch(shuffle_chr=True)
        epoch_finished = False

        while not epoch_finished:
            # Get unique identifiers for validation
            current_biosample_name = list(self.navigation.keys())[self.bios_pointer]
            current_chr = list(self.loci.keys())[self.chr_pointer]
            locus = self.loci[current_chr][self.chr_loci_pointer]
            current_dsf = self.dsf_list[self.dsf_pointer]
            locus_str = f"{locus[0]}:{locus[1]}-{locus[2]}"
            sample_id = (current_biosample_name, locus_str, current_dsf)
            
            x_batch = self.get_batch(side="x")
            if x_batch is None: break
            x_data, x_meta, x_avail, x_dna, control_data, control_meta, control_avail = x_batch

            y_batch = self.get_batch(side="y", y_prompt=True)
            if y_batch is None: break
            y_data, y_meta, y_avail, y_pval, y_peaks, y_dna = y_batch

            sample = {
                "sample_id": sample_id, # For validation
                "x_data": x_data.squeeze(0), "x_meta": x_meta.squeeze(0),
                "x_avail": x_avail.squeeze(0), "x_dna": x_dna.squeeze(0),
                "control_data": control_data.squeeze(0),
                "control_meta": control_meta.squeeze(0),
                "control_avail": control_avail.squeeze(0),
                "y_data": y_data.squeeze(0), "y_meta": y_meta.squeeze(0),
                "y_avail": y_avail.squeeze(0), "y_pval": y_pval.squeeze(0),
                "y_peaks": y_peaks.squeeze(0), "y_dna": y_dna.squeeze(0),
            }
            yield sample
            epoch_finished = self._update_batch_pointers()

# ========= TEST FUNCTIONS =========
def test_CANDIDataHandler():
    print("--- Starting CANDIDataHandler Test ---")
    set_global_seed(42)
    
    # Path to your dataset. This test will fail if the path and its structure are not correct.
    base_path = "/home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED"

    if not os.path.exists(base_path):
        print(f"Error: The specified base_path '{base_path}' does not exist.")
        print("This test requires the CANDI dataset to be present at that location.")
        print("Skipping integration test.")
    else:
        handler = CANDIDataHandler(base_path=base_path, resolution=25, dataset_type="eic")
        handler.setup_datalooper(
            m=10, context_length=600 * 25, bios_batchsize=20, loci_batchsize=1,
            loci_gen_strategy="full_chr", split="train", bios_min_exp_avail_threshold=3,
            must_have_chr_access=True
        )

        print("\n--- Starting Epoch Simulation ---")
        handler.new_epoch()
        epoch_finished = False
        batch_count = 0
        while not epoch_finished:
            # Get X batch (input)
            t0 = time.time()
            x_batch = handler.get_batch(side="x")
            t1 = time.time()
            if x_batch is None: break 
            x_data, x_meta, x_avail, x_dna = x_batch

            # Get Y batch (target)
            t2 = time.time()
            y_batch = handler.get_batch(side="y", y_prompt=True)
            t3 = time.time()
            if y_batch is None: break
            y_data, y_meta, y_avail, y_pval, y_peaks, y_dna = y_batch
            
            print(f"Batch #{batch_count + 1}:")
            print(f"  X Data shape:   {x_data.shape} | X DNA shape: {x_dna.shape}")
            print(f"  Y Data shape:   {y_data.shape} | Y Pval shape: {y_pval.shape}")
            print(f"  Y Peaks shape: {y_peaks.shape} | Y DNA shape: {y_dna.shape}")
            print(f"  X batch loading time: {t1 - t0:.3f} sec | Y batch loading time: {t3 - t2:.3f} sec")
            
            t4 = time.time()
            epoch_finished = handler._update_batch_pointers()
            t5 = time.time()
            # Batch pointer update completed
            
            batch_count += 1
            if batch_count > 50: # Safety break
                print("Stopping test after 50 batches.")
                break
        
        # Epoch simulation finished

def test_parallelization():
    """Main function to launch the test."""
    # --- PLEASE EDIT THESE PARAMETERS ---
    # Path to your dataset
    base_path = "/home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED"
    # Number of GPUs to simulate
    world_size = 2 
    # ------------------------------------

    if not os.path.exists(base_path):
        print(f"❌ Error: The specified base_path '{base_path}' does not exist.")
        return

    # Use a small 'm' for a quick test run
    test_params = {
        "base_path": base_path,
        "dataset_type": "merged",
        "m": 100, # Generate 100 loci for the test
        "context_length": 600 * 25,
        "split": "train",
        "shuffle_bios": True, # Disable shuffle for a deterministic test
        "dsf_list": [1, 2],
        "DNA": True,
        "bios_batchsize": 1,
        "loci_batchsize": 1,
    }
    
    print("🚀 Starting parallelization test...")
    print(f"Simulating {world_size} GPUs with 2 data loader workers each.")
    print(f"Total consumers = {world_size * 2}")

    # Use a queue to gather results from each process
    queue = mp.Queue()
    mp.spawn(main_worker, args=(world_size, test_params, queue), nprocs=world_size, join=True)

    # --- Validate results ---
    all_processed_ids = []
    while not queue.empty():
        all_processed_ids.extend(queue.get())
    
    num_total_ids = len(all_processed_ids)
    num_unique_ids = len(set(all_processed_ids))

    print("\n--- 📊 Validation Results ---")
    print(f"Total samples processed across all workers: {num_total_ids}")
    print(f"Total unique samples processed: {num_unique_ids}")

    if num_total_ids == 0:
        print("🟡 Warning: No data was processed. Check dataset path and parameters.")
    elif num_total_ids == num_unique_ids:
        print("✅ Validation Successful! No duplicate samples were found across processes.")
    else:
        print(f"❌ Validation Failed! Found {num_total_ids - num_unique_ids} duplicate samples.")

def main_worker(rank, world_size, test_params, queue):
    """The function that runs on each simulated GPU process."""
    # --- Setup DDP Environment ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"--> Process [Rank {rank}] started.")

    # --- Instantiate Dataset and DataLoader ---
    # Each process creates its own dataset and dataloader
    dataset = CANDIIterableDataset(**test_params)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=25, num_workers=2)

    # --- Process data and collect sample IDs ---
    processed_ids_for_this_rank = []
    for i, batch in enumerate(data_loader):
        # The batch is a dictionary where each value is a tensor of size [batch_size, ...]
        # For sample_id, it's a tuple of two lists: (list_of_biosamples, list_of_loci)
        batch_ids = list(zip(*batch['sample_id']))
        processed_ids_for_this_rank.extend(batch_ids)

        print(batch.keys())
        
        # This printout explicitly shows each GPU processing its own unique batches in parallel
        batch_size = len(batch_ids)
        first_id = batch_ids[0]
        print(f"  [Rank {rank}] ---> Batch {i}: Received batch of size {batch_size}. First ID: {first_id}")
        
        # Simulate a small amount of GPU work
        time.sleep(0.01)

    # --- Report results and cleanup ---
    print(f"<-- Process [Rank {rank}] finished. It processed {len(processed_ids_for_this_rank)} total samples.")
    queue.put({rank: processed_ids_for_this_rank})
    dist.destroy_process_group()

def test_parallelization_and_uniqueness():
    """Main function to launch the test and validate the results."""
    # --- PLEASE EDIT THESE PARAMETERS ---
    base_path = "/home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED"
    world_size = 2 # Number of GPUs to simulate
    # ------------------------------------

    if not os.path.exists(base_path):
        print(f"❌ Error: The specified base_path '{base_path}' does not exist.")
        return

    # To solve potential shared memory issues on HPC login nodes
    torch.multiprocessing.set_sharing_strategy('file_system')

    test_params = {
        "base_path": base_path,
        "dataset_type": "merged",
        "m": 10,
        "context_length": 600 * 25,
        "split": "train",
        "shuffle_bios": False, # Disable shuffle for a deterministic test
        "dsf_list": [1, 2],
        "DNA": True,
        "bios_batchsize": 1,
        "loci_batchsize": 1,
    }
    
    print("🚀 Starting parallelization and uniqueness test...")
    print(f"Simulating {world_size} GPUs with 2 data loader workers each (Total consumers = {world_size * 2}).")

    queue = mp.Queue()
    mp.spawn(main_worker, args=(world_size, test_params, queue), nprocs=world_size, join=True)

    # --- Validate results from all processes ---
    results_by_rank = {}
    while not queue.empty():
        results_by_rank.update(queue.get())
    
    all_processed_ids = []
    for rank in sorted(results_by_rank.keys()):
        all_processed_ids.extend(results_by_rank[rank])
    
    num_total_ids = len(all_processed_ids)
    num_unique_ids = len(set(all_processed_ids))

    print("\n" + "="*50)
    print("📊 FINAL VALIDATION REPORT")
    print("="*50)
    
    # --- 1. Uniqueness / Redundancy Check ---
    print("\n## 1. Data Uniqueness Check")
    print(f"Total samples processed across all GPUs: {num_total_ids}")
    print(f"Total unique samples found: {num_unique_ids}")
    if num_total_ids > 0 and num_total_ids == num_unique_ids:
        print("✅ **SUCCESS**: All processed samples are unique. No data redundancy was found.")
    else:
        print(f"❌ **FAILURE**: Found {num_total_ids - num_unique_ids} duplicate samples. Data is NOT unique.")

    # --- 2. Efficiency / Parallelization Check ---
    print("\n## 2. Data Distribution Check (for Parallelization Efficiency)")
    if num_total_ids > 0:
        for rank, ids in sorted(results_by_rank.items()):
            percentage = (len(ids) / num_total_ids) * 100
            print(f"  - Rank {rank} processed {len(ids)} samples ({percentage:.2f}%)")
        print("✅ **SUCCESS**: The data was distributed across all processes, enabling parallel training.")
    else:
        print("🟡 **WARNING**: No data was processed. Cannot verify distribution.")
    print("="*50)

if __name__ == "__main__":
    # test_CANDIDataHandler()
    # test_parallelization()
    # test_parallelization_and_uniqueness()

    ds = CANDIDataHandler(
        base_path="/home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED", 
        resolution=25, dataset_type="merged", DNA=True,
        bios_batchsize=1, loci_batchsize=1, dsf_list=[1, 2, 4])

    ds.setup_datalooper(
        m=10, context_length=1200 * 25, bios_batchsize=1, loci_batchsize=1,
        loci_gen_strategy="random", split="train", bios_min_exp_avail_threshold=5,
        shuffle_bios=True, dsf_list=[1, 2, 4], must_have_chr_access=False)
    
    print(len(ds.navigation))
