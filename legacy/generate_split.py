#!/usr/bin/env python3
"""
Script to generate train-test split for merged dataset
"""

import pandas as pd
import numpy as np
import json
import os
import re
from collections import defaultdict, Counter

def main():
    print("Generating train-test split for merged dataset...")
    print("=" * 80)
    
    metadata_path = "data/merged_metadata.csv"
    split_path = "data/train_va_test_split_merged.json"
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    
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
    split_dict = {}
    
    # RNA-seq terms -> test
    for term in rna_seq_terms:
        for name in term_to_names[term]:
            split_dict[name] = "test"
    
    # Stratified test terms -> test
    for term in test_terms:
        for name in term_to_names[term]:
            split_dict[name] = "test"
    
    # Remaining terms -> train
    for term in train_terms:
        for name in term_to_names[term]:
            split_dict[name] = "train"
    
    # Validation
    train_count = sum(1 for v in split_dict.values() if v == "train")
    test_count = sum(1 for v in split_dict.values() if v == "test")
    
    print("\n" + "="*60)
    print("Train-Test Split Summary:")
    print("="*60)
    print(f"Total biosample_names: {len(split_dict)}")
    print(f"Train: {train_count} ({train_count/len(split_dict)*100:.1f}%)")
    print(f"Test: {test_count} ({test_count/len(split_dict)*100:.1f}%)")
    
    # Check for leakage
    train_terms_set = set()
    test_terms_set = set()
    for name, split in split_dict.items():
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
    rna_seq_in_test = sum(1 for name, split in split_dict.items() 
                          if split == "test" and biosample_info[name]['has_rna_seq'])
    print(f"✓ RNA-seq biosample_names in test: {rna_seq_in_test} / {rna_seq_biosample_count}")
    
    # Assay count distribution
    train_assays = [biosample_info[name]['num_assays'] for name, split in split_dict.items() if split == "train"]
    test_assays = [biosample_info[name]['num_assays'] for name, split in split_dict.items() if split == "test"]
    
    print(f"\nAssay count distribution:")
    print(f"  Train: min={min(train_assays)}, max={max(train_assays)}, mean={np.mean(train_assays):.1f}")
    print(f"  Test:  min={min(test_assays)}, max={max(test_assays)}, mean={np.mean(test_assays):.1f}")
    print("="*60)
    
    # Save split dictionary
    with open(split_path, 'w') as f:
        json.dump(split_dict, f, indent=2)
    
    print(f"\nSplit saved to: {split_path}")
    
    # Additional validation
    print("\n" + "=" * 80)
    print("Additional Validation:")
    print("=" * 80)
    
    # Check RNA-seq biosample_names
    rna_seq_names = df[df['assay_name'] == 'RNA-seq']['biosample_name'].unique()
    rna_seq_in_train = sum(1 for name in rna_seq_names if split_dict.get(name) == "train")
    
    if rna_seq_in_train > 0:
        print(f"❌ WARNING: {rna_seq_in_train} RNA-seq biosample_names in train set!")
        for name in rna_seq_names:
            if split_dict.get(name) == "train":
                print(f"    - {name}")
    else:
        print(f"✓ All RNA-seq biosample_names in test set!")
    
    print("\n" + "=" * 80)
    print("Split generation completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()

