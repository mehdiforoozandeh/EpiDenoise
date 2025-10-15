#!/usr/bin/env python3
"""
Visualize the train-test split stratification
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 8

def extract_biosample_term(name):
    match = re.match(r'^(.+?)(?:_grp\d+_rep\d+|_nonrep)$', name)
    return match.group(1) if match else name

# Load data
print("Loading data...")
df = pd.read_csv('data/merged_metadata.csv')

with open('data/train_va_test_split_merged.json', 'r') as f:
    split_dict = json.load(f)

# Compute biosample info
print("Computing biosample features...")
biosample_info = {}
for name in df['biosample_name'].unique():
    name_df = df[df['biosample_name'] == name]
    
    # Get mode sequencing_platform
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
    
    # Get split
    split = split_dict.get(name, 'unknown')
    
    biosample_info[name] = {
        'biosample_term': term,
        'sequencing_platform': platform,
        'lab': lab,
        'num_assays': num_assays,
        'has_rna_seq': has_rna_seq,
        'split': split
    }

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Overall split ratio (top left)
ax1 = fig.add_subplot(gs[0, 0])
train_count = sum(1 for v in split_dict.values() if v == 'train')
test_count = sum(1 for v in split_dict.values() if v == 'test')
colors_split = ['#3498db', '#e74c3c']
wedges, texts, autotexts = ax1.pie([train_count, test_count], 
                                     labels=['Train', 'Test'],
                                     autopct='%1.1f%%',
                                     colors=colors_split,
                                     startangle=90,
                                     textprops={'fontsize': 8, 'weight': 'bold'})
ax1.set_title('Overall Split Ratio\n(n=361 biosample_names)', fontsize=12, weight='bold', pad=20)

# 2. RNA-seq distribution (top middle)
ax2 = fig.add_subplot(gs[0, 1])
rna_seq_in_test = sum(1 for info in biosample_info.values() if info['has_rna_seq'] and info['split'] == 'test')
rna_seq_in_train = sum(1 for info in biosample_info.values() if info['has_rna_seq'] and info['split'] == 'train')
non_rna_test = test_count - rna_seq_in_test
non_rna_train = train_count - rna_seq_in_train

data_rna = {
    'Train': [non_rna_train, rna_seq_in_train],
    'Test': [non_rna_test, rna_seq_in_test]
}
x = np.arange(2)
width = 0.6
colors_rna = ['#95a5a6', '#f39c12']

bottom = np.zeros(2)
for i, (label, color) in enumerate(zip(['No RNA-seq', 'Has RNA-seq'], colors_rna)):
    values = [data_rna['Train'][i], data_rna['Test'][i]]
    ax2.bar(x, values, width, label=label, bottom=bottom, color=color)
    bottom += values

ax2.set_xticks(x)
ax2.set_xticklabels(['Train', 'Test'], fontsize=11, weight='bold')
ax2.set_ylabel('Number of biosample_names', fontsize=11)
ax2.set_title('RNA-seq Distribution', fontsize=12, weight='bold', pad=20)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# 3. Assay count distribution (top right)
ax3 = fig.add_subplot(gs[0, 2])
train_assays = [info['num_assays'] for info in biosample_info.values() if info['split'] == 'train']
test_assays = [info['num_assays'] for info in biosample_info.values() if info['split'] == 'test']

# Bin the assay counts
bins = [0, 6, 9, 12, 28]
bin_labels = ['4-6', '7-9', '10-12', '13+']
train_binned = np.histogram(train_assays, bins=bins)[0]
test_binned = np.histogram(test_assays, bins=bins)[0]

x_assay = np.arange(len(bin_labels))
width = 0.35

ax3.bar(x_assay - width/2, train_binned, width, label='Train', color=colors_split[0], alpha=0.8)
ax3.bar(x_assay + width/2, test_binned, width, label='Test', color=colors_split[1], alpha=0.8)

ax3.set_xticks(x_assay)
ax3.set_xticklabels(bin_labels, fontsize=10)
ax3.set_ylabel('Number of biosample_names', fontsize=11)
ax3.set_xlabel('Number of assays', fontsize=11)
ax3.set_title('Assay Count Distribution', fontsize=12, weight='bold', pad=20)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# 4. Sequencing platform distribution (middle row, spans 2 columns)
ax4 = fig.add_subplot(gs[1, :2])
train_platforms = [info['sequencing_platform'] for info in biosample_info.values() if info['split'] == 'train']
test_platforms = [info['sequencing_platform'] for info in biosample_info.values() if info['split'] == 'test']

train_platform_counts = Counter(train_platforms)
test_platform_counts = Counter(test_platforms)

# Get top platforms
all_platforms = sorted(set(train_platform_counts.keys()) | set(test_platform_counts.keys()))
platform_totals = [(p, train_platform_counts.get(p, 0) + test_platform_counts.get(p, 0)) for p in all_platforms]
platform_totals.sort(key=lambda x: x[1], reverse=True)
top_platforms = [p for p, _ in platform_totals[:8]]

train_vals = [train_platform_counts.get(p, 0) for p in top_platforms]
test_vals = [test_platform_counts.get(p, 0) for p in top_platforms]

x_plat = np.arange(len(top_platforms))
width = 0.35

ax4.bar(x_plat - width/2, train_vals, width, label='Train', color=colors_split[0], alpha=0.8)
ax4.bar(x_plat + width/2, test_vals, width, label='Test', color=colors_split[1], alpha=0.8)

# Shorten platform names for display
short_names = [p.replace('Illumina ', '').replace(' Analyzer', '').replace('Genome ', 'G.') for p in top_platforms]
ax4.set_xticks(x_plat)
ax4.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Number of biosample_names', fontsize=11)
ax4.set_title('Sequencing Platform Distribution (Top 8)', fontsize=12, weight='bold', pad=20)
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# 5. Lab distribution (middle right)
ax5 = fig.add_subplot(gs[1, 2])
train_labs = [info['lab'] for info in biosample_info.values() if info['split'] == 'train']
test_labs = [info['lab'] for info in biosample_info.values() if info['split'] == 'test']

train_lab_counts = Counter(train_labs)
test_lab_counts = Counter(test_labs)

# Get top 6 labs
all_labs = sorted(set(train_lab_counts.keys()) | set(test_lab_counts.keys()))
lab_totals = [(lab, train_lab_counts.get(lab, 0) + test_lab_counts.get(lab, 0)) for lab in all_labs]
lab_totals.sort(key=lambda x: x[1], reverse=True)
top_labs = [lab for lab, _ in lab_totals[:6]]

train_lab_vals = [train_lab_counts.get(lab, 0) for lab in top_labs]
test_lab_vals = [test_lab_counts.get(lab, 0) for lab in top_labs]

y_lab = np.arange(len(top_labs))
height = 0.35

ax5.barh(y_lab + height/2, train_lab_vals, height, label='Train', color=colors_split[0], alpha=0.8)
ax5.barh(y_lab - height/2, test_lab_vals, height, label='Test', color=colors_split[1], alpha=0.8)

# Shorten lab names
short_lab_names = [lab.split(',')[0] for lab in top_labs]
ax5.set_yticks(y_lab)
ax5.set_yticklabels(short_lab_names, fontsize=9)
ax5.set_xlabel('Number of biosample_names', fontsize=11)
ax5.set_title('Lab Distribution (Top 6)', fontsize=12, weight='bold', pad=20)
ax5.legend(loc='lower right', fontsize=10)
ax5.grid(axis='x', alpha=0.3)

# 6. Biosample term statistics (bottom left)
ax6 = fig.add_subplot(gs[2, 0])
train_terms = set()
test_terms = set()
for info in biosample_info.values():
    if info['split'] == 'train':
        train_terms.add(info['biosample_term'])
    else:
        test_terms.add(info['biosample_term'])

term_data = {
    'Train': [len(train_terms), train_count],
    'Test': [len(test_terms), test_count]
}

x_term = np.arange(2)
width = 0.6
colors_term = ['#9b59b6', '#1abc9c']

for i, (label, color) in enumerate(zip(['Unique terms', 'Total samples'], colors_term)):
    values = [term_data['Train'][i] if i == 0 else term_data['Train'][i]/2, 
              term_data['Test'][i] if i == 0 else term_data['Test'][i]/2]
    ax6.bar(x_term, [term_data['Train'][i], term_data['Test'][i]], width, 
            label=label, color=color, alpha=0.7, bottom=[0, 0] if i == 0 else [0, 0])

# Create custom bars
train_bars = [len(train_terms), train_count]
test_bars = [len(test_terms), test_count]

ax6.clear()
x_pos = [0, 1]
bar_width = 0.35

# Terms
ax6.bar([x_pos[0] - bar_width/2], [len(train_terms)], bar_width, label='Unique Terms (Train)', 
        color='#9b59b6', alpha=0.7)
ax6.bar([x_pos[1] - bar_width/2], [len(test_terms)], bar_width, label='Unique Terms (Test)', 
        color='#8e44ad', alpha=0.7)

# Samples
ax6.bar([x_pos[0] + bar_width/2], [train_count], bar_width, label='Total Samples (Train)', 
        color='#3498db', alpha=0.7)
ax6.bar([x_pos[1] + bar_width/2], [test_count], bar_width, label='Total Samples (Test)', 
        color='#e74c3c', alpha=0.7)

ax6.set_xticks(x_pos)
ax6.set_xticklabels(['Train', 'Test'], fontsize=11, weight='bold')
ax6.set_ylabel('Count', fontsize=11)
ax6.set_title('Terms vs Samples', fontsize=12, weight='bold', pad=20)
ax6.legend(loc='upper right', fontsize=9)
ax6.grid(axis='y', alpha=0.3)

# 7. Platform stratification percentages (bottom middle)
ax7 = fig.add_subplot(gs[2, 1])
platform_train_pcts = []
platform_test_pcts = []
platform_names_short = []

for platform in top_platforms[:6]:
    train_c = train_platform_counts.get(platform, 0)
    test_c = test_platform_counts.get(platform, 0)
    total = train_c + test_c
    if total > 0:
        platform_train_pcts.append(train_c / total * 100)
        platform_test_pcts.append(test_c / total * 100)
        short_name = platform.replace('Illumina ', '').replace(' Analyzer', '').replace('Genome ', 'G.')
        platform_names_short.append(short_name)

y_pos = np.arange(len(platform_names_short))
ax7.barh(y_pos, platform_train_pcts, label='Train %', color=colors_split[0], alpha=0.8)
ax7.barh(y_pos, platform_test_pcts, left=platform_train_pcts, label='Test %', 
         color=colors_split[1], alpha=0.8)

ax7.set_yticks(y_pos)
ax7.set_yticklabels(platform_names_short, fontsize=9)
ax7.set_xlabel('Percentage', fontsize=11)
ax7.set_xlim(0, 100)
ax7.set_title('Platform Stratification %', fontsize=12, weight='bold', pad=20)
ax7.legend(loc='lower right', fontsize=10)
ax7.grid(axis='x', alpha=0.3)

# Add reference line at 70%
ax7.axvline(x=70, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 8. Lab stratification percentages (bottom right)
ax8 = fig.add_subplot(gs[2, 2])
lab_train_pcts = []
lab_test_pcts = []
lab_names_short_pct = []

for lab in top_labs:
    train_c = train_lab_counts.get(lab, 0)
    test_c = test_lab_counts.get(lab, 0)
    total = train_c + test_c
    if total > 0:
        lab_train_pcts.append(train_c / total * 100)
        lab_test_pcts.append(test_c / total * 100)
        lab_names_short_pct.append(lab.split(',')[0])

y_pos_lab = np.arange(len(lab_names_short_pct))
ax8.barh(y_pos_lab, lab_train_pcts, label='Train %', color=colors_split[0], alpha=0.8)
ax8.barh(y_pos_lab, lab_test_pcts, left=lab_train_pcts, label='Test %', 
         color=colors_split[1], alpha=0.8)

ax8.set_yticks(y_pos_lab)
ax8.set_yticklabels(lab_names_short_pct, fontsize=9)
ax8.set_xlabel('Percentage', fontsize=11)
ax8.set_xlim(0, 100)
ax8.set_title('Lab Stratification %', fontsize=12, weight='bold', pad=20)
ax8.legend(loc='lower right', fontsize=10)
ax8.grid(axis='x', alpha=0.3)

# Add reference line at 70%
ax8.axvline(x=70, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Add overall title
fig.suptitle('Train-Test Split Stratification Analysis\nMerged Dataset (361 biosample_names, 164 unique terms)', 
             fontsize=16, weight='bold', y=0.98)

# Save figure
output_path = 'figures/train_test_split_stratification.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to: {output_path}")

plt.close()
print("Done!")

