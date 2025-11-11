#!/usr/bin/env python3
"""
Generate and modify custom fill-in-prompt JSON files for CANDI model inference.

This module provides a CLI tool to:
1. Generate default prompt templates from metadata CSVs (merged or eic datasets)
2. Modify existing prompt files to customize metadata for specific assays
3. Save prompt files to the prompts/ directory

The generated JSON files are compatible with the fill_in_prompt_manual method
in data.py, which expects the format:
{
  "assay_name": {
    "depth": <float>,              # Raw depth (will be converted to log2)
    "sequencing_platform": <str>,  # Platform name
    "read_length": <float>,        # Read length in bp
    "run_type": <str>              # "single-ended" or "paired-ended"
  },
  ...
}
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def compute_assay_statistics(metadata_df, assay_name):
    """
    Compute statistics for a single assay from metadata DataFrame.
    
    Args:
        metadata_df: DataFrame with columns including assay_name, depth, 
                     sequencing_platform, read_length, run_type
        assay_name: Name of the assay to compute statistics for
    
    Returns:
        dict with keys: depth (median), sequencing_platform (mode), 
                       read_length (median), run_type (mode)
    """
    assay_df = metadata_df[metadata_df['assay_name'] == assay_name]
    
    if assay_df.empty:
        return None
    
    # Depth: use median (raw depth, not log2)
    depth_vals = assay_df['depth'].dropna().astype(float) if 'depth' in assay_df.columns else pd.Series(dtype=float)
    depth_median = float(np.nanmedian(depth_vals)) if len(depth_vals) > 0 else None
    
    # Sequencing platform: use mode (most frequent)
    platforms = assay_df['sequencing_platform'].dropna().astype(str).values
    platform_mode = Counter(platforms).most_common(1)[0][0] if len(platforms) > 0 else None
    
    # Read length: use median
    read_length_vals = assay_df['read_length'].dropna().astype(float) if 'read_length' in assay_df.columns else pd.Series(dtype=float)
    read_length_median = float(np.nanmedian(read_length_vals)) if len(read_length_vals) > 0 else None
    
    # Run type: use mode (most frequent)
    run_types = assay_df['run_type'].dropna().astype(str).values
    run_type_mode = Counter(run_types).most_common(1)[0][0] if len(run_types) > 0 else None
    
    return {
        "depth": depth_median,
        "sequencing_platform": platform_mode,
        "read_length": read_length_median,
        "run_type": run_type_mode
    }


def generate_default_template(dataset_type, metadata_path=None):
    """
    Generate default prompt template from metadata CSV.
    
    Args:
        dataset_type: Either "merged" or "eic"
        metadata_path: Optional path to metadata CSV. If None, uses default path.
    
    Returns:
        dict mapping assay names to their default metadata statistics
    """
    if metadata_path is None:
        if dataset_type == "merged":
            metadata_path = "data/merged_metadata.csv"
        elif dataset_type == "eic":
            metadata_path = "data/eic_metadata.csv"
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'merged' or 'eic'")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Read metadata CSV
    metadata_df = pd.read_csv(metadata_path)
    
    # Get unique assays
    unique_assays = metadata_df['assay_name'].unique()
    
    # Compute statistics for each assay
    template = {}
    for assay in unique_assays:
        stats = compute_assay_statistics(metadata_df, assay)
        if stats is not None:
            # Only include if all required fields are present
            if all(v is not None for v in stats.values()):
                template[assay] = stats
    
    return template


def load_prompt_file(filepath):
    """Load a prompt JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_prompt_file(prompt_dict, filepath):
    """Save a prompt dictionary to a JSON file with pretty formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(prompt_dict, f, indent=2)


def modify_prompt(prompt_dict, assay_name, depth=None, read_length=None, 
                  sequencing_platform=None, run_type=None):
    """
    Modify a prompt dictionary for a specific assay.
    
    Args:
        prompt_dict: Existing prompt dictionary
        assay_name: Name of assay to modify
        depth: Optional depth value to set
        read_length: Optional read length to set
        sequencing_platform: Optional platform name to set
        run_type: Optional run type to set ("single-ended" or "paired-ended")
    
    Returns:
        Modified prompt dictionary
    """
    if assay_name not in prompt_dict:
        # Create new entry if assay doesn't exist
        prompt_dict[assay_name] = {}
    
    if depth is not None:
        prompt_dict[assay_name]["depth"] = float(depth)
    if read_length is not None:
        prompt_dict[assay_name]["read_length"] = float(read_length)
    if sequencing_platform is not None:
        prompt_dict[assay_name]["sequencing_platform"] = str(sequencing_platform)
    if run_type is not None:
        prompt_dict[assay_name]["run_type"] = str(run_type)
    
    return prompt_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate and modify custom fill-in-prompt JSON files for CANDI model inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default template for merged dataset
  %(prog)s generate --dataset merged

  # Generate default template with custom output filename
  %(prog)s generate --dataset eic --output prompts/my_custom_prompt.json

  # Modify an existing prompt file
  %(prog)s modify --input prompts/merged_mode.json --assay ATAC-seq --depth 100000000 --read-length 150

  # Generate and modify in one command
  %(prog)s generate --dataset merged --modify ATAC-seq --depth 100000000 --output prompts/custom.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate default prompt template from metadata')
    gen_parser.add_argument('--dataset', type=str, choices=['merged', 'eic'], required=True,
                          help='Dataset type: merged or eic')
    gen_parser.add_argument('--output', type=str, default=None,
                          help='Output filename (default: prompts/{dataset}_mode.json)')
    gen_parser.add_argument('--metadata-path', type=str, default=None,
                          help='Custom path to metadata CSV file (overrides default)')
    
    # Add modify options to generate command
    gen_parser.add_argument('--modify', type=str, metavar='ASSAY', default=None,
                          help='Assay name to modify after generation')
    gen_parser.add_argument('--depth', type=float, default=None,
                          help='Depth value for modified assay')
    gen_parser.add_argument('--read-length', type=float, default=None,
                          help='Read length for modified assay')
    gen_parser.add_argument('--sequencing-platform', type=str, default=None,
                          help='Sequencing platform for modified assay')
    gen_parser.add_argument('--run-type', type=str, choices=['single-ended', 'paired-ended'], default=None,
                          help='Run type for modified assay')
    
    # Modify command
    mod_parser = subparsers.add_parser('modify', help='Modify an existing prompt file')
    mod_parser.add_argument('--input', type=str, required=True,
                           help='Input prompt JSON file to modify')
    mod_parser.add_argument('--assay', type=str, required=True,
                           help='Assay name to modify')
    mod_parser.add_argument('--depth', type=float, default=None,
                           help='Depth value to set')
    mod_parser.add_argument('--read-length', type=float, default=None,
                           help='Read length to set')
    mod_parser.add_argument('--sequencing-platform', type=str, default=None,
                           help='Sequencing platform to set')
    mod_parser.add_argument('--run-type', type=str, choices=['single-ended', 'paired-ended'], default=None,
                           help='Run type to set')
    mod_parser.add_argument('--output', type=str, default=None,
                           help='Output filename (default: overwrites input file)')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        # Generate default template
        print(f"Generating default template for {args.dataset} dataset...")
        template = generate_default_template(args.dataset, args.metadata_path)
        print(f"Generated template with {len(template)} assays")
        
        # Apply modifications if requested
        if args.modify:
            print(f"Modifying assay: {args.modify}")
            template = modify_prompt(
                template, 
                args.modify,
                depth=args.depth,
                read_length=args.read_length,
                sequencing_platform=args.sequencing_platform,
                run_type=args.run_type
            )
        
        # Determine output filename
        if args.output:
            output_file = args.output
        else:
            output_file = f"prompts/{args.dataset}_mode.json"
        
        # Ensure prompts directory exists
        os.makedirs("prompts", exist_ok=True)
        
        # Save template
        save_prompt_file(template, output_file)
        print(f"Saved template to {output_file}")
        
    elif args.command == 'modify':
        # Load existing prompt file
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        
        print(f"Loading prompt file: {args.input}")
        prompt_dict = load_prompt_file(args.input)
        
        # Apply modifications
        print(f"Modifying assay: {args.assay}")
        prompt_dict = modify_prompt(
            prompt_dict,
            args.assay,
            depth=args.depth,
            read_length=args.read_length,
            sequencing_platform=args.sequencing_platform,
            run_type=args.run_type
        )
        
        # Determine output filename
        output_file = args.output if args.output else args.input
        
        # Save modified prompt
        save_prompt_file(prompt_dict, output_file)
        print(f"Saved modified prompt to {output_file}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

