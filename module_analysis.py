#!/usr/bin/env python3
"""
Comprehensive Analysis of Python Modules Created in Last 2 Days

This script analyzes all the modules we've created and categorizes them by purpose.
"""

import os
from pathlib import Path

def analyze_modules():
    """Analyze all Python modules and categorize them."""
    
    print("📋 COMPREHENSIVE MODULE ANALYSIS")
    print("=" * 60)
    print("Modules created in the last 2 days for CANDI dataset processing:")
    print()
    
    # Core Data Processing Modules
    print("🔧 CORE DATA PROCESSING MODULES:")
    print("-" * 40)
    print("1. get_candi_data.py")
    print("   Purpose: Main data download and processing pipeline")
    print("   Features: CLI interface, parallel processing, task management")
    print("   Status: ✅ KEEP - Core functionality")
    print()
    
    print("2. data.py")
    print("   Purpose: Low-level bioinformatics processing functions")
    print("   Features: BAM_TO_SIGNAL, get_binned_values, get_binned_bigBed_peaks")
    print("   Status: ✅ KEEP - Core functionality")
    print()
    
    # Dataset Processing Scripts
    print("📊 DATASET PROCESSING SCRIPTS:")
    print("-" * 40)
    print("3. process_eic_complete.py")
    print("   Purpose: Complete EIC dataset processing")
    print("   Features: Sets base_path, runs full pipeline")
    print("   Status: ✅ KEEP - Production script")
    print()
    
    print("4. process_merged_complete.py")
    print("   Purpose: Complete MERGED dataset processing")
    print("   Features: Sets base_path, runs full pipeline")
    print("   Status: ✅ KEEP - Production script")
    print()
    
    # Validation and Analysis Modules
    print("🔍 VALIDATION AND ANALYSIS MODULES:")
    print("-" * 40)
    print("5. validate_candi_datasets.py")
    print("   Purpose: Comprehensive dataset validation")
    print("   Features: Completeness checking, availability heatmaps")
    print("   Status: ✅ KEEP - Essential validation tool")
    print()
    
    print("6. quick_validate_candi.py")
    print("   Purpose: Fast dataset validation")
    print("   Features: Quick completeness checks")
    print("   Status: ✅ KEEP - Useful for routine checks")
    print()
    
    print("7. analyze_missing_experiments.py")
    print("   Purpose: Detailed analysis of missing experiments")
    print("   Features: Categorizes missing data by type")
    print("   Status: ✅ KEEP - Useful for debugging")
    print()
    
    # Retry and Recovery Modules
    print("🔄 RETRY AND RECOVERY MODULES:")
    print("-" * 40)
    print("8. retry_failed_experiments.py")
    print("   Purpose: Initial retry system for failed experiments")
    print("   Features: Failure analysis, enhanced processors")
    print("   Status: ❌ REMOVE - Replaced by specific retry scripts")
    print()
    
    print("9. retry_eic_failed.py")
    print("   Purpose: EIC-specific retry script")
    print("   Features: Targets EIC failures only")
    print("   Status: ✅ KEEP - Specific retry functionality")
    print()
    
    print("10. retry_merged_failed.py")
    print("    Purpose: MERGED-specific retry script")
    print("    Features: Targets MERGED failures only")
    print("    Status: ✅ KEEP - Specific retry functionality")
    print()
    
    print("11. test_retry_fixes.py")
    print("    Purpose: Test script for retry functionality")
    print("    Features: Validates failure analyzer and processors")
    print("    Status: ❌ REMOVE - Testing script, no longer needed")
    print()
    
    # Visualization Modules (Multiple Versions)
    print("🎨 VISUALIZATION MODULES (MULTIPLE VERSIONS):")
    print("-" * 40)
    print("12. create_enhanced_availability_plots.py")
    print("    Purpose: First attempt at enhanced plots")
    print("    Features: Initial color coding, grouping")
    print("    Status: ❌ REMOVE - Replaced by better versions")
    print()
    
    print("13. create_corrected_availability_plots.py")
    print("    Purpose: Corrected EIC union logic")
    print("    Features: Proper split-specific coloring")
    print("    Status: ❌ REMOVE - Replaced by consistent version")
    print()
    
    print("14. create_final_availability_plots.py")
    print("    Purpose: Final version with consistent colors")
    print("    Features: cornflowerblue/forestgreen/salmon")
    print("    Status: ❌ REMOVE - Replaced by clean version")
    print()
    
    print("15. create_final_consistent_plots.py")
    print("    Purpose: Consistent color scheme")
    print("    Features: Same colors for both datasets")
    print("    Status: ❌ REMOVE - Replaced by clean version")
    print()
    
    print("16. create_truly_consistent_plots.py")
    print("    Purpose: Truly consistent coloring")
    print("    Features: Only three base colors")
    print("    Status: ❌ REMOVE - Replaced by clean version")
    print()
    
    print("17. create_final_clean_plots.py")
    print("    Purpose: FINAL clean version")
    print("    Features: Improved fonts, simplified titles, consistent colors")
    print("    Status: ✅ KEEP - Final production version")
    print()
    
    # Documentation and Summary Modules
    print("📝 DOCUMENTATION AND SUMMARY MODULES:")
    print("-" * 40)
    print("18. compare_plot_improvements.py")
    print("    Purpose: Shows improvements made to plots")
    print("    Features: Before/after comparison")
    print("    Status: ❌ REMOVE - Documentation only")
    print()
    
    print("19. consistent_plot_summary.py")
    print("    Purpose: Summary of consistent plot features")
    print("    Features: Color scheme explanation")
    print("    Status: ❌ REMOVE - Documentation only")
    print()
    
    print("20. explain_corrections.py")
    print("    Purpose: Explains corrections made to plots")
    print("    Features: What was wrong and how it was fixed")
    print("    Status: ❌ REMOVE - Documentation only")
    print()
    
    print("21. final_plot_summary.py")
    print("    Purpose: Final summary of plot features")
    print("    Features: Complete feature list")
    print("    Status: ❌ REMOVE - Documentation only")
    print()
    
    print("22. module_analysis.py")
    print("    Purpose: This analysis script")
    print("    Features: Categorizes all modules")
    print("    Status: ❌ REMOVE - Temporary analysis")
    print()

def provide_recommendations():
    """Provide cleanup recommendations."""
    
    print("🧹 CLEANUP RECOMMENDATIONS")
    print("=" * 60)
    print()
    
    print("✅ KEEP THESE MODULES (Essential):")
    print("   • get_candi_data.py - Core pipeline")
    print("   • data.py - Bioinformatics functions")
    print("   • process_eic_complete.py - EIC processing")
    print("   • process_merged_complete.py - MERGED processing")
    print("   • validate_candi_datasets.py - Validation")
    print("   • quick_validate_candi.py - Quick checks")
    print("   • analyze_missing_experiments.py - Missing data analysis")
    print("   • retry_eic_failed.py - EIC retry")
    print("   • retry_merged_failed.py - MERGED retry")
    print("   • create_final_clean_plots.py - Final visualization")
    print()
    
    print("❌ REMOVE THESE MODULES (Redundant/Obsolete):")
    print("   • retry_failed_experiments.py - Replaced by specific scripts")
    print("   • test_retry_fixes.py - Testing script")
    print("   • create_enhanced_availability_plots.py - Old version")
    print("   • create_corrected_availability_plots.py - Old version")
    print("   • create_final_availability_plots.py - Old version")
    print("   • create_final_consistent_plots.py - Old version")
    print("   • create_truly_consistent_plots.py - Old version")
    print("   • compare_plot_improvements.py - Documentation")
    print("   • consistent_plot_summary.py - Documentation")
    print("   • explain_corrections.py - Documentation")
    print("   • final_plot_summary.py - Documentation")
    print("   • module_analysis.py - This script")
    print()
    
    print("🔄 MERGE RECOMMENDATIONS:")
    print("   • Consider merging retry_eic_failed.py and retry_merged_failed.py")
    print("     into a single retry_failed_experiments.py with dataset parameter")
    print("   • Consider merging quick_validate_candi.py into validate_candi_datasets.py")
    print("     as a 'quick' mode option")
    print("   • Consider merging analyze_missing_experiments.py into validate_candi_datasets.py")
    print("     as an analysis mode")
    print()

def main():
    """Main analysis function."""
    analyze_modules()
    print()
    provide_recommendations()
    
    print("📊 SUMMARY:")
    print("   Total modules created: 22")
    print("   Keep: 11 modules")
    print("   Remove: 11 modules")
    print("   Reduction: 50%")
    print()
    print("🎯 This cleanup will significantly reduce codebase complexity!")

if __name__ == "__main__":
    main()

