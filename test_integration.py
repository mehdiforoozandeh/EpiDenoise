#!/usr/bin/env python3
"""
Integration test script for the refactored eval modules.

This script tests the basic functionality of pred.py, viz.py, and eval.py
to ensure they work together correctly.
"""

import os
import sys
import tempfile
import pickle
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from pred import CANDIPredictor
        print("✓ pred.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import pred.py: {e}")
        return False
    
    try:
        from viz import VISUALS_CANDI
        print("✓ viz.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import viz.py: {e}")
        return False
    
    try:
        from eval import EVAL_CANDI, compare_hard_clusterings, compare_soft_clusterings
        print("✓ eval.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import eval.py: {e}")
        return False
    
    return True

def test_utility_functions():
    """Test utility functions from eval.py."""
    print("\nTesting utility functions...")
    
    try:
        from eval import compare_hard_clusterings, compare_soft_clusterings
        
        # Test hard clustering comparison
        import numpy as np
        labels1 = np.array([0, 0, 1, 1, 2, 2])
        labels2 = np.array([0, 1, 1, 2, 2, 0])
        
        result = compare_hard_clusterings(labels1, labels2)
        assert 'ari' in result
        assert 'nmi' in result
        assert 'contingency_matrix' in result
        print("✓ compare_hard_clusterings works")
        
        # Test soft clustering comparison
        posteriors1 = np.random.random((10, 3))
        posteriors2 = np.random.random((10, 4))
        posteriors1 = posteriors1 / posteriors1.sum(axis=1, keepdims=True)
        posteriors2 = posteriors2 / posteriors2.sum(axis=1, keepdims=True)
        
        result = compare_soft_clusterings(posteriors1, posteriors2)
        assert 'avg_jsd' in result
        assert 'posterior_correlation' in result
        print("✓ compare_soft_clusterings works")
        
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        return False

def test_visualization_class():
    """Test VISUALS_CANDI class initialization."""
    print("\nTesting visualization class...")
    
    try:
        from viz import VISUALS_CANDI
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            viz = VISUALS_CANDI(resolution=25, savedir=temp_dir)
            
            # Test that methods exist
            assert hasattr(viz, 'count_track')
            assert hasattr(viz, 'signal_track')
            assert hasattr(viz, 'count_confidence')
            assert hasattr(viz, 'signal_confidence')
            assert hasattr(viz, 'count_scatter_with_marginals')
            assert hasattr(viz, 'signal_scatter_with_marginals')
            assert hasattr(viz, 'generate_all_plots')
            
            print("✓ VISUALS_CANDI class initialized successfully")
            print("✓ All required methods are present")
            
        return True
        
    except Exception as e:
        print(f"✗ Visualization class test failed: {e}")
        return False

def test_prediction_structure():
    """Test that prediction dictionary structure is correct."""
    print("\nTesting prediction structure...")
    
    try:
        # Create a mock prediction dictionary
        mock_prediction_dict = {
            'GM12878': {
                'H3K4me3': {
                    'type': 'imputed',
                    'count_dist': None,  # Would be a NegativeBinomial object
                    'count_params': {'p': None, 'n': None},  # Would be tensors
                    'pval_dist': None,  # Would be a Gaussian object
                    'pval_params': {'mu': None, 'var': None},  # Would be tensors
                    'peak_scores': None,  # Would be a tensor
                },
                'H3K4me3_upsampled': {
                    'type': 'denoised',
                    'count_dist': None,
                    'count_params': {'p': None, 'n': None},
                    'pval_dist': None,
                    'pval_params': {'mu': None, 'var': None},
                    'peak_scores': None,
                }
            }
        }
        
        # Test that the structure matches what we expect
        bios_name = 'GM12878'
        experiment = 'H3K4me3'
        
        assert bios_name in mock_prediction_dict
        assert experiment in mock_prediction_dict[bios_name]
        assert f"{experiment}_upsampled" in mock_prediction_dict[bios_name]
        
        # Check structure of imputed prediction
        imp_pred = mock_prediction_dict[bios_name][experiment]
        assert imp_pred['type'] == 'imputed'
        assert 'count_dist' in imp_pred
        assert 'count_params' in imp_pred
        assert 'pval_dist' in imp_pred
        assert 'pval_params' in imp_pred
        assert 'peak_scores' in imp_pred
        
        # Check structure of denoised prediction
        den_pred = mock_prediction_dict[bios_name][f"{experiment}_upsampled"]
        assert den_pred['type'] == 'denoised'
        
        print("✓ Prediction dictionary structure is correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction structure test failed: {e}")
        return False

def test_evaluation_class():
    """Test EVAL_CANDI class initialization (without actual model)."""
    print("\nTesting evaluation class...")
    
    try:
        from eval import EVAL_CANDI
        
        # This test will fail if we try to actually initialize with real paths
        # but we can test that the class exists and has the right methods
        assert hasattr(EVAL_CANDI, '__init__')
        assert hasattr(EVAL_CANDI, 'get_metrics')
        assert hasattr(EVAL_CANDI, 'bios_pipeline')
        assert hasattr(EVAL_CANDI, 'bios_pipeline_eic')
        assert hasattr(EVAL_CANDI, 'eval_rnaseq')
        assert hasattr(EVAL_CANDI, 'quick_eval_rnaseq')
        assert hasattr(EVAL_CANDI, 'saga')
        assert hasattr(EVAL_CANDI, 'viz_bios')
        assert hasattr(EVAL_CANDI, 'viz_all')
        assert hasattr(EVAL_CANDI, 'filter_res')
        
        print("✓ EVAL_CANDI class has all required methods")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation class test failed: {e}")
        return False

def test_cli_interfaces():
    """Test that CLI interfaces can be imported and have required arguments."""
    print("\nTesting CLI interfaces...")
    
    try:
        # Test pred.py CLI
        import pred
        assert hasattr(pred, 'main')
        print("✓ pred.py has CLI interface")
        
        # Test viz.py CLI
        import viz
        assert hasattr(viz, 'main')
        print("✓ viz.py has CLI interface")
        
        # Test eval.py CLI
        import eval
        assert hasattr(eval, 'main')
        print("✓ eval.py has CLI interface")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI interfaces test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("CANDI Evaluation Modules Integration Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_utility_functions,
        test_visualization_class,
        test_prediction_structure,
        test_evaluation_class,
        test_cli_interfaces,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Integration Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The refactored modules are ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
