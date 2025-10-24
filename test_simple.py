#!/usr/bin/env python3
"""
Simple integration test that avoids scipy dependencies.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports without scipy dependencies."""
    print("Testing basic imports...")
    
    try:
        # Test that we can import the core modules
        import torch
        print("✓ PyTorch imported successfully")
        
        # Test our modules (they should import even if scipy fails later)
        try:
            import pred
            print("✓ pred.py module structure is valid")
        except Exception as e:
            print(f"✗ pred.py import failed: {e}")
            return False
        
        try:
            import viz
            print("✓ viz.py module structure is valid")
        except Exception as e:
            print(f"✗ viz.py import failed: {e}")
            return False
        
        try:
            import eval
            print("✓ eval.py module structure is valid")
        except Exception as e:
            print(f"✗ eval.py import failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'pred.py',
        'viz.py', 
        'eval.py',
        'test_integration.py',
        'test_simple.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            return False
    
    return True

def test_syntax():
    """Test that Python files have valid syntax."""
    print("\nTesting syntax...")
    
    python_files = ['pred.py', 'viz.py', 'eval.py']
    
    for file in python_files:
        try:
            with open(file, 'r') as f:
                compile(f.read(), file, 'exec')
            print(f"✓ {file} has valid syntax")
        except SyntaxError as e:
            print(f"✗ {file} has syntax error: {e}")
            return False
        except Exception as e:
            print(f"✗ {file} compilation failed: {e}")
            return False
    
    return True

def main():
    """Run simple integration tests."""
    print("=" * 50)
    print("Simple Integration Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_syntax,
        test_basic_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Simple Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 Basic tests passed! The refactored modules are structurally sound.")
        print("Note: Full functionality testing requires proper environment setup.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


