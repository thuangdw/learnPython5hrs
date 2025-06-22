#!/usr/bin/env python3
"""
Test Runner for Data Structures and Algorithms Project
Run all tests and show implementation status
"""

import unittest
import sys
import os
from pathlib import Path

def discover_and_run_tests():
    """Discover and run all test files in the project"""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    print("=" * 60)
    print("Data Structures and Algorithms Test Runner")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(project_root)
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION STATUS")
    print("=" * 60)
    
    # Check what's been implemented
    implemented_data_structures = []
    implemented_algorithms = []
    
    # Check data structures
    ds_path = project_root / "datastructure"
    if ds_path.exists():
        for item in ds_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                py_file = item / f"{item.name}.py"
                test_file = item / f"test_{item.name}.py"
                if py_file.exists() and test_file.exists():
                    implemented_data_structures.append(item.name)
    
    # Check algorithms
    algo_path = project_root / "algorithm"
    if algo_path.exists():
        for item in algo_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                py_file = item / f"{item.name}.py"
                test_file = item / f"test_{item.name}.py"
                if py_file.exists() and test_file.exists():
                    implemented_algorithms.append(item.name)
    
    print("\n✅ COMPLETED DATA STRUCTURES:")
    for ds in implemented_data_structures:
        print(f"   • {ds.replace('_', ' ').title()}")
    
    print(f"\n   Total: {len(implemented_data_structures)} data structures")
    
    print("\n✅ COMPLETED ALGORITHMS:")
    for algo in implemented_algorithms:
        print(f"   • {algo.replace('_', ' ').title()}")
    
    print(f"\n   Total: {len(implemented_algorithms)} algorithm categories")
    
    # Show remaining work
    all_data_structures = [
        "arrays", "stacks", "queues", "linked_lists", "trees", 
        "graphs", "hash_tables", "heaps", "sets", "tries"
    ]
    
    all_algorithms = [
        "sorting", "searching", "graph_algorithms", "dynamic_programming",
        "greedy", "divide_conquer", "backtracking", "string_algorithms",
        "mathematical", "bit_manipulation"
    ]
    
    remaining_ds = [ds for ds in all_data_structures if ds not in implemented_data_structures]
    remaining_algo = [algo for algo in all_algorithms if algo not in implemented_algorithms]
    
    if remaining_ds:
        print("\n🚧 REMAINING DATA STRUCTURES:")
        for ds in remaining_ds:
            print(f"   • {ds.replace('_', ' ').title()}")
    
    if remaining_algo:
        print("\n🚧 REMAINING ALGORITHMS:")
        for algo in remaining_algo:
            print(f"   • {algo.replace('_', ' ').title()}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)) * 100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback.split('\\n')[-2]}")
    
    return result.wasSuccessful()


def show_project_structure():
    """Display the current project structure"""
    project_root = Path(__file__).parent
    
    print("\n" + "=" * 60)
    print("PROJECT STRUCTURE")
    print("=" * 60)
    
    def print_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = sorted([item for item in path.iterdir() if not item.name.startswith('.')])
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
    
    print_tree(project_root)


if __name__ == "__main__":
    print("Running comprehensive tests...\n")
    
    # Show project structure
    show_project_structure()
    
    # Run tests
    success = discover_and_run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ Some tests failed. Check the details above.")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 