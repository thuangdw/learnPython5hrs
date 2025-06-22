"""
Algorithm Test Runner
Comprehensive test runner for all algorithm implementations
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add algorithm directories to path
algorithm_dirs = [
    'sorting', 'searching', 'dynamic_programming', 'greedy',
    'divide_and_conquer', 'backtracking', 'string_algorithms',
    'mathematical', 'bit_manipulation', 'graph_algorithms'
]

for dir_name in algorithm_dirs:
    sys.path.append(os.path.join(os.path.dirname(__file__), dir_name))


def run_algorithm_tests():
    """Run all algorithm tests and generate comprehensive report"""
    
    print("=" * 80)
    print("ALGORITHM IMPLEMENTATIONS TEST SUITE")
    print("Senior Python Developer Guide")
    print("=" * 80)
    print()
    
    # Test results tracking
    test_results = {}
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    # Define test modules to run
    test_modules = [
        ('Sorting Algorithms', 'sorting.test_sorting'),
        ('Searching Algorithms', 'searching.test_searching'),
        ('Dynamic Programming', 'dynamic_programming.test_dynamic_programming'),
        ('Greedy Algorithms', 'greedy.test_greedy'),
        ('Divide and Conquer', 'divide_and_conquer.test_divide_and_conquer'),
        ('Backtracking', 'backtracking.test_backtracking'),
        ('String Algorithms', 'string_algorithms.test_string_algorithms'),
        ('Mathematical Algorithms', 'mathematical.test_mathematical'),
        ('Bit Manipulation', 'bit_manipulation.test_bit_manipulation'),
        ('Graph Algorithms', 'graph_algorithms.test_graph_algorithms')
    ]
    
    for category_name, module_name in test_modules:
        print(f"Testing {category_name}...")
        print("-" * 60)
        
        try:
            # Import the test module
            test_module = __import__(module_name, fromlist=[''])
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests with custom result handler
            stream = StringIO()
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=2,
                buffer=True
            )
            
            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()
            
            # Record results
            test_results[category_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
                'execution_time': end_time - start_time,
                'status': 'PASS' if len(result.failures) == 0 and len(result.errors) == 0 else 'FAIL'
            }
            
            # Update totals
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            
            # Print summary for this category
            print(f"  Tests Run: {result.testsRun}")
            print(f"  Failures: {len(result.failures)}")
            print(f"  Errors: {len(result.errors)}")
            print(f"  Success Rate: {test_results[category_name]['success_rate']:.1f}%")
            print(f"  Execution Time: {test_results[category_name]['execution_time']:.3f}s")
            print(f"  Status: {test_results[category_name]['status']}")
            
            # Print failure details if any
            if result.failures:
                print("\n  FAILURES:")
                for test, traceback in result.failures:
                    print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip()}")
            
            if result.errors:
                print("\n  ERRORS:")
                for test, traceback in result.errors:
                    print(f"    - {test}: {traceback.split('Error:')[-1].strip()}")
            
        except ImportError as e:
            print(f"  ERROR: Could not import {module_name}")
            print(f"  Reason: {e}")
            test_results[category_name] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success_rate': 0,
                'execution_time': 0,
                'status': 'IMPORT_ERROR'
            }
            total_errors += 1
        
        except Exception as e:
            print(f"  ERROR: Unexpected error in {category_name}")
            print(f"  Reason: {e}")
            test_results[category_name] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success_rate': 0,
                'execution_time': 0,
                'status': 'ERROR'
            }
            total_errors += 1
        
        print()
    
    # Generate comprehensive summary report
    print("=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    print()
    
    # Summary table
    print(f"{'Category':<25} {'Tests':<8} {'Pass':<8} {'Fail':<8} {'Error':<8} {'Rate':<8} {'Time':<8} {'Status':<12}")
    print("-" * 95)
    
    for category, results in test_results.items():
        passed = results['tests_run'] - results['failures'] - results['errors']
        print(f"{category:<25} {results['tests_run']:<8} {passed:<8} {results['failures']:<8} "
              f"{results['errors']:<8} {results['success_rate']:<7.1f}% {results['execution_time']:<7.3f}s {results['status']:<12}")
    
    print("-" * 95)
    
    # Overall statistics
    total_passed = total_tests - total_failures - total_errors
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"{'TOTAL':<25} {total_tests:<8} {total_passed:<8} {total_failures:<8} "
          f"{total_errors:<8} {overall_success_rate:<7.1f}% {'':<8} {'SUMMARY':<12}")
    
    print()
    print("IMPLEMENTATION STATUS:")
    print("-" * 40)
    
    # Count implementation status
    implemented = sum(1 for results in test_results.values() if results['status'] in ['PASS', 'FAIL'])
    not_implemented = sum(1 for results in test_results.values() if results['status'] in ['IMPORT_ERROR', 'ERROR'])
    
    print(f"✅ Implemented Categories: {implemented}/{len(test_results)}")
    print(f"❌ Not Implemented: {not_implemented}/{len(test_results)}")
    print(f"🎯 Overall Progress: {implemented/len(test_results)*100:.1f}%")
    
    if overall_success_rate >= 90:
        print(f"🏆 Excellent! Success rate: {overall_success_rate:.1f}%")
    elif overall_success_rate >= 75:
        print(f"✅ Good! Success rate: {overall_success_rate:.1f}%")
    elif overall_success_rate >= 50:
        print(f"⚠️  Needs improvement. Success rate: {overall_success_rate:.1f}%")
    else:
        print(f"❌ Poor performance. Success rate: {overall_success_rate:.1f}%")
    
    print()
    print("ALGORITHM CATEGORIES OVERVIEW:")
    print("-" * 40)
    
    categories_info = {
        'Sorting Algorithms': 'Quicksort, Mergesort, Heapsort, etc.',
        'Searching Algorithms': 'Binary search, Linear search, etc.',
        'Dynamic Programming': 'Fibonacci, Knapsack, LCS, etc.',
        'Greedy Algorithms': 'Activity selection, Huffman coding, etc.',
        'Divide and Conquer': 'Merge sort, Maximum subarray, etc.',
        'Backtracking': 'N-Queens, Sudoku solver, etc.',
        'String Algorithms': 'KMP, Rabin-Karp, LPS, etc.',
        'Mathematical Algorithms': 'GCD, Primality tests, etc.',
        'Bit Manipulation': 'Bit operations, XOR tricks, etc.',
        'Graph Algorithms': 'DFS, BFS, Dijkstra, MST, etc.'
    }
    
    for category, description in categories_info.items():
        status = test_results.get(category, {}).get('status', 'UNKNOWN')
        status_icon = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⚠️'
        print(f"{status_icon} {category}: {description}")
    
    print()
    print("=" * 80)
    print("TEST EXECUTION COMPLETED")
    print(f"Total execution time: {sum(r['execution_time'] for r in test_results.values()):.3f} seconds")
    print("=" * 80)
    
    return test_results


def check_implementation_status():
    """Check which algorithm implementations exist"""
    
    print("ALGORITHM IMPLEMENTATION STATUS CHECK")
    print("=" * 50)
    print()
    
    # Check for main implementation files
    implementation_files = [
        ('sorting/sorting.py', 'Sorting Algorithms'),
        ('searching/searching.py', 'Searching Algorithms'),
        ('dynamic_programming/dynamic_programming.py', 'Dynamic Programming'),
        ('greedy/greedy.py', 'Greedy Algorithms'),
        ('divide_and_conquer/divide_and_conquer.py', 'Divide and Conquer'),
        ('backtracking/backtracking.py', 'Backtracking'),
        ('string_algorithms/string_algorithms.py', 'String Algorithms'),
        ('mathematical/mathematical.py', 'Mathematical Algorithms'),
        ('bit_manipulation/bit_manipulation.py', 'Bit Manipulation'),
        ('graph_algorithms/graph_algorithms.py', 'Graph Algorithms')
    ]
    
    implemented_count = 0
    
    for file_path, category_name in implementation_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            try:
                # Try to get file size
                size = os.path.getsize(full_path)
                print(f"✅ {category_name}: {file_path} ({size:,} bytes)")
                implemented_count += 1
            except Exception as e:
                print(f"⚠️  {category_name}: {file_path} (Error: {e})")
        else:
            print(f"❌ {category_name}: {file_path} (Not found)")
    
    print()
    print(f"Implementation Progress: {implemented_count}/{len(implementation_files)} "
          f"({implemented_count/len(implementation_files)*100:.1f}%)")
    
    return implemented_count == len(implementation_files)


def main():
    """Main function to run all tests"""
    
    print("Algorithm Implementation Test Suite")
    print("Starting comprehensive testing...")
    print()
    
    # First check implementation status
    all_implemented = check_implementation_status()
    print()
    
    if not all_implemented:
        print("⚠️  Some implementations are missing. Proceeding with available tests...")
        print()
    
    # Run the tests
    results = run_algorithm_tests()
    
    # Exit with appropriate code
    total_failures = sum(r['failures'] for r in results.values())
    total_errors = sum(r['errors'] for r in results.values())
    
    if total_failures > 0 or total_errors > 0:
        print("Some tests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("All tests passed successfully! 🎉")
        sys.exit(0)


if __name__ == '__main__':
    main() 