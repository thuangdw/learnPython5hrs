"""
Test cases for Sorting Algorithms
Comprehensive testing suite for all sorting implementations
"""

import unittest
import random
import time
from sorting import SortingAlgorithms, SortingUtils


class TestSortingAlgorithms(unittest.TestCase):
    """Test cases for sorting algorithms"""
    
    def setUp(self):
        """Set up test data"""
        self.sorter = SortingAlgorithms()
        self.test_cases = [
            [],  # Empty array
            [1],  # Single element
            [2, 1],  # Two elements
            [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],  # Random array with duplicates
            [1, 2, 3, 4, 5],  # Already sorted
            [5, 4, 3, 2, 1],  # Reverse sorted
            [1, 1, 1, 1, 1],  # All same elements
            [64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42],  # Standard test case
            [-5, -2, 0, 3, 8],  # With negative numbers
            [100, -50, 0, 75, -25, 25]  # Mixed positive and negative
        ]
        
        # Large test case
        self.large_array = SortingUtils.generate_test_data(1000, "random")
    
    def test_quicksort(self):
        """Test quicksort algorithm"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.quicksort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_quicksort_inplace(self):
        """Test in-place quicksort"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                arr_copy = test_case.copy()
                self.sorter.quicksort_inplace(arr_copy)
                expected = sorted(test_case)
                self.assertEqual(arr_copy, expected)
                self.assertTrue(SortingUtils.is_sorted(arr_copy))
    
    def test_quicksort_random(self):
        """Test randomized quicksort"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.quicksort_random(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
                self.assertTrue(SortingUtils.is_sorted(result))
    
    def test_mergesort(self):
        """Test mergesort algorithm"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.mergesort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_heapsort(self):
        """Test heapsort algorithm"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.heapsort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_insertion_sort(self):
        """Test insertion sort algorithm"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.insertion_sort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_selection_sort(self):
        """Test selection sort algorithm"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.selection_sort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_bubble_sort(self):
        """Test bubble sort algorithm"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.bubble_sort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_shell_sort(self):
        """Test shell sort algorithm"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.shell_sort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_bucket_sort(self):
        """Test bucket sort algorithm"""
        # Test with floating point numbers in range [0, 1)
        test_cases = [
            [],
            [0.5],
            [0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [0.5, 0.5, 0.5, 0.5]
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.bucket_sort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_tim_sort(self):
        """Test tim sort algorithm"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.tim_sort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_counting_sort(self):
        """Test counting sort algorithm"""
        # Test with non-negative integers only
        test_cases = [
            [],
            [1],
            [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [1, 1, 1, 1, 1]
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                if test_case:  # Skip empty array for counting sort
                    result = self.sorter.counting_sort(test_case.copy())
                    expected = sorted(test_case)
                    self.assertEqual(result, expected)
    
    def test_radix_sort(self):
        """Test radix sort algorithm"""
        # Test with non-negative integers only
        test_cases = [
            [],
            [1],
            [170, 45, 75, 90, 2, 802, 24, 66],
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [100, 100, 100]
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.radix_sort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_cocktail_sort(self):
        """Test cocktail sort algorithm"""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case):
                result = self.sorter.cocktail_sort(test_case.copy())
                expected = sorted(test_case)
                self.assertEqual(result, expected)
    
    def test_is_sorted(self):
        """Test is_sorted utility function"""
        self.assertTrue(self.sorter.is_sorted([]))
        self.assertTrue(self.sorter.is_sorted([1]))
        self.assertTrue(self.sorter.is_sorted([1, 2, 3, 4, 5]))
        self.assertTrue(self.sorter.is_sorted([1, 1, 1, 1]))
        self.assertFalse(self.sorter.is_sorted([5, 4, 3, 2, 1]))
        self.assertFalse(self.sorter.is_sorted([1, 3, 2, 4]))
    
    def test_custom_comparator(self):
        """Test sorting with custom comparator"""
        test_array = [3, 1, 4, 1, 5, 9, 2, 6]
        
        # Sort in descending order
        result = self.sorter.quicksort_with_comparator(
            test_array.copy(), 
            lambda a, b: a > b
        )
        expected = sorted(test_array, reverse=True)
        self.assertEqual(result, expected)
    
    def test_stability(self):
        """Test sorting stability with tuples"""
        # Test with tuples (value, original_index)
        test_data = [(3, 0), (1, 1), (3, 2), (2, 3), (1, 4)]
        
        # Stable sorts should maintain relative order of equal elements
        stable_result = self.sorter.mergesort_stable(test_data.copy())
        
        # Check that elements with same value maintain original order
        ones = [item for item in stable_result if item[0] == 1]
        threes = [item for item in stable_result if item[0] == 3]
        
        self.assertEqual(ones, [(1, 1), (1, 4)])  # Original order maintained
        self.assertEqual(threes, [(3, 0), (3, 2)])  # Original order maintained
    
    def test_large_array_performance(self):
        """Test performance with large arrays"""
        large_array = [random.randint(1, 1000) for _ in range(1000)]
        
        # Test that sorting completes in reasonable time
        start_time = time.time()
        result = self.sorter.quicksort(large_array.copy())
        end_time = time.time()
        
        self.assertEqual(result, sorted(large_array))
        self.assertLess(end_time - start_time, 1.0)  # Should complete in under 1 second
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test with None values (should handle gracefully or raise appropriate error)
        with self.assertRaises((TypeError, AttributeError)):
            self.sorter.quicksort([1, None, 3])
        
        # Test with very large numbers
        large_numbers = [10**10, 10**9, 10**11]
        result = self.sorter.quicksort(large_numbers.copy())
        self.assertEqual(result, sorted(large_numbers))
    
    def test_in_place_sorting(self):
        """Test in-place sorting algorithms"""
        test_array = [3, 1, 4, 1, 5, 9, 2, 6]
        original_id = id(test_array)
        
        # Test in-place quicksort
        self.sorter.quicksort_in_place(test_array, 0, len(test_array) - 1)
        
        # Array should be sorted and same object
        self.assertEqual(test_array, sorted([3, 1, 4, 1, 5, 9, 2, 6]))
        self.assertEqual(id(test_array), original_id)
    
    def test_partial_sorting(self):
        """Test partial sorting (k smallest/largest elements)"""
        test_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        
        # Get 3 smallest elements
        k_smallest = self.sorter.quickselect(test_array.copy(), 3)
        expected_smallest = sorted(test_array)[:3]
        self.assertEqual(sorted(k_smallest), expected_smallest)
        
        # Get 3 largest elements
        k_largest = self.sorter.quickselect_largest(test_array.copy(), 3)
        expected_largest = sorted(test_array, reverse=True)[:3]
        self.assertEqual(sorted(k_largest, reverse=True), expected_largest)
    
    def test_large_arrays(self):
        """Test sorting algorithms with large arrays"""
        algorithms = [
            'quicksort', 'mergesort', 'heapsort', 'insertion_sort',
            'selection_sort', 'shell_sort', 'tim_sort'
        ]
        
        for algo_name in algorithms:
            with self.subTest(algorithm=algo_name):
                algo_func = getattr(self.sorter, algo_name)
                result = algo_func(self.large_array)
                self.assertTrue(SortingUtils.is_sorted(result))
                self.assertEqual(len(result), len(self.large_array))


class TestSortingUtils(unittest.TestCase):
    """Test cases for sorting utilities"""
    
    def test_is_sorted(self):
        """Test is_sorted function"""
        self.assertTrue(SortingUtils.is_sorted([]))
        self.assertTrue(SortingUtils.is_sorted([1]))
        self.assertTrue(SortingUtils.is_sorted([1, 2, 3, 4, 5]))
        self.assertTrue(SortingUtils.is_sorted([1, 1, 1, 1]))
        self.assertFalse(SortingUtils.is_sorted([3, 1, 2]))
        
        # Test reverse sorting
        self.assertTrue(SortingUtils.is_sorted([5, 4, 3, 2, 1], reverse=True))
        self.assertFalse(SortingUtils.is_sorted([1, 2, 3, 4, 5], reverse=True))
    
    def test_generate_test_data(self):
        """Test test data generation"""
        # Random data
        random_data = SortingUtils.generate_test_data(100, "random")
        self.assertEqual(len(random_data), 100)
        self.assertTrue(all(1 <= x <= 1000 for x in random_data))
        
        # Sorted data
        sorted_data = SortingUtils.generate_test_data(50, "sorted")
        self.assertEqual(sorted_data, list(range(1, 51)))
        
        # Reverse sorted data
        reverse_data = SortingUtils.generate_test_data(30, "reverse_sorted")
        self.assertEqual(reverse_data, list(range(30, 0, -1)))
        
        # Nearly sorted data
        nearly_sorted = SortingUtils.generate_test_data(20, "nearly_sorted")
        self.assertEqual(len(nearly_sorted), 20)
        
        # Duplicates data
        duplicates = SortingUtils.generate_test_data(25, "duplicates")
        self.assertEqual(len(duplicates), 25)
        self.assertTrue(all(1 <= x <= 10 for x in duplicates))
        
        # Invalid data type
        with self.assertRaises(ValueError):
            SortingUtils.generate_test_data(10, "invalid")
    
    def test_benchmark_sort(self):
        """Test sorting benchmark function"""
        test_arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
        sorter = SortingAlgorithms()
        
        result, time_taken = SortingUtils.benchmark_sort(sorter.quicksort, test_arr)
        
        self.assertTrue(SortingUtils.is_sorted(result))
        self.assertIsInstance(time_taken, float)
        self.assertGreaterEqual(time_taken, 0)
    
    def test_compare_sorts(self):
        """Test sorting algorithm comparison"""
        test_arr = SortingUtils.generate_test_data(100, "random")
        algorithms = ['quicksort', 'mergesort', 'heapsort']
        
        results = SortingUtils.compare_sorts(test_arr, algorithms)
        
        self.assertEqual(len(results), len(algorithms))
        
        for algo in algorithms:
            self.assertIn(algo, results)
            self.assertIn('time', results[algo])
            self.assertIn('is_correct', results[algo])
            self.assertTrue(results[algo]['is_correct'])
            self.assertIsInstance(results[algo]['time'], float)


class TestSortingPerformance(unittest.TestCase):
    """Performance tests for sorting algorithms"""
    
    def setUp(self):
        """Set up performance test data"""
        self.sorter = SortingAlgorithms()
        self.small_data = SortingUtils.generate_test_data(100, "random")
        self.medium_data = SortingUtils.generate_test_data(1000, "random")
    
    def test_performance_comparison(self):
        """Compare performance of different algorithms"""
        algorithms = ['quicksort', 'mergesort', 'heapsort', 'tim_sort']
        
        results = SortingUtils.compare_sorts(self.medium_data, algorithms)
        
        # All algorithms should complete successfully
        for algo in algorithms:
            self.assertTrue(results[algo]['is_correct'])
            self.assertLess(results[algo]['time'], 1.0)  # Should complete within 1 second
    
    def test_best_case_performance(self):
        """Test performance on already sorted data"""
        sorted_data = SortingUtils.generate_test_data(1000, "sorted")
        
        # Insertion sort should be very fast on sorted data
        result, time_taken = SortingUtils.benchmark_sort(self.sorter.insertion_sort, sorted_data)
        self.assertTrue(SortingUtils.is_sorted(result))
        
        # Quick sort should also be reasonable
        result, time_taken = SortingUtils.benchmark_sort(self.sorter.quicksort, sorted_data)
        self.assertTrue(SortingUtils.is_sorted(result))
    
    def test_worst_case_handling(self):
        """Test algorithms handle worst-case scenarios"""
        reverse_data = SortingUtils.generate_test_data(500, "reverse_sorted")
        
        # Test that algorithms still work correctly on reverse-sorted data
        algorithms = ['quicksort', 'mergesort', 'heapsort']
        
        for algo_name in algorithms:
            with self.subTest(algorithm=algo_name):
                algo_func = getattr(self.sorter, algo_name)
                result = algo_func(reverse_data)
                self.assertTrue(SortingUtils.is_sorted(result))


class TestSortingEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up edge case test data"""
        self.sorter = SortingAlgorithms()
    
    def test_empty_arrays(self):
        """Test sorting empty arrays"""
        empty = []
        algorithms = [
            'quicksort', 'mergesort', 'heapsort', 'insertion_sort',
            'selection_sort', 'bubble_sort', 'shell_sort', 'tim_sort'
        ]
        
        for algo_name in algorithms:
            with self.subTest(algorithm=algo_name):
                algo_func = getattr(self.sorter, algo_name)
                result = algo_func(empty)
                self.assertEqual(result, [])
    
    def test_single_element(self):
        """Test sorting single-element arrays"""
        single = [42]
        algorithms = [
            'quicksort', 'mergesort', 'heapsort', 'insertion_sort',
            'selection_sort', 'bubble_sort', 'shell_sort', 'tim_sort'
        ]
        
        for algo_name in algorithms:
            with self.subTest(algorithm=algo_name):
                algo_func = getattr(self.sorter, algo_name)
                result = algo_func(single)
                self.assertEqual(result, [42])
    
    def test_duplicate_elements(self):
        """Test sorting arrays with many duplicates"""
        duplicates = [5] * 100 + [3] * 50 + [7] * 25
        random.shuffle(duplicates)
        
        algorithms = ['quicksort', 'mergesort', 'heapsort', 'tim_sort']
        
        for algo_name in algorithms:
            with self.subTest(algorithm=algo_name):
                algo_func = getattr(self.sorter, algo_name)
                result = algo_func(duplicates)
                self.assertTrue(SortingUtils.is_sorted(result))
                self.assertEqual(len(result), len(duplicates))
    
    def test_negative_numbers(self):
        """Test sorting arrays with negative numbers"""
        mixed = [-5, -1, 0, 3, -10, 7, -2, 1]
        
        algorithms = [
            'quicksort', 'mergesort', 'heapsort', 'insertion_sort',
            'selection_sort', 'bubble_sort', 'shell_sort', 'tim_sort'
        ]
        
        for algo_name in algorithms:
            with self.subTest(algorithm=algo_name):
                algo_func = getattr(self.sorter, algo_name)
                result = algo_func(mixed)
                expected = sorted(mixed)
                self.assertEqual(result, expected)
    
    def test_large_numbers(self):
        """Test sorting arrays with very large numbers"""
        large_nums = [10**9, 10**6, 10**12, 10**3, 10**15]
        
        result = self.sorter.quicksort(large_nums)
        expected = sorted(large_nums)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main() 