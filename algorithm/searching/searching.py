"""
Searching Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of various searching algorithms
with detailed time and space complexity analysis.
"""

from typing import List, Any, Optional, Callable
import math


class SearchingAlgorithms:
    """Collection of searching algorithms with performance analysis"""
    
    @staticmethod
    def binary_search(arr: List[Any], target: Any) -> int:
        """
        Binary Search - For sorted arrays
        Time: O(log n), Space: O(1)
        Returns index of target or -1 if not found
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    @staticmethod
    def binary_search_recursive(arr: List[Any], target: Any, left: int = 0, right: int = None) -> int:
        """Recursive binary search"""
        if right is None:
            right = len(arr) - 1
        
        if left > right:
            return -1
        
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return SearchingAlgorithms.binary_search_recursive(arr, target, mid + 1, right)
        else:
            return SearchingAlgorithms.binary_search_recursive(arr, target, left, mid - 1)
    
    @staticmethod
    def find_first_occurrence(arr: List[Any], target: Any) -> int:
        """Find the first occurrence of target in sorted array"""
        left, right = 0, len(arr) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                result = mid
                right = mid - 1  # Continue searching left
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    @staticmethod
    def find_last_occurrence(arr: List[Any], target: Any) -> int:
        """Find the last occurrence of target in sorted array"""
        left, right = 0, len(arr) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                result = mid
                left = mid + 1  # Continue searching right
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    @staticmethod
    def count_occurrences(arr: List[Any], target: Any) -> int:
        """Count occurrences of target in sorted array"""
        first = SearchingAlgorithms.find_first_occurrence(arr, target)
        if first == -1:
            return 0
        
        last = SearchingAlgorithms.find_last_occurrence(arr, target)
        return last - first + 1
    
    @staticmethod
    def ternary_search(arr: List[Any], target: Any) -> int:
        """
        Ternary Search - For sorted arrays
        Time: O(log₃ n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            
            if arr[mid1] == target:
                return mid1
            if arr[mid2] == target:
                return mid2
            
            if target < arr[mid1]:
                right = mid1 - 1
            elif target > arr[mid2]:
                left = mid2 + 1
            else:
                left = mid1 + 1
                right = mid2 - 1
        
        return -1
    
    @staticmethod
    def exponential_search(arr: List[Any], target: Any) -> int:
        """
        Exponential Search - For sorted unbounded arrays
        Time: O(log n), Space: O(1)
        First finds range, then binary search
        """
        if not arr:
            return -1
        
        if arr[0] == target:
            return 0
        
        # Find range for binary search
        bound = 1
        while bound < len(arr) and arr[bound] < target:
            bound *= 2
        
        # Binary search in the found range
        left = bound // 2
        right = min(bound, len(arr) - 1)
        
        return SearchingAlgorithms._binary_search_range(arr, target, left, right)
    
    @staticmethod
    def _binary_search_range(arr: List[Any], target: Any, left: int, right: int) -> int:
        """Binary search in a specific range"""
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    @staticmethod
    def interpolation_search(arr: List[int], target: int) -> int:
        """
        Interpolation Search - For uniformly distributed sorted arrays
        Time: O(log log n) average, O(n) worst case
        Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right and target >= arr[left] and target <= arr[right]:
            # If array has only one element
            if left == right:
                if arr[left] == target:
                    return left
                return -1
            
            # Calculate position using interpolation formula
            pos = left + int(((target - arr[left]) / (arr[right] - arr[left])) * (right - left))
            
            # Ensure pos is within bounds
            pos = max(left, min(pos, right))
            
            if arr[pos] == target:
                return pos
            elif arr[pos] < target:
                left = pos + 1
            else:
                right = pos - 1
        
        return -1
    
    @staticmethod
    def jump_search(arr: List[Any], target: Any) -> int:
        """
        Jump Search - For sorted arrays
        Time: O(√n), Space: O(1)
        """
        n = len(arr)
        if n == 0:
            return -1
        
        jump = int(math.sqrt(n))
        prev = 0
        
        while arr[min(jump, n) - 1] < target:
            prev = jump
            jump += int(math.sqrt(n))
            if prev >= n:
                return -1
        
        while arr[prev] < target:
            prev += 1
            if prev == min(jump, n):
                return -1
        
        if arr[prev] == target:
            return prev
        
        return -1
    
    @staticmethod
    def fibonacci_search(arr: List[Any], target: Any) -> int:
        """
        Fibonacci Search - For sorted arrays
        Time: O(log n), Space: O(1)
        Uses Fibonacci numbers to divide array
        """
        n = len(arr)
        if n == 0:
            return -1
        
        # Initialize Fibonacci numbers
        fib_m2 = 0  # (m-2)th Fibonacci number
        fib_m1 = 1  # (m-1)th Fibonacci number
        fib_m = fib_m2 + fib_m1  # mth Fibonacci number
        
        # Find smallest Fibonacci number >= n
        while fib_m < n:
            fib_m2 = fib_m1
            fib_m1 = fib_m
            fib_m = fib_m2 + fib_m1
        
        # Marks the eliminated range from front
        offset = -1
        
        # While there are elements to be inspected
        while fib_m > 1:
            # Check if fib_m2 is a valid location
            i = min(offset + fib_m2, n - 1)
            
            # If target is greater than the value at index fib_m2, cut the subarray from offset to i
            if arr[i] < target:
                fib_m = fib_m1
                fib_m1 = fib_m2
                fib_m2 = fib_m - fib_m1
                offset = i
            
            # If target is less than the value at index fib_m2, cut the subarray after i+1
            elif arr[i] > target:
                fib_m = fib_m2
                fib_m1 = fib_m1 - fib_m2
                fib_m2 = fib_m - fib_m1
            
            # Element found
            else:
                return i
        
        # Comparing the last element with target
        if fib_m1 and offset + 1 < n and arr[offset + 1] == target:
            return offset + 1
        
        return -1
    
    @staticmethod
    def linear_search(arr: List[Any], target: Any) -> int:
        """
        Linear Search - For unsorted arrays
        Time: O(n), Space: O(1)
        """
        for i, element in enumerate(arr):
            if element == target:
                return i
        return -1
    
    @staticmethod
    def find_all_occurrences(arr: List[Any], target: Any) -> List[int]:
        """Find all occurrences of target in array"""
        occurrences = []
        for i, element in enumerate(arr):
            if element == target:
                occurrences.append(i)
        return occurrences
    
    @staticmethod
    def search_in_rotated_array(arr: List[int], target: int) -> int:
        """
        Search in rotated sorted array
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            
            # Check which half is sorted
            if arr[left] <= arr[mid]:  # Left half is sorted
                if arr[left] <= target < arr[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:  # Right half is sorted
                if arr[mid] < target <= arr[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    @staticmethod
    def find_peak_element(arr: List[int]) -> int:
        """
        Find peak element in array (element greater than neighbors)
        Time: O(log n), Space: O(1)
        """
        if not arr:
            return -1
        
        n = len(arr)
        if n == 1:
            return 0
        
        left, right = 0, n - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            # Check if mid is peak
            if ((mid == 0 or arr[mid] >= arr[mid - 1]) and 
                (mid == n - 1 or arr[mid] >= arr[mid + 1])):
                return mid
            
            # If left neighbor is greater, peak lies in left half
            if mid > 0 and arr[mid - 1] > arr[mid]:
                right = mid - 1
            # Otherwise, peak lies in right half
            else:
                left = mid + 1
        
        return -1
    
    @staticmethod
    def search_2d_matrix(matrix: List[List[int]], target: int) -> bool:
        """
        Search in 2D matrix where each row and column is sorted
        Time: O(m + n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        rows, cols = len(matrix), len(matrix[0])
        row, col = 0, cols - 1
        
        while row < rows and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False
    
    @staticmethod
    def search_2d_matrix_binary(matrix: List[List[int]], target: int) -> bool:
        """
        Search in 2D matrix using binary search
        Time: O(log(m*n)), Space: O(1)
        Treats 2D matrix as 1D sorted array
        """
        if not matrix or not matrix[0]:
            return False
        
        rows, cols = len(matrix), len(matrix[0])
        left, right = 0, rows * cols - 1
        
        while left <= right:
            mid = (left + right) // 2
            mid_value = matrix[mid // cols][mid % cols]
            
            if mid_value == target:
                return True
            elif mid_value < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return False


class SearchingUtils:
    """Utility functions for searching algorithms"""
    
    @staticmethod
    def is_sorted(arr: List[Any]) -> bool:
        """Check if array is sorted"""
        return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
    
    @staticmethod
    def benchmark_search(search_func: Callable, arr: List[Any], target: Any) -> tuple:
        """Benchmark a search function"""
        import time
        
        start_time = time.time()
        result = search_func(arr, target)
        end_time = time.time()
        
        return result, end_time - start_time
    
    @staticmethod
    def compare_searches(arr: List[Any], target: Any, algorithms: List[str] = None) -> dict:
        """Compare multiple search algorithms"""
        if algorithms is None:
            algorithms = ['binary_search', 'linear_search', 'ternary_search']
        
        results = {}
        searcher = SearchingAlgorithms()
        
        for algo in algorithms:
            if hasattr(searcher, algo):
                func = getattr(searcher, algo)
                try:
                    result, time_taken = SearchingUtils.benchmark_search(func, arr, target)
                    results[algo] = {
                        'result': result,
                        'time': time_taken,
                        'found': result != -1
                    }
                except Exception as e:
                    results[algo] = {
                        'result': -1,
                        'time': 0,
                        'found': False,
                        'error': str(e)
                    }
        
        return results
    
    @staticmethod
    def generate_search_data(size: int, sorted_data: bool = True) -> List[int]:
        """Generate test data for search algorithms"""
        import random
        
        if sorted_data:
            return list(range(1, size + 1))
        else:
            data = list(range(1, size + 1))
            random.shuffle(data)
            return data
    
    @staticmethod
    def binary_search_variants(arr: List[Any], target: Any) -> dict:
        """Test all binary search variants"""
        return {
            'binary_search': SearchingAlgorithms.binary_search(arr, target),
            'binary_search_recursive': SearchingAlgorithms.binary_search_recursive(arr, target),
            'first_occurrence': SearchingAlgorithms.find_first_occurrence(arr, target),
            'last_occurrence': SearchingAlgorithms.find_last_occurrence(arr, target),
            'count_occurrences': SearchingAlgorithms.count_occurrences(arr, target)
        }


# Example usage and demonstrations
if __name__ == "__main__":
    # Test data
    sorted_arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    target = 7
    
    print(f"Sorted array: {sorted_arr}")
    print(f"Target: {target}")
    
    searcher = SearchingAlgorithms()
    
    # Test different search algorithms
    print(f"\nBinary Search: {searcher.binary_search(sorted_arr, target)}")
    print(f"Binary Search Recursive: {searcher.binary_search_recursive(sorted_arr, target)}")
    print(f"Ternary Search: {searcher.ternary_search(sorted_arr, target)}")
    print(f"Exponential Search: {searcher.exponential_search(sorted_arr, target)}")
    print(f"Interpolation Search: {searcher.interpolation_search(sorted_arr, target)}")
    print(f"Jump Search: {searcher.jump_search(sorted_arr, target)}")
    print(f"Fibonacci Search: {searcher.fibonacci_search(sorted_arr, target)}")
    print(f"Linear Search: {searcher.linear_search(sorted_arr, target)}")
    
    # Test array with duplicates
    dup_arr = [1, 2, 2, 2, 3, 4, 4, 5]
    target_dup = 2
    print(f"\nArray with duplicates: {dup_arr}")
    print(f"Target: {target_dup}")
    print(f"First occurrence: {searcher.find_first_occurrence(dup_arr, target_dup)}")
    print(f"Last occurrence: {searcher.find_last_occurrence(dup_arr, target_dup)}")
    print(f"Count occurrences: {searcher.count_occurrences(dup_arr, target_dup)}")
    print(f"All occurrences: {searcher.find_all_occurrences(dup_arr, target_dup)}")
    
    # Test rotated array
    rotated = [4, 5, 6, 7, 0, 1, 2]
    target_rot = 0
    print(f"\nRotated array: {rotated}")
    print(f"Search {target_rot}: {searcher.search_in_rotated_array(rotated, target_rot)}")
    
    # Test peak finding
    peak_arr = [1, 3, 20, 4, 1, 0]
    print(f"\nPeak array: {peak_arr}")
    print(f"Peak element index: {searcher.find_peak_element(peak_arr)}")
    
    # Test 2D matrix search
    matrix = [
        [1,  4,  7,  11],
        [2,  5,  8,  12],
        [3,  6,  9,  16],
        [10, 13, 14, 17]
    ]
    target_2d = 5
    print(f"\n2D Matrix search for {target_2d}: {searcher.search_2d_matrix(matrix, target_2d)}")
    print(f"2D Binary search for {target_2d}: {searcher.search_2d_matrix_binary(matrix, target_2d)}")
    
    # Performance comparison
    print("\n--- Performance Comparison ---")
    large_arr = SearchingUtils.generate_search_data(10000)
    target_large = 5000
    
    comparison = SearchingUtils.compare_searches(
        large_arr, target_large, 
        ['binary_search', 'linear_search', 'ternary_search', 'jump_search']
    )
    
    for algo, stats in comparison.items():
        if 'error' not in stats:
            print(f"{algo}: Found={stats['found']}, Time={stats['time']:.6f}s")
        else:
            print(f"{algo}: Error - {stats['error']}")
    
    # Binary search variants
    print("\n--- Binary Search Variants ---")
    variants = SearchingUtils.binary_search_variants(dup_arr, target_dup)
    for variant, result in variants.items():
        print(f"{variant}: {result}") 