"""
Sorting Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of various sorting algorithms
with detailed time and space complexity analysis.
"""

import random
from typing import List, Callable, Any
import heapq


class SortingAlgorithms:
    """Collection of sorting algorithms with performance analysis"""
    
    @staticmethod
    def quicksort(arr: List[Any]) -> List[Any]:
        """
        Quick Sort - Divide and conquer algorithm
        Average: O(n log n), Worst: O(n²), Space: O(log n)
        """
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return SortingAlgorithms.quicksort(left) + middle + SortingAlgorithms.quicksort(right)
    
    @staticmethod
    def quicksort_inplace(arr: List[Any], low: int = 0, high: int = None) -> None:
        """In-place quicksort with better space complexity"""
        if high is None:
            high = len(arr) - 1
        
        if low < high:
            pivot_index = SortingAlgorithms._partition(arr, low, high)
            SortingAlgorithms.quicksort_inplace(arr, low, pivot_index - 1)
            SortingAlgorithms.quicksort_inplace(arr, pivot_index + 1, high)
    
    @staticmethod
    def _partition(arr: List[Any], low: int, high: int) -> int:
        """Partition function for quicksort"""
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    @staticmethod
    def quicksort_random(arr: List[Any]) -> List[Any]:
        """Randomized quicksort to avoid worst-case scenarios"""
        if len(arr) <= 1:
            return arr
        
        pivot_idx = random.randint(0, len(arr) - 1)
        pivot = arr[pivot_idx]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return SortingAlgorithms.quicksort_random(left) + middle + SortingAlgorithms.quicksort_random(right)
    
    @staticmethod
    def mergesort(arr: List[Any]) -> List[Any]:
        """
        Merge Sort - Stable divide and conquer algorithm
        Time: O(n log n), Space: O(n)
        """
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = SortingAlgorithms.mergesort(arr[:mid])
        right = SortingAlgorithms.mergesort(arr[mid:])
        
        return SortingAlgorithms._merge(left, right)
    
    @staticmethod
    def _merge(left: List[Any], right: List[Any]) -> List[Any]:
        """Merge function for mergesort"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    @staticmethod
    def heapsort(arr: List[Any]) -> List[Any]:
        """
        Heap Sort - In-place sorting using heap
        Time: O(n log n), Space: O(1)
        """
        arr = arr.copy()  # Don't modify original
        n = len(arr)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            SortingAlgorithms._heapify(arr, n, i)
        
        # Extract elements from heap one by one
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]  # Move current root to end
            SortingAlgorithms._heapify(arr, i, 0)
        
        return arr
    
    @staticmethod
    def _heapify(arr: List[Any], n: int, i: int) -> None:
        """Heapify function for heapsort"""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] > arr[largest]:
            largest = left
        
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            SortingAlgorithms._heapify(arr, n, largest)
    
    @staticmethod
    def counting_sort(arr: List[int], max_val: int = None) -> List[int]:
        """
        Counting Sort - For integers in a specific range
        Time: O(n + k), Space: O(k)
        Where k is the range of input values
        """
        if not arr:
            return arr
        
        if max_val is None:
            max_val = max(arr)
        
        min_val = min(arr)
        range_val = max_val - min_val + 1
        count = [0] * range_val
        
        # Count occurrences
        for num in arr:
            count[num - min_val] += 1
        
        # Reconstruct sorted array
        result = []
        for i, freq in enumerate(count):
            result.extend([i + min_val] * freq)
        
        return result
    
    @staticmethod
    def radix_sort(arr: List[int]) -> List[int]:
        """
        Radix Sort - For non-negative integers
        Time: O(d * (n + k)), Space: O(n + k)
        Where d is the number of digits and k is the range of digits
        """
        if not arr:
            return arr
        
        # Find the maximum number to know number of digits
        max_num = max(arr)
        
        # Do counting sort for every digit
        exp = 1
        while max_num // exp > 0:
            arr = SortingAlgorithms._counting_sort_by_digit(arr, exp)
            exp *= 10
        
        return arr
    
    @staticmethod
    def _counting_sort_by_digit(arr: List[int], exp: int) -> List[int]:
        """Counting sort by specific digit for radix sort"""
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        
        # Count occurrences of each digit
        for num in arr:
            index = (num // exp) % 10
            count[index] += 1
        
        # Change count[i] to actual position of this digit in output[]
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        # Build the output array
        for i in range(n - 1, -1, -1):
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
        
        return output
    
    @staticmethod
    def insertion_sort(arr: List[Any]) -> List[Any]:
        """
        Insertion Sort - Simple sorting algorithm
        Time: O(n²), Space: O(1)
        Best case: O(n) for nearly sorted arrays
        """
        arr = arr.copy()
        
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            
            # Move elements greater than key one position ahead
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            
            arr[j + 1] = key
        
        return arr
    
    @staticmethod
    def selection_sort(arr: List[Any]) -> List[Any]:
        """
        Selection Sort - Simple sorting algorithm
        Time: O(n²), Space: O(1)
        """
        arr = arr.copy()
        
        for i in range(len(arr)):
            min_idx = i
            
            # Find minimum element in remaining array
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            # Swap the found minimum element with first element
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        
        return arr
    
    @staticmethod
    def bubble_sort(arr: List[Any]) -> List[Any]:
        """
        Bubble Sort - Simple sorting algorithm
        Time: O(n²), Space: O(1)
        """
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            swapped = False
            
            # Last i elements are already sorted
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            
            # If no swapping occurred, array is sorted
            if not swapped:
                break
        
        return arr
    
    @staticmethod
    def shell_sort(arr: List[Any]) -> List[Any]:
        """
        Shell Sort - Improved insertion sort
        Time: O(n log n) to O(n²), Space: O(1)
        """
        arr = arr.copy()
        n = len(arr)
        gap = n // 2
        
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                
                while j >= gap and arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    j -= gap
                
                arr[j] = temp
            
            gap //= 2
        
        return arr
    
    @staticmethod
    def bucket_sort(arr: List[float], num_buckets: int = None) -> List[float]:
        """
        Bucket Sort - For uniformly distributed data
        Time: O(n + k), Space: O(n + k)
        Average case, worst case can be O(n²)
        """
        if not arr:
            return arr
        
        if num_buckets is None:
            num_buckets = len(arr)
        
        # Create empty buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # Distribute elements into buckets
        max_val = max(arr)
        min_val = min(arr)
        range_val = max_val - min_val
        
        for num in arr:
            if range_val == 0:
                bucket_idx = 0
            else:
                bucket_idx = int((num - min_val) / range_val * (num_buckets - 1))
            buckets[bucket_idx].append(num)
        
        # Sort individual buckets and concatenate
        result = []
        for bucket in buckets:
            if bucket:
                result.extend(sorted(bucket))
        
        return result
    
    @staticmethod
    def tim_sort(arr: List[Any]) -> List[Any]:
        """
        Tim Sort - Python's built-in sort (simplified version)
        Hybrid stable sorting algorithm
        Time: O(n log n), Space: O(n)
        """
        # This is a simplified version - actual Timsort is much more complex
        MIN_MERGE = 32
        
        def insertion_sort_range(arr, left, right):
            for i in range(left + 1, right + 1):
                key = arr[i]
                j = i - 1
                while j >= left and arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
        
        def merge(arr, left, mid, right):
            left_part = arr[left:mid + 1]
            right_part = arr[mid + 1:right + 1]
            
            i = j = 0
            k = left
            
            while i < len(left_part) and j < len(right_part):
                if left_part[i] <= right_part[j]:
                    arr[k] = left_part[i]
                    i += 1
                else:
                    arr[k] = right_part[j]
                    j += 1
                k += 1
            
            while i < len(left_part):
                arr[k] = left_part[i]
                i += 1
                k += 1
            
            while j < len(right_part):
                arr[k] = right_part[j]
                j += 1
                k += 1
        
        arr = arr.copy()
        n = len(arr)
        
        # Sort individual subarrays of size MIN_MERGE using insertion sort
        for start in range(0, n, MIN_MERGE):
            end = min(start + MIN_MERGE - 1, n - 1)
            insertion_sort_range(arr, start, end)
        
        # Start merging from size MIN_MERGE
        size = MIN_MERGE
        while size < n:
            for start in range(0, n, size * 2):
                mid = start + size - 1
                end = min(start + size * 2 - 1, n - 1)
                
                if mid < end:
                    merge(arr, start, mid, end)
            
            size *= 2
        
        return arr


class SortingUtils:
    """Utility functions for sorting algorithms"""
    
    @staticmethod
    def is_sorted(arr: List[Any], reverse: bool = False) -> bool:
        """Check if array is sorted"""
        if len(arr) <= 1:
            return True
        
        if reverse:
            return all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))
        else:
            return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
    
    @staticmethod
    def generate_test_data(size: int, data_type: str = "random") -> List[int]:
        """Generate test data for sorting algorithms"""
        if data_type == "random":
            return [random.randint(1, 1000) for _ in range(size)]
        elif data_type == "sorted":
            return list(range(1, size + 1))
        elif data_type == "reverse_sorted":
            return list(range(size, 0, -1))
        elif data_type == "nearly_sorted":
            arr = list(range(1, size + 1))
            # Swap a few random elements
            for _ in range(size // 10):
                i, j = random.randint(0, size - 1), random.randint(0, size - 1)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
        elif data_type == "duplicates":
            return [random.randint(1, 10) for _ in range(size)]
        else:
            raise ValueError("Invalid data type")
    
    @staticmethod
    def benchmark_sort(sort_func: Callable, arr: List[Any]) -> tuple:
        """Benchmark a sorting function"""
        import time
        
        start_time = time.time()
        sorted_arr = sort_func(arr)
        end_time = time.time()
        
        return sorted_arr, end_time - start_time
    
    @staticmethod
    def compare_sorts(arr: List[Any], algorithms: List[str] = None) -> dict:
        """Compare multiple sorting algorithms"""
        if algorithms is None:
            algorithms = ['quicksort', 'mergesort', 'heapsort', 'insertion_sort']
        
        results = {}
        sort_algos = SortingAlgorithms()
        
        for algo in algorithms:
            if hasattr(sort_algos, algo):
                func = getattr(sort_algos, algo)
                sorted_arr, time_taken = SortingUtils.benchmark_sort(func, arr)
                results[algo] = {
                    'time': time_taken,
                    'is_correct': SortingUtils.is_sorted(sorted_arr)
                }
        
        return results


# Example usage and demonstrations
if __name__ == "__main__":
    # Test data
    test_arr = [64, 34, 25, 12, 22, 11, 90, 5]
    print(f"Original array: {test_arr}")
    
    # Test different sorting algorithms
    sorter = SortingAlgorithms()
    
    print(f"Quick Sort: {sorter.quicksort(test_arr)}")
    print(f"Merge Sort: {sorter.mergesort(test_arr)}")
    print(f"Heap Sort: {sorter.heapsort(test_arr)}")
    print(f"Insertion Sort: {sorter.insertion_sort(test_arr)}")
    print(f"Selection Sort: {sorter.selection_sort(test_arr)}")
    print(f"Bubble Sort: {sorter.bubble_sort(test_arr)}")
    print(f"Shell Sort: {sorter.shell_sort(test_arr)}")
    
    # Test integer-specific sorts
    int_arr = [170, 45, 75, 90, 2, 802, 24, 66]
    print(f"\nInteger array: {int_arr}")
    print(f"Counting Sort: {sorter.counting_sort(int_arr)}")
    print(f"Radix Sort: {sorter.radix_sort(int_arr)}")
    
    # Test float array for bucket sort
    float_arr = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
    print(f"\nFloat array: {float_arr}")
    print(f"Bucket Sort: {sorter.bucket_sort(float_arr)}")
    
    # Performance comparison
    print("\n--- Performance Comparison ---")
    large_arr = SortingUtils.generate_test_data(1000, "random")
    comparison = SortingUtils.compare_sorts(large_arr, ['quicksort', 'mergesort', 'heapsort'])
    
    for algo, stats in comparison.items():
        print(f"{algo}: {stats['time']:.6f}s, Correct: {stats['is_correct']}") 