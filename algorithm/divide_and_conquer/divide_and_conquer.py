"""
Divide and Conquer Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of classic divide and conquer algorithms
with detailed time and space complexity analysis.
"""

from typing import List, Tuple
import math


class DivideAndConquer:
    """Collection of divide and conquer algorithms"""
    
    @staticmethod
    def merge_sort(arr: List[int]) -> List[int]:
        """
        Merge Sort using Divide and Conquer
        Time: O(n log n), Space: O(n)
        """
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = DivideAndConquer.merge_sort(arr[:mid])
        right = DivideAndConquer.merge_sort(arr[mid:])
        
        return DivideAndConquer._merge(left, right)
    
    @staticmethod
    def _merge(left: List[int], right: List[int]) -> List[int]:
        """Helper function to merge two sorted arrays"""
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
    def quick_sort(arr: List[int]) -> List[int]:
        """
        Quick Sort using Divide and Conquer
        Average: O(n log n), Worst: O(n²), Space: O(log n)
        """
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return (DivideAndConquer.quick_sort(left) + 
                middle + 
                DivideAndConquer.quick_sort(right))
    
    @staticmethod
    def maximum_subarray(arr: List[int]) -> Tuple[int, int, int]:
        """
        Maximum Subarray using Divide and Conquer
        Time: O(n log n), Space: O(log n)
        Returns (max_sum, start_index, end_index)
        """
        def max_crossing_sum(arr, low, mid, high):
            """Find max sum crossing the midpoint"""
            left_sum = float('-inf')
            sum_val = 0
            max_left = mid
            
            for i in range(mid, low - 1, -1):
                sum_val += arr[i]
                if sum_val > left_sum:
                    left_sum = sum_val
                    max_left = i
            
            right_sum = float('-inf')
            sum_val = 0
            max_right = mid + 1
            
            for i in range(mid + 1, high + 1):
                sum_val += arr[i]
                if sum_val > right_sum:
                    right_sum = sum_val
                    max_right = i
            
            return left_sum + right_sum, max_left, max_right
        
        def max_subarray_rec(arr, low, high):
            if low == high:
                return arr[low], low, high
            
            mid = (low + high) // 2
            
            left_sum, left_start, left_end = max_subarray_rec(arr, low, mid)
            right_sum, right_start, right_end = max_subarray_rec(arr, mid + 1, high)
            cross_sum, cross_start, cross_end = max_crossing_sum(arr, low, mid, high)
            
            if left_sum >= right_sum and left_sum >= cross_sum:
                return left_sum, left_start, left_end
            elif right_sum >= left_sum and right_sum >= cross_sum:
                return right_sum, right_start, right_end
            else:
                return cross_sum, cross_start, cross_end
        
        if not arr:
            return 0, -1, -1
        
        return max_subarray_rec(arr, 0, len(arr) - 1)
    
    @staticmethod
    def closest_pair_of_points(points: List[Tuple[float, float]]) -> Tuple[float, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Closest Pair of Points using Divide and Conquer
        Time: O(n log n), Space: O(n)
        Returns (distance, (point1, point2))
        """
        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def brute_force(points):
            """Brute force for small arrays"""
            min_dist = float('inf')
            pair = None
            n = len(points)
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = distance(points[i], points[j])
                    if dist < min_dist:
                        min_dist = dist
                        pair = (points[i], points[j])
            
            return min_dist, pair
        
        def closest_pair_rec(px, py):
            n = len(px)
            
            # Base case for small arrays
            if n <= 3:
                return brute_force(px)
            
            # Divide
            mid = n // 2
            midpoint = px[mid]
            
            pyl = [point for point in py if point[0] <= midpoint[0]]
            pyr = [point for point in py if point[0] > midpoint[0]]
            
            # Conquer
            dl, pair_l = closest_pair_rec(px[:mid], pyl)
            dr, pair_r = closest_pair_rec(px[mid:], pyr)
            
            # Find minimum of the two halves
            if dl <= dr:
                d = dl
                min_pair = pair_l
            else:
                d = dr
                min_pair = pair_r
            
            # Check points near the dividing line
            strip = [point for point in py if abs(point[0] - midpoint[0]) < d]
            
            for i in range(len(strip)):
                j = i + 1
                while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
                    dist = distance(strip[i], strip[j])
                    if dist < d:
                        d = dist
                        min_pair = (strip[i], strip[j])
                    j += 1
            
            return d, min_pair
        
        if len(points) < 2:
            return float('inf'), (None, None)
        
        # Sort points by x and y coordinates
        px = sorted(points, key=lambda p: p[0])
        py = sorted(points, key=lambda p: p[1])
        
        return closest_pair_rec(px, py)
    
    @staticmethod
    def matrix_multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """
        Matrix Multiplication using Divide and Conquer (Strassen's Algorithm simplified)
        Time: O(n^2.807), Space: O(n²)
        """
        n = len(A)
        
        # Base case
        if n == 1:
            return [[A[0][0] * B[0][0]]]
        
        # Initialize result matrix
        C = [[0 for _ in range(n)] for _ in range(n)]
        
        # Divide matrices into quarters
        mid = n // 2
        
        # Create submatrices
        A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
        A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
        A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
        A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
        
        B11 = [[B[i][j] for j in range(mid)] for i in range(mid)]
        B12 = [[B[i][j] for j in range(mid, n)] for i in range(mid)]
        B21 = [[B[i][j] for j in range(mid)] for i in range(mid, n)]
        B22 = [[B[i][j] for j in range(mid, n)] for i in range(mid, n)]
        
        # Recursive calls
        C11 = DivideAndConquer._matrix_add(
            DivideAndConquer.matrix_multiply(A11, B11),
            DivideAndConquer.matrix_multiply(A12, B21)
        )
        C12 = DivideAndConquer._matrix_add(
            DivideAndConquer.matrix_multiply(A11, B12),
            DivideAndConquer.matrix_multiply(A12, B22)
        )
        C21 = DivideAndConquer._matrix_add(
            DivideAndConquer.matrix_multiply(A21, B11),
            DivideAndConquer.matrix_multiply(A22, B21)
        )
        C22 = DivideAndConquer._matrix_add(
            DivideAndConquer.matrix_multiply(A21, B12),
            DivideAndConquer.matrix_multiply(A22, B22)
        )
        
        # Combine results
        for i in range(mid):
            for j in range(mid):
                C[i][j] = C11[i][j]
                C[i][j + mid] = C12[i][j]
                C[i + mid][j] = C21[i][j]
                C[i + mid][j + mid] = C22[i][j]
        
        return C
    
    @staticmethod
    def _matrix_add(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """Helper function to add two matrices"""
        n = len(A)
        C = [[0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                C[i][j] = A[i][j] + B[i][j]
        
        return C
    
    @staticmethod
    def power(base: int, exponent: int) -> int:
        """
        Fast Exponentiation using Divide and Conquer
        Time: O(log n), Space: O(log n)
        """
        if exponent == 0:
            return 1
        if exponent == 1:
            return base
        
        if exponent % 2 == 0:
            half_power = DivideAndConquer.power(base, exponent // 2)
            return half_power * half_power
        else:
            return base * DivideAndConquer.power(base, exponent - 1)
    
    @staticmethod
    def count_inversions(arr: List[int]) -> int:
        """
        Count Inversions using Divide and Conquer
        Time: O(n log n), Space: O(n)
        An inversion is when arr[i] > arr[j] and i < j
        """
        def merge_and_count(arr, temp, left, mid, right):
            i, j, k = left, mid + 1, left
            inv_count = 0
            
            while i <= mid and j <= right:
                if arr[i] <= arr[j]:
                    temp[k] = arr[i]
                    i += 1
                else:
                    temp[k] = arr[j]
                    inv_count += (mid - i + 1)  # All elements from i to mid are greater than arr[j]
                    j += 1
                k += 1
            
            # Copy remaining elements
            while i <= mid:
                temp[k] = arr[i]
                i += 1
                k += 1
            
            while j <= right:
                temp[k] = arr[j]
                j += 1
                k += 1
            
            # Copy back to original array
            for i in range(left, right + 1):
                arr[i] = temp[i]
            
            return inv_count
        
        def merge_sort_and_count(arr, temp, left, right):
            inv_count = 0
            if left < right:
                mid = (left + right) // 2
                
                inv_count += merge_sort_and_count(arr, temp, left, mid)
                inv_count += merge_sort_and_count(arr, temp, mid + 1, right)
                inv_count += merge_and_count(arr, temp, left, mid, right)
            
            return inv_count
        
        temp = [0] * len(arr)
        arr_copy = arr.copy()  # Don't modify original array
        return merge_sort_and_count(arr_copy, temp, 0, len(arr) - 1)
    
    @staticmethod
    def binary_search(arr: List[int], target: int) -> int:
        """
        Binary Search using Divide and Conquer
        Time: O(log n), Space: O(log n)
        """
        def binary_search_rec(arr, target, left, right):
            if left > right:
                return -1
            
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] > target:
                return binary_search_rec(arr, target, left, mid - 1)
            else:
                return binary_search_rec(arr, target, mid + 1, right)
        
        return binary_search_rec(arr, target, 0, len(arr) - 1)


# Example usage
if __name__ == "__main__":
    dc = DivideAndConquer()
    
    # Merge Sort
    arr = [64, 34, 25, 12, 22, 11, 90, 5]
    print(f"Original array: {arr}")
    sorted_arr = dc.merge_sort(arr)
    print(f"Merge sorted: {sorted_arr}")
    
    # Quick Sort
    quick_sorted = dc.quick_sort(arr)
    print(f"Quick sorted: {quick_sorted}")
    
    # Maximum Subarray
    subarray_arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum, start, end = dc.maximum_subarray(subarray_arr)
    print(f"Maximum subarray sum: {max_sum} from index {start} to {end}")
    print(f"Subarray: {subarray_arr[start:end+1]}")
    
    # Closest Pair of Points
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    min_dist, closest_pair = dc.closest_pair_of_points(points)
    print(f"Closest pair distance: {min_dist:.2f}")
    print(f"Closest pair: {closest_pair}")
    
    # Matrix Multiplication
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = dc.matrix_multiply(A, B)
    print(f"Matrix A: {A}")
    print(f"Matrix B: {B}")
    print(f"A × B = {C}")
    
    # Fast Exponentiation
    base, exp = 2, 10
    result = dc.power(base, exp)
    print(f"{base}^{exp} = {result}")
    
    # Count Inversions
    inv_arr = [2, 3, 8, 6, 1]
    inversions = dc.count_inversions(inv_arr)
    print(f"Number of inversions in {inv_arr}: {inversions}")
    
    # Binary Search
    search_arr = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    index = dc.binary_search(search_arr, target)
    print(f"Binary search for {target} in {search_arr}: index {index}") 