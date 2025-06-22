"""
Arrays and Lists Implementation
Comprehensive implementation of array/list operations and algorithms
"""

class DynamicArray:
    """Custom dynamic array implementation to demonstrate concepts"""
    
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.size = 0
        self.data = [None] * capacity
    
    def __getitem__(self, index):
        """Access element at index - O(1)"""
        if 0 <= index < self.size:
            return self.data[index]
        raise IndexError("Index out of range")
    
    def __setitem__(self, index, value):
        """Set element at index - O(1)"""
        if 0 <= index < self.size:
            self.data[index] = value
        else:
            raise IndexError("Index out of range")
    
    def append(self, value):
        """Append element - O(1) amortized"""
        if self.size >= self.capacity:
            self._resize()
        self.data[self.size] = value
        self.size += 1
    
    def insert(self, index, value):
        """Insert element at index - O(n)"""
        if index < 0 or index > self.size:
            raise IndexError("Index out of range")
        
        if self.size >= self.capacity:
            self._resize()
        
        # Shift elements to the right
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i - 1]
        
        self.data[index] = value
        self.size += 1
    
    def delete(self, index):
        """Delete element at index - O(n)"""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        
        # Shift elements to the left
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]
        
        self.size -= 1
        return self.data[index]
    
    def _resize(self):
        """Double the capacity when needed"""
        old_data = self.data
        self.capacity *= 2
        self.data = [None] * self.capacity
        
        for i in range(self.size):
            self.data[i] = old_data[i]
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        return str([self.data[i] for i in range(self.size)])


def rotate_array(arr, k):
    """
    Rotate array k positions to the right
    Time: O(n), Space: O(1)
    """
    if not arr:
        return arr
    
    n = len(arr)
    k = k % n
    
    # Reverse entire array
    reverse_array(arr, 0, n - 1)
    # Reverse first k elements
    reverse_array(arr, 0, k - 1)
    # Reverse remaining elements
    reverse_array(arr, k, n - 1)
    
    return arr


def reverse_array(arr, start, end):
    """Helper function to reverse array between indices"""
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1


def find_maximum_subarray_sum(arr):
    """
    Kadane's Algorithm - Maximum subarray sum
    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0
    
    max_ending_here = max_so_far = arr[0]
    
    for i in range(1, len(arr)):
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        max_so_far = max(max_so_far, max_ending_here)
    
    return max_so_far


def two_sum(nums, target):
    """
    Find two numbers that add up to target
    Time: O(n), Space: O(n)
    """
    num_map = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    return []


def remove_duplicates(arr):
    """
    Remove duplicates from sorted array
    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0
    
    write_index = 1
    
    for read_index in range(1, len(arr)):
        if arr[read_index] != arr[read_index - 1]:
            arr[write_index] = arr[read_index]
            write_index += 1
    
    return write_index


def merge_sorted_arrays(arr1, arr2):
    """
    Merge two sorted arrays
    Time: O(m + n), Space: O(m + n)
    """
    result = []
    i = j = 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    # Add remaining elements
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    
    return result


def find_peak_element(arr):
    """
    Find a peak element in array
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left


class Matrix:
    """2D Matrix operations"""
    
    def __init__(self, rows, cols, default_value=0):
        self.rows = rows
        self.cols = cols
        self.data = [[default_value for _ in range(cols)] for _ in range(rows)]
    
    def get(self, row, col):
        """Get element at (row, col)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.data[row][col]
        raise IndexError("Matrix indices out of range")
    
    def set(self, row, col, value):
        """Set element at (row, col)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.data[row][col] = value
        else:
            raise IndexError("Matrix indices out of range")
    
    def transpose(self):
        """Return transpose of matrix"""
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.set(j, i, self.get(i, j))
        return result
    
    def multiply(self, other):
        """Matrix multiplication"""
        if self.cols != other.rows:
            raise ValueError("Incompatible matrix dimensions")
        
        result = Matrix(self.rows, other.cols)
        
        for i in range(self.rows):
            for j in range(other.cols):
                sum_val = 0
                for k in range(self.cols):
                    sum_val += self.get(i, k) * other.get(k, j)
                result.set(i, j, sum_val)
        
        return result
    
    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])


def spiral_matrix_traversal(matrix):
    """
    Traverse matrix in spiral order
    Time: O(m * n), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Traverse down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Traverse left (if we still have rows)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Traverse up (if we still have columns)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result


# List comprehension examples and utilities
def list_comprehension_examples():
    """Examples of advanced list comprehensions"""
    
    # Basic comprehensions
    squares = [x**2 for x in range(10)]
    evens = [x for x in range(20) if x % 2 == 0]
    
    # Nested comprehensions
    matrix = [[i * j for j in range(3)] for i in range(3)]
    
    # Flattening nested lists
    nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flattened = [item for sublist in nested for item in sublist]
    
    # Dictionary comprehensions
    square_dict = {x: x**2 for x in range(5)}
    
    # Set comprehensions
    unique_squares = {x**2 for x in [-2, -1, 0, 1, 2]}
    
    return {
        'squares': squares,
        'evens': evens,
        'matrix': matrix,
        'flattened': flattened,
        'square_dict': square_dict,
        'unique_squares': unique_squares
    }


if __name__ == "__main__":
    # Demo usage
    arr = DynamicArray()
    for i in range(5):
        arr.append(i)
    print(f"Dynamic Array: {arr}")
    
    # Rotation example
    nums = [1, 2, 3, 4, 5, 6, 7]
    rotated = rotate_array(nums.copy(), 3)
    print(f"Rotated array: {rotated}")
    
    # Matrix example
    mat = Matrix(3, 3)
    for i in range(3):
        for j in range(3):
            mat.set(i, j, i + j)
    print(f"Matrix:\n{mat}") 