"""
Test cases for Arrays and Lists Implementation
"""

import unittest
from arrays import (
    DynamicArray, rotate_array, find_maximum_subarray_sum, 
    two_sum, remove_duplicates, merge_sorted_arrays, 
    find_peak_element, Matrix, spiral_matrix_traversal,
    list_comprehension_examples
)


class TestDynamicArray(unittest.TestCase):
    
    def test_initialization(self):
        arr = DynamicArray()
        self.assertEqual(len(arr), 0)
        self.assertEqual(arr.capacity, 10)
    
    def test_append_and_access(self):
        arr = DynamicArray()
        for i in range(5):
            arr.append(i)
        
        self.assertEqual(len(arr), 5)
        for i in range(5):
            self.assertEqual(arr[i], i)
    
    def test_insert(self):
        arr = DynamicArray()
        for i in range(5):
            arr.append(i)
        
        arr.insert(2, 99)
        self.assertEqual(arr[2], 99)
        self.assertEqual(len(arr), 6)
    
    def test_delete(self):
        arr = DynamicArray()
        for i in range(5):
            arr.append(i)
        
        arr.delete(2)
        self.assertEqual(len(arr), 4)
        self.assertEqual(arr[2], 3)
    
    def test_resize(self):
        arr = DynamicArray(capacity=2)
        for i in range(5):
            arr.append(i)
        
        self.assertEqual(len(arr), 5)
        self.assertGreaterEqual(arr.capacity, 5)
    
    def test_index_error(self):
        arr = DynamicArray()
        arr.append(1)
        
        with self.assertRaises(IndexError):
            _ = arr[5]
        
        with self.assertRaises(IndexError):
            arr[5] = 10


class TestArrayOperations(unittest.TestCase):
    
    def test_rotate_array(self):
        arr = [1, 2, 3, 4, 5, 6, 7]
        result = rotate_array(arr.copy(), 3)
        expected = [5, 6, 7, 1, 2, 3, 4]
        self.assertEqual(result, expected)
    
    def test_rotate_array_edge_cases(self):
        # Empty array
        self.assertEqual(rotate_array([], 3), [])
        
        # Rotation greater than array length
        arr = [1, 2, 3]
        result = rotate_array(arr.copy(), 5)  # 5 % 3 = 2
        expected = [2, 3, 1]
        self.assertEqual(result, expected)
    
    def test_maximum_subarray_sum(self):
        arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
        result = find_maximum_subarray_sum(arr)
        self.assertEqual(result, 6)  # [4, -1, 2, 1]
        
        # All negative numbers
        arr = [-5, -2, -8, -1]
        result = find_maximum_subarray_sum(arr)
        self.assertEqual(result, -1)
    
    def test_two_sum(self):
        nums = [2, 7, 11, 15]
        target = 9
        result = two_sum(nums, target)
        self.assertEqual(result, [0, 1])
        
        # No solution
        nums = [1, 2, 3]
        target = 7
        result = two_sum(nums, target)
        self.assertEqual(result, [])
    
    def test_remove_duplicates(self):
        arr = [1, 1, 2, 2, 3, 4, 4, 5]
        new_length = remove_duplicates(arr)
        self.assertEqual(new_length, 5)
        self.assertEqual(arr[:new_length], [1, 2, 3, 4, 5])
    
    def test_merge_sorted_arrays(self):
        arr1 = [1, 3, 5, 7]
        arr2 = [2, 4, 6, 8]
        result = merge_sorted_arrays(arr1, arr2)
        expected = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(result, expected)
        
        # Empty arrays
        self.assertEqual(merge_sorted_arrays([], [1, 2]), [1, 2])
        self.assertEqual(merge_sorted_arrays([1, 2], []), [1, 2])
    
    def test_find_peak_element(self):
        arr = [1, 2, 3, 1]
        peak_index = find_peak_element(arr)
        self.assertEqual(peak_index, 2)
        
        arr = [1, 2, 1, 3, 5, 6, 4]
        peak_index = find_peak_element(arr)
        self.assertIn(peak_index, [1, 5])  # Multiple peaks possible


class TestMatrix(unittest.TestCase):
    
    def test_matrix_creation(self):
        mat = Matrix(3, 3, 5)
        self.assertEqual(mat.rows, 3)
        self.assertEqual(mat.cols, 3)
        self.assertEqual(mat.get(0, 0), 5)
    
    def test_matrix_operations(self):
        mat = Matrix(2, 2)
        mat.set(0, 0, 1)
        mat.set(0, 1, 2)
        mat.set(1, 0, 3)
        mat.set(1, 1, 4)
        
        self.assertEqual(mat.get(0, 0), 1)
        self.assertEqual(mat.get(1, 1), 4)
    
    def test_matrix_transpose(self):
        mat = Matrix(2, 3)
        mat.set(0, 0, 1)
        mat.set(0, 1, 2)
        mat.set(0, 2, 3)
        mat.set(1, 0, 4)
        mat.set(1, 1, 5)
        mat.set(1, 2, 6)
        
        transposed = mat.transpose()
        self.assertEqual(transposed.rows, 3)
        self.assertEqual(transposed.cols, 2)
        self.assertEqual(transposed.get(0, 0), 1)
        self.assertEqual(transposed.get(2, 1), 6)
    
    def test_matrix_multiplication(self):
        mat1 = Matrix(2, 2)
        mat1.set(0, 0, 1)
        mat1.set(0, 1, 2)
        mat1.set(1, 0, 3)
        mat1.set(1, 1, 4)
        
        mat2 = Matrix(2, 2)
        mat2.set(0, 0, 5)
        mat2.set(0, 1, 6)
        mat2.set(1, 0, 7)
        mat2.set(1, 1, 8)
        
        result = mat1.multiply(mat2)
        self.assertEqual(result.get(0, 0), 19)  # 1*5 + 2*7
        self.assertEqual(result.get(0, 1), 22)  # 1*6 + 2*8
        self.assertEqual(result.get(1, 0), 43)  # 3*5 + 4*7
        self.assertEqual(result.get(1, 1), 50)  # 3*6 + 4*8
    
    def test_matrix_index_error(self):
        mat = Matrix(2, 2)
        
        with self.assertRaises(IndexError):
            mat.get(3, 0)
        
        with self.assertRaises(IndexError):
            mat.set(0, 3, 5)


class TestSpiralTraversal(unittest.TestCase):
    
    def test_spiral_traversal(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        result = spiral_matrix_traversal(matrix)
        expected = [1, 2, 3, 6, 9, 8, 7, 4, 5]
        self.assertEqual(result, expected)
    
    def test_spiral_traversal_rectangle(self):
        matrix = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ]
        result = spiral_matrix_traversal(matrix)
        expected = [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
        self.assertEqual(result, expected)
    
    def test_spiral_traversal_single_row(self):
        matrix = [[1, 2, 3, 4]]
        result = spiral_matrix_traversal(matrix)
        expected = [1, 2, 3, 4]
        self.assertEqual(result, expected)
    
    def test_spiral_traversal_single_column(self):
        matrix = [[1], [2], [3], [4]]
        result = spiral_matrix_traversal(matrix)
        expected = [1, 2, 3, 4]
        self.assertEqual(result, expected)
    
    def test_spiral_traversal_empty(self):
        self.assertEqual(spiral_matrix_traversal([]), [])
        self.assertEqual(spiral_matrix_traversal([[]]), [])


class TestListComprehensions(unittest.TestCase):
    
    def test_comprehension_examples(self):
        examples = list_comprehension_examples()
        
        self.assertEqual(examples['squares'], [0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
        self.assertEqual(examples['evens'], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
        self.assertEqual(examples['flattened'], [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(examples['square_dict'], {0: 0, 1: 1, 2: 4, 3: 9, 4: 16})
        self.assertEqual(examples['unique_squares'], {0, 1, 4})


if __name__ == '__main__':
    unittest.main() 