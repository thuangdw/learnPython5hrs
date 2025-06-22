"""
Test cases for Heaps Implementation
"""

import unittest
from heaps import (
    MinHeap, MaxHeap, PriorityQueue, MedianFinder, BinaryHeap,
    heap_sort, find_k_largest, find_k_smallest, merge_k_sorted_lists,
    huffman_coding_example
)


class TestMinHeap(unittest.TestCase):
    
    def test_insert_and_peek(self):
        heap = MinHeap()
        
        heap.insert(5)
        heap.insert(3)
        heap.insert(8)
        heap.insert(1)
        
        self.assertEqual(heap.peek(), 1)
        self.assertEqual(heap.size(), 4)
    
    def test_extract_min(self):
        heap = MinHeap()
        
        values = [5, 3, 8, 1, 9, 2]
        for val in values:
            heap.insert(val)
        
        extracted = []
        while not heap.is_empty():
            extracted.append(heap.extract_min())
        
        self.assertEqual(extracted, [1, 2, 3, 5, 8, 9])
    
    def test_empty_heap_operations(self):
        heap = MinHeap()
        
        self.assertTrue(heap.is_empty())
        self.assertEqual(heap.size(), 0)
        
        with self.assertRaises(IndexError):
            heap.peek()
        
        with self.assertRaises(IndexError):
            heap.extract_min()
    
    def test_decrease_key(self):
        heap = MinHeap()
        
        heap.insert(10)
        heap.insert(5)
        heap.insert(15)
        
        # Decrease key at index 2 (value 15) to 3
        heap.decrease_key(2, 3)
        
        self.assertEqual(heap.peek(), 3)
    
    def test_delete(self):
        heap = MinHeap()
        
        values = [5, 3, 8, 1]
        for val in values:
            heap.insert(val)
        
        # Delete element at index 1
        heap.delete(1)
        
        self.assertEqual(heap.size(), 3)
        # Extract all to verify heap property maintained
        extracted = []
        while not heap.is_empty():
            extracted.append(heap.extract_min())
        
        # Should be sorted and missing one element
        self.assertEqual(len(extracted), 3)
        self.assertEqual(extracted, sorted(extracted))


class TestMaxHeap(unittest.TestCase):
    
    def test_insert_and_peek(self):
        heap = MaxHeap()
        
        heap.insert(5)
        heap.insert(3)
        heap.insert(8)
        heap.insert(1)
        
        self.assertEqual(heap.peek(), 8)
        self.assertEqual(heap.size(), 4)
    
    def test_extract_max(self):
        heap = MaxHeap()
        
        values = [5, 3, 8, 1, 9, 2]
        for val in values:
            heap.insert(val)
        
        extracted = []
        while not heap.is_empty():
            extracted.append(heap.extract_max())
        
        self.assertEqual(extracted, [9, 8, 5, 3, 2, 1])
    
    def test_empty_heap_operations(self):
        heap = MaxHeap()
        
        self.assertTrue(heap.is_empty())
        
        with self.assertRaises(IndexError):
            heap.peek()
        
        with self.assertRaises(IndexError):
            heap.extract_max()


class TestPriorityQueue(unittest.TestCase):
    
    def test_basic_operations(self):
        pq = PriorityQueue()
        
        pq.push("Task C", 3)
        pq.push("Task A", 1)
        pq.push("Task B", 2)
        
        self.assertFalse(pq.is_empty())
        self.assertEqual(pq.size(), 3)
        
        # Should return highest priority (lowest number) first
        item, priority = pq.pop()
        self.assertEqual(item, "Task A")
        self.assertEqual(priority, 1)
    
    def test_priority_order(self):
        pq = PriorityQueue()
        
        tasks = [("Low", 5), ("High", 1), ("Medium", 3), ("Urgent", 0)]
        
        for task, priority in tasks:
            pq.push(task, priority)
        
        expected_order = ["Urgent", "High", "Medium", "Low"]
        actual_order = []
        
        while not pq.is_empty():
            item, _ = pq.pop()
            actual_order.append(item)
        
        self.assertEqual(actual_order, expected_order)
    
    def test_peek(self):
        pq = PriorityQueue()
        
        pq.push("Task", 5)
        
        item, priority = pq.peek()
        self.assertEqual(item, "Task")
        self.assertEqual(priority, 5)
        self.assertEqual(pq.size(), 1)  # Should not remove item
    
    def test_empty_queue_operations(self):
        pq = PriorityQueue()
        
        self.assertTrue(pq.is_empty())
        
        with self.assertRaises(IndexError):
            pq.pop()
        
        with self.assertRaises(IndexError):
            pq.peek()


class TestMedianFinder(unittest.TestCase):
    
    def test_odd_number_of_elements(self):
        mf = MedianFinder()
        
        numbers = [1, 3, 5]
        for num in numbers:
            mf.add_number(num)
        
        self.assertEqual(mf.find_median(), 3)
    
    def test_even_number_of_elements(self):
        mf = MedianFinder()
        
        numbers = [1, 2, 3, 4]
        for num in numbers:
            mf.add_number(num)
        
        self.assertEqual(mf.find_median(), 2.5)
    
    def test_single_element(self):
        mf = MedianFinder()
        
        mf.add_number(5)
        self.assertEqual(mf.find_median(), 5)
    
    def test_streaming_median(self):
        mf = MedianFinder()
        
        # Test median after each addition
        mf.add_number(1)
        self.assertEqual(mf.find_median(), 1)
        
        mf.add_number(2)
        self.assertEqual(mf.find_median(), 1.5)
        
        mf.add_number(3)
        self.assertEqual(mf.find_median(), 2)
        
        mf.add_number(4)
        self.assertEqual(mf.find_median(), 2.5)


class TestBinaryHeap(unittest.TestCase):
    
    def test_min_heap_behavior(self):
        heap = BinaryHeap()  # Default is min heap
        
        values = [5, 3, 8, 1, 9, 2]
        for val in values:
            heap.push(val)
        
        extracted = []
        while not heap.is_empty():
            extracted.append(heap.pop())
        
        self.assertEqual(extracted, [1, 2, 3, 5, 8, 9])
    
    def test_max_heap_behavior(self):
        heap = BinaryHeap(reverse=True)  # Max heap
        
        values = [5, 3, 8, 1, 9, 2]
        for val in values:
            heap.push(val)
        
        extracted = []
        while not heap.is_empty():
            extracted.append(heap.pop())
        
        self.assertEqual(extracted, [9, 8, 5, 3, 2, 1])
    
    def test_custom_key_function(self):
        # Heap of strings by length
        heap = BinaryHeap(key_func=len)
        
        words = ["apple", "hi", "banana", "a", "elephant"]
        for word in words:
            heap.push(word)
        
        extracted = []
        while not heap.is_empty():
            extracted.append(heap.pop())
        
        # Should be ordered by string length
        expected_lengths = [1, 2, 5, 6, 8]  # a, hi, apple, banana, elephant
        actual_lengths = [len(word) for word in extracted]
        self.assertEqual(actual_lengths, expected_lengths)


class TestHeapAlgorithms(unittest.TestCase):
    
    def test_heap_sort(self):
        arr = [64, 34, 25, 12, 22, 11, 90]
        sorted_arr = heap_sort(arr.copy())
        
        self.assertEqual(sorted_arr, [11, 12, 22, 25, 34, 64, 90])
        # Original array should be modified in-place
        self.assertEqual(arr, [64, 34, 25, 12, 22, 11, 90])  # Original unchanged
    
    def test_heap_sort_empty_array(self):
        arr = []
        sorted_arr = heap_sort(arr)
        self.assertEqual(sorted_arr, [])
    
    def test_heap_sort_single_element(self):
        arr = [42]
        sorted_arr = heap_sort(arr)
        self.assertEqual(sorted_arr, [42])
    
    def test_find_k_largest(self):
        arr = [3, 1, 6, 5, 2, 4]
        k_largest = find_k_largest(arr, 3)
        
        self.assertEqual(k_largest, [6, 5, 4])
    
    def test_find_k_largest_edge_cases(self):
        arr = [1, 2, 3]
        
        # k = 0
        self.assertEqual(find_k_largest(arr, 0), [])
        
        # k > array length
        result = find_k_largest(arr, 5)
        self.assertEqual(sorted(result, reverse=True), [3, 2, 1])
    
    def test_find_k_smallest(self):
        arr = [3, 1, 6, 5, 2, 4]
        k_smallest = find_k_smallest(arr, 3)
        
        self.assertEqual(k_smallest, [1, 2, 3])
    
    def test_find_k_smallest_edge_cases(self):
        arr = [3, 2, 1]
        
        # k = 0
        self.assertEqual(find_k_smallest(arr, 0), [])
        
        # k > array length
        result = find_k_smallest(arr, 5)
        self.assertEqual(result, [1, 2, 3])
    
    def test_merge_k_sorted_lists(self):
        lists = [
            [1, 4, 5],
            [1, 3, 4],
            [2, 6]
        ]
        
        merged = merge_k_sorted_lists(lists)
        self.assertEqual(merged, [1, 1, 2, 3, 4, 4, 5, 6])
    
    def test_merge_k_sorted_lists_empty_lists(self):
        lists = [[], [1], []]
        merged = merge_k_sorted_lists(lists)
        self.assertEqual(merged, [1])
    
    def test_merge_k_sorted_lists_single_list(self):
        lists = [[1, 2, 3, 4]]
        merged = merge_k_sorted_lists(lists)
        self.assertEqual(merged, [1, 2, 3, 4])


class TestHuffmanCoding(unittest.TestCase):
    
    def test_huffman_coding(self):
        build_huffman_tree, get_codes = huffman_coding_example()
        
        text = "hello"
        tree = build_huffman_tree(text)
        codes = get_codes(tree)
        
        # Check that we have codes for all characters
        unique_chars = set(text)
        self.assertEqual(set(codes.keys()), unique_chars)
        
        # Check that codes are valid (no code is prefix of another)
        code_values = list(codes.values())
        for i, code1 in enumerate(code_values):
            for j, code2 in enumerate(code_values):
                if i != j:
                    self.assertFalse(code1.startswith(code2))
                    self.assertFalse(code2.startswith(code1))
    
    def test_huffman_single_character(self):
        build_huffman_tree, get_codes = huffman_coding_example()
        
        text = "aaaa"
        tree = build_huffman_tree(text)
        codes = get_codes(tree)
        
        # Single character should get code '0'
        self.assertEqual(codes['a'], '0')
    
    def test_huffman_empty_text(self):
        build_huffman_tree, get_codes = huffman_coding_example()
        
        text = ""
        tree = build_huffman_tree(text)
        codes = get_codes(tree)
        
        self.assertEqual(codes, {})


class TestHeapStress(unittest.TestCase):
    
    def test_large_heap_operations(self):
        heap = MinHeap()
        
        # Insert many elements
        n = 1000
        import random
        values = list(range(n))
        random.shuffle(values)
        
        for val in values:
            heap.insert(val)
        
        # Extract all elements - should be in sorted order
        extracted = []
        while not heap.is_empty():
            extracted.append(heap.extract_min())
        
        self.assertEqual(extracted, list(range(n)))
    
    def test_priority_queue_stress(self):
        pq = PriorityQueue()
        
        # Add many tasks with random priorities
        n = 500
        import random
        
        for i in range(n):
            priority = random.randint(1, 100)
            pq.push(f"task{i}", priority)
        
        # Extract all - should be in priority order
        prev_priority = -1
        while not pq.is_empty():
            _, priority = pq.pop()
            self.assertGreaterEqual(priority, prev_priority)
            prev_priority = priority


if __name__ == '__main__':
    unittest.main() 