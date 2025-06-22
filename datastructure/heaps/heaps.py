"""
Heaps Implementation
Comprehensive implementation of heap data structures and algorithms
"""

import heapq
from typing import List, Any, Callable


class MinHeap:
    """Min-heap implementation using array representation"""
    
    def __init__(self):
        self.heap = []
    
    def _parent(self, i):
        """Get parent index"""
        return (i - 1) // 2
    
    def _left_child(self, i):
        """Get left child index"""
        return 2 * i + 1
    
    def _right_child(self, i):
        """Get right child index"""
        return 2 * i + 2
    
    def _swap(self, i, j):
        """Swap elements at indices i and j"""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def _heapify_up(self, i):
        """Restore heap property upward from index i"""
        while i > 0 and self.heap[i] < self.heap[self._parent(i)]:
            self._swap(i, self._parent(i))
            i = self._parent(i)
    
    def _heapify_down(self, i):
        """Restore heap property downward from index i"""
        size = len(self.heap)
        
        while True:
            smallest = i
            left = self._left_child(i)
            right = self._right_child(i)
            
            if left < size and self.heap[left] < self.heap[smallest]:
                smallest = left
            
            if right < size and self.heap[right] < self.heap[smallest]:
                smallest = right
            
            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break
    
    def insert(self, value):
        """Insert value into heap - O(log n)"""
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def extract_min(self):
        """Remove and return minimum element - O(log n)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return min_val
    
    def peek(self):
        """Get minimum element without removing - O(1)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def decrease_key(self, i, new_value):
        """Decrease key at index i to new_value"""
        if new_value > self.heap[i]:
            raise ValueError("New value is larger than current value")
        
        self.heap[i] = new_value
        self._heapify_up(i)
    
    def delete(self, i):
        """Delete element at index i"""
        if i >= len(self.heap):
            raise IndexError("Index out of range")
        
        if i == len(self.heap) - 1:
            self.heap.pop()
            return
        
        self.heap[i] = self.heap.pop()
        
        # Need to heapify both up and down
        if i > 0 and self.heap[i] < self.heap[self._parent(i)]:
            self._heapify_up(i)
        else:
            self._heapify_down(i)
    
    def size(self):
        """Get heap size"""
        return len(self.heap)
    
    def is_empty(self):
        """Check if heap is empty"""
        return len(self.heap) == 0
    
    def __str__(self):
        return str(self.heap)


class MaxHeap:
    """Max-heap implementation"""
    
    def __init__(self):
        self.heap = []
    
    def _parent(self, i):
        return (i - 1) // 2
    
    def _left_child(self, i):
        return 2 * i + 1
    
    def _right_child(self, i):
        return 2 * i + 2
    
    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def _heapify_up(self, i):
        """Restore max-heap property upward"""
        while i > 0 and self.heap[i] > self.heap[self._parent(i)]:
            self._swap(i, self._parent(i))
            i = self._parent(i)
    
    def _heapify_down(self, i):
        """Restore max-heap property downward"""
        size = len(self.heap)
        
        while True:
            largest = i
            left = self._left_child(i)
            right = self._right_child(i)
            
            if left < size and self.heap[left] > self.heap[largest]:
                largest = left
            
            if right < size and self.heap[right] > self.heap[largest]:
                largest = right
            
            if largest != i:
                self._swap(i, largest)
                i = largest
            else:
                break
    
    def insert(self, value):
        """Insert value into max-heap"""
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def extract_max(self):
        """Remove and return maximum element"""
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return max_val
    
    def peek(self):
        """Get maximum element without removing"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0


class PriorityQueue:
    """Priority queue using min-heap (lower values = higher priority)"""
    
    def __init__(self):
        self.heap = []
        self.index = 0  # For tie-breaking
    
    def push(self, item, priority):
        """Add item with given priority"""
        heapq.heappush(self.heap, (priority, self.index, item))
        self.index += 1
    
    def pop(self):
        """Remove and return highest priority item"""
        if not self.heap:
            raise IndexError("Priority queue is empty")
        
        priority, _, item = heapq.heappop(self.heap)
        return item, priority
    
    def peek(self):
        """Get highest priority item without removing"""
        if not self.heap:
            raise IndexError("Priority queue is empty")
        
        priority, _, item = self.heap[0]
        return item, priority
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def size(self):
        return len(self.heap)


class MedianFinder:
    """Find median of a stream of numbers using two heaps"""
    
    def __init__(self):
        self.max_heap = []  # For smaller half (use negative values for max behavior)
        self.min_heap = []  # For larger half
    
    def add_number(self, num):
        """Add a number to the data structure"""
        # Add to max_heap first (smaller half)
        heapq.heappush(self.max_heap, -num)
        
        # Move the largest from max_heap to min_heap
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        
        # Balance the heaps
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def find_median(self):
        """Find the median of all numbers added so far"""
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0


def heap_sort(arr):
    """
    Heap sort algorithm
    Time: O(n log n), Space: O(1)
    """
    def heapify(arr, n, i):
        """Heapify subtree rooted at index i"""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] > arr[largest]:
            largest = left
        
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move current root to end
        heapify(arr, i, 0)  # Heapify reduced heap
    
    return arr


def find_k_largest(arr, k):
    """
    Find k largest elements using min-heap
    Time: O(n log k), Space: O(k)
    """
    if k <= 0:
        return []
    
    heap = []
    
    for num in arr:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)
    
    return sorted(heap, reverse=True)


def find_k_smallest(arr, k):
    """
    Find k smallest elements using max-heap
    Time: O(n log k), Space: O(k)
    """
    if k <= 0:
        return []
    
    heap = []
    
    for num in arr:
        if len(heap) < k:
            heapq.heappush(heap, -num)  # Use negative for max-heap behavior
        elif num < -heap[0]:
            heapq.heapreplace(heap, -num)
    
    return sorted([-x for x in heap])


def merge_k_sorted_lists(lists):
    """
    Merge k sorted lists using min-heap
    Time: O(n log k), Space: O(k)
    """
    heap = []
    result = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from the same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result


class BinaryHeap:
    """Generic binary heap with custom comparison function"""
    
    def __init__(self, key_func=None, reverse=False):
        self.heap = []
        self.key_func = key_func or (lambda x: x)
        self.reverse = reverse
    
    def _compare(self, a, b):
        """Compare two elements based on key function and reverse flag"""
        key_a, key_b = self.key_func(a), self.key_func(b)
        if self.reverse:
            return key_a > key_b
        return key_a < key_b
    
    def _heapify_up(self, i):
        """Restore heap property upward"""
        while i > 0:
            parent = (i - 1) // 2
            if self._compare(self.heap[i], self.heap[parent]):
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break
    
    def _heapify_down(self, i):
        """Restore heap property downward"""
        size = len(self.heap)
        
        while True:
            target = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < size and self._compare(self.heap[left], self.heap[target]):
                target = left
            
            if right < size and self._compare(self.heap[right], self.heap[target]):
                target = right
            
            if target != i:
                self.heap[i], self.heap[target] = self.heap[target], self.heap[i]
                i = target
            else:
                break
    
    def push(self, item):
        """Add item to heap"""
        self.heap.append(item)
        self._heapify_up(len(self.heap) - 1)
    
    def pop(self):
        """Remove and return top element"""
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        top = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return top
    
    def peek(self):
        """Get top element without removing"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0


def huffman_coding_example():
    """Example of Huffman coding using heaps"""
    import heapq
    from collections import defaultdict, Counter
    
    class Node:
        def __init__(self, char=None, freq=0, left=None, right=None):
            self.char = char
            self.freq = freq
            self.left = left
            self.right = right
        
        def __lt__(self, other):
            return self.freq < other.freq
    
    def build_huffman_tree(text):
        """Build Huffman tree from text"""
        if not text:
            return None
        
        # Count frequencies
        freq = Counter(text)
        
        # Create leaf nodes and add to heap
        heap = [Node(char, freq) for char, freq in freq.items()]
        heapq.heapify(heap)
        
        # Build tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = Node(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0]
    
    def get_codes(root):
        """Get Huffman codes from tree"""
        if not root:
            return {}
        
        codes = {}
        
        def dfs(node, code):
            if node.char:  # Leaf node
                codes[node.char] = code or '0'  # Handle single character case
            else:
                if node.left:
                    dfs(node.left, code + '0')
                if node.right:
                    dfs(node.right, code + '1')
        
        dfs(root, '')
        return codes
    
    return build_huffman_tree, get_codes


if __name__ == "__main__":
    # Demo usage
    min_heap = MinHeap()
    
    # Insert elements
    for val in [3, 1, 6, 5, 2, 4]:
        min_heap.insert(val)
    
    print(f"Min heap: {min_heap}")
    print(f"Extract min: {min_heap.extract_min()}")
    print(f"After extraction: {min_heap}")
    
    # Priority queue example
    pq = PriorityQueue()
    pq.push("Task C", 3)
    pq.push("Task A", 1)
    pq.push("Task B", 2)
    
    while not pq.is_empty():
        task, priority = pq.pop()
        print(f"Processing {task} (priority {priority})")
    
    # Median finder example
    mf = MedianFinder()
    for num in [1, 2, 3, 4, 5]:
        mf.add_number(num)
        print(f"Added {num}, median: {mf.find_median()}")
    
    # Heap sort example
    arr = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr = heap_sort(arr.copy())
    print(f"Original: {arr}")
    print(f"Heap sorted: {sorted_arr}")
    
    # Find k largest
    k_largest = find_k_largest([3, 1, 6, 5, 2, 4], 3)
    print(f"3 largest elements: {k_largest}") 