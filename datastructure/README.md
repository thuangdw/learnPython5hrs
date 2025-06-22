# Data Structures - Senior Python Developer Guide

## Table of Contents
1. [Arrays and Lists](#arrays-and-lists)
2. [Stacks](#stacks)
3. [Queues](#queues)
4. [Linked Lists](#linked-lists)
5. [Trees](#trees)
6. [Graphs](#graphs)
7. [Hash Tables](#hash-tables)
8. [Heaps](#heaps)
9. [Sets](#sets)
10. [Tries](#tries)

## Arrays and Lists

### Dynamic Arrays (Python Lists)
```python
# List operations and time complexities
arr = [1, 2, 3, 4, 5]

# Access: O(1)
print(arr[0])

# Append: O(1) amortized
arr.append(6)

# Insert: O(n)
arr.insert(2, 10)

# Delete: O(n)
arr.pop(2)

# List comprehensions
squares = [x**2 for x in range(10)]
filtered = [x for x in arr if x % 2 == 0]
```

### Array Rotation
```python
def rotate_array(arr, k):
    """Rotate array k positions to the right"""
    n = len(arr)
    k = k % n
    return arr[-k:] + arr[:-k]

# Example usage
nums = [1, 2, 3, 4, 5, 6, 7]
rotated = rotate_array(nums, 3)  # [5, 6, 7, 1, 2, 3, 4]
```

## Stacks

### Stack Implementation
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Using built-in list as stack
stack = []
stack.append(1)  # push
stack.append(2)  # push
top = stack.pop()  # pop
```

### Stack Applications
```python
def is_valid_parentheses(s):
    """Check if parentheses are balanced"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return len(stack) == 0

# Example
print(is_valid_parentheses("({[]})"))  # True
```

## Queues

### Queue Implementation
```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Queue is empty")
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Queue is empty")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Using deque for optimal performance
queue = deque()
queue.append(1)      # enqueue
queue.append(2)      # enqueue
first = queue.popleft()  # dequeue
```

### Priority Queue
```python
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0
    
    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1
    
    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item
    
    def is_empty(self):
        return len(self.heap) == 0

# Example usage
pq = PriorityQueue()
pq.push("task1", 3)
pq.push("task2", 1)
pq.push("task3", 2)
print(pq.pop())  # "task2" (lowest priority first)
```

## Linked Lists

### Singly Linked List
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
    
    def insert_at_end(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def delete(self, val):
        if not self.head:
            return
        
        if self.head.val == val:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        
        if current.next:
            current.next = current.next.next
    
    def display(self):
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result

# Common linked list operations
def reverse_linked_list(head):
    """Reverse a linked list"""
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev

def detect_cycle(head):
    """Floyd's cycle detection algorithm"""
    if not head or not head.next:
        return False
    
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False
```

## Trees

### Binary Tree
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    def __init__(self):
        self.root = None
    
    def inorder_traversal(self, root):
        """Left -> Root -> Right"""
        result = []
        if root:
            result.extend(self.inorder_traversal(root.left))
            result.append(root.val)
            result.extend(self.inorder_traversal(root.right))
        return result
    
    def preorder_traversal(self, root):
        """Root -> Left -> Right"""
        result = []
        if root:
            result.append(root.val)
            result.extend(self.preorder_traversal(root.left))
            result.extend(self.preorder_traversal(root.right))
        return result
    
    def postorder_traversal(self, root):
        """Left -> Right -> Root"""
        result = []
        if root:
            result.extend(self.postorder_traversal(root.left))
            result.extend(self.postorder_traversal(root.right))
            result.append(root.val)
        return result
    
    def level_order_traversal(self, root):
        """Breadth-first traversal"""
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level)
        
        return result
```

### Binary Search Tree
```python
class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, root, val):
        if not root:
            return TreeNode(val)
        
        if val < root.val:
            root.left = self._insert_recursive(root.left, val)
        else:
            root.right = self._insert_recursive(root.right, val)
        
        return root
    
    def search(self, val):
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, root, val):
        if not root or root.val == val:
            return root
        
        if val < root.val:
            return self._search_recursive(root.left, val)
        return self._search_recursive(root.right, val)
    
    def delete(self, val):
        self.root = self._delete_recursive(self.root, val)
    
    def _delete_recursive(self, root, val):
        if not root:
            return root
        
        if val < root.val:
            root.left = self._delete_recursive(root.left, val)
        elif val > root.val:
            root.right = self._delete_recursive(root.right, val)
        else:
            # Node to be deleted found
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            
            # Node with two children
            min_larger_node = self._find_min(root.right)
            root.val = min_larger_node.val
            root.right = self._delete_recursive(root.right, min_larger_node.val)
        
        return root
    
    def _find_min(self, root):
        while root.left:
            root = root.left
        return root
```

## Graphs

### Graph Representation
```python
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
    
    def dfs(self, start, visited=None):
        """Depth-First Search"""
        if visited is None:
            visited = set()
        
        visited.add(start)
        result = [start]
        
        for neighbor in self.graph[start]:
            if neighbor not in visited:
                result.extend(self.dfs(neighbor, visited))
        
        return result
    
    def bfs(self, start):
        """Breadth-First Search"""
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def has_cycle(self):
        """Detect cycle in directed graph"""
        color = {}  # 0: white, 1: gray, 2: black
        
        def dfs_cycle(node):
            if node in color:
                return color[node] == 1  # Gray means cycle
            
            color[node] = 1  # Mark as gray
            
            for neighbor in self.graph[node]:
                if dfs_cycle(neighbor):
                    return True
            
            color[node] = 2  # Mark as black
            return False
        
        for node in self.graph:
            if node not in color:
                if dfs_cycle(node):
                    return True
        
        return False

# Dijkstra's Algorithm
import heapq

def dijkstra(graph, start):
    """Find shortest paths from start to all other vertices"""
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

## Hash Tables

### Hash Table Implementation
```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        bucket.append((key, value))
    
    def get(self, key):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(key)
    
    def remove(self, key):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return
        
        raise KeyError(key)

# Dictionary operations and time complexities
my_dict = {}
my_dict['key1'] = 'value1'  # O(1) average
value = my_dict['key1']     # O(1) average
del my_dict['key1']         # O(1) average

# Dictionary comprehensions
squares_dict = {x: x**2 for x in range(5)}
filtered_dict = {k: v for k, v in my_dict.items() if condition}
```

## Heaps

### Min Heap Implementation
```python
import heapq

class MinHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        heapq.heappush(self.heap, val)
    
    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)
        raise IndexError("Heap is empty")
    
    def peek(self):
        if self.heap:
            return self.heap[0]
        raise IndexError("Heap is empty")
    
    def size(self):
        return len(self.heap)

# Max Heap (using negative values)
class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        heapq.heappush(self.heap, -val)
    
    def pop(self):
        if self.heap:
            return -heapq.heappop(self.heap)
        raise IndexError("Heap is empty")
    
    def peek(self):
        if self.heap:
            return -self.heap[0]
        raise IndexError("Heap is empty")

# Heap operations
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
min_val = heapq.heappop(heap)  # Returns 1

# Convert list to heap
nums = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(nums)  # O(n) time complexity
```

## Sets

### Set Operations
```python
# Set creation and operations
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Union
union_set = set1 | set2  # or set1.union(set2)

# Intersection
intersection_set = set1 & set2  # or set1.intersection(set2)

# Difference
difference_set = set1 - set2  # or set1.difference(set2)

# Symmetric difference
sym_diff_set = set1 ^ set2  # or set1.symmetric_difference(set2)

# Set comprehensions
even_squares = {x**2 for x in range(10) if x % 2 == 0}

# Frozen sets (immutable)
frozen_set = frozenset([1, 2, 3, 4])
```

## Tries

### Trie Implementation
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def delete(self, word):
        def _delete_recursive(node, word, index):
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                node.is_end_of_word = False
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete_child = _delete_recursive(node.children[char], word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end_of_word
            
            return False
        
        _delete_recursive(self.root, word, 0)

# Example usage
trie = Trie()
trie.insert("apple")
trie.insert("app")
print(trie.search("app"))     # True
print(trie.search("apple"))   # True
print(trie.starts_with("ap")) # True
```

## Performance Considerations

### Time Complexities Summary
- **List Access**: O(1)
- **List Append**: O(1) amortized
- **List Insert/Delete**: O(n)
- **Dict Access/Insert/Delete**: O(1) average, O(n) worst case
- **Set Operations**: O(1) average for add/remove/contains
- **Heap Push/Pop**: O(log n)
- **BST Operations**: O(log n) average, O(n) worst case

### Space Complexities
- **Arrays**: O(n)
- **Linked Lists**: O(n)
- **Trees**: O(n)
- **Graphs**: O(V + E) where V is vertices, E is edges
- **Hash Tables**: O(n)

### Best Practices
1. Use appropriate data structures for specific use cases
2. Consider time vs space trade-offs
3. Understand when to use built-in Python data structures vs custom implementations
4. Profile your code to identify bottlenecks
5. Choose the right data structure based on the most common operations 