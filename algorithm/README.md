# Algorithms - Senior Python Developer Guide

## Table of Contents
1. [Sorting Algorithms](#sorting-algorithms)
2. [Searching Algorithms](#searching-algorithms)
3. [Graph Algorithms](#graph-algorithms)
4. [Dynamic Programming](#dynamic-programming)
5. [Greedy Algorithms](#greedy-algorithms)
6. [Divide and Conquer](#divide-and-conquer)
7. [Backtracking](#backtracking)
8. [String Algorithms](#string-algorithms)
9. [Mathematical Algorithms](#mathematical-algorithms)
10. [Bit Manipulation](#bit-manipulation)

## Sorting Algorithms

### Quick Sort
```python
def quicksort(arr):
    """
    Average: O(n log n), Worst: O(n²), Space: O(log n)
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

def quicksort_inplace(arr, low=0, high=None):
    """In-place version with better space complexity"""
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

### Merge Sort
```python
def mergesort(arr):
    """
    Time: O(n log n), Space: O(n)
    Stable sorting algorithm
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
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
```

### Heap Sort
```python
import heapq

def heapsort(arr):
    """
    Time: O(n log n), Space: O(1)
    Not stable but in-place
    """
    # Build max heap
    for i in range(len(arr) // 2 - 1, -1, -1):
        heapify(arr, len(arr), i)
    
    # Extract elements from heap one by one
    for i in range(len(arr) - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move current root to end
        heapify(arr, i, 0)
    
    return arr

def heapify(arr, n, i):
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

# Using Python's heapq for simpler implementation
def heapsort_simple(arr):
    return [heapq.heappop(arr) for _ in range(len(arr))]
```

### Counting Sort
```python
def counting_sort(arr, max_val=None):
    """
    Time: O(n + k), Space: O(k)
    Where k is the range of input values
    Works only with integers in a specific range
    """
    if not arr:
        return arr
    
    if max_val is None:
        max_val = max(arr)
    
    count = [0] * (max_val + 1)
    
    # Count occurrences
    for num in arr:
        count[num] += 1
    
    # Reconstruct sorted array
    result = []
    for i, freq in enumerate(count):
        result.extend([i] * freq)
    
    return result
```

## Searching Algorithms

### Binary Search
```python
def binary_search(arr, target):
    """
    Time: O(log n), Space: O(1)
    Array must be sorted
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

def binary_search_recursive(arr, target, left=0, right=None):
    """Recursive version"""
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Binary search variants
def find_first_occurrence(arr, target):
    """Find the first occurrence of target"""
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

def find_last_occurrence(arr, target):
    """Find the last occurrence of target"""
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
```

### Ternary Search
```python
def ternary_search(arr, target):
    """
    Time: O(log₃ n), Space: O(1)
    For sorted arrays, divides into three parts
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
```

## Graph Algorithms

### Depth-First Search (DFS)
```python
from collections import defaultdict

def dfs_iterative(graph, start):
    """Iterative DFS using stack"""
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            
            # Add neighbors in reverse order for consistent traversal
            for neighbor in reversed(graph[vertex]):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result

def dfs_recursive(graph, start, visited=None, result=None):
    """Recursive DFS"""
    if visited is None:
        visited = set()
    if result is None:
        result = []
    
    visited.add(start)
    result.append(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, result)
    
    return result
```

### Breadth-First Search (BFS)
```python
from collections import deque

def bfs(graph, start):
    """BFS traversal"""
    visited = set()
    queue = deque([start])
    result = []
    
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return result

def bfs_shortest_path(graph, start, end):
    """Find shortest path using BFS"""
    if start == end:
        return [start]
    
    visited = set()
    queue = deque([(start, [start])])
    
    while queue:
        vertex, path = queue.popleft()
        
        if vertex not in visited:
            visited.add(vertex)
            
            for neighbor in graph[vertex]:
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    
    return None  # No path found
```

### Dijkstra's Algorithm
```python
import heapq

def dijkstra(graph, start):
    """
    Find shortest paths from start to all vertices
    Time: O((V + E) log V), Space: O(V)
    """
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    previous = {vertex: None for vertex in graph}
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_vertex in visited:
            continue
        
        visited.add(current_vertex)
        
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))
    
    return distances, previous

def reconstruct_path(previous, start, end):
    """Reconstruct path from Dijkstra's result"""
    path = []
    current = end
    
    while current is not None:
        path.insert(0, current)
        current = previous[current]
    
    return path if path[0] == start else None
```

### Floyd-Warshall Algorithm
```python
def floyd_warshall(graph):
    """
    All-pairs shortest paths
    Time: O(V³), Space: O(V²)
    """
    vertices = list(graph.keys())
    dist = {}
    
    # Initialize distances
    for i in vertices:
        dist[i] = {}
        for j in vertices:
            if i == j:
                dist[i][j] = 0
            elif j in graph[i]:
                dist[i][j] = graph[i][j]
            else:
                dist[i][j] = float('infinity')
    
    # Floyd-Warshall algorithm
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist
```

### Topological Sort
```python
def topological_sort_dfs(graph):
    """Topological sort using DFS"""
    visited = set()
    stack = []
    
    def dfs(vertex):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(vertex)
    
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)
    
    return stack[::-1]

def topological_sort_kahn(graph):
    """Kahn's algorithm for topological sorting"""
    in_degree = {vertex: 0 for vertex in graph}
    
    # Calculate in-degrees
    for vertex in graph:
        for neighbor in graph[vertex]:
            in_degree[neighbor] += 1
    
    # Find vertices with no incoming edges
    queue = deque([vertex for vertex in in_degree if in_degree[vertex] == 0])
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == len(graph) else None  # Cycle detection
```

## Dynamic Programming

### Fibonacci Sequence
```python
def fibonacci_dp(n):
    """
    Time: O(n), Space: O(n)
    Bottom-up approach
    """
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

def fibonacci_optimized(n):
    """
    Time: O(n), Space: O(1)
    Space-optimized version
    """
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

# Memoization approach
def fibonacci_memo(n, memo={}):
    """Top-down approach with memoization"""
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

### 0/1 Knapsack Problem
```python
def knapsack_01(weights, values, capacity):
    """
    Time: O(n * W), Space: O(n * W)
    where n is number of items, W is capacity
    """
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w - weights[i-1]],  # Include item
                    dp[i-1][w]  # Exclude item
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

def knapsack_01_optimized(weights, values, capacity):
    """Space-optimized version: O(W) space"""
    dp = [0 for _ in range(capacity + 1)]
    
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
```

### Longest Common Subsequence (LCS)
```python
def lcs_length(text1, text2):
    """
    Time: O(m * n), Space: O(m * n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def lcs_string(text1, text2):
    """Return the actual LCS string"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))
```

### Edit Distance (Levenshtein Distance)
```python
def edit_distance(word1, word2):
    """
    Time: O(m * n), Space: O(m * n)
    Minimum operations to convert word1 to word2
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )
    
    return dp[m][n]
```

### Coin Change Problem
```python
def coin_change(coins, amount):
    """
    Minimum number of coins to make amount
    Time: O(amount * len(coins)), Space: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def coin_change_ways(coins, amount):
    """Number of ways to make amount"""
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]
```

## Greedy Algorithms

### Activity Selection Problem
```python
def activity_selection(activities):
    """
    Select maximum number of non-overlapping activities
    Time: O(n log n), Space: O(1)
    activities: list of (start, end) tuples
    """
    # Sort by end time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_end_time = activities[0][1]
    
    for start, end in activities[1:]:
        if start >= last_end_time:
            selected.append((start, end))
            last_end_time = end
    
    return selected
```

### Fractional Knapsack
```python
def fractional_knapsack(items, capacity):
    """
    items: list of (value, weight) tuples
    Time: O(n log n), Space: O(1)
    """
    # Sort by value-to-weight ratio
    items.sort(key=lambda x: x[0]/x[1], reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    
    for value, weight in items:
        if weight <= remaining_capacity:
            # Take the whole item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction of the item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            break
    
    return total_value
```

### Huffman Coding
```python
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

def huffman_coding(text):
    """
    Generate Huffman codes for characters in text
    Time: O(n log n), Space: O(n)
    """
    if not text:
        return {}
    
    # Count character frequencies
    freq = Counter(text)
    
    # Create priority queue with leaf nodes
    heap = [Node(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = Node(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    # Generate codes
    root = heap[0]
    codes = {}
    
    def generate_codes(node, code=""):
        if node.char:  # Leaf node
            codes[node.char] = code or "0"  # Handle single character
        else:
            generate_codes(node.left, code + "0")
            generate_codes(node.right, code + "1")
    
    generate_codes(root)
    return codes
```

## Divide and Conquer

### Maximum Subarray (Kadane's Algorithm)
```python
def max_subarray_kadane(nums):
    """
    Kadane's algorithm - actually a DP approach
    Time: O(n), Space: O(1)
    """
    max_ending_here = max_so_far = nums[0]
    
    for i in range(1, len(nums)):
        max_ending_here = max(nums[i], max_ending_here + nums[i])
        max_so_far = max(max_so_far, max_ending_here)
    
    return max_so_far

def max_subarray_divide_conquer(nums, left=0, right=None):
    """
    Divide and conquer approach
    Time: O(n log n), Space: O(log n)
    """
    if right is None:
        right = len(nums) - 1
    
    if left == right:
        return nums[left]
    
    mid = (left + right) // 2
    
    # Maximum subarray in left half
    left_max = max_subarray_divide_conquer(nums, left, mid)
    
    # Maximum subarray in right half
    right_max = max_subarray_divide_conquer(nums, mid + 1, right)
    
    # Maximum subarray crossing the midpoint
    left_sum = float('-inf')
    current_sum = 0
    for i in range(mid, left - 1, -1):
        current_sum += nums[i]
        left_sum = max(left_sum, current_sum)
    
    right_sum = float('-inf')
    current_sum = 0
    for i in range(mid + 1, right + 1):
        current_sum += nums[i]
        right_sum = max(right_sum, current_sum)
    
    cross_sum = left_sum + right_sum
    
    return max(left_max, right_max, cross_sum)
```

### Closest Pair of Points
```python
import math

def closest_pair(points):
    """
    Find closest pair of points in 2D plane
    Time: O(n log n), Space: O(n)
    """
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def closest_pair_rec(px, py):
        n = len(px)
        
        # Base case for small arrays
        if n <= 3:
            min_dist = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    min_dist = min(min_dist, distance(px[i], px[j]))
            return min_dist
        
        # Divide
        mid = n // 2
        midpoint = px[mid]
        
        pyl = [point for point in py if point[0] <= midpoint[0]]
        pyr = [point for point in py if point[0] > midpoint[0]]
        
        # Conquer
        dl = closest_pair_rec(px[:mid], pyl)
        dr = closest_pair_rec(px[mid:], pyr)
        
        # Find minimum of the two
        d = min(dl, dr)
        
        # Check points near the dividing line
        strip = [point for point in py if abs(point[0] - midpoint[0]) < d]
        
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
                d = min(d, distance(strip[i], strip[j]))
                j += 1
        
        return d
    
    # Sort points by x and y coordinates
    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    
    return closest_pair_rec(px, py)
```

## Backtracking

### N-Queens Problem
```python
def solve_n_queens(n):
    """
    Place n queens on n×n chessboard
    Time: O(n!), Space: O(n)
    """
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i] == col:
                return False
        
        # Check diagonal
        for i in range(row):
            if abs(board[i] - col) == abs(i - row):
                return False
        
        return True
    
    def backtrack(board, row):
        if row == n:
            result.append(board[:])
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(board, row + 1)
                board[row] = -1  # Backtrack
    
    result = []
    board = [-1] * n
    backtrack(board, 0)
    return result

def print_queens_solution(solution):
    """Print a readable solution"""
    n = len(solution)
    for row in range(n):
        line = ""
        for col in range(n):
            if solution[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line)
    print()
```

### Sudoku Solver
```python
def solve_sudoku(board):
    """
    Solve 9x9 Sudoku puzzle
    Time: O(9^(n²)), Space: O(n²)
    """
    def is_valid(board, row, col, num):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            
                            if solve():
                                return True
                            
                            board[i][j] = 0  # Backtrack
                    
                    return False
        return True
    
    solve()
    return board
```

### Generate All Permutations
```python
def permutations(nums):
    """
    Generate all permutations of nums
    Time: O(n! × n), Space: O(n)
    """
    def backtrack(current_perm):
        if len(current_perm) == len(nums):
            result.append(current_perm[:])
            return
        
        for num in nums:
            if num not in current_perm:
                current_perm.append(num)
                backtrack(current_perm)
                current_perm.pop()  # Backtrack
    
    result = []
    backtrack([])
    return result

def permutations_iterative(nums):
    """Iterative approach using built-in itertools"""
    from itertools import permutations as perm
    return list(perm(nums))
```

## String Algorithms

### KMP Algorithm (Pattern Matching)
```python
def kmp_search(text, pattern):
    """
    Knuth-Morris-Pratt pattern matching
    Time: O(n + m), Space: O(m)
    """
    def compute_lps(pattern):
        """Compute Longest Prefix Suffix array"""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    n, m = len(text), len(pattern)
    lps = compute_lps(pattern)
    
    i = j = 0  # Indices for text and pattern
    matches = []
    
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches
```

### Rabin-Karp Algorithm
```python
def rabin_karp_search(text, pattern, prime=101):
    """
    Rabin-Karp rolling hash pattern matching
    Time: O(n + m) average, O(nm) worst case
    """
    n, m = len(text), len(pattern)
    d = 256  # Number of characters in alphabet
    
    pattern_hash = 0
    text_hash = 0
    h = 1
    matches = []
    
    # Calculate h = d^(m-1) % prime
    for i in range(m - 1):
        h = (h * d) % prime
    
    # Calculate hash values for pattern and first window
    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
        text_hash = (d * text_hash + ord(text[i])) % prime
    
    # Slide pattern over text
    for i in range(n - m + 1):
        # Check if hash values match
        if pattern_hash == text_hash:
            # Check characters one by one
            if text[i:i + m] == pattern:
                matches.append(i)
        
        # Calculate hash for next window
        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if text_hash < 0:
                text_hash += prime
    
    return matches
```

### Longest Palindromic Substring
```python
def longest_palindrome_expand(s):
    """
    Expand around centers approach
    Time: O(n²), Space: O(1)
    """
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    start = 0
    max_len = 0
    
    for i in range(len(s)):
        # Odd length palindromes
        len1 = expand_around_center(i, i)
        # Even length palindromes
        len2 = expand_around_center(i, i + 1)
        
        current_max = max(len1, len2)
        if current_max > max_len:
            max_len = current_max
            start = i - (current_max - 1) // 2
    
    return s[start:start + max_len]

def longest_palindrome_manacher(s):
    """
    Manacher's algorithm - O(n) time
    """
    # Preprocess string
    processed = '#'.join('^{}$'.format(s))
    n = len(processed)
    P = [0] * n
    center = right = 0
    
    for i in range(1, n - 1):
        # Mirror of i
        mirror = 2 * center - i
        
        if i < right:
            P[i] = min(right - i, P[mirror])
        
        # Try to expand palindrome centered at i
        try:
            while processed[i + (1 + P[i])] == processed[i - (1 + P[i])]:
                P[i] += 1
        except IndexError:
            pass
        
        # If palindrome centered at i extends past right, adjust center and right
        if i + P[i] > right:
            center, right = i, i + P[i]
    
    # Find the longest palindrome
    max_len = max(P)
    center_index = P.index(max_len)
    start = (center_index - max_len) // 2
    
    return s[start:start + max_len]
```

## Mathematical Algorithms

### Euclidean Algorithm (GCD)
```python
def gcd(a, b):
    """
    Greatest Common Divisor
    Time: O(log(min(a, b))), Space: O(1)
    """
    while b:
        a, b = b, a % b
    return a

def gcd_recursive(a, b):
    """Recursive version"""
    if b == 0:
        return a
    return gcd_recursive(b, a % b)

def lcm(a, b):
    """Least Common Multiple"""
    return abs(a * b) // gcd(a, b)

def extended_gcd(a, b):
    """Extended Euclidean Algorithm"""
    if a == 0:
        return b, 0, 1
    
    gcd_val, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return gcd_val, x, y
```

### Sieve of Eratosthenes
```python
def sieve_of_eratosthenes(n):
    """
    Find all prime numbers up to n
    Time: O(n log log n), Space: O(n)
    """
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, n + 1) if is_prime[i]]

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    
    return True
```

### Fast Exponentiation
```python
def power(base, exp, mod=None):
    """
    Fast exponentiation using binary exponentiation
    Time: O(log exp), Space: O(1)
    """
    result = 1
    base = base % mod if mod else base
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod if mod else result * base
        
        exp = exp // 2
        base = (base * base) % mod if mod else base * base
    
    return result

def matrix_power(matrix, n):
    """Matrix exponentiation for Fibonacci-like sequences"""
    def matrix_multiply(A, B):
        return [[A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
                [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
    
    if n == 0:
        return [[1, 0], [0, 1]]  # Identity matrix
    
    result = [[1, 0], [0, 1]]
    base = matrix
    
    while n > 0:
        if n % 2 == 1:
            result = matrix_multiply(result, base)
        base = matrix_multiply(base, base)
        n //= 2
    
    return result
```

## Bit Manipulation

### Basic Bit Operations
```python
def bit_operations_examples():
    """Common bit manipulation operations"""
    n = 12  # Binary: 1100
    
    # Check if bit at position i is set
    def is_bit_set(num, i):
        return (num & (1 << i)) != 0
    
    # Set bit at position i
    def set_bit(num, i):
        return num | (1 << i)
    
    # Clear bit at position i
    def clear_bit(num, i):
        return num & ~(1 << i)
    
    # Toggle bit at position i
    def toggle_bit(num, i):
        return num ^ (1 << i)
    
    # Count number of set bits (Brian Kernighan's algorithm)
    def count_set_bits(num):
        count = 0
        while num:
            num &= num - 1  # Clear the lowest set bit
            count += 1
        return count
    
    # Check if number is power of 2
    def is_power_of_two(num):
        return num > 0 and (num & (num - 1)) == 0
    
    return {
        'is_bit_set': is_bit_set(n, 2),
        'set_bit': set_bit(n, 1),
        'clear_bit': clear_bit(n, 3),
        'toggle_bit': toggle_bit(n, 0),
        'count_set_bits': count_set_bits(n),
        'is_power_of_two': is_power_of_two(16)
    }

def find_single_number(nums):
    """
    Find the single number in array where every other number appears twice
    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def find_two_single_numbers(nums):
    """
    Find two single numbers in array where every other number appears twice
    """
    xor = 0
    for num in nums:
        xor ^= num
    
    # Find rightmost set bit
    rightmost_set_bit = xor & -xor
    
    num1 = num2 = 0
    for num in nums:
        if num & rightmost_set_bit:
            num1 ^= num
        else:
            num2 ^= num
    
    return num1, num2

def reverse_bits(n):
    """Reverse bits of a 32-bit unsigned integer"""
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
```

### Bit Manipulation Problems
```python
def subset_generation(nums):
    """
    Generate all subsets using bit manipulation
    Time: O(2^n * n), Space: O(2^n * n)
    """
    n = len(nums)
    subsets = []
    
    for i in range(1 << n):  # 2^n subsets
        subset = []
        for j in range(n):
            if i & (1 << j):
                subset.append(nums[j])
        subsets.append(subset)
    
    return subsets

def hamming_distance(x, y):
    """
    Number of positions where bits are different
    Time: O(1), Space: O(1)
    """
    xor = x ^ y
    distance = 0
    
    while xor:
        distance += xor & 1
        xor >>= 1
    
    return distance

def missing_number(nums):
    """
    Find missing number in array [0, 1, 2, ..., n]
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    expected_xor = 0
    actual_xor = 0
    
    for i in range(n + 1):
        expected_xor ^= i
    
    for num in nums:
        actual_xor ^= num
    
    return expected_xor ^ actual_xor
```

## Time Complexity Analysis

### Common Time Complexities
```python
"""
O(1) - Constant Time:
- Array access, hash table operations (average case)
- Stack push/pop, queue enqueue/dequeue

O(log n) - Logarithmic Time:
- Binary search, heap operations
- Balanced tree operations (search, insert, delete)

O(n) - Linear Time:
- Linear search, array traversal
- Single pass through data

O(n log n) - Linearithmic Time:
- Efficient sorting algorithms (merge sort, heap sort, quick sort average)
- Divide and conquer algorithms

O(n²) - Quadratic Time:
- Nested loops, bubble sort, selection sort
- Many brute force solutions

O(2^n) - Exponential Time:
- Recursive algorithms without memoization (Fibonacci)
- Generating all subsets

O(n!) - Factorial Time:
- Generating all permutations
- Traveling salesman problem (brute force)
"""
```

## Best Practices

### Algorithm Selection Guidelines
1. **Choose the right algorithm for the problem size**:
   - Small datasets: Simple algorithms (O(n²)) may be faster due to lower constants
   - Large datasets: Efficient algorithms (O(n log n)) are essential

2. **Consider space-time trade-offs**:
   - Memoization: Trade space for time
   - In-place algorithms: Trade time for space

3. **Understand your data**:
   - Nearly sorted data: Insertion sort can be O(n)
   - Many duplicates: Three-way quicksort
   - Known range: Counting sort

4. **Profile and measure**:
   - Use timing and profiling tools
   - Consider real-world constraints and requirements

5. **Implement and test**:
   - Start with a working solution
   - Optimize based on actual performance needs
   - Test edge cases and boundary conditions 