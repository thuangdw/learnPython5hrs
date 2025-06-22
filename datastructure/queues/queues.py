"""
Queue Implementation
Comprehensive implementation of queue data structures and algorithms
"""

from collections import deque
import heapq


class Queue:
    """Basic queue implementation using deque for optimal performance"""
    
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        """Add item to rear of queue - O(1)"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item - O(1)"""
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Queue is empty")
    
    def front(self):
        """Return front item without removing - O(1)"""
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Queue is empty")
    
    def rear(self):
        """Return rear item without removing - O(1)"""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Queue is empty")
    
    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items - O(1)"""
        return len(self.items)
    
    def __str__(self):
        return f"Queue({list(self.items)})"


class CircularQueue:
    """Circular queue implementation with fixed size"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enqueue(self, item):
        """Add item to queue"""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1
    
    def dequeue(self):
        """Remove and return front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def peek_front(self):
        """Return front item without removing"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.queue[self.front]
    
    def is_empty(self):
        """Check if queue is empty"""
        return self.size == 0
    
    def is_full(self):
        """Check if queue is full"""
        return self.size == self.capacity
    
    def __str__(self):
        if self.is_empty():
            return "CircularQueue([])"
        
        items = []
        index = self.front
        for _ in range(self.size):
            items.append(self.queue[index])
            index = (index + 1) % self.capacity
        
        return f"CircularQueue({items})"


class PriorityQueue:
    """Priority queue implementation using heap"""
    
    def __init__(self):
        self.heap = []
        self.count = 0
    
    def push(self, item, priority):
        """Add item with priority (lower number = higher priority)"""
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1
    
    def pop(self):
        """Remove and return highest priority item"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        priority, count, item = heapq.heappop(self.heap)
        return item
    
    def peek(self):
        """Return highest priority item without removing"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        priority, count, item = self.heap[0]
        return item
    
    def is_empty(self):
        """Check if priority queue is empty"""
        return len(self.heap) == 0
    
    def size(self):
        """Return number of items"""
        return len(self.heap)


class Deque:
    """Double-ended queue implementation"""
    
    def __init__(self):
        self.items = deque()
    
    def add_front(self, item):
        """Add item to front - O(1)"""
        self.items.appendleft(item)
    
    def add_rear(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)
    
    def remove_front(self):
        """Remove and return front item - O(1)"""
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Deque is empty")
    
    def remove_rear(self):
        """Remove and return rear item - O(1)"""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Deque is empty")
    
    def peek_front(self):
        """Return front item without removing"""
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Deque is empty")
    
    def peek_rear(self):
        """Return rear item without removing"""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Deque is empty")
    
    def is_empty(self):
        """Check if deque is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items"""
        return len(self.items)
    
    def __str__(self):
        return f"Deque({list(self.items)})"


class QueueUsingStacks:
    """Implement queue using two stacks"""
    
    def __init__(self):
        self.stack1 = []  # for enqueue
        self.stack2 = []  # for dequeue
    
    def enqueue(self, item):
        """Add item to queue"""
        self.stack1.append(item)
    
    def dequeue(self):
        """Remove and return front item"""
        if not self.stack2:
            # Transfer all elements from stack1 to stack2
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        if not self.stack2:
            raise IndexError("Queue is empty")
        
        return self.stack2.pop()
    
    def front(self):
        """Return front item without removing"""
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        if not self.stack2:
            raise IndexError("Queue is empty")
        
        return self.stack2[-1]
    
    def is_empty(self):
        """Check if queue is empty"""
        return len(self.stack1) == 0 and len(self.stack2) == 0


def sliding_window_maximum(nums, k):
    """
    Find maximum in each sliding window of size k
    Time: O(n), Space: O(k)
    """
    if not nums or k == 0:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices of elements smaller than current element
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add maximum of current window to result
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


def first_negative_in_window(arr, k):
    """
    Find first negative number in each window of size k
    Time: O(n), Space: O(k)
    """
    dq = deque()  # Store indices of negative numbers
    result = []
    
    # Process first window
    for i in range(k):
        if arr[i] < 0:
            dq.append(i)
    
    # First negative in first window
    if dq:
        result.append(arr[dq[0]])
    else:
        result.append(0)
    
    # Process remaining windows
    for i in range(k, len(arr)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Add current element if negative
        if arr[i] < 0:
            dq.append(i)
        
        # First negative in current window
        if dq:
            result.append(arr[dq[0]])
        else:
            result.append(0)
    
    return result


class MovingAverage:
    """Calculate moving average of numbers in a sliding window"""
    
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.window_sum = 0
    
    def next(self, val):
        """Add next value and return moving average"""
        self.queue.append(val)
        self.window_sum += val
        
        # Remove oldest value if window exceeds size
        if len(self.queue) > self.size:
            self.window_sum -= self.queue.popleft()
        
        return self.window_sum / len(self.queue)


def generate_binary_numbers(n):
    """
    Generate binary representations of numbers 1 to n using queue
    Time: O(n), Space: O(n)
    """
    if n <= 0:
        return []
    
    queue = Queue()
    result = []
    
    queue.enqueue("1")
    
    for i in range(n):
        # Get front of queue
        binary = queue.dequeue()
        result.append(binary)
        
        # Generate next level
        queue.enqueue(binary + "0")
        queue.enqueue(binary + "1")
    
    return result


def interleave_queue(queue):
    """
    Interleave first half with second half of queue
    Time: O(n), Space: O(n)
    """
    if queue.size() % 2 != 0:
        raise ValueError("Queue size must be even")
    
    stack = []
    half_size = queue.size() // 2
    
    # Put first half in stack
    for _ in range(half_size):
        stack.append(queue.dequeue())
    
    # Put stack elements back to queue
    while stack:
        queue.enqueue(stack.pop())
    
    # Move first half to rear
    for _ in range(half_size):
        queue.enqueue(queue.dequeue())
    
    # Put first half in stack again
    for _ in range(half_size):
        stack.append(queue.dequeue())
    
    # Interleave
    while stack:
        queue.enqueue(stack.pop())
        queue.enqueue(queue.dequeue())


def reverse_queue(queue):
    """
    Reverse a queue using recursion
    Time: O(n), Space: O(n)
    """
    if queue.is_empty():
        return
    
    # Remove front element
    item = queue.dequeue()
    
    # Reverse remaining queue
    reverse_queue(queue)
    
    # Add front element to rear
    queue.enqueue(item)


if __name__ == "__main__":
    # Demo usage
    queue = Queue()
    for i in range(5):
        queue.enqueue(i)
    print(f"Queue: {queue}")
    print(f"Front: {queue.front()}")
    print(f"Dequeue: {queue.dequeue()}")
    
    # Priority queue example
    pq = PriorityQueue()
    pq.push("task1", 3)
    pq.push("task2", 1)
    pq.push("task3", 2)
    print(f"Priority queue pop: {pq.pop()}")  # task2 (priority 1)
    
    # Sliding window maximum
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    result = sliding_window_maximum(nums, 3)
    print(f"Sliding window maximum: {result}")
    
    # Generate binary numbers
    binary_nums = generate_binary_numbers(5)
    print(f"Binary numbers 1-5: {binary_nums}") 