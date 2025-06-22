"""
Test cases for Queue Implementation
"""

import unittest
from queues import (
    Queue, CircularQueue, PriorityQueue, Deque, QueueUsingStacks,
    sliding_window_maximum, first_negative_in_window, MovingAverage,
    generate_binary_numbers, interleave_queue, reverse_queue
)


class TestQueue(unittest.TestCase):
    
    def test_basic_operations(self):
        queue = Queue()
        self.assertTrue(queue.is_empty())
        self.assertEqual(queue.size(), 0)
        
        queue.enqueue(1)
        queue.enqueue(2)
        queue.enqueue(3)
        
        self.assertFalse(queue.is_empty())
        self.assertEqual(queue.size(), 3)
        self.assertEqual(queue.front(), 1)
        self.assertEqual(queue.rear(), 3)
        
        self.assertEqual(queue.dequeue(), 1)
        self.assertEqual(queue.dequeue(), 2)
        self.assertEqual(queue.size(), 1)
    
    def test_empty_queue_exceptions(self):
        queue = Queue()
        
        with self.assertRaises(IndexError):
            queue.dequeue()
        
        with self.assertRaises(IndexError):
            queue.front()
        
        with self.assertRaises(IndexError):
            queue.rear()


class TestCircularQueue(unittest.TestCase):
    
    def test_basic_operations(self):
        queue = CircularQueue(3)
        self.assertTrue(queue.is_empty())
        self.assertFalse(queue.is_full())
        
        queue.enqueue(1)
        queue.enqueue(2)
        queue.enqueue(3)
        
        self.assertTrue(queue.is_full())
        self.assertEqual(queue.peek_front(), 1)
        
        self.assertEqual(queue.dequeue(), 1)
        self.assertFalse(queue.is_full())
        
        queue.enqueue(4)
        self.assertEqual(queue.dequeue(), 2)
        self.assertEqual(queue.dequeue(), 3)
        self.assertEqual(queue.dequeue(), 4)
    
    def test_overflow_and_underflow(self):
        queue = CircularQueue(2)
        
        queue.enqueue(1)
        queue.enqueue(2)
        
        with self.assertRaises(OverflowError):
            queue.enqueue(3)
        
        queue.dequeue()
        queue.dequeue()
        
        with self.assertRaises(IndexError):
            queue.dequeue()


class TestPriorityQueue(unittest.TestCase):
    
    def test_priority_ordering(self):
        pq = PriorityQueue()
        
        pq.push("task3", 3)
        pq.push("task1", 1)
        pq.push("task2", 2)
        
        self.assertEqual(pq.peek(), "task1")
        self.assertEqual(pq.pop(), "task1")
        self.assertEqual(pq.pop(), "task2")
        self.assertEqual(pq.pop(), "task3")
    
    def test_same_priority(self):
        pq = PriorityQueue()
        
        pq.push("first", 1)
        pq.push("second", 1)
        
        # First in should come out first for same priority
        self.assertEqual(pq.pop(), "first")
        self.assertEqual(pq.pop(), "second")
    
    def test_empty_priority_queue(self):
        pq = PriorityQueue()
        
        with self.assertRaises(IndexError):
            pq.pop()
        
        with self.assertRaises(IndexError):
            pq.peek()


class TestDeque(unittest.TestCase):
    
    def test_front_operations(self):
        dq = Deque()
        
        dq.add_front(1)
        dq.add_front(2)
        dq.add_front(3)
        
        self.assertEqual(dq.peek_front(), 3)
        self.assertEqual(dq.remove_front(), 3)
        self.assertEqual(dq.remove_front(), 2)
        self.assertEqual(dq.remove_front(), 1)
    
    def test_rear_operations(self):
        dq = Deque()
        
        dq.add_rear(1)
        dq.add_rear(2)
        dq.add_rear(3)
        
        self.assertEqual(dq.peek_rear(), 3)
        self.assertEqual(dq.remove_rear(), 3)
        self.assertEqual(dq.remove_rear(), 2)
        self.assertEqual(dq.remove_rear(), 1)
    
    def test_mixed_operations(self):
        dq = Deque()
        
        dq.add_front(2)
        dq.add_rear(3)
        dq.add_front(1)
        dq.add_rear(4)
        
        self.assertEqual(dq.remove_front(), 1)
        self.assertEqual(dq.remove_rear(), 4)
        self.assertEqual(dq.remove_front(), 2)
        self.assertEqual(dq.remove_rear(), 3)


class TestQueueUsingStacks(unittest.TestCase):
    
    def test_queue_operations(self):
        queue = QueueUsingStacks()
        self.assertTrue(queue.is_empty())
        
        queue.enqueue(1)
        queue.enqueue(2)
        queue.enqueue(3)
        
        self.assertFalse(queue.is_empty())
        self.assertEqual(queue.front(), 1)
        self.assertEqual(queue.dequeue(), 1)
        self.assertEqual(queue.dequeue(), 2)
        self.assertEqual(queue.front(), 3)


class TestSlidingWindowMaximum(unittest.TestCase):
    
    def test_sliding_window_maximum(self):
        nums = [1, 3, -1, -3, 5, 3, 6, 7]
        result = sliding_window_maximum(nums, 3)
        expected = [3, 3, 5, 5, 6, 7]
        self.assertEqual(result, expected)
    
    def test_single_element_window(self):
        nums = [1, 2, 3, 4, 5]
        result = sliding_window_maximum(nums, 1)
        self.assertEqual(result, nums)
    
    def test_window_size_equals_array_length(self):
        nums = [1, 3, 2, 5, 4]
        result = sliding_window_maximum(nums, 5)
        self.assertEqual(result, [5])


class TestFirstNegativeInWindow(unittest.TestCase):
    
    def test_first_negative_in_window(self):
        arr = [12, -1, -7, 8, -15, 30, 16, 28]
        result = first_negative_in_window(arr, 3)
        expected = [-1, -1, -7, -15, -15, 0]
        self.assertEqual(result, expected)
    
    def test_no_negatives(self):
        arr = [1, 2, 3, 4, 5]
        result = first_negative_in_window(arr, 3)
        expected = [0, 0, 0]
        self.assertEqual(result, expected)


class TestMovingAverage(unittest.TestCase):
    
    def test_moving_average(self):
        ma = MovingAverage(3)
        
        self.assertEqual(ma.next(1), 1.0)
        self.assertEqual(ma.next(10), 5.5)
        self.assertEqual(ma.next(3), 14.0 / 3)
        self.assertEqual(ma.next(5), 6.0)  # (10 + 3 + 5) / 3


class TestGenerateBinaryNumbers(unittest.TestCase):
    
    def test_generate_binary_numbers(self):
        result = generate_binary_numbers(5)
        expected = ["1", "10", "11", "100", "101"]
        self.assertEqual(result, expected)
    
    def test_zero_and_negative(self):
        self.assertEqual(generate_binary_numbers(0), [])
        self.assertEqual(generate_binary_numbers(-1), [])


class TestInterleaveQueue(unittest.TestCase):
    
    def test_interleave_queue(self):
        queue = Queue()
        for i in range(1, 7):  # [1, 2, 3, 4, 5, 6]
            queue.enqueue(i)
        
        interleave_queue(queue)
        
        result = []
        while not queue.is_empty():
            result.append(queue.dequeue())
        
        expected = [1, 4, 2, 5, 3, 6]
        self.assertEqual(result, expected)
    
    def test_odd_size_queue(self):
        queue = Queue()
        for i in range(1, 6):  # [1, 2, 3, 4, 5] - odd size
            queue.enqueue(i)
        
        with self.assertRaises(ValueError):
            interleave_queue(queue)


class TestReverseQueue(unittest.TestCase):
    
    def test_reverse_queue(self):
        queue = Queue()
        for i in range(1, 6):  # [1, 2, 3, 4, 5]
            queue.enqueue(i)
        
        reverse_queue(queue)
        
        result = []
        while not queue.is_empty():
            result.append(queue.dequeue())
        
        expected = [5, 4, 3, 2, 1]
        self.assertEqual(result, expected)
    
    def test_empty_queue_reverse(self):
        queue = Queue()
        reverse_queue(queue)  # Should not raise error
        self.assertTrue(queue.is_empty())


if __name__ == '__main__':
    unittest.main() 