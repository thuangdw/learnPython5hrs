"""
Test cases for Linked Lists Implementation
"""

import unittest
from linked_lists import (
    ListNode, SinglyLinkedList, DoublyLinkedList, CircularLinkedList,
    reverse_linked_list, reverse_linked_list_recursive, detect_cycle,
    find_cycle_start, merge_two_sorted_lists, find_middle_node,
    remove_nth_from_end, is_palindrome, intersection_of_two_lists,
    add_two_numbers
)


class TestSinglyLinkedList(unittest.TestCase):
    
    def test_insert_operations(self):
        sll = SinglyLinkedList()
        
        # Insert at beginning
        sll.insert_at_beginning(1)
        sll.insert_at_beginning(0)
        self.assertEqual(sll.display(), [0, 1])
        
        # Insert at end
        sll.insert_at_end(2)
        sll.insert_at_end(3)
        self.assertEqual(sll.display(), [0, 1, 2, 3])
        
        # Insert at position
        sll.insert_at_position(2, 99)
        self.assertEqual(sll.display(), [0, 1, 99, 2, 3])
    
    def test_delete_operations(self):
        sll = SinglyLinkedList()
        for i in range(5):
            sll.insert_at_end(i)
        
        # Delete by value
        self.assertTrue(sll.delete_by_value(2))
        self.assertEqual(sll.display(), [0, 1, 3, 4])
        self.assertFalse(sll.delete_by_value(10))
        
        # Delete at position
        val = sll.delete_at_position(1)
        self.assertEqual(val, 1)
        self.assertEqual(sll.display(), [0, 3, 4])
    
    def test_search_and_get(self):
        sll = SinglyLinkedList()
        for i in range(5):
            sll.insert_at_end(i * 2)
        
        self.assertEqual(sll.search(4), 2)
        self.assertEqual(sll.search(10), -1)
        
        self.assertEqual(sll.get(2), 4)
        with self.assertRaises(IndexError):
            sll.get(10)
    
    def test_reverse(self):
        sll = SinglyLinkedList()
        for i in range(5):
            sll.insert_at_end(i)
        
        sll.reverse()
        self.assertEqual(sll.display(), [4, 3, 2, 1, 0])
    
    def test_size_tracking(self):
        sll = SinglyLinkedList()
        self.assertEqual(len(sll), 0)
        
        sll.insert_at_end(1)
        sll.insert_at_end(2)
        self.assertEqual(len(sll), 2)
        
        sll.delete_by_value(1)
        self.assertEqual(len(sll), 1)


class TestDoublyLinkedList(unittest.TestCase):
    
    def test_insert_operations(self):
        dll = DoublyLinkedList()
        
        dll.insert_at_beginning(1)
        dll.insert_at_beginning(0)
        dll.insert_at_end(2)
        dll.insert_at_end(3)
        
        self.assertEqual(dll.display_forward(), [0, 1, 2, 3])
        self.assertEqual(dll.display_backward(), [3, 2, 1, 0])
    
    def test_delete_operations(self):
        dll = DoublyLinkedList()
        for i in range(5):
            dll.insert_at_end(i)
        
        self.assertTrue(dll.delete_by_value(2))
        self.assertEqual(dll.display_forward(), [0, 1, 3, 4])
        
        # Delete head
        self.assertTrue(dll.delete_by_value(0))
        self.assertEqual(dll.display_forward(), [1, 3, 4])
        
        # Delete tail
        self.assertTrue(dll.delete_by_value(4))
        self.assertEqual(dll.display_forward(), [1, 3])
    
    def test_bidirectional_traversal(self):
        dll = DoublyLinkedList()
        for i in range(3):
            dll.insert_at_end(i)
        
        forward = dll.display_forward()
        backward = dll.display_backward()
        
        self.assertEqual(forward, [0, 1, 2])
        self.assertEqual(backward, [2, 1, 0])


class TestCircularLinkedList(unittest.TestCase):
    
    def test_insert_and_display(self):
        cll = CircularLinkedList()
        
        cll.insert(0)
        cll.insert(1)
        cll.insert(2)
        
        self.assertEqual(cll.display(), [0, 1, 2])
        self.assertEqual(len(cll), 3)
    
    def test_delete_operations(self):
        cll = CircularLinkedList()
        for i in range(3):
            cll.insert(i)
        
        self.assertTrue(cll.delete(1))
        self.assertEqual(cll.display(), [0, 2])
        
        # Delete all nodes
        self.assertTrue(cll.delete(0))
        self.assertTrue(cll.delete(2))
        self.assertEqual(cll.display(), [])


class TestLinkedListAlgorithms(unittest.TestCase):
    
    def create_list_from_array(self, arr):
        """Helper to create linked list from array"""
        if not arr:
            return None
        
        head = ListNode(arr[0])
        current = head
        for val in arr[1:]:
            current.next = ListNode(val)
            current = current.next
        return head
    
    def list_to_array(self, head):
        """Helper to convert linked list to array"""
        result = []
        current = head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def test_reverse_linked_list(self):
        head = self.create_list_from_array([1, 2, 3, 4, 5])
        
        # Test iterative reverse
        reversed_head = reverse_linked_list(head)
        self.assertEqual(self.list_to_array(reversed_head), [5, 4, 3, 2, 1])
        
        # Test recursive reverse
        head = self.create_list_from_array([1, 2, 3, 4, 5])
        reversed_head = reverse_linked_list_recursive(head)
        self.assertEqual(self.list_to_array(reversed_head), [5, 4, 3, 2, 1])
    
    def test_detect_cycle(self):
        # Create list without cycle
        head = self.create_list_from_array([1, 2, 3, 4])
        self.assertFalse(detect_cycle(head))
        
        # Create list with cycle
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = head.next  # Create cycle
        
        self.assertTrue(detect_cycle(head))
    
    def test_find_cycle_start(self):
        # Create list with cycle
        head = ListNode(1)
        cycle_start = ListNode(2)
        head.next = cycle_start
        cycle_start.next = ListNode(3)
        cycle_start.next.next = ListNode(4)
        cycle_start.next.next.next = cycle_start  # Create cycle
        
        found_start = find_cycle_start(head)
        self.assertEqual(found_start, cycle_start)
    
    def test_merge_two_sorted_lists(self):
        l1 = self.create_list_from_array([1, 2, 4])
        l2 = self.create_list_from_array([1, 3, 4])
        
        merged = merge_two_sorted_lists(l1, l2)
        self.assertEqual(self.list_to_array(merged), [1, 1, 2, 3, 4, 4])
    
    def test_find_middle_node(self):
        # Odd length
        head = self.create_list_from_array([1, 2, 3, 4, 5])
        middle = find_middle_node(head)
        self.assertEqual(middle.val, 3)
        
        # Even length
        head = self.create_list_from_array([1, 2, 3, 4])
        middle = find_middle_node(head)
        self.assertEqual(middle.val, 2)
    
    def test_remove_nth_from_end(self):
        head = self.create_list_from_array([1, 2, 3, 4, 5])
        result = remove_nth_from_end(head, 2)
        self.assertEqual(self.list_to_array(result), [1, 2, 3, 5])
        
        # Remove first node
        head = self.create_list_from_array([1, 2])
        result = remove_nth_from_end(head, 2)
        self.assertEqual(self.list_to_array(result), [2])
    
    def test_is_palindrome(self):
        # Palindrome
        head = self.create_list_from_array([1, 2, 2, 1])
        self.assertTrue(is_palindrome(head))
        
        # Not palindrome
        head = self.create_list_from_array([1, 2, 3])
        self.assertFalse(is_palindrome(head))
        
        # Single node
        head = self.create_list_from_array([1])
        self.assertTrue(is_palindrome(head))
    
    def test_intersection_of_two_lists(self):
        # Create intersection
        common = ListNode(8)
        common.next = ListNode(4)
        common.next.next = ListNode(5)
        
        headA = ListNode(4)
        headA.next = ListNode(1)
        headA.next.next = common
        
        headB = ListNode(5)
        headB.next = ListNode(6)
        headB.next.next = ListNode(1)
        headB.next.next.next = common
        
        intersection = intersection_of_two_lists(headA, headB)
        self.assertEqual(intersection, common)
        
        # No intersection
        headA = self.create_list_from_array([1, 2, 3])
        headB = self.create_list_from_array([4, 5, 6])
        intersection = intersection_of_two_lists(headA, headB)
        self.assertIsNone(intersection)
    
    def test_add_two_numbers(self):
        # 342 + 465 = 807
        l1 = self.create_list_from_array([2, 4, 3])  # represents 342
        l2 = self.create_list_from_array([5, 6, 4])  # represents 465
        
        result = add_two_numbers(l1, l2)
        self.assertEqual(self.list_to_array(result), [7, 0, 8])  # represents 807
        
        # Different lengths
        l1 = self.create_list_from_array([9, 9])  # represents 99
        l2 = self.create_list_from_array([1])     # represents 1
        
        result = add_two_numbers(l1, l2)
        self.assertEqual(self.list_to_array(result), [0, 0, 1])  # represents 100


if __name__ == '__main__':
    unittest.main() 