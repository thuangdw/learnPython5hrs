"""
Linked Lists Implementation
Comprehensive implementation of linked list data structures and algorithms
"""


class ListNode:
    """Node for singly linked list"""
    
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __str__(self):
        return str(self.val)


class DoublyListNode:
    """Node for doubly linked list"""
    
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
    
    def __str__(self):
        return str(self.val)


class SinglyLinkedList:
    """Singly linked list implementation"""
    
    def __init__(self):
        self.head = None
        self.size = 0
    
    def insert_at_beginning(self, val):
        """Insert at beginning - O(1)"""
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def insert_at_end(self, val):
        """Insert at end - O(n)"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def insert_at_position(self, position, val):
        """Insert at specific position - O(n)"""
        if position < 0 or position > self.size:
            raise IndexError("Position out of range")
        
        if position == 0:
            self.insert_at_beginning(val)
            return
        
        new_node = ListNode(val)
        current = self.head
        for _ in range(position - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    def delete_by_value(self, val):
        """Delete first occurrence of value - O(n)"""
        if not self.head:
            return False
        
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        
        if current.next:
            current.next = current.next.next
            self.size -= 1
            return True
        
        return False
    
    def delete_at_position(self, position):
        """Delete node at position - O(n)"""
        if position < 0 or position >= self.size:
            raise IndexError("Position out of range")
        
        if position == 0:
            val = self.head.val
            self.head = self.head.next
            self.size -= 1
            return val
        
        current = self.head
        for _ in range(position - 1):
            current = current.next
        
        val = current.next.val
        current.next = current.next.next
        self.size -= 1
        return val
    
    def search(self, val):
        """Search for value - O(n)"""
        current = self.head
        position = 0
        
        while current:
            if current.val == val:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def get(self, position):
        """Get value at position - O(n)"""
        if position < 0 or position >= self.size:
            raise IndexError("Position out of range")
        
        current = self.head
        for _ in range(position):
            current = current.next
        
        return current.val
    
    def reverse(self):
        """Reverse the linked list - O(n)"""
        prev = None
        current = self.head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        self.head = prev
    
    def display(self):
        """Return list representation - O(n)"""
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        return str(self.display())


class DoublyLinkedList:
    """Doubly linked list implementation"""
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def insert_at_beginning(self, val):
        """Insert at beginning - O(1)"""
        new_node = DoublyListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, val):
        """Insert at end - O(1)"""
        new_node = DoublyListNode(val)
        
        if not self.tail:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def delete_by_value(self, val):
        """Delete first occurrence of value - O(n)"""
        current = self.head
        
        while current:
            if current.val == val:
                # Update previous node
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                
                # Update next node
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                
                self.size -= 1
                return True
            
            current = current.next
        
        return False
    
    def display_forward(self):
        """Display from head to tail - O(n)"""
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def display_backward(self):
        """Display from tail to head - O(n)"""
        result = []
        current = self.tail
        while current:
            result.append(current.val)
            current = current.prev
        return result
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        return str(self.display_forward())


class CircularLinkedList:
    """Circular linked list implementation"""
    
    def __init__(self):
        self.head = None
        self.size = 0
    
    def insert(self, val):
        """Insert node - O(1) if we maintain tail pointer"""
        new_node = ListNode(val)
        
        if not self.head:
            self.head = new_node
            new_node.next = new_node  # Point to itself
        else:
            # Find the last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            new_node.next = self.head
            current.next = new_node
        
        self.size += 1
    
    def delete(self, val):
        """Delete node with value - O(n)"""
        if not self.head:
            return False
        
        # Special case: only one node
        if self.head.next == self.head and self.head.val == val:
            self.head = None
            self.size -= 1
            return True
        
        # Find the node to delete and its previous node
        current = self.head
        prev = None
        
        # Find the last node (previous to head)
        while current.next != self.head:
            current = current.next
        prev = current
        current = self.head
        
        # Search for the node to delete
        while True:
            if current.val == val:
                if current == self.head:
                    self.head = current.next
                prev.next = current.next
                self.size -= 1
                return True
            
            prev = current
            current = current.next
            
            if current == self.head:
                break
        
        return False
    
    def display(self):
        """Display the circular list - O(n)"""
        if not self.head:
            return []
        
        result = []
        current = self.head
        
        while True:
            result.append(current.val)
            current = current.next
            if current == self.head:
                break
        
        return result
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        return str(self.display())


# Linked List Algorithms
def reverse_linked_list(head):
    """
    Reverse a linked list iteratively
    Time: O(n), Space: O(1)
    """
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev


def reverse_linked_list_recursive(head):
    """
    Reverse a linked list recursively
    Time: O(n), Space: O(n)
    """
    if not head or not head.next:
        return head
    
    reversed_head = reverse_linked_list_recursive(head.next)
    head.next.next = head
    head.next = None
    
    return reversed_head


def detect_cycle(head):
    """
    Floyd's cycle detection algorithm (Tortoise and Hare)
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False


def find_cycle_start(head):
    """
    Find the start of cycle in linked list
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return None
    
    # Detect cycle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow


def merge_two_sorted_lists(l1, l2):
    """
    Merge two sorted linked lists
    Time: O(m + n), Space: O(1)
    """
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    # Attach remaining nodes
    current.next = l1 or l2
    
    return dummy.next


def find_middle_node(head):
    """
    Find middle node using slow/fast pointers
    Time: O(n), Space: O(1)
    """
    if not head:
        return None
    
    slow = fast = head
    
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow


def remove_nth_from_end(head, n):
    """
    Remove nth node from end
    Time: O(L), Space: O(1) where L is length
    """
    dummy = ListNode(0)
    dummy.next = head
    first = second = dummy
    
    # Move first pointer n+1 steps ahead
    for _ in range(n + 1):
        first = first.next
    
    # Move both pointers until first reaches end
    while first:
        first = first.next
        second = second.next
    
    # Remove the nth node from end
    second.next = second.next.next
    
    return dummy.next


def is_palindrome(head):
    """
    Check if linked list is palindrome
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return True
    
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second_half = reverse_linked_list(slow.next)
    slow.next = None
    
    # Compare two halves
    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next
    
    return True


def intersection_of_two_lists(headA, headB):
    """
    Find intersection of two linked lists
    Time: O(m + n), Space: O(1)
    """
    if not headA or not headB:
        return None
    
    pA, pB = headA, headB
    
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    
    return pA


def add_two_numbers(l1, l2):
    """
    Add two numbers represented as linked lists
    Time: O(max(m, n)), Space: O(max(m, n))
    """
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        current.next = ListNode(total % 10)
        
        current = current.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next


if __name__ == "__main__":
    # Demo usage
    sll = SinglyLinkedList()
    for i in range(5):
        sll.insert_at_end(i)
    print(f"Singly Linked List: {sll}")
    
    sll.reverse()
    print(f"Reversed: {sll}")
    
    # Doubly linked list
    dll = DoublyLinkedList()
    for i in range(3):
        dll.insert_at_end(i)
    print(f"Doubly Linked List: {dll}")
    print(f"Backward: {dll.display_backward()}")
    
    # Circular linked list
    cll = CircularLinkedList()
    for i in range(3):
        cll.insert(i)
    print(f"Circular Linked List: {cll}") 