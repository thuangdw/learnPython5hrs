"""
Stack Implementation
Comprehensive implementation of stack data structure and algorithms
"""

class Stack:
    """Stack implementation using list"""
    
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack - O(1)"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item - O(1)"""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def peek(self):
        """Return top item without removing - O(1)"""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")
    
    def is_empty(self):
        """Check if stack is empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items - O(1)"""
        return len(self.items)
    
    def __str__(self):
        return f"Stack({self.items})"


class MinStack:
    """Stack that supports getMin() operation in O(1)"""
    
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        """Push value onto stack"""
        self.stack.append(val)
        
        # Push to min_stack if it's empty or val is <= current min
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        """Pop value from stack"""
        if not self.stack:
            raise IndexError("Stack is empty")
        
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val
    
    def top(self):
        """Get top element"""
        if not self.stack:
            raise IndexError("Stack is empty")
        return self.stack[-1]
    
    def get_min(self):
        """Get minimum element in O(1)"""
        if not self.min_stack:
            raise IndexError("Stack is empty")
        return self.min_stack[-1]


def is_valid_parentheses(s):
    """
    Check if parentheses are balanced
    Time: O(n), Space: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return len(stack) == 0


def evaluate_postfix(expression):
    """
    Evaluate postfix expression
    Time: O(n), Space: O(n)
    """
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in expression.split():
        if token in operators:
            if len(stack) < 2:
                raise ValueError("Invalid postfix expression")
            
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                if b == 0:
                    raise ValueError("Division by zero")
                result = a / b
            
            stack.append(result)
        else:
            try:
                stack.append(float(token))
            except ValueError:
                raise ValueError(f"Invalid token: {token}")
    
    if len(stack) != 1:
        raise ValueError("Invalid postfix expression")
    
    return stack[0]


def infix_to_postfix(expression):
    """
    Convert infix expression to postfix
    Time: O(n), Space: O(n)
    """
    stack = []
    postfix = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    
    for char in expression:
        if char.isalnum():
            postfix.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            if stack:
                stack.pop()  # Remove '('
        elif char in precedence:
            while (stack and stack[-1] != '(' and 
                   stack[-1] in precedence and
                   precedence[stack[-1]] >= precedence[char]):
                postfix.append(stack.pop())
            stack.append(char)
    
    while stack:
        postfix.append(stack.pop())
    
    return ''.join(postfix)


def next_greater_element(nums):
    """
    Find next greater element for each element
    Time: O(n), Space: O(n)
    """
    stack = []
    result = [-1] * len(nums)
    
    # Traverse from right to left
    for i in range(len(nums) - 1, -1, -1):
        # Pop elements smaller than current element
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        
        # If stack is not empty, top is the next greater element
        if stack:
            result[i] = stack[-1]
        
        # Push current element
        stack.append(nums[i])
    
    return result


def largest_rectangle_histogram(heights):
    """
    Find largest rectangle area in histogram
    Time: O(n), Space: O(n)
    """
    stack = []
    max_area = 0
    
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    
    # Process remaining elements in stack
    while stack:
        h = heights[stack.pop()]
        w = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, h * w)
    
    return max_area


def daily_temperatures(temperatures):
    """
    Find how many days until warmer temperature
    Time: O(n), Space: O(n)
    """
    stack = []
    result = [0] * len(temperatures)
    
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
    
    return result


class StackUsingQueues:
    """Implement stack using two queues"""
    
    def __init__(self):
        from collections import deque
        self.q1 = deque()
        self.q2 = deque()
    
    def push(self, x):
        """Push element to stack"""
        self.q2.append(x)
        
        # Transfer all elements from q1 to q2
        while self.q1:
            self.q2.append(self.q1.popleft())
        
        # Swap q1 and q2
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self):
        """Pop element from stack"""
        if not self.q1:
            raise IndexError("Stack is empty")
        return self.q1.popleft()
    
    def top(self):
        """Get top element"""
        if not self.q1:
            raise IndexError("Stack is empty")
        return self.q1[0]
    
    def empty(self):
        """Check if stack is empty"""
        return len(self.q1) == 0


def simplify_path(path):
    """
    Simplify Unix file path
    Time: O(n), Space: O(n)
    """
    stack = []
    components = path.split('/')
    
    for component in components:
        if component == '' or component == '.':
            continue
        elif component == '..':
            if stack:
                stack.pop()
        else:
            stack.append(component)
    
    return '/' + '/'.join(stack)


def decode_string(s):
    """
    Decode string with pattern k[encoded_string]
    Time: O(n), Space: O(n)
    """
    stack = []
    current_num = 0
    current_string = ""
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append((current_string, current_num))
            current_string = ""
            current_num = 0
        elif char == ']':
            prev_string, num = stack.pop()
            current_string = prev_string + current_string * num
        else:
            current_string += char
    
    return current_string


if __name__ == "__main__":
    # Demo usage
    stack = Stack()
    for i in range(5):
        stack.push(i)
    print(f"Stack: {stack}")
    print(f"Top: {stack.peek()}")
    print(f"Pop: {stack.pop()}")
    
    # Parentheses validation
    print(f"Valid parentheses '()[]{{}}': {is_valid_parentheses('()[]{}')}")
    print(f"Valid parentheses '([)]': {is_valid_parentheses('([)]')}")
    
    # Postfix evaluation
    print(f"Postfix '3 4 + 2 *': {evaluate_postfix('3 4 + 2 *')}")
    
    # Next greater element
    nums = [4, 5, 2, 25]
    print(f"Next greater elements for {nums}: {next_greater_element(nums)}") 