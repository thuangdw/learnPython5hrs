"""
Test cases for Stack Implementation
"""

import unittest
from stacks import (
    Stack, MinStack, is_valid_parentheses, evaluate_postfix,
    infix_to_postfix, next_greater_element, largest_rectangle_histogram,
    daily_temperatures, StackUsingQueues, simplify_path, decode_string
)


class TestStack(unittest.TestCase):
    
    def test_basic_operations(self):
        stack = Stack()
        self.assertTrue(stack.is_empty())
        self.assertEqual(stack.size(), 0)
        
        stack.push(1)
        stack.push(2)
        stack.push(3)
        
        self.assertFalse(stack.is_empty())
        self.assertEqual(stack.size(), 3)
        self.assertEqual(stack.peek(), 3)
        
        self.assertEqual(stack.pop(), 3)
        self.assertEqual(stack.pop(), 2)
        self.assertEqual(stack.size(), 1)
    
    def test_empty_stack_exceptions(self):
        stack = Stack()
        
        with self.assertRaises(IndexError):
            stack.pop()
        
        with self.assertRaises(IndexError):
            stack.peek()


class TestMinStack(unittest.TestCase):
    
    def test_min_stack_operations(self):
        min_stack = MinStack()
        
        min_stack.push(-2)
        min_stack.push(0)
        min_stack.push(-3)
        
        self.assertEqual(min_stack.get_min(), -3)
        self.assertEqual(min_stack.pop(), -3)
        self.assertEqual(min_stack.top(), 0)
        self.assertEqual(min_stack.get_min(), -2)
    
    def test_min_stack_duplicate_mins(self):
        min_stack = MinStack()
        
        min_stack.push(1)
        min_stack.push(1)
        min_stack.push(2)
        
        self.assertEqual(min_stack.get_min(), 1)
        min_stack.pop()
        self.assertEqual(min_stack.get_min(), 1)
        min_stack.pop()
        self.assertEqual(min_stack.get_min(), 1)


class TestParenthesesValidation(unittest.TestCase):
    
    def test_valid_parentheses(self):
        self.assertTrue(is_valid_parentheses("()"))
        self.assertTrue(is_valid_parentheses("()[]{}"))
        self.assertTrue(is_valid_parentheses("{[()]}"))
        self.assertTrue(is_valid_parentheses(""))
    
    def test_invalid_parentheses(self):
        self.assertFalse(is_valid_parentheses("(]"))
        self.assertFalse(is_valid_parentheses("([)]"))
        self.assertFalse(is_valid_parentheses("(("))
        self.assertFalse(is_valid_parentheses("))"))
        self.assertFalse(is_valid_parentheses("(()"))


class TestPostfixEvaluation(unittest.TestCase):
    
    def test_basic_operations(self):
        self.assertEqual(evaluate_postfix("3 4 +"), 7)
        self.assertEqual(evaluate_postfix("3 4 -"), -1)
        self.assertEqual(evaluate_postfix("3 4 *"), 12)
        self.assertEqual(evaluate_postfix("8 4 /"), 2)
    
    def test_complex_expression(self):
        self.assertEqual(evaluate_postfix("3 4 + 2 *"), 14)
        self.assertEqual(evaluate_postfix("15 7 1 1 + - / 3 * 2 1 1 + + -"), 5)
    
    def test_invalid_expressions(self):
        with self.assertRaises(ValueError):
            evaluate_postfix("3 +")  # Not enough operands
        
        with self.assertRaises(ValueError):
            evaluate_postfix("3 4 5 +")  # Too many operands
        
        with self.assertRaises(ValueError):
            evaluate_postfix("3 0 /")  # Division by zero


class TestInfixToPostfix(unittest.TestCase):
    
    def test_simple_conversions(self):
        self.assertEqual(infix_to_postfix("A+B"), "AB+")
        self.assertEqual(infix_to_postfix("A+B*C"), "ABC*+")
        self.assertEqual(infix_to_postfix("(A+B)*C"), "AB+C*")
    
    def test_complex_conversions(self):
        self.assertEqual(infix_to_postfix("A+B*C-D"), "ABC*+D-")
        self.assertEqual(infix_to_postfix("(A+B)*(C-D)"), "AB+CD-*")


class TestNextGreaterElement(unittest.TestCase):
    
    def test_next_greater_element(self):
        nums = [4, 5, 2, 25]
        result = next_greater_element(nums)
        expected = [5, 25, 25, -1]
        self.assertEqual(result, expected)
    
    def test_decreasing_array(self):
        nums = [5, 4, 3, 2, 1]
        result = next_greater_element(nums)
        expected = [-1, -1, -1, -1, -1]
        self.assertEqual(result, expected)
    
    def test_increasing_array(self):
        nums = [1, 2, 3, 4, 5]
        result = next_greater_element(nums)
        expected = [2, 3, 4, 5, -1]
        self.assertEqual(result, expected)


class TestLargestRectangleHistogram(unittest.TestCase):
    
    def test_largest_rectangle(self):
        heights = [2, 1, 5, 6, 2, 3]
        result = largest_rectangle_histogram(heights)
        self.assertEqual(result, 10)  # Rectangle with height 5, width 2
    
    def test_single_bar(self):
        heights = [5]
        result = largest_rectangle_histogram(heights)
        self.assertEqual(result, 5)
    
    def test_increasing_heights(self):
        heights = [1, 2, 3, 4, 5]
        result = largest_rectangle_histogram(heights)
        self.assertEqual(result, 9)  # Rectangle with height 3, width 3


class TestDailyTemperatures(unittest.TestCase):
    
    def test_daily_temperatures(self):
        temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
        result = daily_temperatures(temperatures)
        expected = [1, 1, 4, 2, 1, 1, 0, 0]
        self.assertEqual(result, expected)
    
    def test_decreasing_temperatures(self):
        temperatures = [30, 25, 20, 15]
        result = daily_temperatures(temperatures)
        expected = [0, 0, 0, 0]
        self.assertEqual(result, expected)


class TestStackUsingQueues(unittest.TestCase):
    
    def test_stack_operations(self):
        stack = StackUsingQueues()
        self.assertTrue(stack.empty())
        
        stack.push(1)
        stack.push(2)
        stack.push(3)
        
        self.assertFalse(stack.empty())
        self.assertEqual(stack.top(), 3)
        self.assertEqual(stack.pop(), 3)
        self.assertEqual(stack.pop(), 2)
        self.assertEqual(stack.top(), 1)


class TestSimplifyPath(unittest.TestCase):
    
    def test_path_simplification(self):
        self.assertEqual(simplify_path("/home/"), "/home")
        self.assertEqual(simplify_path("/../"), "/")
        self.assertEqual(simplify_path("/home//foo/"), "/home/foo")
        self.assertEqual(simplify_path("/a/./b/../../c/"), "/c")
        self.assertEqual(simplify_path("/a/../../b/../c//.//"), "/c")
        self.assertEqual(simplify_path("/a//b////c/d//././/.."), "/a/b/c")


class TestDecodeString(unittest.TestCase):
    
    def test_decode_string(self):
        self.assertEqual(decode_string("3[a]2[bc]"), "aaabcbc")
        self.assertEqual(decode_string("2[abc]3[cd]ef"), "abcabccdcdcdef")
        self.assertEqual(decode_string("abc3[cd]xyz"), "abccdcdcdxyz")
    
    def test_nested_decode(self):
        self.assertEqual(decode_string("2[a2[bc]]"), "abcbcabcbc")
        self.assertEqual(decode_string("3[a2[c]]"), "accaccacc")


if __name__ == '__main__':
    unittest.main() 