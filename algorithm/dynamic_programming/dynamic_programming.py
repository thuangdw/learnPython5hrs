"""
Dynamic Programming Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of classic dynamic programming problems
with detailed time and space complexity analysis.
"""

from typing import List, Dict, Tuple
import sys


class DynamicProgramming:
    """Collection of dynamic programming algorithms"""
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """
        Fibonacci with memoization
        Time: O(n), Space: O(n)
        """
        if n <= 1:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
    
    @staticmethod
    def fibonacci_optimized(n: int) -> int:
        """
        Space-optimized fibonacci
        Time: O(n), Space: O(1)
        """
        if n <= 1:
            return n
        
        prev2, prev1 = 0, 1
        
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    @staticmethod
    def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
        """
        0/1 Knapsack Problem
        Time: O(n * W), Space: O(n * W)
        """
        n = len(weights)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]],
                        dp[i - 1][w]
                    )
                else:
                    dp[i][w] = dp[i - 1][w]
        
        return dp[n][capacity]
    
    @staticmethod
    def knapsack_01_optimized(weights: List[int], values: List[int], capacity: int) -> int:
        """
        Space-optimized 0/1 Knapsack
        Time: O(n * W), Space: O(W)
        """
        dp = [0] * (capacity + 1)
        
        for i in range(len(weights)):
            for w in range(capacity, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
        
        return dp[capacity]
    
    @staticmethod
    def longest_common_subsequence(text1: str, text2: str) -> int:
        """
        Longest Common Subsequence
        Time: O(m * n), Space: O(m * n)
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    @staticmethod
    def edit_distance(word1: str, word2: str) -> int:
        """
        Edit Distance (Levenshtein Distance)
        Time: O(m * n), Space: O(m * n)
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
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],    # deletion
                        dp[i][j - 1],    # insertion
                        dp[i - 1][j - 1] # substitution
                    )
        
        return dp[m][n]
    
    @staticmethod
    def coin_change(coins: List[int], amount: int) -> int:
        """
        Coin Change - Minimum coins needed
        Time: O(amount * len(coins)), Space: O(amount)
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    @staticmethod
    def coin_change_ways(coins: List[int], amount: int) -> int:
        """
        Coin Change - Number of ways to make amount
        Time: O(amount * len(coins)), Space: O(amount)
        """
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] += dp[x - coin]
        
        return dp[amount]
    
    @staticmethod
    def longest_increasing_subsequence(nums: List[int]) -> int:
        """
        Longest Increasing Subsequence
        Time: O(n²), Space: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    @staticmethod
    def lis_binary_search(nums: List[int]) -> int:
        """
        LIS using binary search
        Time: O(n log n), Space: O(n)
        """
        if not nums:
            return 0
        
        tails = []
        
        for num in nums:
            left, right = 0, len(tails)
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            
            if left == len(tails):
                tails.append(num)
            else:
                tails[left] = num
        
        return len(tails)
    
    @staticmethod
    def maximum_subarray(nums: List[int]) -> int:
        """
        Maximum Subarray Sum (Kadane's Algorithm)
        Time: O(n), Space: O(1)
        """
        max_sum = current_sum = nums[0]
        
        for i in range(1, len(nums)):
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    @staticmethod
    def house_robber(nums: List[int]) -> int:
        """
        House Robber Problem
        Time: O(n), Space: O(1)
        """
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        prev2 = nums[0]
        prev1 = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            current = max(prev1, prev2 + nums[i])
            prev2, prev1 = prev1, current
        
        return prev1
    
    @staticmethod
    def unique_paths(m: int, n: int) -> int:
        """
        Unique Paths in Grid
        Time: O(m * n), Space: O(n)
        """
        dp = [1] * n
        
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        
        return dp[n - 1]
    
    @staticmethod
    def word_break(s: str, word_dict: List[str]) -> bool:
        """
        Word Break Problem
        Time: O(n² * m), Space: O(n)
        """
        word_set = set(word_dict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[len(s)]
    
    @staticmethod
    def palindrome_partitioning_min_cuts(s: str) -> int:
        """
        Minimum cuts for palindrome partitioning
        Time: O(n²), Space: O(n²)
        """
        n = len(s)
        
        # Precompute palindrome table
        is_palindrome = [[False] * n for _ in range(n)]
        for i in range(n):
            is_palindrome[i][i] = True
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    if length == 2:
                        is_palindrome[i][j] = True
                    else:
                        is_palindrome[i][j] = is_palindrome[i + 1][j - 1]
        
        # DP for minimum cuts
        dp = [0] * n
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0
            else:
                dp[i] = float('inf')
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n - 1]


# Example usage
if __name__ == "__main__":
    dp = DynamicProgramming()
    
    # Test Fibonacci
    print(f"Fibonacci(10): {dp.fibonacci(10)}")
    print(f"Fibonacci Optimized(10): {dp.fibonacci_optimized(10)}")
    
    # Test Knapsack
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    capacity = 7
    print(f"0/1 Knapsack: {dp.knapsack_01(weights, values, capacity)}")
    
    # Test LCS
    text1, text2 = "abcde", "ace"
    print(f"LCS length: {dp.longest_common_subsequence(text1, text2)}")
    
    # Test Edit Distance
    word1, word2 = "horse", "ros"
    print(f"Edit Distance: {dp.edit_distance(word1, word2)}")
    
    # Test Coin Change
    coins = [1, 3, 4]
    amount = 6
    print(f"Min coins for {amount}: {dp.coin_change(coins, amount)}")
    print(f"Ways to make {amount}: {dp.coin_change_ways(coins, amount)}")
    
    # Test LIS
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print(f"LIS length: {dp.longest_increasing_subsequence(nums)}")
    print(f"LIS (optimized): {dp.lis_binary_search(nums)}")
    
    # Test Maximum Subarray
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"Maximum subarray sum: {dp.maximum_subarray(arr)}")
    
    # Test House Robber
    houses = [2, 7, 9, 3, 1]
    print(f"Max money robbed: {dp.house_robber(houses)}")
    
    # Test Unique Paths
    print(f"Unique paths (3x7): {dp.unique_paths(3, 7)}")
    
    # Test Word Break
    s = "leetcode"
    word_dict = ["leet", "code"]
    print(f"Word break possible: {dp.word_break(s, word_dict)}")
    
    # Test Palindrome Partitioning
    palindrome_str = "aab"
    print(f"Min cuts for palindrome: {dp.palindrome_partitioning_min_cuts(palindrome_str)}") 