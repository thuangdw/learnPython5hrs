"""
Backtracking Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of classic backtracking problems
with detailed time and space complexity analysis.
"""

from typing import List, Set, Tuple


class BacktrackingAlgorithms:
    """Collection of backtracking algorithms"""
    
    @staticmethod
    def n_queens(n: int) -> List[List[str]]:
        """
        N-Queens Problem
        Time: O(N!), Space: O(N)
        Place N queens on NxN board such that none attack each other
        """
        def is_safe(board, row, col):
            # Check column
            for i in range(row):
                if board[i][col] == 'Q':
                    return False
            
            # Check diagonal (top-left to bottom-right)
            for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
                if board[i][j] == 'Q':
                    return False
            
            # Check diagonal (top-right to bottom-left)
            for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
                if board[i][j] == 'Q':
                    return False
            
            return True
        
        def solve(board, row):
            if row == n:
                return [[''.join(row) for row in board]]
            
            solutions = []
            for col in range(n):
                if is_safe(board, row, col):
                    board[row][col] = 'Q'
                    solutions.extend(solve(board, row + 1))
                    board[row][col] = '.'  # Backtrack
            
            return solutions
        
        board = [['.' for _ in range(n)] for _ in range(n)]
        return solve(board, 0)
    
    @staticmethod
    def solve_sudoku(board: List[List[str]]) -> bool:
        """
        Sudoku Solver
        Time: O(9^(n*n)), Space: O(n*n)
        Solve 9x9 Sudoku puzzle
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
                    if board[i][j] == '.':
                        for num in '123456789':
                            if is_valid(board, i, j, num):
                                board[i][j] = num
                                if solve():
                                    return True
                                board[i][j] = '.'  # Backtrack
                        return False
            return True
        
        return solve()
    
    @staticmethod
    def generate_permutations(nums: List[int]) -> List[List[int]]:
        """
        Generate all permutations
        Time: O(N! * N), Space: O(N)
        """
        def backtrack(current_perm, remaining):
            if not remaining:
                result.append(current_perm[:])
                return
            
            for i in range(len(remaining)):
                current_perm.append(remaining[i])
                backtrack(current_perm, remaining[:i] + remaining[i+1:])
                current_perm.pop()  # Backtrack
        
        result = []
        backtrack([], nums)
        return result
    
    @staticmethod
    def generate_combinations(n: int, k: int) -> List[List[int]]:
        """
        Generate all combinations of k numbers from 1 to n
        Time: O(C(n,k) * k), Space: O(k)
        """
        def backtrack(start, current_comb):
            if len(current_comb) == k:
                result.append(current_comb[:])
                return
            
            for i in range(start, n + 1):
                current_comb.append(i)
                backtrack(i + 1, current_comb)
                current_comb.pop()  # Backtrack
        
        result = []
        backtrack(1, [])
        return result
    
    @staticmethod
    def word_search(board: List[List[str]], word: str) -> bool:
        """
        Word Search in 2D board
        Time: O(N * 4^L), Space: O(L)
        where N is number of cells, L is length of word
        """
        if not board or not board[0]:
            return False
        
        rows, cols = len(board), len(board[0])
        
        def backtrack(row, col, index):
            if index == len(word):
                return True
            
            if (row < 0 or row >= rows or col < 0 or col >= cols or
                board[row][col] != word[index]):
                return False
            
            # Mark as visited
            temp = board[row][col]
            board[row][col] = '#'
            
            # Explore all 4 directions
            found = (backtrack(row + 1, col, index + 1) or
                    backtrack(row - 1, col, index + 1) or
                    backtrack(row, col + 1, index + 1) or
                    backtrack(row, col - 1, index + 1))
            
            # Backtrack
            board[row][col] = temp
            
            return found
        
        for i in range(rows):
            for j in range(cols):
                if backtrack(i, j, 0):
                    return True
        
        return False
    
    @staticmethod
    def subset_sum(nums: List[int], target: int) -> bool:
        """
        Subset Sum Problem
        Time: O(2^N), Space: O(N)
        Check if there's a subset that sums to target
        """
        def backtrack(index, current_sum):
            if current_sum == target:
                return True
            if index >= len(nums) or current_sum > target:
                return False
            
            # Include current number
            if backtrack(index + 1, current_sum + nums[index]):
                return True
            
            # Exclude current number
            return backtrack(index + 1, current_sum)
        
        return backtrack(0, 0)
    
    @staticmethod
    def generate_parentheses(n: int) -> List[str]:
        """
        Generate all valid parentheses combinations
        Time: O(4^n / √n), Space: O(n)
        """
        def backtrack(current, open_count, close_count):
            if len(current) == 2 * n:
                result.append(current)
                return
            
            # Add opening parenthesis
            if open_count < n:
                backtrack(current + '(', open_count + 1, close_count)
            
            # Add closing parenthesis
            if close_count < open_count:
                backtrack(current + ')', open_count, close_count + 1)
        
        result = []
        backtrack('', 0, 0)
        return result
    
    @staticmethod
    def palindrome_partitioning(s: str) -> List[List[str]]:
        """
        Palindrome Partitioning
        Time: O(N * 2^N), Space: O(N)
        Partition string into palindromic substrings
        """
        def is_palindrome(string):
            return string == string[::-1]
        
        def backtrack(start, current_partition):
            if start == len(s):
                result.append(current_partition[:])
                return
            
            for end in range(start, len(s)):
                substring = s[start:end+1]
                if is_palindrome(substring):
                    current_partition.append(substring)
                    backtrack(end + 1, current_partition)
                    current_partition.pop()  # Backtrack
        
        result = []
        backtrack(0, [])
        return result
    
    @staticmethod
    def letter_combinations(digits: str) -> List[str]:
        """
        Letter Combinations of Phone Number
        Time: O(4^N), Space: O(N)
        """
        if not digits:
            return []
        
        phone_map = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        
        def backtrack(index, current_combination):
            if index == len(digits):
                result.append(current_combination)
                return
            
            digit = digits[index]
            for letter in phone_map[digit]:
                backtrack(index + 1, current_combination + letter)
        
        result = []
        backtrack(0, '')
        return result
    
    @staticmethod
    def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
        """
        Combination Sum (with repetition allowed)
        Time: O(N^(T/M)), Space: O(T/M)
        where T is target, M is minimal value in candidates
        """
        def backtrack(start, current_comb, current_sum):
            if current_sum == target:
                result.append(current_comb[:])
                return
            if current_sum > target:
                return
            
            for i in range(start, len(candidates)):
                current_comb.append(candidates[i])
                backtrack(i, current_comb, current_sum + candidates[i])
                current_comb.pop()  # Backtrack
        
        candidates.sort()  # Sort to optimize pruning
        result = []
        backtrack(0, [], 0)
        return result
    
    @staticmethod
    def rat_in_maze(maze: List[List[int]]) -> List[str]:
        """
        Rat in a Maze Problem
        Time: O(4^(N*N)), Space: O(N*N)
        Find all paths from top-left to bottom-right
        """
        n = len(maze)
        if n == 0 or maze[0][0] == 0 or maze[n-1][n-1] == 0:
            return []
        
        def is_safe(x, y, visited):
            return (0 <= x < n and 0 <= y < n and 
                   maze[x][y] == 1 and not visited[x][y])
        
        def solve(x, y, path, visited):
            if x == n-1 and y == n-1:
                result.append(path)
                return
            
            visited[x][y] = True
            
            # Down
            if is_safe(x+1, y, visited):
                solve(x+1, y, path + 'D', visited)
            
            # Left
            if is_safe(x, y-1, visited):
                solve(x, y-1, path + 'L', visited)
            
            # Right
            if is_safe(x, y+1, visited):
                solve(x, y+1, path + 'R', visited)
            
            # Up
            if is_safe(x-1, y, visited):
                solve(x-1, y, path + 'U', visited)
            
            visited[x][y] = False  # Backtrack
        
        result = []
        visited = [[False for _ in range(n)] for _ in range(n)]
        solve(0, 0, '', visited)
        return sorted(result)


# Example usage
if __name__ == "__main__":
    bt = BacktrackingAlgorithms()
    
    # N-Queens
    print("4-Queens solutions:")
    queens_solutions = bt.n_queens(4)
    for i, solution in enumerate(queens_solutions):
        print(f"Solution {i+1}:")
        for row in solution:
            print(row)
        print()
    
    # Sudoku
    sudoku_board = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    print("Sudoku solved:", bt.solve_sudoku(sudoku_board))
    
    # Permutations
    nums = [1, 2, 3]
    perms = bt.generate_permutations(nums)
    print(f"Permutations of {nums}: {perms}")
    
    # Combinations
    combs = bt.generate_combinations(4, 2)
    print(f"Combinations C(4,2): {combs}")
    
    # Word Search
    board = [
        ['A','B','C','E'],
        ['S','F','C','S'],
        ['A','D','E','E']
    ]
    word = "ABCCED"
    print(f"Word '{word}' found: {bt.word_search(board, word)}")
    
    # Subset Sum
    nums = [3, 34, 4, 12, 5, 2]
    target = 9
    print(f"Subset sum {target} exists: {bt.subset_sum(nums, target)}")
    
    # Generate Parentheses
    n = 3
    parentheses = bt.generate_parentheses(n)
    print(f"Valid parentheses for n={n}: {parentheses}")
    
    # Palindrome Partitioning
    s = "aab"
    palindromes = bt.palindrome_partitioning(s)
    print(f"Palindrome partitions of '{s}': {palindromes}")
    
    # Letter Combinations
    digits = "23"
    letters = bt.letter_combinations(digits)
    print(f"Letter combinations for '{digits}': {letters}")
    
    # Combination Sum
    candidates = [2, 3, 6, 7]
    target = 7
    comb_sums = bt.combination_sum(candidates, target)
    print(f"Combination sums for target {target}: {comb_sums}")
    
    # Rat in Maze
    maze = [
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 1]
    ]
    paths = bt.rat_in_maze(maze)
    print(f"Rat maze paths: {paths}") 