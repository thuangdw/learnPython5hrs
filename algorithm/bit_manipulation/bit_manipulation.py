"""
Bit Manipulation Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of bit manipulation algorithms and tricks
with detailed time and space complexity analysis.
"""

from typing import List, Tuple


class BitManipulation:
    """Collection of bit manipulation algorithms and tricks"""
    
    @staticmethod
    def set_bit(num: int, position: int) -> int:
        """
        Set bit at given position to 1
        Time: O(1), Space: O(1)
        """
        return num | (1 << position)
    
    @staticmethod
    def clear_bit(num: int, position: int) -> int:
        """
        Clear bit at given position (set to 0)
        Time: O(1), Space: O(1)
        """
        return num & ~(1 << position)
    
    @staticmethod
    def toggle_bit(num: int, position: int) -> int:
        """
        Toggle bit at given position
        Time: O(1), Space: O(1)
        """
        return num ^ (1 << position)
    
    @staticmethod
    def check_bit(num: int, position: int) -> bool:
        """
        Check if bit at given position is set
        Time: O(1), Space: O(1)
        """
        return (num & (1 << position)) != 0
    
    @staticmethod
    def count_set_bits(num: int) -> int:
        """
        Count number of set bits (Brian Kernighan's algorithm)
        Time: O(number of set bits), Space: O(1)
        """
        count = 0
        while num:
            num &= num - 1  # Remove the rightmost set bit
            count += 1
        return count
    
    @staticmethod
    def count_set_bits_builtin(num: int) -> int:
        """
        Count set bits using built-in function
        Time: O(1), Space: O(1)
        """
        return bin(num).count('1')
    
    @staticmethod
    def is_power_of_two(num: int) -> bool:
        """
        Check if number is power of 2
        Time: O(1), Space: O(1)
        """
        return num > 0 and (num & (num - 1)) == 0
    
    @staticmethod
    def next_power_of_two(num: int) -> int:
        """
        Find next power of 2 greater than or equal to num
        Time: O(log n), Space: O(1)
        """
        if num <= 1:
            return 1
        
        num -= 1
        num |= num >> 1
        num |= num >> 2
        num |= num >> 4
        num |= num >> 8
        num |= num >> 16
        num |= num >> 32  # For 64-bit numbers
        
        return num + 1
    
    @staticmethod
    def reverse_bits(num: int, bit_length: int = 32) -> int:
        """
        Reverse bits of a number
        Time: O(log n), Space: O(1)
        """
        result = 0
        for _ in range(bit_length):
            result = (result << 1) | (num & 1)
            num >>= 1
        return result
    
    @staticmethod
    def swap_bits(num: int, pos1: int, pos2: int) -> int:
        """
        Swap bits at two positions
        Time: O(1), Space: O(1)
        """
        # Check if bits are different
        if ((num >> pos1) & 1) != ((num >> pos2) & 1):
            # Toggle both bits
            num ^= (1 << pos1) | (1 << pos2)
        return num
    
    @staticmethod
    def find_missing_number(nums: List[int]) -> int:
        """
        Find missing number in array [0, n] using XOR
        Time: O(n), Space: O(1)
        """
        n = len(nums)
        result = n  # Start with n
        
        for i in range(n):
            result ^= i ^ nums[i]
        
        return result
    
    @staticmethod
    def single_number(nums: List[int]) -> int:
        """
        Find single number when all others appear twice
        Time: O(n), Space: O(1)
        """
        result = 0
        for num in nums:
            result ^= num
        return result
    
    @staticmethod
    def single_number_three_times(nums: List[int]) -> int:
        """
        Find single number when all others appear three times
        Time: O(n), Space: O(1)
        """
        ones = twos = 0
        
        for num in nums:
            ones = (ones ^ num) & ~twos
            twos = (twos ^ num) & ~ones
        
        return ones
    
    @staticmethod
    def two_single_numbers(nums: List[int]) -> Tuple[int, int]:
        """
        Find two single numbers when all others appear twice
        Time: O(n), Space: O(1)
        """
        # XOR all numbers
        xor_all = 0
        for num in nums:
            xor_all ^= num
        
        # Find rightmost set bit
        rightmost_set_bit = xor_all & -xor_all
        
        # Divide numbers into two groups and XOR separately
        num1 = num2 = 0
        for num in nums:
            if num & rightmost_set_bit:
                num1 ^= num
            else:
                num2 ^= num
        
        return num1, num2
    
    @staticmethod
    def subset_generation(nums: List[int]) -> List[List[int]]:
        """
        Generate all subsets using bit manipulation
        Time: O(n * 2^n), Space: O(n * 2^n)
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
    
    @staticmethod
    def gray_code(n: int) -> List[int]:
        """
        Generate Gray code sequence
        Time: O(2^n), Space: O(2^n)
        """
        if n == 0:
            return [0]
        
        result = [0, 1]
        
        for i in range(2, n + 1):
            # Add MSB to existing codes in reverse order
            for j in range(len(result) - 1, -1, -1):
                result.append(result[j] | (1 << (i - 1)))
        
        return result
    
    @staticmethod
    def hamming_distance(x: int, y: int) -> int:
        """
        Calculate Hamming distance between two numbers
        Time: O(log n), Space: O(1)
        """
        return BitManipulation.count_set_bits(x ^ y)
    
    @staticmethod
    def total_hamming_distance(nums: List[int]) -> int:
        """
        Calculate total Hamming distance between all pairs
        Time: O(n), Space: O(1)
        """
        total = 0
        n = len(nums)
        
        for i in range(32):  # Assuming 32-bit integers
            count_ones = 0
            for num in nums:
                if num & (1 << i):
                    count_ones += 1
            
            count_zeros = n - count_ones
            total += count_ones * count_zeros
        
        return total
    
    @staticmethod
    def maximum_xor(nums: List[int]) -> int:
        """
        Find maximum XOR of any two numbers in array
        Time: O(n), Space: O(1)
        """
        max_xor = 0
        mask = 0
        
        for i in range(31, -1, -1):  # From MSB to LSB
            mask |= (1 << i)
            prefixes = {num & mask for num in nums}
            
            candidate = max_xor | (1 << i)
            
            # Check if candidate can be formed
            for prefix in prefixes:
                if candidate ^ prefix in prefixes:
                    max_xor = candidate
                    break
        
        return max_xor
    
    @staticmethod
    def bitwise_and_range(left: int, right: int) -> int:
        """
        Bitwise AND of numbers in range [left, right]
        Time: O(log n), Space: O(1)
        """
        shift = 0
        while left != right:
            left >>= 1
            right >>= 1
            shift += 1
        
        return left << shift
    
    @staticmethod
    def count_bits_dp(n: int) -> List[int]:
        """
        Count bits for numbers 0 to n using DP
        Time: O(n), Space: O(n)
        """
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            dp[i] = dp[i >> 1] + (i & 1)
        
        return dp
    
    @staticmethod
    def multiply_by_power_of_two(num: int, power: int) -> int:
        """
        Multiply number by 2^power using bit shift
        Time: O(1), Space: O(1)
        """
        return num << power
    
    @staticmethod
    def divide_by_power_of_two(num: int, power: int) -> int:
        """
        Divide number by 2^power using bit shift
        Time: O(1), Space: O(1)
        """
        return num >> power
    
    @staticmethod
    def is_even(num: int) -> bool:
        """
        Check if number is even using bitwise AND
        Time: O(1), Space: O(1)
        """
        return (num & 1) == 0
    
    @staticmethod
    def is_odd(num: int) -> bool:
        """
        Check if number is odd using bitwise AND
        Time: O(1), Space: O(1)
        """
        return (num & 1) == 1
    
    @staticmethod
    def swap_without_temp(a: int, b: int) -> Tuple[int, int]:
        """
        Swap two numbers without temporary variable
        Time: O(1), Space: O(1)
        """
        if a != b:  # Avoid issues when a and b are the same
            a ^= b
            b ^= a
            a ^= b
        return a, b
    
    @staticmethod
    def absolute_value(num: int) -> int:
        """
        Get absolute value using bit manipulation
        Time: O(1), Space: O(1)
        """
        mask = num >> 31  # Get sign bit (works for 32-bit signed integers)
        return (num + mask) ^ mask
    
    @staticmethod
    def min_without_branching(a: int, b: int) -> int:
        """
        Find minimum without branching
        Time: O(1), Space: O(1)
        """
        return b ^ ((a ^ b) & -(a < b))
    
    @staticmethod
    def max_without_branching(a: int, b: int) -> int:
        """
        Find maximum without branching
        Time: O(1), Space: O(1)
        """
        return a ^ ((a ^ b) & -(a < b))
    
    @staticmethod
    def binary_to_decimal(binary_str: str) -> int:
        """
        Convert binary string to decimal using bit manipulation
        Time: O(n), Space: O(1)
        """
        result = 0
        for bit in binary_str:
            result = (result << 1) + int(bit)
        return result
    
    @staticmethod
    def decimal_to_binary(num: int) -> str:
        """
        Convert decimal to binary string
        Time: O(log n), Space: O(log n)
        """
        if num == 0:
            return "0"
        
        binary = ""
        while num > 0:
            binary = str(num & 1) + binary
            num >>= 1
        
        return binary


# Example usage and demonstrations
if __name__ == "__main__":
    bm = BitManipulation()
    
    # Basic bit operations
    num = 12  # Binary: 1100
    print(f"Original number: {num} (binary: {bin(num)})")
    
    # Set bit at position 1
    set_result = bm.set_bit(num, 1)
    print(f"Set bit 1: {set_result} (binary: {bin(set_result)})")
    
    # Clear bit at position 3
    clear_result = bm.clear_bit(num, 3)
    print(f"Clear bit 3: {clear_result} (binary: {bin(clear_result)})")
    
    # Toggle bit at position 0
    toggle_result = bm.toggle_bit(num, 0)
    print(f"Toggle bit 0: {toggle_result} (binary: {bin(toggle_result)})")
    
    # Check bit at position 2
    check_result = bm.check_bit(num, 2)
    print(f"Bit 2 is set: {check_result}")
    
    # Count set bits
    count = bm.count_set_bits(num)
    print(f"Number of set bits: {count}")
    
    # Power of 2 operations
    print(f"\nPower of 2 operations:")
    print(f"Is 16 power of 2? {bm.is_power_of_two(16)}")
    print(f"Is 15 power of 2? {bm.is_power_of_two(15)}")
    print(f"Next power of 2 after 10: {bm.next_power_of_two(10)}")
    
    # Reverse bits
    reversed_bits = bm.reverse_bits(12, 8)
    print(f"Reverse bits of 12 (8-bit): {reversed_bits} (binary: {bin(reversed_bits)})")
    
    # XOR applications
    print(f"\nXOR applications:")
    
    # Missing number
    missing_array = [0, 1, 3, 4, 5]  # Missing 2
    missing = bm.find_missing_number(missing_array)
    print(f"Missing number: {missing}")
    
    # Single number
    single_array = [2, 2, 1, 1, 4, 4, 3]  # 3 appears once
    single = bm.single_number(single_array)
    print(f"Single number: {single}")
    
    # Two single numbers
    two_single_array = [1, 2, 1, 3, 2, 5]  # 3 and 5 appear once
    num1, num2 = bm.two_single_numbers(two_single_array)
    print(f"Two single numbers: {num1}, {num2}")
    
    # Subset generation
    subset_nums = [1, 2, 3]
    subsets = bm.subset_generation(subset_nums)
    print(f"All subsets of {subset_nums}: {subsets}")
    
    # Gray code
    gray_codes = bm.gray_code(3)
    print(f"Gray code for n=3: {gray_codes}")
    print(f"Gray code binary: {[bin(code) for code in gray_codes]}")
    
    # Hamming distance
    hamming_dist = bm.hamming_distance(1, 4)  # 001 vs 100
    print(f"Hamming distance between 1 and 4: {hamming_dist}")
    
    # Maximum XOR
    xor_array = [3, 10, 5, 25, 2, 8]
    max_xor = bm.maximum_xor(xor_array)
    print(f"Maximum XOR in {xor_array}: {max_xor}")
    
    # Bitwise AND range
    and_range = bm.bitwise_and_range(5, 7)
    print(f"Bitwise AND of range [5, 7]: {and_range}")
    
    # Count bits DP
    count_bits = bm.count_bits_dp(5)
    print(f"Count bits 0 to 5: {count_bits}")
    
    # Bit manipulation tricks
    print(f"\nBit manipulation tricks:")
    
    # Multiplication and division by powers of 2
    print(f"12 * 2^3 = {bm.multiply_by_power_of_two(12, 3)}")
    print(f"32 / 2^2 = {bm.divide_by_power_of_two(32, 2)}")
    
    # Even/odd check
    print(f"Is 7 even? {bm.is_even(7)}")
    print(f"Is 8 odd? {bm.is_odd(8)}")
    
    # Swap without temp
    a, b = 5, 10
    swapped_a, swapped_b = bm.swap_without_temp(a, b)
    print(f"Swap {a} and {b}: {swapped_a}, {swapped_b}")
    
    # Absolute value
    abs_val = bm.absolute_value(-15)
    print(f"Absolute value of -15: {abs_val}")
    
    # Min/Max without branching
    min_val = bm.min_without_branching(10, 20)
    max_val = bm.max_without_branching(10, 20)
    print(f"Min of 10 and 20: {min_val}")
    print(f"Max of 10 and 20: {max_val}")
    
    # Binary conversion
    binary_str = "1101"
    decimal = bm.binary_to_decimal(binary_str)
    print(f"Binary '{binary_str}' to decimal: {decimal}")
    
    binary_result = bm.decimal_to_binary(13)
    print(f"Decimal 13 to binary: '{binary_result}'") 