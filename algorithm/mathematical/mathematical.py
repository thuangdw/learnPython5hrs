"""
Mathematical Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of classic mathematical algorithms
with detailed time and space complexity analysis.
"""

from typing import List, Tuple, Dict
import math
import random


class MathematicalAlgorithms:
    """Collection of mathematical algorithms"""
    
    @staticmethod
    def euclidean_gcd(a: int, b: int) -> int:
        """
        Euclidean Algorithm for GCD
        Time: O(log(min(a,b))), Space: O(1)
        """
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def extended_euclidean_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean Algorithm
        Time: O(log(min(a,b))), Space: O(1)
        Returns (gcd, x, y) where ax + by = gcd(a,b)
        """
        if b == 0:
            return a, 1, 0
        
        gcd, x1, y1 = MathematicalAlgorithms.extended_euclidean_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        
        return gcd, x, y
    
    @staticmethod
    def lcm(a: int, b: int) -> int:
        """
        Least Common Multiple using GCD
        Time: O(log(min(a,b))), Space: O(1)
        """
        return abs(a * b) // MathematicalAlgorithms.euclidean_gcd(a, b)
    
    @staticmethod
    def sieve_of_eratosthenes(n: int) -> List[int]:
        """
        Sieve of Eratosthenes for finding all primes up to n
        Time: O(n log log n), Space: O(n)
        """
        if n < 2:
            return []
        
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    @staticmethod
    def segmented_sieve(low: int, high: int) -> List[int]:
        """
        Segmented Sieve for finding primes in range [low, high]
        Time: O((high-low+1) log log high), Space: O(sqrt(high))
        """
        if low < 2:
            low = 2
        
        # Find all primes up to sqrt(high)
        limit = int(math.sqrt(high)) + 1
        primes = MathematicalAlgorithms.sieve_of_eratosthenes(limit)
        
        # Create boolean array for range [low, high]
        is_prime = [True] * (high - low + 1)
        
        # Mark multiples of primes in the range
        for prime in primes:
            # Find the minimum number in [low, high] that is multiple of prime
            start = max(prime * prime, (low + prime - 1) // prime * prime)
            
            for j in range(start, high + 1, prime):
                is_prime[j - low] = False
        
        # Special case for 2
        if low == 2:
            result = [2]
        else:
            result = []
        
        # Collect primes
        for i in range(max(3, low), high + 1, 2):
            if is_prime[i - low]:
                result.append(i)
        
        return result
    
    @staticmethod
    def is_prime_trial_division(n: int) -> bool:
        """
        Prime check using trial division
        Time: O(sqrt(n)), Space: O(1)
        """
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        
        return True
    
    @staticmethod
    def miller_rabin_primality_test(n: int, k: int = 5) -> bool:
        """
        Miller-Rabin Probabilistic Primality Test
        Time: O(k log³ n), Space: O(1)
        k is the number of rounds (higher k = more accurate)
        """
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as d * 2^r
        r = 0
        d = n - 1
        while d % 2 == 0:
            d //= 2
            r += 1
        
        # Witness loop
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    @staticmethod
    def fast_exponentiation(base: int, exponent: int, modulus: int = None) -> int:
        """
        Fast Exponentiation using binary method
        Time: O(log exponent), Space: O(1)
        """
        if modulus is None:
            result = 1
            base = base
            
            while exponent > 0:
                if exponent % 2 == 1:
                    result *= base
                exponent //= 2
                base *= base
            
            return result
        else:
            return pow(base, exponent, modulus)  # Python's built-in is optimized
    
    @staticmethod
    def modular_inverse(a: int, m: int) -> int:
        """
        Modular Multiplicative Inverse using Extended Euclidean Algorithm
        Time: O(log m), Space: O(1)
        Returns x such that (a * x) % m = 1
        """
        gcd, x, _ = MathematicalAlgorithms.extended_euclidean_gcd(a, m)
        
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        
        return (x % m + m) % m
    
    @staticmethod
    def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
        """
        Chinese Remainder Theorem
        Time: O(n log M), Space: O(1)
        where M is the product of all moduli
        """
        if len(remainders) != len(moduli):
            raise ValueError("Number of remainders and moduli must be equal")
        
        # Check if moduli are pairwise coprime
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                if MathematicalAlgorithms.euclidean_gcd(moduli[i], moduli[j]) != 1:
                    raise ValueError("Moduli must be pairwise coprime")
        
        total = 0
        product = 1
        for m in moduli:
            product *= m
        
        for i in range(len(remainders)):
            partial_product = product // moduli[i]
            inverse = MathematicalAlgorithms.modular_inverse(partial_product, moduli[i])
            total += remainders[i] * partial_product * inverse
        
        return total % product
    
    @staticmethod
    def factorial_iterative(n: int) -> int:
        """
        Factorial using iteration
        Time: O(n), Space: O(1)
        """
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        
        result = 1
        for i in range(2, n + 1):
            result *= i
        
        return result
    
    @staticmethod
    def factorial_recursive(n: int) -> int:
        """
        Factorial using recursion
        Time: O(n), Space: O(n)
        """
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n <= 1:
            return 1
        
        return n * MathematicalAlgorithms.factorial_recursive(n - 1)
    
    @staticmethod
    def binomial_coefficient(n: int, k: int) -> int:
        """
        Binomial Coefficient C(n,k) = n! / (k! * (n-k)!)
        Time: O(min(k, n-k)), Space: O(1)
        """
        if k > n or k < 0:
            return 0
        
        k = min(k, n - k)  # Take advantage of symmetry
        
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        
        return result
    
    @staticmethod
    def fibonacci_matrix(n: int) -> int:
        """
        Fibonacci using matrix exponentiation
        Time: O(log n), Space: O(1)
        """
        if n <= 1:
            return n
        
        def matrix_multiply(A, B):
            return [
                [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
                [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
            ]
        
        def matrix_power(matrix, power):
            if power == 1:
                return matrix
            
            if power % 2 == 0:
                half_power = matrix_power(matrix, power // 2)
                return matrix_multiply(half_power, half_power)
            else:
                return matrix_multiply(matrix, matrix_power(matrix, power - 1))
        
        fib_matrix = [[1, 1], [1, 0]]
        result_matrix = matrix_power(fib_matrix, n)
        
        return result_matrix[0][1]
    
    @staticmethod
    def catalan_number(n: int) -> int:
        """
        nth Catalan Number using binomial coefficient
        Time: O(n), Space: O(1)
        Catalan(n) = C(2n, n) / (n + 1)
        """
        if n <= 1:
            return 1
        
        return MathematicalAlgorithms.binomial_coefficient(2 * n, n) // (n + 1)
    
    @staticmethod
    def euler_totient(n: int) -> int:
        """
        Euler's Totient Function φ(n)
        Time: O(sqrt(n)), Space: O(1)
        Returns count of integers ≤ n that are coprime to n
        """
        result = n
        p = 2
        
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        
        if n > 1:
            result -= result // n
        
        return result
    
    @staticmethod
    def pollard_rho_factorization(n: int) -> int:
        """
        Pollard's Rho Algorithm for integer factorization
        Time: O(n^(1/4)), Space: O(1)
        Returns a non-trivial factor of n
        """
        if n % 2 == 0:
            return 2
        
        x = random.randint(2, n - 2)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = MathematicalAlgorithms.euclidean_gcd(abs(x - y), n)
        
        return d if d != n else None
    
    @staticmethod
    def baby_step_giant_step(g: int, h: int, p: int) -> int:
        """
        Baby-step Giant-step Algorithm for discrete logarithm
        Time: O(sqrt(p)), Space: O(sqrt(p))
        Finds x such that g^x ≡ h (mod p)
        """
        n = int(math.sqrt(p)) + 1
        
        # Baby steps
        baby_steps = {}
        gamma = 1
        for j in range(n):
            if gamma == h:
                return j
            baby_steps[gamma] = j
            gamma = (gamma * g) % p
        
        # Giant steps
        factor = pow(g, n * (p - 2), p)  # g^(-n) mod p
        y = h
        
        for i in range(n):
            if y in baby_steps:
                return i * n + baby_steps[y]
            y = (y * factor) % p
        
        return None  # No solution found
    
    @staticmethod
    def josephus_problem(n: int, k: int) -> int:
        """
        Josephus Problem - find the survivor position
        Time: O(n), Space: O(1)
        """
        result = 0
        for i in range(2, n + 1):
            result = (result + k) % i
        return result + 1  # Convert to 1-indexed


# Example usage
if __name__ == "__main__":
    ma = MathematicalAlgorithms()
    
    # GCD and LCM
    a, b = 48, 18
    gcd = ma.euclidean_gcd(a, b)
    lcm = ma.lcm(a, b)
    print(f"GCD({a}, {b}) = {gcd}")
    print(f"LCM({a}, {b}) = {lcm}")
    
    # Extended GCD
    gcd, x, y = ma.extended_euclidean_gcd(a, b)
    print(f"Extended GCD: {a}*{x} + {b}*{y} = {gcd}")
    
    # Sieve of Eratosthenes
    primes = ma.sieve_of_eratosthenes(30)
    print(f"Primes up to 30: {primes}")
    
    # Segmented Sieve
    seg_primes = ma.segmented_sieve(50, 100)
    print(f"Primes from 50 to 100: {seg_primes}")
    
    # Primality Tests
    test_num = 97
    print(f"Is {test_num} prime (trial division)? {ma.is_prime_trial_division(test_num)}")
    print(f"Is {test_num} prime (Miller-Rabin)? {ma.miller_rabin_primality_test(test_num)}")
    
    # Fast Exponentiation
    base, exp, mod = 2, 10, 1000
    result = ma.fast_exponentiation(base, exp, mod)
    print(f"{base}^{exp} mod {mod} = {result}")
    
    # Modular Inverse
    try:
        inv = ma.modular_inverse(3, 11)
        print(f"Modular inverse of 3 mod 11 = {inv}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Chinese Remainder Theorem
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    crt_result = ma.chinese_remainder_theorem(remainders, moduli)
    print(f"CRT solution: x ≡ {crt_result} (mod {math.prod(moduli)})")
    
    # Factorial
    n = 10
    fact_iter = ma.factorial_iterative(n)
    fact_rec = ma.factorial_recursive(n)
    print(f"{n}! = {fact_iter} (iterative) = {fact_rec} (recursive)")
    
    # Binomial Coefficient
    n, k = 10, 4
    binom = ma.binomial_coefficient(n, k)
    print(f"C({n}, {k}) = {binom}")
    
    # Fibonacci (matrix method)
    fib_n = 20
    fib_result = ma.fibonacci_matrix(fib_n)
    print(f"Fibonacci({fib_n}) = {fib_result}")
    
    # Catalan Number
    cat_n = 5
    catalan = ma.catalan_number(cat_n)
    print(f"Catalan({cat_n}) = {catalan}")
    
    # Euler's Totient
    totient_n = 12
    totient = ma.euler_totient(totient_n)
    print(f"φ({totient_n}) = {totient}")
    
    # Josephus Problem
    josephus_n, josephus_k = 7, 3
    survivor = ma.josephus_problem(josephus_n, josephus_k)
    print(f"Josephus({josephus_n}, {josephus_k}) = position {survivor}")
    
    # Pollard's Rho (for composite numbers)
    composite = 8051  # 97 * 83
    factor = ma.pollard_rho_factorization(composite)
    if factor:
        print(f"Factor of {composite}: {factor}")
    
    # Baby-step Giant-step (example with small numbers)
    g, h, p = 2, 3, 5
    discrete_log = ma.baby_step_giant_step(g, h, p)
    if discrete_log is not None:
        print(f"Discrete log: {g}^{discrete_log} ≡ {h} (mod {p})") 