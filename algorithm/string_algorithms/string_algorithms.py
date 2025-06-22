"""
String Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of classic string processing algorithms
with detailed time and space complexity analysis.
"""

from typing import List, Dict, Tuple
from collections import defaultdict


class StringAlgorithms:
    """Collection of string processing algorithms"""
    
    @staticmethod
    def kmp_search(text: str, pattern: str) -> List[int]:
        """
        KMP (Knuth-Morris-Pratt) Pattern Matching
        Time: O(n + m), Space: O(m)
        Returns list of starting indices where pattern is found
        """
        def compute_lps(pattern):
            """Compute Longest Proper Prefix which is also Suffix array"""
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1
            
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            
            return lps
        
        if not pattern:
            return []
        
        n, m = len(text), len(pattern)
        lps = compute_lps(pattern)
        result = []
        
        i = j = 0  # i for text, j for pattern
        
        while i < n:
            if text[i] == pattern[j]:
                i += 1
                j += 1
            
            if j == m:
                result.append(i - j)
                j = lps[j - 1]
            elif i < n and text[i] != pattern[j]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return result
    
    @staticmethod
    def rabin_karp_search(text: str, pattern: str, prime: int = 101) -> List[int]:
        """
        Rabin-Karp Pattern Matching using Rolling Hash
        Time: O(n + m) average, O(nm) worst case, Space: O(1)
        """
        def hash_value(s, start, end, prime):
            """Calculate hash value for substring"""
            h = 0
            for i in range(start, end):
                h = (h * 256 + ord(s[i])) % prime
            return h
        
        if not pattern:
            return []
        
        n, m = len(text), len(pattern)
        if m > n:
            return []
        
        result = []
        pattern_hash = hash_value(pattern, 0, m, prime)
        text_hash = hash_value(text, 0, m, prime)
        
        # Calculate 256^(m-1) % prime for rolling hash
        h = 1
        for _ in range(m - 1):
            h = (h * 256) % prime
        
        # Check first window
        if pattern_hash == text_hash and text[:m] == pattern:
            result.append(0)
        
        # Roll the hash over the text
        for i in range(1, n - m + 1):
            # Remove leading character and add trailing character
            text_hash = (256 * (text_hash - ord(text[i - 1]) * h) + ord(text[i + m - 1])) % prime
            
            # If hash values match, check actual strings
            if pattern_hash == text_hash and text[i:i + m] == pattern:
                result.append(i)
        
        return result
    
    @staticmethod
    def longest_palindromic_substring(s: str) -> str:
        """
        Find longest palindromic substring using Expand Around Centers
        Time: O(n²), Space: O(1)
        """
        if not s:
            return ""
        
        start = 0
        max_len = 1
        
        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        for i in range(len(s)):
            # Check for odd length palindromes (center at i)
            len1 = expand_around_center(i, i)
            
            # Check for even length palindromes (center between i and i+1)
            len2 = expand_around_center(i, i + 1)
            
            current_max = max(len1, len2)
            
            if current_max > max_len:
                max_len = current_max
                start = i - (current_max - 1) // 2
        
        return s[start:start + max_len]
    
    @staticmethod
    def manacher_algorithm(s: str) -> str:
        """
        Manacher's Algorithm for longest palindromic substring
        Time: O(n), Space: O(n)
        """
        if not s:
            return ""
        
        # Preprocess string: "abc" -> "^#a#b#c#$"
        processed = "^#" + "#".join(s) + "#$"
        n = len(processed)
        P = [0] * n  # Array to store palindrome lengths
        center = right = 0  # Center and right boundary of current palindrome
        
        max_len = 0
        center_index = 0
        
        for i in range(1, n - 1):
            # Mirror of i with respect to center
            mirror = 2 * center - i
            
            # If i is within right boundary, use previously computed values
            if i < right:
                P[i] = min(right - i, P[mirror])
            
            # Try to expand palindrome centered at i
            try:
                while processed[i + P[i] + 1] == processed[i - P[i] - 1]:
                    P[i] += 1
            except IndexError:
                pass
            
            # If palindrome centered at i extends past right, adjust center and right
            if i + P[i] > right:
                center, right = i, i + P[i]
            
            # Update maximum length palindrome
            if P[i] > max_len:
                max_len = P[i]
                center_index = i
        
        # Extract the longest palindrome
        start = (center_index - max_len) // 2
        return s[start:start + max_len]
    
    @staticmethod
    def z_algorithm(s: str) -> List[int]:
        """
        Z Algorithm for pattern matching
        Time: O(n), Space: O(n)
        Returns Z array where Z[i] = length of longest substring starting from i which is also prefix
        """
        n = len(s)
        if n == 0:
            return []
        
        Z = [0] * n
        Z[0] = n
        
        left = right = 0
        
        for i in range(1, n):
            if i <= right:
                Z[i] = min(right - i + 1, Z[i - left])
            
            while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
                Z[i] += 1
            
            if i + Z[i] - 1 > right:
                left, right = i, i + Z[i] - 1
        
        return Z
    
    @staticmethod
    def longest_common_subsequence_string(text1: str, text2: str) -> str:
        """
        Find actual LCS string (not just length)
        Time: O(mn), Space: O(mn)
        """
        m, n = len(text1), len(text2)
        dp = [["" for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + text1[i - 1]
                else:
                    dp[i][j] = dp[i - 1][j] if len(dp[i - 1][j]) > len(dp[i][j - 1]) else dp[i][j - 1]
        
        return dp[m][n]
    
    @staticmethod
    def edit_distance_operations(word1: str, word2: str) -> List[str]:
        """
        Edit Distance with actual operations
        Time: O(mn), Space: O(mn)
        Returns list of operations to transform word1 to word2
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        
        # Backtrack to find operations
        operations = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and word1[i - 1] == word2[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                operations.append(f"Replace '{word1[i - 1]}' with '{word2[j - 1]}' at position {i - 1}")
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                operations.append(f"Delete '{word1[i - 1]}' at position {i - 1}")
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                operations.append(f"Insert '{word2[j - 1]}' at position {i}")
                j -= 1
        
        return operations[::-1]  # Reverse to get correct order
    
    @staticmethod
    def suffix_array(s: str) -> List[int]:
        """
        Build suffix array using simple sorting
        Time: O(n² log n), Space: O(n)
        For better complexity, use advanced algorithms like DC3 or SA-IS
        """
        n = len(s)
        suffixes = [(s[i:], i) for i in range(n)]
        suffixes.sort()
        return [suffix[1] for suffix in suffixes]
    
    @staticmethod
    def lcp_array(s: str, suffix_arr: List[int]) -> List[int]:
        """
        Build LCP (Longest Common Prefix) array from suffix array
        Time: O(n), Space: O(n)
        """
        n = len(s)
        lcp = [0] * n
        inv_suffix = [0] * n
        
        # Build inverse suffix array
        for i in range(n):
            inv_suffix[suffix_arr[i]] = i
        
        k = 0
        for i in range(n):
            if inv_suffix[i] == n - 1:
                k = 0
                continue
            
            j = suffix_arr[inv_suffix[i] + 1]
            
            while i + k < n and j + k < n and s[i + k] == s[j + k]:
                k += 1
            
            lcp[inv_suffix[i]] = k
            
            if k > 0:
                k -= 1
        
        return lcp
    
    @staticmethod
    def boyer_moore_search(text: str, pattern: str) -> List[int]:
        """
        Boyer-Moore Pattern Matching (simplified version with bad character rule)
        Time: O(nm) worst case, O(n/m) best case, Space: O(σ)
        """
        def build_bad_char_table(pattern):
            """Build bad character table"""
            table = {}
            for i, char in enumerate(pattern):
                table[char] = i
            return table
        
        if not pattern:
            return []
        
        n, m = len(text), len(pattern)
        bad_char = build_bad_char_table(pattern)
        result = []
        
        shift = 0
        while shift <= n - m:
            j = m - 1
            
            # Keep reducing j while characters match
            while j >= 0 and pattern[j] == text[shift + j]:
                j -= 1
            
            if j < 0:
                # Pattern found
                result.append(shift)
                # Shift pattern to align with next possible match
                shift += m - bad_char.get(text[shift + m], -1) if shift + m < n else 1
            else:
                # Shift pattern based on bad character rule
                shift += max(1, j - bad_char.get(text[shift + j], -1))
        
        return result
    
    @staticmethod
    def aho_corasick_search(text: str, patterns: List[str]) -> Dict[str, List[int]]:
        """
        Aho-Corasick Algorithm for multiple pattern matching
        Time: O(n + m + z), Space: O(m)
        where n = len(text), m = sum of pattern lengths, z = number of matches
        """
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.failure = None
                self.output = []
        
        # Build trie
        root = TrieNode()
        
        for pattern in patterns:
            node = root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.output.append(pattern)
        
        # Build failure links (simplified version)
        from collections import deque
        queue = deque()
        
        # Set failure links for depth 1
        for child in root.children.values():
            child.failure = root
            queue.append(child)
        
        # Set failure links for deeper levels
        while queue:
            current = queue.popleft()
            
            for char, child in current.children.items():
                queue.append(child)
                
                # Find failure link
                failure = current.failure
                while failure and char not in failure.children:
                    failure = failure.failure
                
                child.failure = failure.children.get(char, root) if failure else root
                
                # Add output from failure link
                child.output.extend(child.failure.output)
        
        # Search for patterns
        result = {pattern: [] for pattern in patterns}
        node = root
        
        for i, char in enumerate(text):
            # Follow failure links until we find a match or reach root
            while node and char not in node.children:
                node = node.failure
            
            if not node:
                node = root
                continue
            
            node = node.children[char]
            
            # Report all patterns ending at this position
            for pattern in node.output:
                result[pattern].append(i - len(pattern) + 1)
        
        return result


# Example usage
if __name__ == "__main__":
    sa = StringAlgorithms()
    
    # KMP Search
    text = "ABABDABACDABABCABCABCABCABC"
    pattern = "ABABCABCABCABC"
    kmp_matches = sa.kmp_search(text, pattern)
    print(f"KMP matches for '{pattern}' in text: {kmp_matches}")
    
    # Rabin-Karp Search
    rk_matches = sa.rabin_karp_search(text, "ABC")
    print(f"Rabin-Karp matches for 'ABC': {rk_matches}")
    
    # Longest Palindromic Substring
    palindrome_text = "babad"
    longest_palindrome = sa.longest_palindromic_substring(palindrome_text)
    print(f"Longest palindrome in '{palindrome_text}': '{longest_palindrome}'")
    
    # Manacher's Algorithm
    manacher_result = sa.manacher_algorithm("racecar")
    print(f"Manacher result for 'racecar': '{manacher_result}'")
    
    # Z Algorithm
    z_text = "aabcaabxaaaz"
    z_array = sa.z_algorithm(z_text)
    print(f"Z array for '{z_text}': {z_array}")
    
    # LCS String
    lcs_result = sa.longest_common_subsequence_string("ABCDGH", "AEDFHR")
    print(f"LCS of 'ABCDGH' and 'AEDFHR': '{lcs_result}'")
    
    # Edit Distance Operations
    operations = sa.edit_distance_operations("kitten", "sitting")
    print(f"Edit operations from 'kitten' to 'sitting':")
    for op in operations:
        print(f"  {op}")
    
    # Suffix Array
    suffix_text = "banana"
    suffix_arr = sa.suffix_array(suffix_text)
    print(f"Suffix array for '{suffix_text}': {suffix_arr}")
    
    # LCP Array
    lcp_arr = sa.lcp_array(suffix_text, suffix_arr)
    print(f"LCP array: {lcp_arr}")
    
    # Boyer-Moore Search
    bm_matches = sa.boyer_moore_search("HERE IS A SIMPLE EXAMPLE", "EXAMPLE")
    print(f"Boyer-Moore matches: {bm_matches}")
    
    # Aho-Corasick Search
    patterns = ["he", "she", "his", "hers"]
    ac_text = "ushers"
    ac_matches = sa.aho_corasick_search(ac_text, patterns)
    print(f"Aho-Corasick matches in '{ac_text}': {ac_matches}") 