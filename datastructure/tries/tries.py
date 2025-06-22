"""
Tries Implementation
Comprehensive implementation of trie (prefix tree) data structures and algorithms
"""


class TrieNode:
    """Node for Trie data structure"""
    
    def __init__(self):
        self.children = {}  # Dictionary to store child nodes
        self.is_end_of_word = False
        self.word_count = 0  # Count of words ending at this node
        self.prefix_count = 0  # Count of words passing through this node


class Trie:
    """
    Trie (Prefix Tree) implementation
    Efficient for string operations like autocomplete, spell checking
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.size = 0  # Total number of unique words
    
    def insert(self, word):
        """
        Insert word into trie
        Time: O(m), Space: O(m) where m is word length
        """
        if not word:
            return
        
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.prefix_count += 1
        
        if not node.is_end_of_word:
            self.size += 1
        
        node.is_end_of_word = True
        node.word_count += 1
    
    def search(self, word):
        """
        Search for exact word in trie
        Time: O(m), Space: O(1)
        """
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix):
        """
        Check if any word starts with given prefix
        Time: O(m), Space: O(1)
        """
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix):
        """Helper method to find node for given prefix"""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node
    
    def delete(self, word):
        """
        Delete word from trie
        Time: O(m), Space: O(m) for recursion
        """
        if not self.search(word):
            return False
        
        def _delete_recursive(node, word, index):
            if index == len(word):
                # Reached end of word
                node.is_end_of_word = False
                node.word_count = 0
                # Return True if node has no children (can be deleted)
                return len(node.children) == 0
            
            char = word[index]
            child_node = node.children[char]
            child_node.prefix_count -= 1
            
            should_delete_child = _delete_recursive(child_node, word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                # Return True if current node can also be deleted
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        _delete_recursive(self.root, word, 0)
        self.size -= 1
        return True
    
    def get_all_words_with_prefix(self, prefix):
        """
        Get all words that start with given prefix
        Time: O(p + n) where p is prefix length, n is number of nodes in subtree
        """
        node = self._find_node(prefix)
        if not node:
            return []
        
        words = []
        self._collect_words(node, prefix, words)
        return words
    
    def _collect_words(self, node, current_word, words):
        """Helper to collect all words from a node using DFS"""
        if node.is_end_of_word:
            words.append(current_word)
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, words)
    
    def count_words_with_prefix(self, prefix):
        """Count number of words with given prefix"""
        node = self._find_node(prefix)
        return node.prefix_count if node else 0
    
    def longest_common_prefix(self):
        """Find longest common prefix of all words in trie"""
        if self.size == 0:
            return ""
        
        node = self.root
        prefix = ""
        
        while (len(node.children) == 1 and 
               not node.is_end_of_word and 
               node != self.root):
            char = next(iter(node.children))
            prefix += char
            node = node.children[char]
        
        return prefix
    
    def auto_complete(self, prefix, max_suggestions=10):
        """
        Get auto-complete suggestions for given prefix
        Returns up to max_suggestions words
        """
        suggestions = self.get_all_words_with_prefix(prefix)
        return suggestions[:max_suggestions]
    
    def __len__(self):
        return self.size
    
    def __contains__(self, word):
        return self.search(word)


class SuffixTrie:
    """
    Suffix Trie for pattern matching
    Stores all suffixes of a string
    """
    
    def __init__(self, text):
        self.text = text
        self.trie = Trie()
        self._build_suffix_trie()
    
    def _build_suffix_trie(self):
        """Build suffix trie from text"""
        for i in range(len(self.text)):
            suffix = self.text[i:] + "$"  # Add end marker
            self.trie.insert(suffix)
    
    def search_pattern(self, pattern):
        """
        Search for pattern in text using suffix trie
        Time: O(m) where m is pattern length
        """
        return self.trie.starts_with(pattern)
    
    def find_all_occurrences(self, pattern):
        """Find all starting positions of pattern in text"""
        if not self.search_pattern(pattern):
            return []
        
        positions = []
        suffixes = self.trie.get_all_words_with_prefix(pattern)
        
        for suffix in suffixes:
            # Calculate original position
            position = len(self.text) - len(suffix) + 1
            if position < len(self.text):
                positions.append(position)
        
        return sorted(positions)


class CompressedTrie:
    """
    Compressed Trie (Patricia Tree/Radix Tree)
    More space-efficient than regular trie
    """
    
    class CompressedTrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
            self.edge_label = ""  # Label on edge leading to this node
    
    def __init__(self):
        self.root = self.CompressedTrieNode()
        self.size = 0
    
    def insert(self, word):
        """Insert word into compressed trie"""
        if not word:
            return
        
        node = self.root
        i = 0
        
        while i < len(word):
            char = word[i]
            
            if char not in node.children:
                # Create new node with remaining word as edge label
                new_node = self.CompressedTrieNode()
                new_node.edge_label = word[i:]
                new_node.is_end_of_word = True
                node.children[char] = new_node
                self.size += 1
                return
            
            child = node.children[char]
            edge_label = child.edge_label
            
            # Find common prefix between remaining word and edge label
            j = 0
            while (j < len(edge_label) and 
                   i + j < len(word) and 
                   edge_label[j] == word[i + j]):
                j += 1
            
            if j == len(edge_label):
                # Edge label is prefix of remaining word
                node = child
                i += j
            elif j == len(word) - i:
                # Remaining word is prefix of edge label
                # Split the edge
                new_node = self.CompressedTrieNode()
                new_node.edge_label = word[i:]
                new_node.is_end_of_word = True
                
                child.edge_label = edge_label[j:]
                new_node.children[edge_label[j]] = child
                node.children[char] = new_node
                self.size += 1
                return
            else:
                # Split the edge
                intermediate_node = self.CompressedTrieNode()
                intermediate_node.edge_label = edge_label[:j]
                
                # Update existing child
                child.edge_label = edge_label[j:]
                intermediate_node.children[edge_label[j]] = child
                
                # Create new child for remaining word
                new_node = self.CompressedTrieNode()
                new_node.edge_label = word[i + j:]
                new_node.is_end_of_word = True
                intermediate_node.children[word[i + j]] = new_node
                
                node.children[char] = intermediate_node
                self.size += 1
                return
        
        # Word completely processed
        if not node.is_end_of_word:
            self.size += 1
        node.is_end_of_word = True
    
    def search(self, word):
        """Search for word in compressed trie"""
        node = self.root
        i = 0
        
        while i < len(word):
            char = word[i]
            
            if char not in node.children:
                return False
            
            child = node.children[char]
            edge_label = child.edge_label
            
            # Check if word matches edge label
            if i + len(edge_label) > len(word):
                return False
            
            if word[i:i + len(edge_label)] != edge_label:
                return False
            
            node = child
            i += len(edge_label)
        
        return node.is_end_of_word


class TrieWithWildcard:
    """
    Trie that supports wildcard search with '.' matching any character
    """
    
    def __init__(self):
        self.trie = Trie()
    
    def insert(self, word):
        """Insert word into trie"""
        self.trie.insert(word)
    
    def search_with_wildcard(self, pattern):
        """
        Search for pattern with wildcards
        '.' matches any single character
        """
        return self._search_recursive(self.trie.root, pattern, 0)
    
    def _search_recursive(self, node, pattern, index):
        """Helper for wildcard search"""
        if index == len(pattern):
            return node.is_end_of_word
        
        char = pattern[index]
        
        if char == '.':
            # Wildcard - try all children
            for child in node.children.values():
                if self._search_recursive(child, pattern, index + 1):
                    return True
            return False
        else:
            # Regular character
            if char not in node.children:
                return False
            return self._search_recursive(node.children[char], pattern, index + 1)


def build_trie_from_words(words):
    """Build trie from list of words"""
    trie = Trie()
    for word in words:
        trie.insert(word)
    return trie


def find_shortest_unique_prefix(words):
    """
    Find shortest unique prefix for each word
    Time: O(n * m) where n is number of words, m is average word length
    """
    trie = Trie()
    
    # Insert all words
    for word in words:
        trie.insert(word)
    
    prefixes = {}
    
    for word in words:
        node = trie.root
        prefix = ""
        
        for char in word:
            prefix += char
            node = node.children[char]
            
            # If only one word passes through this node, we found unique prefix
            if node.prefix_count == 1:
                prefixes[word] = prefix
                break
        
        # If no unique prefix found, use entire word
        if word not in prefixes:
            prefixes[word] = word
    
    return prefixes


def word_break_dp(s, word_dict):
    """
    Check if string can be segmented into words from dictionary using DP
    Can be optimized using trie for word lookup
    """
    trie = build_trie_from_words(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and trie.search(s[j:i]):
                dp[i] = True
                break
    
    return dp[n]


def count_distinct_substrings(s):
    """
    Count distinct substrings using suffix trie
    Time: O(n²), Space: O(n²)
    """
    suffix_trie = SuffixTrie(s)
    
    def count_nodes(node):
        count = 1  # Count current node
        for child in node.children.values():
            count += count_nodes(child)
        return count
    
    # Subtract 1 for root node
    return count_nodes(suffix_trie.trie.root) - 1


if __name__ == "__main__":
    # Demo usage
    trie = Trie()
    
    # Insert words
    words = ["apple", "app", "apricot", "application", "banana", "band", "bandana"]
    for word in words:
        trie.insert(word)
    
    print(f"Trie size: {len(trie)}")
    print(f"Search 'app': {trie.search('app')}")
    print(f"Search 'apply': {trie.search('apply')}")
    print(f"Starts with 'app': {trie.starts_with('app')}")
    
    # Auto-complete
    suggestions = trie.auto_complete("ap", 5)
    print(f"Auto-complete for 'ap': {suggestions}")
    
    # Count words with prefix
    count = trie.count_words_with_prefix("app")
    print(f"Words with prefix 'app': {count}")
    
    # Suffix trie example
    text = "banana"
    suffix_trie = SuffixTrie(text)
    
    pattern = "ana"
    print(f"\nPattern '{pattern}' in '{text}': {suffix_trie.search_pattern(pattern)}")
    positions = suffix_trie.find_all_occurrences(pattern)
    print(f"Positions of '{pattern}': {positions}")
    
    # Wildcard search
    wildcard_trie = TrieWithWildcard()
    for word in ["cat", "car", "card", "care", "careful"]:
        wildcard_trie.insert(word)
    
    print(f"\nWildcard search 'ca.': {wildcard_trie.search_with_wildcard('ca.')}")
    print(f"Wildcard search 'car.': {wildcard_trie.search_with_wildcard('car.')}")
    
    # Shortest unique prefixes
    test_words = ["cat", "car", "card", "care", "careful", "can"]
    prefixes = find_shortest_unique_prefix(test_words)
    print(f"\nShortest unique prefixes: {prefixes}")
    
    # Word break example
    s = "applepie"
    word_dict = ["apple", "pie", "app", "le"]
    can_break = word_break_dp(s, word_dict)
    print(f"Can break '{s}' with {word_dict}: {can_break}")
    
    # Count distinct substrings
    test_string = "abc"
    distinct_count = count_distinct_substrings(test_string)
    print(f"Distinct substrings in '{test_string}': {distinct_count}") 