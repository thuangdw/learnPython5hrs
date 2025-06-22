"""
Sets Implementation
Comprehensive implementation of set data structures and algorithms
"""


class HashSet:
    """Set implementation using hash table with separate chaining"""
    
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, item):
        """Hash function for items"""
        return hash(item) % self.capacity
    
    def _resize(self):
        """Resize hash table when load factor exceeds threshold"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all existing items
        for bucket in old_buckets:
            for item in bucket:
                self.add(item)
    
    def add(self, item):
        """Add item to set - O(1) average"""
        index = self._hash(item)
        bucket = self.buckets[index]
        
        if item not in bucket:
            bucket.append(item)
            self.size += 1
            
            # Check if resize is needed
            if self.size > self.capacity * self.load_factor_threshold:
                self._resize()
    
    def remove(self, item):
        """Remove item from set - O(1) average"""
        index = self._hash(item)
        bucket = self.buckets[index]
        
        if item in bucket:
            bucket.remove(item)
            self.size -= 1
        else:
            raise KeyError(f"Item '{item}' not found in set")
    
    def discard(self, item):
        """Remove item if present, no error if not found"""
        try:
            self.remove(item)
        except KeyError:
            pass
    
    def contains(self, item):
        """Check if item is in set - O(1) average"""
        index = self._hash(item)
        bucket = self.buckets[index]
        return item in bucket
    
    def __contains__(self, item):
        """Support 'in' operator"""
        return self.contains(item)
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        """Make set iterable"""
        for bucket in self.buckets:
            for item in bucket:
                yield item
    
    def to_list(self):
        """Convert set to list"""
        return list(self)
    
    def is_empty(self):
        """Check if set is empty"""
        return self.size == 0
    
    def clear(self):
        """Remove all items from set"""
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
    
    def union(self, other):
        """Return union of two sets"""
        result = HashSet()
        
        # Add all items from self
        for item in self:
            result.add(item)
        
        # Add all items from other
        for item in other:
            result.add(item)
        
        return result
    
    def intersection(self, other):
        """Return intersection of two sets"""
        result = HashSet()
        
        # Add items that are in both sets
        for item in self:
            if item in other:
                result.add(item)
        
        return result
    
    def difference(self, other):
        """Return difference of two sets (self - other)"""
        result = HashSet()
        
        # Add items that are in self but not in other
        for item in self:
            if item not in other:
                result.add(item)
        
        return result
    
    def symmetric_difference(self, other):
        """Return symmetric difference of two sets"""
        result = HashSet()
        
        # Add items that are in self but not in other
        for item in self:
            if item not in other:
                result.add(item)
        
        # Add items that are in other but not in self
        for item in other:
            if item not in self:
                result.add(item)
        
        return result
    
    def is_subset(self, other):
        """Check if self is subset of other"""
        for item in self:
            if item not in other:
                return False
        return True
    
    def is_superset(self, other):
        """Check if self is superset of other"""
        return other.is_subset(self)
    
    def is_disjoint(self, other):
        """Check if sets have no common elements"""
        for item in self:
            if item in other:
                return False
        return True
    
    def __str__(self):
        items = list(self)
        return "{" + ", ".join(str(item) for item in items) + "}"


class DisjointSet:
    """Disjoint Set (Union-Find) data structure with path compression and union by rank"""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def make_set(self, x):
        """Create a new set containing only x"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
    
    def find(self, x):
        """
        Find the representative of the set containing x
        Uses path compression for optimization
        Time: O(α(n)) amortized, where α is inverse Ackermann function
        """
        if x not in self.parent:
            self.make_set(x)
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        
        return self.parent[x]
    
    def union(self, x, y):
        """
        Union the sets containing x and y
        Uses union by rank for optimization
        Time: O(α(n)) amortized
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return  # Already in same set
        
        # Union by rank: attach smaller tree under root of larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def connected(self, x, y):
        """Check if x and y are in the same set"""
        return self.find(x) == self.find(y)
    
    def count_sets(self):
        """Count number of disjoint sets"""
        roots = set()
        for x in self.parent:
            roots.add(self.find(x))
        return len(roots)
    
    def get_sets(self):
        """Get all disjoint sets as dictionary"""
        sets = {}
        for x in self.parent:
            root = self.find(x)
            if root not in sets:
                sets[root] = []
            sets[root].append(x)
        return sets


class BloomFilter:
    """
    Bloom filter - probabilistic data structure for set membership testing
    False positives possible, false negatives never
    """
    
    def __init__(self, capacity, error_rate=0.01):
        """
        Initialize bloom filter
        capacity: expected number of elements
        error_rate: desired false positive rate
        """
        import math
        
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Calculate optimal bit array size and number of hash functions
        self.bit_size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.hash_count = int(self.bit_size * math.log(2) / capacity)
        
        # Initialize bit array
        self.bit_array = [0] * self.bit_size
        self.item_count = 0
    
    def _hash_functions(self, item):
        """Generate multiple hash values for an item"""
        import hashlib
        
        # Convert item to string for hashing
        item_str = str(item).encode('utf-8')
        
        # Use different hash functions
        hashes = []
        for i in range(self.hash_count):
            # Combine hash with index to get different hash functions
            hash_input = item_str + str(i).encode('utf-8')
            hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
            hashes.append(hash_val % self.bit_size)
        
        return hashes
    
    def add(self, item):
        """Add item to bloom filter"""
        for hash_val in self._hash_functions(item):
            self.bit_array[hash_val] = 1
        self.item_count += 1
    
    def contains(self, item):
        """
        Check if item might be in the set
        Returns True if item might be present (could be false positive)
        Returns False if item is definitely not present
        """
        for hash_val in self._hash_functions(item):
            if self.bit_array[hash_val] == 0:
                return False
        return True
    
    def __contains__(self, item):
        return self.contains(item)
    
    def current_error_rate(self):
        """Estimate current false positive rate"""
        import math
        
        if self.item_count == 0:
            return 0.0
        
        # Calculate actual false positive rate based on current state
        ones_ratio = sum(self.bit_array) / len(self.bit_array)
        return ones_ratio ** self.hash_count


def power_set(s):
    """
    Generate power set (all subsets) of a set
    Time: O(2^n), Space: O(2^n)
    """
    s_list = list(s)
    n = len(s_list)
    power_set_list = []
    
    # Generate all 2^n subsets using bit manipulation
    for i in range(2 ** n):
        subset = []
        for j in range(n):
            if i & (1 << j):  # Check if j-th bit is set
                subset.append(s_list[j])
        power_set_list.append(subset)
    
    return power_set_list


def cartesian_product(set1, set2):
    """
    Generate cartesian product of two sets
    Time: O(|set1| * |set2|), Space: O(|set1| * |set2|)
    """
    result = []
    for item1 in set1:
        for item2 in set2:
            result.append((item1, item2))
    return result


def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity coefficient between two sets
    J(A,B) = |A ∩ B| / |A ∪ B|
    """
    if isinstance(set1, HashSet):
        set1 = set(set1.to_list())
    if isinstance(set2, HashSet):
        set2 = set(set2.to_list())
    
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    
    if union_size == 0:
        return 1.0 if intersection_size == 0 else 0.0
    
    return intersection_size / union_size


def set_cover_greedy(universe, subsets):
    """
    Greedy approximation algorithm for set cover problem
    Time: O(|subsets| * |universe|)
    """
    uncovered = set(universe)
    cover = []
    
    while uncovered:
        # Find subset that covers the most uncovered elements
        best_subset = None
        best_coverage = 0
        
        for i, subset in enumerate(subsets):
            coverage = len(set(subset).intersection(uncovered))
            if coverage > best_coverage:
                best_coverage = coverage
                best_subset = i
        
        if best_subset is None:
            break  # No more coverage possible
        
        # Add best subset to cover
        cover.append(best_subset)
        uncovered -= set(subsets[best_subset])
    
    return cover


class BitSet:
    """Bit set implementation for efficient set operations on integers"""
    
    def __init__(self, max_value=1000):
        self.max_value = max_value
        self.bits = [0] * ((max_value // 64) + 1)  # Use 64-bit integers
    
    def _get_word_and_bit(self, value):
        """Get word index and bit position for a value"""
        if value < 0 or value > self.max_value:
            raise ValueError(f"Value must be between 0 and {self.max_value}")
        
        word_index = value // 64
        bit_position = value % 64
        return word_index, bit_position
    
    def add(self, value):
        """Add value to bit set"""
        word_index, bit_position = self._get_word_and_bit(value)
        self.bits[word_index] |= (1 << bit_position)
    
    def remove(self, value):
        """Remove value from bit set"""
        word_index, bit_position = self._get_word_and_bit(value)
        self.bits[word_index] &= ~(1 << bit_position)
    
    def contains(self, value):
        """Check if value is in bit set"""
        word_index, bit_position = self._get_word_and_bit(value)
        return (self.bits[word_index] & (1 << bit_position)) != 0
    
    def __contains__(self, value):
        return self.contains(value)
    
    def union(self, other):
        """Bitwise OR with another bit set"""
        result = BitSet(max(self.max_value, other.max_value))
        for i in range(min(len(self.bits), len(other.bits))):
            result.bits[i] = self.bits[i] | other.bits[i]
        return result
    
    def intersection(self, other):
        """Bitwise AND with another bit set"""
        result = BitSet(max(self.max_value, other.max_value))
        for i in range(min(len(self.bits), len(other.bits))):
            result.bits[i] = self.bits[i] & other.bits[i]
        return result
    
    def difference(self, other):
        """Bitwise difference (self - other)"""
        result = BitSet(max(self.max_value, other.max_value))
        for i in range(min(len(self.bits), len(other.bits))):
            result.bits[i] = self.bits[i] & ~other.bits[i]
        return result
    
    def count(self):
        """Count number of set bits"""
        count = 0
        for word in self.bits:
            # Brian Kernighan's algorithm for counting set bits
            while word:
                count += 1
                word &= word - 1
        return count
    
    def to_list(self):
        """Convert to list of integers"""
        result = []
        for value in range(self.max_value + 1):
            if self.contains(value):
                result.append(value)
        return result


if __name__ == "__main__":
    # Demo usage
    s1 = HashSet()
    s2 = HashSet()
    
    # Add elements
    for i in [1, 2, 3, 4]:
        s1.add(i)
    
    for i in [3, 4, 5, 6]:
        s2.add(i)
    
    print(f"Set 1: {s1}")
    print(f"Set 2: {s2}")
    print(f"Union: {s1.union(s2)}")
    print(f"Intersection: {s1.intersection(s2)}")
    print(f"Difference: {s1.difference(s2)}")
    print(f"Symmetric difference: {s1.symmetric_difference(s2)}")
    
    # Disjoint set example
    ds = DisjointSet()
    
    # Create sets
    for i in range(6):
        ds.make_set(i)
    
    # Union operations
    ds.union(0, 1)
    ds.union(2, 3)
    ds.union(4, 5)
    ds.union(1, 2)
    
    print(f"\nDisjoint sets: {ds.get_sets()}")
    print(f"0 and 3 connected: {ds.connected(0, 3)}")
    print(f"0 and 4 connected: {ds.connected(0, 4)}")
    
    # Bloom filter example
    bf = BloomFilter(1000, 0.01)
    
    # Add items
    words = ["apple", "banana", "cherry", "date"]
    for word in words:
        bf.add(word)
    
    # Test membership
    print(f"\n'apple' in bloom filter: {'apple' in bf}")
    print(f"'grape' in bloom filter: {'grape' in bf}")
    
    # Power set example
    small_set = [1, 2, 3]
    ps = power_set(small_set)
    print(f"\nPower set of {small_set}: {ps}")
    
    # Jaccard similarity
    set_a = {1, 2, 3, 4}
    set_b = {3, 4, 5, 6}
    similarity = jaccard_similarity(set_a, set_b)
    print(f"Jaccard similarity: {similarity:.2f}") 