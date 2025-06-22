"""
Hash Tables Implementation
Comprehensive implementation of hash table data structures and algorithms
"""


class HashTable:
    """Hash table implementation with separate chaining for collision resolution"""
    
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, key):
        """Simple hash function using built-in hash()"""
        return hash(key) % self.capacity
    
    def _resize(self):
        """Resize hash table when load factor exceeds threshold"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all existing key-value pairs
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    def put(self, key, value):
        """Insert or update key-value pair - O(1) average"""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        # Check if key already exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Update existing
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self.size += 1
        
        # Check if resize is needed
        if self.size > self.capacity * self.load_factor_threshold:
            self._resize()
    
    def get(self, key):
        """Get value by key - O(1) average"""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        """Delete key-value pair - O(1) average"""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return v
        
        raise KeyError(f"Key '{key}' not found")
    
    def contains(self, key):
        """Check if key exists - O(1) average"""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def keys(self):
        """Get all keys"""
        result = []
        for bucket in self.buckets:
            for key, _ in bucket:
                result.append(key)
        return result
    
    def values(self):
        """Get all values"""
        result = []
        for bucket in self.buckets:
            for _, value in bucket:
                result.append(value)
        return result
    
    def items(self):
        """Get all key-value pairs"""
        result = []
        for bucket in self.buckets:
            for key, value in bucket:
                result.append((key, value))
        return result
    
    def load_factor(self):
        """Calculate current load factor"""
        return self.size / self.capacity
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        items = self.items()
        return "{" + ", ".join(f"'{k}': {v}" for k, v in items) + "}"


class LinearProbingHashTable:
    """Hash table with linear probing for collision resolution"""
    
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        self.load_factor_threshold = 0.5  # Lower threshold for open addressing
    
    def _hash(self, key):
        """Hash function"""
        return hash(key) % self.capacity
    
    def _resize(self):
        """Resize and rehash all elements"""
        old_keys = self.keys
        old_values = self.values
        old_deleted = self.deleted
        
        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        
        for i in range(len(old_keys)):
            if old_keys[i] is not None and not old_deleted[i]:
                self.put(old_keys[i], old_values[i])
    
    def _find_slot(self, key):
        """Find slot for key using linear probing"""
        index = self._hash(key)
        original_index = index
        
        while True:
            if self.keys[index] is None or self.deleted[index]:
                return index  # Empty slot found
            if self.keys[index] == key:
                return index  # Key found
            
            index = (index + 1) % self.capacity
            if index == original_index:
                raise Exception("Hash table is full")
    
    def put(self, key, value):
        """Insert or update key-value pair"""
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        index = self._find_slot(key)
        
        if self.keys[index] is None or self.deleted[index]:
            self.size += 1
        
        self.keys[index] = key
        self.values[index] = value
        self.deleted[index] = False
    
    def get(self, key):
        """Get value by key"""
        index = self._hash(key)
        original_index = index
        
        while self.keys[index] is not None:
            if self.keys[index] == key and not self.deleted[index]:
                return self.values[index]
            
            index = (index + 1) % self.capacity
            if index == original_index:
                break
        
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        """Delete key-value pair using lazy deletion"""
        index = self._hash(key)
        original_index = index
        
        while self.keys[index] is not None:
            if self.keys[index] == key and not self.deleted[index]:
                self.deleted[index] = True
                self.size -= 1
                return self.values[index]
            
            index = (index + 1) % self.capacity
            if index == original_index:
                break
        
        raise KeyError(f"Key '{key}' not found")
    
    def contains(self, key):
        """Check if key exists"""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def __len__(self):
        return self.size


class QuadraticProbingHashTable:
    """Hash table with quadratic probing for collision resolution"""
    
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        self.load_factor_threshold = 0.5
    
    def _hash(self, key):
        """Hash function"""
        return hash(key) % self.capacity
    
    def _find_slot(self, key):
        """Find slot using quadratic probing: h(k) + i²"""
        index = self._hash(key)
        
        for i in range(self.capacity):
            probe_index = (index + i * i) % self.capacity
            
            if self.keys[probe_index] is None or self.deleted[probe_index]:
                return probe_index
            if self.keys[probe_index] == key:
                return probe_index
        
        raise Exception("Hash table is full")
    
    def put(self, key, value):
        """Insert or update key-value pair"""
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        index = self._find_slot(key)
        
        if self.keys[index] is None or self.deleted[index]:
            self.size += 1
        
        self.keys[index] = key
        self.values[index] = value
        self.deleted[index] = False
    
    def _resize(self):
        """Resize and rehash"""
        old_keys = self.keys
        old_values = self.values
        old_deleted = self.deleted
        
        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        
        for i in range(len(old_keys)):
            if old_keys[i] is not None and not old_deleted[i]:
                self.put(old_keys[i], old_values[i])
    
    def get(self, key):
        """Get value by key"""
        index = self._hash(key)
        
        for i in range(self.capacity):
            probe_index = (index + i * i) % self.capacity
            
            if self.keys[probe_index] is None:
                break
            if self.keys[probe_index] == key and not self.deleted[probe_index]:
                return self.values[probe_index]
        
        raise KeyError(f"Key '{key}' not found")
    
    def __len__(self):
        return self.size


class DoubleHashingTable:
    """Hash table with double hashing for collision resolution"""
    
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        self.load_factor_threshold = 0.5
    
    def _hash1(self, key):
        """Primary hash function"""
        return hash(key) % self.capacity
    
    def _hash2(self, key):
        """Secondary hash function"""
        # Use a prime number less than capacity
        prime = 7
        return prime - (hash(key) % prime)
    
    def _find_slot(self, key):
        """Find slot using double hashing"""
        index = self._hash1(key)
        step = self._hash2(key)
        
        for i in range(self.capacity):
            probe_index = (index + i * step) % self.capacity
            
            if self.keys[probe_index] is None or self.deleted[probe_index]:
                return probe_index
            if self.keys[probe_index] == key:
                return probe_index
        
        raise Exception("Hash table is full")
    
    def put(self, key, value):
        """Insert or update key-value pair"""
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        index = self._find_slot(key)
        
        if self.keys[index] is None or self.deleted[index]:
            self.size += 1
        
        self.keys[index] = key
        self.values[index] = value
        self.deleted[index] = False
    
    def _resize(self):
        """Resize and rehash"""
        old_keys = self.keys
        old_values = self.values
        old_deleted = self.deleted
        
        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.deleted = [False] * self.capacity
        
        for i in range(len(old_keys)):
            if old_keys[i] is not None and not old_deleted[i]:
                self.put(old_keys[i], old_values[i])
    
    def get(self, key):
        """Get value by key"""
        index = self._hash1(key)
        step = self._hash2(key)
        
        for i in range(self.capacity):
            probe_index = (index + i * step) % self.capacity
            
            if self.keys[probe_index] is None:
                break
            if self.keys[probe_index] == key and not self.deleted[probe_index]:
                return self.values[probe_index]
        
        raise KeyError(f"Key '{key}' not found")
    
    def __len__(self):
        return self.size


class LRUCache:
    """Least Recently Used Cache using hash table + doubly linked list"""
    
    class Node:
        def __init__(self, key=None, value=None):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Create dummy head and tail nodes
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove an existing node"""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node):
        """Move node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove last node before tail"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key):
        """Get value and mark as recently used"""
        node = self.cache.get(key)
        
        if not node:
            return -1
        
        # Move to head (recently used)
        self._move_to_head(node)
        return node.value
    
    def put(self, key, value):
        """Put key-value pair"""
        node = self.cache.get(key)
        
        if not node:
            new_node = self.Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove LRU
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            self.cache[key] = new_node
            self._add_node(new_node)
        else:
            # Update existing
            node.value = value
            self._move_to_head(node)


def consistent_hashing_example():
    """Example of consistent hashing for distributed systems"""
    import hashlib
    
    class ConsistentHash:
        def __init__(self, nodes=None, replicas=3):
            self.replicas = replicas
            self.ring = {}
            self.sorted_keys = []
            
            if nodes:
                for node in nodes:
                    self.add_node(node)
        
        def _hash(self, key):
            """Hash function using MD5"""
            return int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        def add_node(self, node):
            """Add a node to the ring"""
            for i in range(self.replicas):
                key = self._hash(f"{node}:{i}")
                self.ring[key] = node
                self.sorted_keys.append(key)
            
            self.sorted_keys.sort()
        
        def remove_node(self, node):
            """Remove a node from the ring"""
            for i in range(self.replicas):
                key = self._hash(f"{node}:{i}")
                del self.ring[key]
                self.sorted_keys.remove(key)
        
        def get_node(self, key):
            """Get the node responsible for a key"""
            if not self.ring:
                return None
            
            hash_key = self._hash(key)
            
            # Find the first node >= hash_key
            for ring_key in self.sorted_keys:
                if ring_key >= hash_key:
                    return self.ring[ring_key]
            
            # Wrap around to the first node
            return self.ring[self.sorted_keys[0]]
    
    return ConsistentHash


if __name__ == "__main__":
    # Demo usage
    ht = HashTable()
    
    # Add some key-value pairs
    ht.put("name", "Alice")
    ht.put("age", 30)
    ht.put("city", "New York")
    
    print(f"Hash Table: {ht}")
    print(f"Name: {ht.get('name')}")
    print(f"Contains 'age': {ht.contains('age')}")
    print(f"Load factor: {ht.load_factor():.2f}")
    
    # LRU Cache example
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(f"Get 1: {cache.get(1)}")  # Returns 1
    cache.put(3, 3)  # Evicts key 2
    print(f"Get 2: {cache.get(2)}")  # Returns -1 (not found)
    
    # Consistent hashing example
    ConsistentHash = consistent_hashing_example()
    ch = ConsistentHash(['server1', 'server2', 'server3'])
    print(f"Key 'user123' maps to: {ch.get_node('user123')}")
    print(f"Key 'data456' maps to: {ch.get_node('data456')}") 