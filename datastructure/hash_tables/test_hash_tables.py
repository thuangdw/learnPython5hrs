"""
Test cases for Hash Tables Implementation
"""

import unittest
from hash_tables import (
    HashTable, LinearProbingHashTable, QuadraticProbingHashTable,
    DoubleHashingTable, LRUCache, consistent_hashing_example
)


class TestHashTable(unittest.TestCase):
    
    def test_basic_operations(self):
        ht = HashTable()
        
        # Test put and get
        ht.put("name", "Alice")
        ht.put("age", 30)
        ht.put("city", "New York")
        
        self.assertEqual(ht.get("name"), "Alice")
        self.assertEqual(ht.get("age"), 30)
        self.assertEqual(ht.get("city"), "New York")
        self.assertEqual(len(ht), 3)
    
    def test_update_existing_key(self):
        ht = HashTable()
        
        ht.put("key", "value1")
        self.assertEqual(ht.get("key"), "value1")
        
        ht.put("key", "value2")  # Update
        self.assertEqual(ht.get("key"), "value2")
        self.assertEqual(len(ht), 1)  # Size shouldn't change
    
    def test_delete(self):
        ht = HashTable()
        
        ht.put("key1", "value1")
        ht.put("key2", "value2")
        
        deleted_value = ht.delete("key1")
        self.assertEqual(deleted_value, "value1")
        self.assertEqual(len(ht), 1)
        
        with self.assertRaises(KeyError):
            ht.get("key1")
    
    def test_delete_nonexistent_key(self):
        ht = HashTable()
        
        with self.assertRaises(KeyError):
            ht.delete("nonexistent")
    
    def test_contains(self):
        ht = HashTable()
        
        ht.put("key", "value")
        self.assertTrue(ht.contains("key"))
        self.assertFalse(ht.contains("nonexistent"))
    
    def test_keys_values_items(self):
        ht = HashTable()
        
        ht.put("a", 1)
        ht.put("b", 2)
        ht.put("c", 3)
        
        keys = ht.keys()
        values = ht.values()
        items = ht.items()
        
        self.assertEqual(set(keys), {"a", "b", "c"})
        self.assertEqual(set(values), {1, 2, 3})
        self.assertEqual(set(items), {("a", 1), ("b", 2), ("c", 3)})
    
    def test_load_factor(self):
        ht = HashTable(initial_capacity=4)
        
        # Initially empty
        self.assertEqual(ht.load_factor(), 0.0)
        
        # Add some items
        ht.put("a", 1)
        ht.put("b", 2)
        self.assertEqual(ht.load_factor(), 0.5)
    
    def test_resize(self):
        ht = HashTable(initial_capacity=4)
        
        # Add enough items to trigger resize
        for i in range(10):
            ht.put(f"key{i}", i)
        
        # All items should still be accessible
        for i in range(10):
            self.assertEqual(ht.get(f"key{i}"), i)
        
        # Capacity should have increased
        self.assertGreater(ht.capacity, 4)


class TestLinearProbingHashTable(unittest.TestCase):
    
    def test_basic_operations(self):
        ht = LinearProbingHashTable()
        
        ht.put("key1", "value1")
        ht.put("key2", "value2")
        
        self.assertEqual(ht.get("key1"), "value1")
        self.assertEqual(ht.get("key2"), "value2")
        self.assertEqual(len(ht), 2)
    
    def test_collision_handling(self):
        ht = LinearProbingHashTable(initial_capacity=4)
        
        # These keys might hash to same position
        ht.put("key1", "value1")
        ht.put("key2", "value2")
        ht.put("key3", "value3")
        
        self.assertEqual(ht.get("key1"), "value1")
        self.assertEqual(ht.get("key2"), "value2")
        self.assertEqual(ht.get("key3"), "value3")
    
    def test_delete_with_lazy_deletion(self):
        ht = LinearProbingHashTable()
        
        ht.put("key1", "value1")
        ht.put("key2", "value2")
        
        deleted_value = ht.delete("key1")
        self.assertEqual(deleted_value, "value1")
        self.assertEqual(len(ht), 1)
        
        with self.assertRaises(KeyError):
            ht.get("key1")
        
        # Should still be able to access key2
        self.assertEqual(ht.get("key2"), "value2")
    
    def test_contains(self):
        ht = LinearProbingHashTable()
        
        ht.put("key", "value")
        self.assertTrue(ht.contains("key"))
        self.assertFalse(ht.contains("nonexistent"))


class TestQuadraticProbingHashTable(unittest.TestCase):
    
    def test_basic_operations(self):
        ht = QuadraticProbingHashTable()
        
        ht.put("key1", "value1")
        ht.put("key2", "value2")
        
        self.assertEqual(ht.get("key1"), "value1")
        self.assertEqual(ht.get("key2"), "value2")
        self.assertEqual(len(ht), 2)
    
    def test_collision_handling(self):
        ht = QuadraticProbingHashTable(initial_capacity=8)
        
        # Add multiple items to test quadratic probing
        for i in range(5):
            ht.put(f"key{i}", f"value{i}")
        
        for i in range(5):
            self.assertEqual(ht.get(f"key{i}"), f"value{i}")


class TestDoubleHashingTable(unittest.TestCase):
    
    def test_basic_operations(self):
        ht = DoubleHashingTable()
        
        ht.put("key1", "value1")
        ht.put("key2", "value2")
        
        self.assertEqual(ht.get("key1"), "value1")
        self.assertEqual(ht.get("key2"), "value2")
        self.assertEqual(len(ht), 2)
    
    def test_collision_handling(self):
        ht = DoubleHashingTable(initial_capacity=8)
        
        # Add multiple items to test double hashing
        for i in range(5):
            ht.put(f"key{i}", f"value{i}")
        
        for i in range(5):
            self.assertEqual(ht.get(f"key{i}"), f"value{i}")


class TestLRUCache(unittest.TestCase):
    
    def test_basic_operations(self):
        cache = LRUCache(2)
        
        cache.put(1, 1)
        cache.put(2, 2)
        
        self.assertEqual(cache.get(1), 1)
        self.assertEqual(cache.get(2), 2)
    
    def test_lru_eviction(self):
        cache = LRUCache(2)
        
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(cache.get(1), 1)  # Make 1 recently used
        
        cache.put(3, 3)  # Should evict 2
        self.assertEqual(cache.get(1), 1)  # Should still be there
        self.assertEqual(cache.get(3), 3)  # Should be there
        self.assertEqual(cache.get(2), -1)  # Should be evicted
    
    def test_update_existing_key(self):
        cache = LRUCache(2)
        
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(1, 10)  # Update existing key
        
        self.assertEqual(cache.get(1), 10)
        self.assertEqual(cache.get(2), 2)
    
    def test_get_updates_recency(self):
        cache = LRUCache(2)
        
        cache.put(1, 1)
        cache.put(2, 2)
        cache.get(1)  # Make 1 recently used
        cache.put(3, 3)  # Should evict 2, not 1
        
        self.assertEqual(cache.get(1), 1)
        self.assertEqual(cache.get(3), 3)
        self.assertEqual(cache.get(2), -1)  # Should be evicted


class TestConsistentHashing(unittest.TestCase):
    
    def test_consistent_hashing(self):
        ConsistentHash = consistent_hashing_example()
        ch = ConsistentHash(['server1', 'server2', 'server3'])
        
        # Test that keys map to servers
        key1_server = ch.get_node('key1')
        key2_server = ch.get_node('key2')
        
        self.assertIn(key1_server, ['server1', 'server2', 'server3'])
        self.assertIn(key2_server, ['server1', 'server2', 'server3'])
    
    def test_add_remove_node(self):
        ConsistentHash = consistent_hashing_example()
        ch = ConsistentHash(['server1', 'server2'])
        
        # Add a new server
        ch.add_node('server3')
        
        # Test that keys still map to valid servers
        key_server = ch.get_node('test_key')
        self.assertIn(key_server, ['server1', 'server2', 'server3'])
        
        # Remove a server
        ch.remove_node('server2')
        
        # Test that keys map to remaining servers
        key_server = ch.get_node('test_key')
        self.assertIn(key_server, ['server1', 'server3'])
    
    def test_empty_ring(self):
        ConsistentHash = consistent_hashing_example()
        ch = ConsistentHash([])
        
        self.assertIsNone(ch.get_node('test_key'))


class TestHashTableStress(unittest.TestCase):
    
    def test_large_dataset(self):
        ht = HashTable()
        
        # Insert many items
        n = 1000
        for i in range(n):
            ht.put(f"key{i}", i)
        
        # Verify all items
        for i in range(n):
            self.assertEqual(ht.get(f"key{i}"), i)
        
        self.assertEqual(len(ht), n)
    
    def test_mixed_operations(self):
        ht = HashTable()
        
        # Mixed put, get, delete operations
        for i in range(100):
            ht.put(f"key{i}", i)
        
        # Delete every other key
        for i in range(0, 100, 2):
            ht.delete(f"key{i}")
        
        # Verify remaining keys
        for i in range(1, 100, 2):
            self.assertEqual(ht.get(f"key{i}"), i)
        
        # Verify deleted keys are gone
        for i in range(0, 100, 2):
            with self.assertRaises(KeyError):
                ht.get(f"key{i}")
        
        self.assertEqual(len(ht), 50)


if __name__ == '__main__':
    unittest.main() 