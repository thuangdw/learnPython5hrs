"""
Test cases for Sets Implementation
"""

import unittest
from sets import (
    HashSet, DisjointSet, BloomFilter, BitSet,
    power_set, cartesian_product, jaccard_similarity,
    set_cover_greedy
)


class TestHashSet(unittest.TestCase):
    
    def test_basic_operations(self):
        s = HashSet()
        
        # Test add
        s.add(1)
        s.add(2)
        s.add(3)
        
        self.assertEqual(len(s), 3)
        self.assertIn(1, s)
        self.assertIn(2, s)
        self.assertIn(3, s)
        self.assertNotIn(4, s)
    
    def test_add_duplicate(self):
        s = HashSet()
        
        s.add(1)
        s.add(1)  # Duplicate
        
        self.assertEqual(len(s), 1)
        self.assertIn(1, s)
    
    def test_remove(self):
        s = HashSet()
        
        s.add(1)
        s.add(2)
        
        s.remove(1)
        self.assertEqual(len(s), 1)
        self.assertNotIn(1, s)
        self.assertIn(2, s)
    
    def test_remove_nonexistent(self):
        s = HashSet()
        
        with self.assertRaises(KeyError):
            s.remove(1)
    
    def test_discard(self):
        s = HashSet()
        
        s.add(1)
        s.discard(1)  # Should remove
        self.assertEqual(len(s), 0)
        
        s.discard(2)  # Should not raise error
        self.assertEqual(len(s), 0)
    
    def test_contains(self):
        s = HashSet()
        
        s.add("hello")
        self.assertTrue(s.contains("hello"))
        self.assertFalse(s.contains("world"))
    
    def test_iteration(self):
        s = HashSet()
        
        items = [1, 2, 3, 4, 5]
        for item in items:
            s.add(item)
        
        result = list(s)
        self.assertEqual(set(result), set(items))
    
    def test_to_list(self):
        s = HashSet()
        
        items = [1, 2, 3]
        for item in items:
            s.add(item)
        
        result = s.to_list()
        self.assertEqual(set(result), set(items))
    
    def test_is_empty(self):
        s = HashSet()
        
        self.assertTrue(s.is_empty())
        
        s.add(1)
        self.assertFalse(s.is_empty())
    
    def test_clear(self):
        s = HashSet()
        
        s.add(1)
        s.add(2)
        s.clear()
        
        self.assertEqual(len(s), 0)
        self.assertTrue(s.is_empty())


class TestSetOperations(unittest.TestCase):
    
    def setUp(self):
        self.s1 = HashSet()
        self.s2 = HashSet()
        
        for i in [1, 2, 3, 4]:
            self.s1.add(i)
        
        for i in [3, 4, 5, 6]:
            self.s2.add(i)
    
    def test_union(self):
        result = self.s1.union(self.s2)
        expected = {1, 2, 3, 4, 5, 6}
        
        self.assertEqual(set(result.to_list()), expected)
    
    def test_intersection(self):
        result = self.s1.intersection(self.s2)
        expected = {3, 4}
        
        self.assertEqual(set(result.to_list()), expected)
    
    def test_difference(self):
        result = self.s1.difference(self.s2)
        expected = {1, 2}
        
        self.assertEqual(set(result.to_list()), expected)
    
    def test_symmetric_difference(self):
        result = self.s1.symmetric_difference(self.s2)
        expected = {1, 2, 5, 6}
        
        self.assertEqual(set(result.to_list()), expected)
    
    def test_is_subset(self):
        s3 = HashSet()
        s3.add(1)
        s3.add(2)
        
        self.assertTrue(s3.is_subset(self.s1))
        self.assertFalse(self.s1.is_subset(s3))
    
    def test_is_superset(self):
        s3 = HashSet()
        s3.add(1)
        s3.add(2)
        
        self.assertTrue(self.s1.is_superset(s3))
        self.assertFalse(s3.is_superset(self.s1))
    
    def test_is_disjoint(self):
        s3 = HashSet()
        s3.add(7)
        s3.add(8)
        
        self.assertTrue(self.s1.is_disjoint(s3))
        self.assertFalse(self.s1.is_disjoint(self.s2))


class TestDisjointSet(unittest.TestCase):
    
    def test_make_set_and_find(self):
        ds = DisjointSet()
        
        ds.make_set(1)
        ds.make_set(2)
        
        self.assertEqual(ds.find(1), 1)
        self.assertEqual(ds.find(2), 2)
    
    def test_union(self):
        ds = DisjointSet()
        
        for i in range(5):
            ds.make_set(i)
        
        ds.union(0, 1)
        ds.union(2, 3)
        
        # 0 and 1 should be connected
        self.assertTrue(ds.connected(0, 1))
        # 2 and 3 should be connected
        self.assertTrue(ds.connected(2, 3))
        # 0 and 2 should not be connected
        self.assertFalse(ds.connected(0, 2))
    
    def test_path_compression(self):
        ds = DisjointSet()
        
        # Create a chain: 0 -> 1 -> 2 -> 3
        for i in range(4):
            ds.make_set(i)
        
        ds.union(0, 1)
        ds.union(1, 2)
        ds.union(2, 3)
        
        # All should have same root
        root = ds.find(0)
        self.assertEqual(ds.find(1), root)
        self.assertEqual(ds.find(2), root)
        self.assertEqual(ds.find(3), root)
    
    def test_count_sets(self):
        ds = DisjointSet()
        
        for i in range(5):
            ds.make_set(i)
        
        self.assertEqual(ds.count_sets(), 5)
        
        ds.union(0, 1)
        self.assertEqual(ds.count_sets(), 4)
        
        ds.union(2, 3)
        self.assertEqual(ds.count_sets(), 3)
    
    def test_get_sets(self):
        ds = DisjointSet()
        
        for i in range(4):
            ds.make_set(i)
        
        ds.union(0, 1)
        ds.union(2, 3)
        
        sets = ds.get_sets()
        self.assertEqual(len(sets), 2)
        
        # Convert to sets for easier comparison
        set_values = [set(values) for values in sets.values()]
        self.assertIn({0, 1}, set_values)
        self.assertIn({2, 3}, set_values)


class TestBloomFilter(unittest.TestCase):
    
    def test_basic_operations(self):
        bf = BloomFilter(1000, 0.01)
        
        # Add items
        items = ["apple", "banana", "cherry"]
        for item in items:
            bf.add(item)
        
        # Test membership
        for item in items:
            self.assertTrue(item in bf)
        
        # Test non-membership (might have false positives)
        self.assertFalse("grape" in bf)  # Very likely to be false
    
    def test_false_positives_no_false_negatives(self):
        bf = BloomFilter(100, 0.1)
        
        # Add some items
        added_items = ["item1", "item2", "item3"]
        for item in added_items:
            bf.add(item)
        
        # All added items should be found (no false negatives)
        for item in added_items:
            self.assertTrue(item in bf)
        
        # Some non-added items might be found (false positives allowed)
        # But we can't test this reliably as it depends on hash functions
    
    def test_current_error_rate(self):
        bf = BloomFilter(100, 0.01)
        
        # Initially no items
        self.assertEqual(bf.current_error_rate(), 0.0)
        
        # Add some items
        for i in range(10):
            bf.add(f"item{i}")
        
        # Error rate should be > 0
        self.assertGreater(bf.current_error_rate(), 0.0)


class TestBitSet(unittest.TestCase):
    
    def test_basic_operations(self):
        bs = BitSet(100)
        
        # Add values
        bs.add(5)
        bs.add(10)
        bs.add(15)
        
        # Test membership
        self.assertTrue(5 in bs)
        self.assertTrue(10 in bs)
        self.assertTrue(15 in bs)
        self.assertFalse(20 in bs)
    
    def test_remove(self):
        bs = BitSet(100)
        
        bs.add(5)
        self.assertTrue(5 in bs)
        
        bs.remove(5)
        self.assertFalse(5 in bs)
    
    def test_union(self):
        bs1 = BitSet(100)
        bs2 = BitSet(100)
        
        bs1.add(1)
        bs1.add(2)
        bs2.add(2)
        bs2.add(3)
        
        result = bs1.union(bs2)
        
        self.assertTrue(1 in result)
        self.assertTrue(2 in result)
        self.assertTrue(3 in result)
        self.assertFalse(4 in result)
    
    def test_intersection(self):
        bs1 = BitSet(100)
        bs2 = BitSet(100)
        
        bs1.add(1)
        bs1.add(2)
        bs2.add(2)
        bs2.add(3)
        
        result = bs1.intersection(bs2)
        
        self.assertFalse(1 in result)
        self.assertTrue(2 in result)
        self.assertFalse(3 in result)
    
    def test_difference(self):
        bs1 = BitSet(100)
        bs2 = BitSet(100)
        
        bs1.add(1)
        bs1.add(2)
        bs2.add(2)
        bs2.add(3)
        
        result = bs1.difference(bs2)
        
        self.assertTrue(1 in result)
        self.assertFalse(2 in result)
        self.assertFalse(3 in result)
    
    def test_count(self):
        bs = BitSet(100)
        
        self.assertEqual(bs.count(), 0)
        
        bs.add(1)
        bs.add(5)
        bs.add(10)
        
        self.assertEqual(bs.count(), 3)
    
    def test_to_list(self):
        bs = BitSet(100)
        
        values = [1, 5, 10, 50]
        for val in values:
            bs.add(val)
        
        result = bs.to_list()
        self.assertEqual(result, values)
    
    def test_boundary_values(self):
        bs = BitSet(100)
        
        # Test boundary values
        bs.add(0)
        bs.add(100)
        
        self.assertTrue(0 in bs)
        self.assertTrue(100 in bs)
        
        # Test out of range
        with self.assertRaises(ValueError):
            bs.add(101)
        
        with self.assertRaises(ValueError):
            bs.add(-1)


class TestSetAlgorithms(unittest.TestCase):
    
    def test_power_set(self):
        s = [1, 2, 3]
        ps = power_set(s)
        
        expected = [
            [],
            [1],
            [2],
            [1, 2],
            [3],
            [1, 3],
            [2, 3],
            [1, 2, 3]
        ]
        
        # Convert to sets for comparison (order doesn't matter)
        ps_sets = [set(subset) for subset in ps]
        expected_sets = [set(subset) for subset in expected]
        
        self.assertEqual(len(ps_sets), len(expected_sets))
        for expected_set in expected_sets:
            self.assertIn(expected_set, ps_sets)
    
    def test_power_set_empty(self):
        ps = power_set([])
        self.assertEqual(ps, [[]])
    
    def test_cartesian_product(self):
        set1 = [1, 2]
        set2 = ['a', 'b']
        
        result = cartesian_product(set1, set2)
        expected = [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
        
        self.assertEqual(set(result), set(expected))
    
    def test_cartesian_product_empty(self):
        result = cartesian_product([], [1, 2])
        self.assertEqual(result, [])
        
        result = cartesian_product([1, 2], [])
        self.assertEqual(result, [])
    
    def test_jaccard_similarity(self):
        set1 = {1, 2, 3, 4}
        set2 = {3, 4, 5, 6}
        
        # Intersection: {3, 4} (size 2)
        # Union: {1, 2, 3, 4, 5, 6} (size 6)
        # Jaccard: 2/6 = 1/3
        
        similarity = jaccard_similarity(set1, set2)
        self.assertAlmostEqual(similarity, 1/3, places=10)
    
    def test_jaccard_similarity_identical(self):
        set1 = {1, 2, 3}
        set2 = {1, 2, 3}
        
        similarity = jaccard_similarity(set1, set2)
        self.assertEqual(similarity, 1.0)
    
    def test_jaccard_similarity_disjoint(self):
        set1 = {1, 2}
        set2 = {3, 4}
        
        similarity = jaccard_similarity(set1, set2)
        self.assertEqual(similarity, 0.0)
    
    def test_jaccard_similarity_empty(self):
        set1 = set()
        set2 = set()
        
        similarity = jaccard_similarity(set1, set2)
        self.assertEqual(similarity, 1.0)  # Both empty
    
    def test_set_cover_greedy(self):
        universe = [1, 2, 3, 4, 5]
        subsets = [
            [1, 2, 3],
            [2, 4],
            [3, 4],
            [4, 5]
        ]
        
        cover = set_cover_greedy(universe, subsets)
        
        # Verify cover is valid
        covered_elements = set()
        for subset_idx in cover:
            covered_elements.update(subsets[subset_idx])
        
        self.assertEqual(covered_elements, set(universe))
    
    def test_set_cover_greedy_minimal_example(self):
        universe = [1, 2]
        subsets = [
            [1, 2],
            [1],
            [2]
        ]
        
        cover = set_cover_greedy(universe, subsets)
        
        # Should choose subset 0 (covers both elements)
        self.assertEqual(len(cover), 1)
        self.assertIn(0, cover)


class TestSetStress(unittest.TestCase):
    
    def test_large_hashset(self):
        s = HashSet()
        
        n = 1000
        for i in range(n):
            s.add(i)
        
        self.assertEqual(len(s), n)
        
        for i in range(n):
            self.assertIn(i, s)
    
    def test_disjoint_set_stress(self):
        ds = DisjointSet()
        
        n = 1000
        for i in range(n):
            ds.make_set(i)
        
        # Union adjacent elements
        for i in range(n - 1):
            ds.union(i, i + 1)
        
        # All should be connected
        root = ds.find(0)
        for i in range(n):
            self.assertEqual(ds.find(i), root)
        
        self.assertEqual(ds.count_sets(), 1)


if __name__ == '__main__':
    unittest.main() 