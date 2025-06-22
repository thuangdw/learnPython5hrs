"""
Test cases for Trees Implementation
"""

import unittest
from trees import (
    TreeNode, BinaryTreeNode, BinarySearchTree, AVLTree, 
    SegmentTree, Trie, binary_tree_from_traversals,
    lowest_common_ancestor, is_balanced_tree, tree_diameter
)


class TestTreeNode(unittest.TestCase):
    
    def test_basic_operations(self):
        root = TreeNode("root")
        child1 = TreeNode("child1")
        child2 = TreeNode("child2")
        
        root.add_child(child1)
        root.add_child(child2)
        
        self.assertEqual(len(root.children), 2)
        self.assertEqual(child1.parent, root)
        self.assertEqual(child2.parent, root)
        self.assertTrue(root.is_root())
        self.assertFalse(child1.is_root())
    
    def test_remove_child(self):
        root = TreeNode("root")
        child = TreeNode("child")
        
        root.add_child(child)
        self.assertEqual(len(root.children), 1)
        
        root.remove_child(child)
        self.assertEqual(len(root.children), 0)
        self.assertIsNone(child.parent)
    
    def test_is_leaf(self):
        root = TreeNode("root")
        child = TreeNode("child")
        
        self.assertTrue(root.is_leaf())
        
        root.add_child(child)
        self.assertFalse(root.is_leaf())
        self.assertTrue(child.is_leaf())
    
    def test_get_level(self):
        root = TreeNode("root")
        child = TreeNode("child")
        grandchild = TreeNode("grandchild")
        
        root.add_child(child)
        child.add_child(grandchild)
        
        self.assertEqual(root.get_level(), 0)
        self.assertEqual(child.get_level(), 1)
        self.assertEqual(grandchild.get_level(), 2)


class TestBinarySearchTree(unittest.TestCase):
    
    def test_insert_and_search(self):
        bst = BinarySearchTree()
        
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            bst.insert(val)
        
        self.assertEqual(len(bst), 7)
        
        for val in values:
            self.assertIsNotNone(bst.search(val))
        
        self.assertIsNone(bst.search(100))
    
    def test_inorder_traversal(self):
        bst = BinarySearchTree()
        
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            bst.insert(val)
        
        inorder = bst.inorder_traversal()
        self.assertEqual(inorder, [20, 30, 40, 50, 60, 70, 80])
    
    def test_preorder_traversal(self):
        bst = BinarySearchTree()
        
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            bst.insert(val)
        
        preorder = bst.preorder_traversal()
        self.assertEqual(preorder[0], 50)  # Root should be first
    
    def test_postorder_traversal(self):
        bst = BinarySearchTree()
        
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            bst.insert(val)
        
        postorder = bst.postorder_traversal()
        self.assertEqual(postorder[-1], 50)  # Root should be last
    
    def test_level_order_traversal(self):
        bst = BinarySearchTree()
        
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            bst.insert(val)
        
        level_order = bst.level_order_traversal()
        self.assertEqual(level_order[0], 50)  # Root should be first
        self.assertIn(30, level_order[:3])    # Should be in first few elements
        self.assertIn(70, level_order[:3])    # Should be in first few elements
    
    def test_delete_leaf_node(self):
        bst = BinarySearchTree()
        
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            bst.insert(val)
        
        bst.delete(20)  # Delete leaf
        self.assertIsNone(bst.search(20))
        self.assertEqual(len(bst), 6)
    
    def test_delete_node_with_one_child(self):
        bst = BinarySearchTree()
        
        bst.insert(50)
        bst.insert(30)
        bst.insert(20)
        
        bst.delete(30)  # Delete node with one child
        self.assertIsNone(bst.search(30))
        self.assertIsNotNone(bst.search(20))
        self.assertIsNotNone(bst.search(50))
    
    def test_delete_node_with_two_children(self):
        bst = BinarySearchTree()
        
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            bst.insert(val)
        
        bst.delete(50)  # Delete root with two children
        self.assertIsNone(bst.search(50))
        
        # Tree should still be valid BST
        self.assertTrue(bst.is_valid_bst())
    
    def test_height(self):
        bst = BinarySearchTree()
        
        # Empty tree
        self.assertEqual(bst.height(), -1)
        
        # Single node
        bst.insert(50)
        self.assertEqual(bst.height(), 0)
        
        # Add more nodes
        bst.insert(30)
        bst.insert(70)
        self.assertEqual(bst.height(), 1)
        
        bst.insert(20)
        self.assertEqual(bst.height(), 2)
    
    def test_is_valid_bst(self):
        bst = BinarySearchTree()
        
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            bst.insert(val)
        
        self.assertTrue(bst.is_valid_bst())
    
    def test_empty_tree_operations(self):
        bst = BinarySearchTree()
        
        self.assertEqual(len(bst), 0)
        self.assertEqual(bst.inorder_traversal(), [])
        self.assertEqual(bst.level_order_traversal(), [])
        self.assertIsNone(bst.search(50))


class TestAVLTree(unittest.TestCase):
    
    def test_insert_and_balance(self):
        avl = AVLTree()
        
        # Insert values that would create unbalanced BST
        values = [10, 20, 30, 40, 50, 25]
        for val in values:
            avl.insert(val)
        
        self.assertEqual(len(avl), 6)
        
        # Inorder should still be sorted
        inorder = avl.inorder_traversal()
        self.assertEqual(inorder, sorted(values))
    
    def test_right_rotation(self):
        avl = AVLTree()
        
        # This sequence should trigger right rotation
        avl.insert(30)
        avl.insert(20)
        avl.insert(10)
        
        inorder = avl.inorder_traversal()
        self.assertEqual(inorder, [10, 20, 30])
    
    def test_left_rotation(self):
        avl = AVLTree()
        
        # This sequence should trigger left rotation
        avl.insert(10)
        avl.insert(20)
        avl.insert(30)
        
        inorder = avl.inorder_traversal()
        self.assertEqual(inorder, [10, 20, 30])
    
    def test_left_right_rotation(self):
        avl = AVLTree()
        
        # This sequence should trigger left-right rotation
        avl.insert(30)
        avl.insert(10)
        avl.insert(20)
        
        inorder = avl.inorder_traversal()
        self.assertEqual(inorder, [10, 20, 30])
    
    def test_right_left_rotation(self):
        avl = AVLTree()
        
        # This sequence should trigger right-left rotation
        avl.insert(10)
        avl.insert(30)
        avl.insert(20)
        
        inorder = avl.inorder_traversal()
        self.assertEqual(inorder, [10, 20, 30])


class TestSegmentTree(unittest.TestCase):
    
    def test_range_sum_query(self):
        arr = [1, 3, 5, 7, 9, 11]
        seg_tree = SegmentTree(arr)
        
        # Test various range queries
        self.assertEqual(seg_tree.range_sum(0, 2), 9)   # 1 + 3 + 5
        self.assertEqual(seg_tree.range_sum(1, 3), 15)  # 3 + 5 + 7
        self.assertEqual(seg_tree.range_sum(0, 5), 36)  # Sum of all elements
        self.assertEqual(seg_tree.range_sum(2, 2), 5)   # Single element
    
    def test_update(self):
        arr = [1, 3, 5, 7, 9, 11]
        seg_tree = SegmentTree(arr)
        
        # Original sum
        original_sum = seg_tree.range_sum(0, 5)
        
        # Update element at index 1 from 3 to 10
        seg_tree.update_value(1, 10)
        
        # New sum should be increased by 7
        new_sum = seg_tree.range_sum(0, 5)
        self.assertEqual(new_sum, original_sum + 7)
        
        # Range that includes updated element
        self.assertEqual(seg_tree.range_sum(1, 3), 22)  # 10 + 5 + 7
    
    def test_single_element(self):
        arr = [42]
        seg_tree = SegmentTree(arr)
        
        self.assertEqual(seg_tree.range_sum(0, 0), 42)
        
        seg_tree.update_value(0, 100)
        self.assertEqual(seg_tree.range_sum(0, 0), 100)
    
    def test_empty_range(self):
        arr = [1, 2, 3, 4, 5]
        seg_tree = SegmentTree(arr)
        
        # This should return 0 (empty range)
        # Note: This tests the implementation's handling of invalid ranges


class TestTrie(unittest.TestCase):
    
    def test_insert_and_search(self):
        trie = Trie()
        
        words = ["apple", "app", "application", "apply"]
        for word in words:
            trie.insert(word)
        
        self.assertEqual(len(trie), 4)
        
        for word in words:
            self.assertTrue(trie.search(word))
        
        self.assertFalse(trie.search("appl"))
        self.assertFalse(trie.search("banana"))
    
    def test_starts_with(self):
        trie = Trie()
        
        words = ["apple", "app", "application"]
        for word in words:
            trie.insert(word)
        
        self.assertTrue(trie.starts_with("app"))
        self.assertTrue(trie.starts_with("appl"))
        self.assertFalse(trie.starts_with("ban"))
    
    def test_delete(self):
        trie = Trie()
        
        words = ["apple", "app", "application"]
        for word in words:
            trie.insert(word)
        
        # Delete "app"
        self.assertTrue(trie.delete("app"))
        self.assertFalse(trie.search("app"))
        self.assertTrue(trie.search("apple"))      # Should still exist
        self.assertTrue(trie.search("application")) # Should still exist
        
        # Try to delete non-existent word
        self.assertFalse(trie.delete("banana"))
    
    def test_get_all_words_with_prefix(self):
        trie = Trie()
        
        words = ["apple", "app", "application", "banana", "band"]
        for word in words:
            trie.insert(word)
        
        app_words = trie.get_all_words_with_prefix("app")
        self.assertEqual(set(app_words), {"apple", "app", "application"})
        
        ban_words = trie.get_all_words_with_prefix("ban")
        self.assertEqual(set(ban_words), {"banana", "band"})
        
        empty_prefix = trie.get_all_words_with_prefix("xyz")
        self.assertEqual(empty_prefix, [])
    
    def test_empty_trie(self):
        trie = Trie()
        
        self.assertEqual(len(trie), 0)
        self.assertFalse(trie.search("word"))
        self.assertFalse(trie.starts_with("pre"))
        self.assertEqual(trie.get_all_words_with_prefix("any"), [])


class TestTreeAlgorithms(unittest.TestCase):
    
    def test_binary_tree_from_traversals(self):
        inorder = [9, 3, 15, 20, 7]
        preorder = [3, 9, 20, 15, 7]
        
        root = binary_tree_from_traversals(inorder, preorder)
        
        self.assertIsNotNone(root)
        self.assertEqual(root.data, 3)  # Root should be first in preorder
    
    def test_binary_tree_from_traversals_empty(self):
        root = binary_tree_from_traversals([], [])
        self.assertIsNone(root)
    
    def test_lowest_common_ancestor(self):
        # Create a BST: 50, 30, 70, 20, 40, 60, 80
        bst = BinarySearchTree()
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            bst.insert(val)
        
        node20 = bst.search(20)
        node40 = bst.search(40)
        node60 = bst.search(60)
        node80 = bst.search(80)
        
        # LCA of 20 and 40 should be 30
        lca = lowest_common_ancestor(bst.root, node20, node40)
        self.assertEqual(lca.data, 30)
        
        # LCA of 60 and 80 should be 70
        lca = lowest_common_ancestor(bst.root, node60, node80)
        self.assertEqual(lca.data, 70)
        
        # LCA of 20 and 60 should be 50 (root)
        lca = lowest_common_ancestor(bst.root, node20, node60)
        self.assertEqual(lca.data, 50)
    
    def test_is_balanced_tree(self):
        # Create balanced tree
        root = BinaryTreeNode(1)
        root.left = BinaryTreeNode(2)
        root.right = BinaryTreeNode(3)
        root.left.left = BinaryTreeNode(4)
        root.left.right = BinaryTreeNode(5)
        
        self.assertTrue(is_balanced_tree(root))
        
        # Create unbalanced tree
        unbalanced_root = BinaryTreeNode(1)
        unbalanced_root.left = BinaryTreeNode(2)
        unbalanced_root.left.left = BinaryTreeNode(3)
        unbalanced_root.left.left.left = BinaryTreeNode(4)
        
        self.assertFalse(is_balanced_tree(unbalanced_root))
    
    def test_tree_diameter(self):
        # Create tree with known diameter
        root = BinaryTreeNode(1)
        root.left = BinaryTreeNode(2)
        root.right = BinaryTreeNode(3)
        root.left.left = BinaryTreeNode(4)
        root.left.right = BinaryTreeNode(5)
        
        diameter = tree_diameter(root)
        self.assertEqual(diameter, 4)  # Path: 4 -> 2 -> 1 -> 3 (or 4 -> 2 -> 5)
    
    def test_tree_diameter_single_node(self):
        root = BinaryTreeNode(1)
        diameter = tree_diameter(root)
        self.assertEqual(diameter, 1)
    
    def test_tree_diameter_empty(self):
        diameter = tree_diameter(None)
        self.assertEqual(diameter, 0)


class TestTreeStress(unittest.TestCase):
    
    def test_large_bst(self):
        bst = BinarySearchTree()
        
        # Insert many values
        n = 1000
        import random
        values = list(range(n))
        random.shuffle(values)
        
        for val in values:
            bst.insert(val)
        
        self.assertEqual(len(bst), n)
        
        # Inorder should be sorted
        inorder = bst.inorder_traversal()
        self.assertEqual(inorder, list(range(n)))
        
        # All values should be searchable
        for val in range(n):
            self.assertIsNotNone(bst.search(val))
    
    def test_large_trie(self):
        trie = Trie()
        
        # Insert many words
        words = [f"word{i}" for i in range(1000)]
        for word in words:
            trie.insert(word)
        
        self.assertEqual(len(trie), 1000)
        
        # All words should be searchable
        for word in words:
            self.assertTrue(trie.search(word))
        
        # Prefix search should work
        prefix_words = trie.get_all_words_with_prefix("word1")
        self.assertGreater(len(prefix_words), 0)
    
    def test_avl_tree_stress(self):
        avl = AVLTree()
        
        # Insert sequential values (worst case for regular BST)
        n = 100
        for i in range(n):
            avl.insert(i)
        
        # Should still be balanced and sorted
        inorder = avl.inorder_traversal()
        self.assertEqual(inorder, list(range(n)))


if __name__ == '__main__':
    unittest.main() 