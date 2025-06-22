"""
Trees Implementation
Comprehensive implementation of tree data structures and algorithms
"""

from collections import deque


class TreeNode:
    """Basic tree node"""
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child):
        """Remove a child node"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
    
    def is_leaf(self):
        """Check if node is a leaf"""
        return len(self.children) == 0
    
    def is_root(self):
        """Check if node is root"""
        return self.parent is None
    
    def get_level(self):
        """Get level of node (root is level 0)"""
        level = 0
        current = self
        while current.parent:
            level += 1
            current = current.parent
        return level


class BinaryTreeNode:
    """Binary tree node"""
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    
    def __str__(self):
        return str(self.data)


class BinarySearchTree:
    """Binary Search Tree implementation"""
    
    def __init__(self):
        self.root = None
        self.size = 0
    
    def insert(self, data):
        """Insert data into BST - O(log n) average, O(n) worst"""
        if not self.root:
            self.root = BinaryTreeNode(data)
            self.size += 1
            return
        
        self._insert_recursive(self.root, data)
        self.size += 1
    
    def _insert_recursive(self, node, data):
        """Helper method for recursive insertion"""
        if data <= node.data:
            if node.left is None:
                node.left = BinaryTreeNode(data)
            else:
                self._insert_recursive(node.left, data)
        else:
            if node.right is None:
                node.right = BinaryTreeNode(data)
            else:
                self._insert_recursive(node.right, data)
    
    def search(self, data):
        """Search for data in BST - O(log n) average, O(n) worst"""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node, data):
        """Helper method for recursive search"""
        if not node or node.data == data:
            return node
        
        if data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)
    
    def delete(self, data):
        """Delete data from BST"""
        self.root = self._delete_recursive(self.root, data)
        self.size -= 1
    
    def _delete_recursive(self, node, data):
        """Helper method for recursive deletion"""
        if not node:
            return node
        
        if data < node.data:
            node.left = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right = self._delete_recursive(node.right, data)
        else:
            # Node to be deleted found
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            
            # Node has two children
            # Find inorder successor (smallest in right subtree)
            successor = self._find_min(node.right)
            node.data = successor.data
            node.right = self._delete_recursive(node.right, successor.data)
        
        return node
    
    def _find_min(self, node):
        """Find minimum value node in subtree"""
        while node.left:
            node = node.left
        return node
    
    def _find_max(self, node):
        """Find maximum value node in subtree"""
        while node.right:
            node = node.right
        return node
    
    def inorder_traversal(self):
        """Inorder traversal (left, root, right) - gives sorted order"""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        """Helper for inorder traversal"""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)
    
    def preorder_traversal(self):
        """Preorder traversal (root, left, right)"""
        result = []
        self._preorder_recursive(self.root, result)
        return result
    
    def _preorder_recursive(self, node, result):
        """Helper for preorder traversal"""
        if node:
            result.append(node.data)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)
    
    def postorder_traversal(self):
        """Postorder traversal (left, right, root)"""
        result = []
        self._postorder_recursive(self.root, result)
        return result
    
    def _postorder_recursive(self, node, result):
        """Helper for postorder traversal"""
        if node:
            self._postorder_recursive(node.left, result)
            self._postorder_recursive(node.right, result)
            result.append(node.data)
    
    def level_order_traversal(self):
        """Level order traversal (breadth-first)"""
        if not self.root:
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            result.append(node.data)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return result
    
    def height(self):
        """Get height of tree"""
        return self._height_recursive(self.root)
    
    def _height_recursive(self, node):
        """Helper for height calculation"""
        if not node:
            return -1
        
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        
        return max(left_height, right_height) + 1
    
    def is_valid_bst(self):
        """Check if tree is a valid BST"""
        return self._is_valid_bst_recursive(self.root, float('-inf'), float('inf'))
    
    def _is_valid_bst_recursive(self, node, min_val, max_val):
        """Helper for BST validation"""
        if not node:
            return True
        
        if node.data <= min_val or node.data >= max_val:
            return False
        
        return (self._is_valid_bst_recursive(node.left, min_val, node.data) and
                self._is_valid_bst_recursive(node.right, node.data, max_val))
    
    def __len__(self):
        return self.size


class AVLTree:
    """AVL Tree (Self-balancing BST) implementation"""
    
    class AVLNode:
        def __init__(self, data):
            self.data = data
            self.left = None
            self.right = None
            self.height = 1
    
    def __init__(self):
        self.root = None
        self.size = 0
    
    def _get_height(self, node):
        """Get height of node"""
        if not node:
            return 0
        return node.height
    
    def _get_balance(self, node):
        """Get balance factor of node"""
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _update_height(self, node):
        """Update height of node"""
        if node:
            node.height = max(self._get_height(node.left), 
                            self._get_height(node.right)) + 1
    
    def _rotate_right(self, y):
        """Right rotation"""
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        self._update_height(y)
        self._update_height(x)
        
        return x
    
    def _rotate_left(self, x):
        """Left rotation"""
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        self._update_height(x)
        self._update_height(y)
        
        return y
    
    def insert(self, data):
        """Insert data maintaining AVL property"""
        self.root = self._insert_recursive(self.root, data)
        self.size += 1
    
    def _insert_recursive(self, node, data):
        """Helper for AVL insertion"""
        # Normal BST insertion
        if not node:
            return self.AVLNode(data)
        
        if data <= node.data:
            node.left = self._insert_recursive(node.left, data)
        else:
            node.right = self._insert_recursive(node.right, data)
        
        # Update height
        self._update_height(node)
        
        # Get balance factor
        balance = self._get_balance(node)
        
        # Left Left Case
        if balance > 1 and data <= node.left.data:
            return self._rotate_right(node)
        
        # Right Right Case
        if balance < -1 and data > node.right.data:
            return self._rotate_left(node)
        
        # Left Right Case
        if balance > 1 and data > node.left.data:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right Left Case
        if balance < -1 and data <= node.right.data:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def inorder_traversal(self):
        """Inorder traversal of AVL tree"""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)


class SegmentTree:
    """Segment Tree for range queries and updates"""
    
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        """Build segment tree"""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build(arr, 2 * node + 1, start, mid)
            self.build(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def update(self, node, start, end, idx, val):
        """Update value at index idx"""
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self.update(2 * node + 1, start, mid, idx, val)
            else:
                self.update(2 * node + 2, mid + 1, end, idx, val)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def query(self, node, start, end, l, r):
        """Query sum in range [l, r]"""
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_sum = self.query(2 * node + 1, start, mid, l, r)
        right_sum = self.query(2 * node + 2, mid + 1, end, l, r)
        return left_sum + right_sum
    
    def update_value(self, idx, val):
        """Public method to update value"""
        self.update(0, 0, self.n - 1, idx, val)
    
    def range_sum(self, l, r):
        """Public method to query range sum"""
        return self.query(0, 0, self.n - 1, l, r)


class Trie:
    """Trie (Prefix Tree) implementation"""
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
            self.word_count = 0  # Number of words ending at this node
    
    def __init__(self):
        self.root = self.TrieNode()
        self.size = 0
    
    def insert(self, word):
        """Insert word into trie - O(m) where m is word length"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.size += 1
        
        node.is_end_of_word = True
        node.word_count += 1
    
    def search(self, word):
        """Search for exact word in trie - O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix):
        """Check if any word starts with prefix - O(m)"""
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix):
        """Helper to find node for given prefix"""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node
    
    def delete(self, word):
        """Delete word from trie"""
        def _delete_recursive(node, word, index):
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                
                node.is_end_of_word = False
                node.word_count = 0
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete_child = _delete_recursive(node.children[char], word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        if self.search(word):
            _delete_recursive(self.root, word, 0)
            self.size -= 1
    
    def get_all_words_with_prefix(self, prefix):
        """Get all words that start with given prefix"""
        node = self._find_node(prefix)
        if not node:
            return []
        
        words = []
        self._collect_words(node, prefix, words)
        return words
    
    def _collect_words(self, node, current_word, words):
        """Helper to collect all words from a node"""
        if node.is_end_of_word:
            words.append(current_word)
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, words)
    
    def __len__(self):
        return self.size


def binary_tree_from_traversals(inorder, preorder):
    """Construct binary tree from inorder and preorder traversals"""
    if not inorder or not preorder:
        return None
    
    # First element in preorder is always root
    root_val = preorder[0]
    root = BinaryTreeNode(root_val)
    
    # Find root position in inorder
    root_index = inorder.index(root_val)
    
    # Recursively construct left and right subtrees
    root.left = binary_tree_from_traversals(
        inorder[:root_index], 
        preorder[1:root_index + 1]
    )
    root.right = binary_tree_from_traversals(
        inorder[root_index + 1:], 
        preorder[root_index + 1:]
    )
    
    return root


def lowest_common_ancestor(root, p, q):
    """Find lowest common ancestor of two nodes in BST"""
    if not root:
        return None
    
    # If both nodes are smaller than root, LCA is in left subtree
    if p.data < root.data and q.data < root.data:
        return lowest_common_ancestor(root.left, p, q)
    
    # If both nodes are greater than root, LCA is in right subtree
    if p.data > root.data and q.data > root.data:
        return lowest_common_ancestor(root.right, p, q)
    
    # If one is smaller and one is greater, root is LCA
    return root


def is_balanced_tree(root):
    """Check if binary tree is height-balanced"""
    def check_balance(node):
        if not node:
            return 0, True
        
        left_height, left_balanced = check_balance(node.left)
        right_height, right_balanced = check_balance(node.right)
        
        balanced = (left_balanced and right_balanced and 
                   abs(left_height - right_height) <= 1)
        
        height = max(left_height, right_height) + 1
        
        return height, balanced
    
    _, balanced = check_balance(root)
    return balanced


def tree_diameter(root):
    """Find diameter of binary tree (longest path between any two nodes)"""
    def diameter_helper(node):
        if not node:
            return 0, 0  # height, diameter
        
        left_height, left_diameter = diameter_helper(node.left)
        right_height, right_diameter = diameter_helper(node.right)
        
        current_height = max(left_height, right_height) + 1
        current_diameter = max(left_diameter, right_diameter, 
                             left_height + right_height + 1)
        
        return current_height, current_diameter
    
    _, diameter = diameter_helper(root)
    return diameter


if __name__ == "__main__":
    # Demo usage
    bst = BinarySearchTree()
    
    # Insert values
    values = [50, 30, 70, 20, 40, 60, 80]
    for val in values:
        bst.insert(val)
    
    print(f"BST size: {len(bst)}")
    print(f"Inorder traversal: {bst.inorder_traversal()}")
    print(f"Preorder traversal: {bst.preorder_traversal()}")
    print(f"Level order traversal: {bst.level_order_traversal()}")
    print(f"Tree height: {bst.height()}")
    print(f"Is valid BST: {bst.is_valid_bst()}")
    
    # Search example
    print(f"Search 40: {bst.search(40) is not None}")
    print(f"Search 90: {bst.search(90) is not None}")
    
    # AVL Tree example
    avl = AVLTree()
    for val in [10, 20, 30, 40, 50, 25]:
        avl.insert(val)
    
    print(f"\nAVL inorder: {avl.inorder_traversal()}")
    
    # Trie example
    trie = Trie()
    words = ["apple", "app", "apricot", "banana", "band"]
    
    for word in words:
        trie.insert(word)
    
    print(f"\nTrie size: {len(trie)}")
    print(f"Search 'app': {trie.search('app')}")
    print(f"Starts with 'ap': {trie.starts_with('ap')}")
    print(f"Words with prefix 'ap': {trie.get_all_words_with_prefix('ap')}")
    
    # Segment Tree example
    arr = [1, 3, 5, 7, 9, 11]
    seg_tree = SegmentTree(arr)
    
    print(f"\nSegment Tree:")
    print(f"Sum of range [1, 3]: {seg_tree.range_sum(1, 3)}")
    seg_tree.update_value(1, 10)
    print(f"After update, sum of range [1, 3]: {seg_tree.range_sum(1, 3)}") 