"""
Test cases for Graphs Implementation
"""

import unittest
from graphs import (
    Graph, WeightedGraph, BipartiteGraph,
    dfs_recursive, dfs_iterative, bfs,
    has_cycle_undirected, has_cycle_directed,
    topological_sort, find_connected_components,
    kruskal_mst, floyd_warshall
)


class TestGraph(unittest.TestCase):
    
    def test_basic_operations(self):
        g = Graph()
        
        # Add vertices and edges
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')
        
        self.assertIn('A', g.get_vertices())
        self.assertIn('B', g.get_vertices())
        self.assertTrue(g.has_edge('A', 'B'))
        self.assertTrue(g.has_edge('B', 'A'))  # Undirected
        self.assertFalse(g.has_edge('A', 'D'))
    
    def test_directed_graph(self):
        g = Graph(directed=True)
        
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        
        self.assertTrue(g.has_edge('A', 'B'))
        self.assertFalse(g.has_edge('B', 'A'))  # Directed
    
    def test_remove_edge(self):
        g = Graph()
        
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        
        self.assertTrue(g.has_edge('A', 'B'))
        g.remove_edge('A', 'B')
        self.assertFalse(g.has_edge('A', 'B'))
        self.assertTrue(g.has_edge('A', 'C'))
    
    def test_get_neighbors(self):
        g = Graph()
        
        g.add_edge('A', 'B', 2)
        g.add_edge('A', 'C', 3)
        
        neighbors = g.get_neighbors('A')
        self.assertEqual(len(neighbors), 2)
        self.assertIn(('B', 2), neighbors)
        self.assertIn(('C', 3), neighbors)
    
    def test_get_edges(self):
        g = Graph()
        
        g.add_edge('A', 'B', 1)
        g.add_edge('B', 'C', 2)
        
        edges = g.get_edges()
        self.assertEqual(len(edges), 2)
        self.assertIn(('A', 'B', 1), edges)
        self.assertIn(('B', 'C', 2), edges)


class TestWeightedGraph(unittest.TestCase):
    
    def test_dijkstra(self):
        wg = WeightedGraph()
        
        wg.add_edge('A', 'B', 4)
        wg.add_edge('A', 'C', 2)
        wg.add_edge('B', 'C', 1)
        wg.add_edge('B', 'D', 5)
        wg.add_edge('C', 'D', 8)
        wg.add_edge('C', 'E', 10)
        wg.add_edge('D', 'E', 2)
        
        distances, previous = wg.dijkstra('A')
        
        self.assertEqual(distances['A'], 0)
        self.assertEqual(distances['B'], 3)  # A->C->B
        self.assertEqual(distances['C'], 2)  # A->C
        self.assertEqual(distances['D'], 8)  # A->C->B->D
        self.assertEqual(distances['E'], 10) # A->C->B->D->E
    
    def test_shortest_path(self):
        wg = WeightedGraph()
        
        wg.add_edge('A', 'B', 1)
        wg.add_edge('B', 'C', 2)
        wg.add_edge('A', 'C', 5)
        
        path, distance = wg.get_shortest_path('A', 'C')
        self.assertEqual(path, ['A', 'B', 'C'])
        self.assertEqual(distance, 3)
    
    def test_no_path(self):
        wg = WeightedGraph()
        
        wg.add_edge('A', 'B', 1)
        wg.add_vertex('C')  # Isolated vertex
        
        path, distance = wg.get_shortest_path('A', 'C')
        self.assertIsNone(path)
        self.assertEqual(distance, float('inf'))


class TestGraphTraversal(unittest.TestCase):
    
    def setUp(self):
        self.g = Graph()
        self.g.add_edge('A', 'B')
        self.g.add_edge('A', 'C')
        self.g.add_edge('B', 'D')
        self.g.add_edge('C', 'E')
    
    def test_dfs_recursive(self):
        result = dfs_recursive(self.g, 'A')
        self.assertIn('A', result)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], 'A')  # Start vertex should be first
    
    def test_dfs_iterative(self):
        result = dfs_iterative(self.g, 'A')
        self.assertIn('A', result)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], 'A')
    
    def test_bfs(self):
        result = bfs(self.g, 'A')
        self.assertEqual(result[0], 'A')
        self.assertEqual(len(result), 5)
        # BFS should visit neighbors before their children
        a_index = result.index('A')
        b_index = result.index('B')
        c_index = result.index('C')
        d_index = result.index('D')
        e_index = result.index('E')
        
        self.assertTrue(a_index < b_index < d_index)
        self.assertTrue(a_index < c_index < e_index)


class TestCycleDetection(unittest.TestCase):
    
    def test_cycle_undirected_with_cycle(self):
        g = Graph()
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')  # Creates cycle
        
        self.assertTrue(has_cycle_undirected(g))
    
    def test_cycle_undirected_no_cycle(self):
        g = Graph()
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')
        
        self.assertFalse(has_cycle_undirected(g))
    
    def test_cycle_directed_with_cycle(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')  # Creates cycle
        
        self.assertTrue(has_cycle_directed(g))
    
    def test_cycle_directed_no_cycle(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')
        
        self.assertFalse(has_cycle_directed(g))


class TestTopologicalSort(unittest.TestCase):
    
    def test_topological_sort(self):
        g = Graph(directed=True)
        g.add_edge('A', 'C')
        g.add_edge('B', 'C')
        g.add_edge('B', 'D')
        g.add_edge('C', 'E')
        g.add_edge('D', 'F')
        g.add_edge('E', 'F')
        
        result = topological_sort(g)
        
        # Check that dependencies are respected
        a_index = result.index('A')
        b_index = result.index('B')
        c_index = result.index('C')
        d_index = result.index('D')
        e_index = result.index('E')
        f_index = result.index('F')
        
        self.assertTrue(a_index < c_index < e_index < f_index)
        self.assertTrue(b_index < c_index)
        self.assertTrue(b_index < d_index < f_index)
    
    def test_topological_sort_undirected_error(self):
        g = Graph(directed=False)
        g.add_edge('A', 'B')
        
        with self.assertRaises(ValueError):
            topological_sort(g)


class TestConnectedComponents(unittest.TestCase):
    
    def test_single_component(self):
        g = Graph()
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'D')
        
        components = find_connected_components(g)
        self.assertEqual(len(components), 1)
        self.assertEqual(len(components[0]), 4)
    
    def test_multiple_components(self):
        g = Graph()
        g.add_edge('A', 'B')
        g.add_edge('C', 'D')
        g.add_vertex('E')  # Isolated vertex
        
        components = find_connected_components(g)
        self.assertEqual(len(components), 3)
        
        # Sort components by size for consistent testing
        components.sort(key=len)
        self.assertEqual(len(components[0]), 1)  # E
        self.assertEqual(len(components[1]), 2)  # A,B
        self.assertEqual(len(components[2]), 2)  # C,D
    
    def test_connected_components_directed_error(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        
        with self.assertRaises(ValueError):
            find_connected_components(g)


class TestKruskalMST(unittest.TestCase):
    
    def test_kruskal_mst(self):
        g = Graph()
        g.add_edge('A', 'B', 4)
        g.add_edge('A', 'C', 2)
        g.add_edge('B', 'C', 1)
        g.add_edge('B', 'D', 5)
        g.add_edge('C', 'D', 8)
        g.add_edge('C', 'E', 10)
        g.add_edge('D', 'E', 2)
        
        mst, total_weight = kruskal_mst(g)
        
        self.assertEqual(len(mst), 4)  # n-1 edges for n vertices
        self.assertEqual(total_weight, 9)  # Expected MST weight
        
        # Check that MST contains expected edges
        mst_edges = [(u, v) for u, v, w in mst]
        self.assertIn(('B', 'C'), mst_edges)  # Weight 1
        self.assertIn(('A', 'C'), mst_edges)  # Weight 2
        self.assertIn(('D', 'E'), mst_edges)  # Weight 2
    
    def test_kruskal_mst_directed_error(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B', 1)
        
        with self.assertRaises(ValueError):
            kruskal_mst(g)


class TestFloydWarshall(unittest.TestCase):
    
    def test_floyd_warshall(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B', 3)
        g.add_edge('A', 'C', 8)
        g.add_edge('A', 'E', -4)
        g.add_edge('B', 'D', 1)
        g.add_edge('B', 'E', 7)
        g.add_edge('C', 'B', 4)
        g.add_edge('D', 'A', 2)
        g.add_edge('D', 'C', -5)
        g.add_edge('E', 'D', 6)
        
        distances = floyd_warshall(g)
        
        # Check some known shortest distances
        self.assertEqual(distances['A']['A'], 0)
        self.assertEqual(distances['A']['B'], 1)  # A->E->D->C->B
        self.assertEqual(distances['A']['C'], -3)  # A->E->D->C
        self.assertEqual(distances['A']['D'], 2)   # A->E->D
        self.assertEqual(distances['A']['E'], -4)  # A->E


class TestBipartiteGraph(unittest.TestCase):
    
    def test_bipartite_graph(self):
        bg = BipartiteGraph()
        bg.add_edge('A', 'X')
        bg.add_edge('A', 'Y')
        bg.add_edge('B', 'X')
        bg.add_edge('C', 'Y')
        bg.add_edge('C', 'Z')
        
        self.assertTrue(bg.is_bipartite())
        
        set1, set2 = bg.get_bipartition()
        self.assertIsNotNone(set1)
        self.assertIsNotNone(set2)
        
        # One set should contain {A, B, C}, other should contain {X, Y, Z}
        if 'A' in set1:
            self.assertIn('B', set1)
            self.assertIn('C', set1)
            self.assertIn('X', set2)
            self.assertIn('Y', set2)
            self.assertIn('Z', set2)
        else:
            self.assertIn('A', set2)
            self.assertIn('B', set2)
            self.assertIn('C', set2)
            self.assertIn('X', set1)
            self.assertIn('Y', set1)
            self.assertIn('Z', set1)
    
    def test_non_bipartite_graph(self):
        bg = BipartiteGraph()
        bg.add_edge('A', 'B')
        bg.add_edge('B', 'C')
        bg.add_edge('C', 'A')  # Odd cycle - not bipartite
        
        self.assertFalse(bg.is_bipartite())
        
        set1, set2 = bg.get_bipartition()
        self.assertIsNone(set1)
        self.assertIsNone(set2)


if __name__ == '__main__':
    unittest.main() 