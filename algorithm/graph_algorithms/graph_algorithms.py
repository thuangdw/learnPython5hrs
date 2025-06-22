"""
Graph Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of essential graph algorithms
with detailed time and space complexity analysis.
"""

from typing import Dict, List, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq
import math


class Graph:
    """Graph representation using adjacency list"""
    
    def __init__(self, directed: bool = False):
        """
        Initialize graph
        
        Args:
            directed: Whether the graph is directed
        """
        self.adjacency_list = defaultdict(list)
        self.directed = directed
        self.vertices = set()
    
    def add_vertex(self, vertex):
        """Add a vertex to the graph"""
        self.vertices.add(vertex)
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
    
    def add_edge(self, u, v, weight: int = 1):
        """Add an edge between vertices u and v"""
        self.add_vertex(u)
        self.add_vertex(v)
        self.adjacency_list[u].append((v, weight))
        
        if not self.directed:
            self.adjacency_list[v].append((u, weight))
    
    def get_vertices(self) -> Set:
        """Get all vertices in the graph"""
        return self.vertices.copy()
    
    def get_neighbors(self, vertex) -> List[Tuple]:
        """Get neighbors of a vertex"""
        return self.adjacency_list[vertex]


class GraphAlgorithms:
    """Collection of graph algorithms"""
    
    @staticmethod
    def dfs_recursive(graph: Graph, start, visited: Set = None, result: List = None) -> List:
        """
        Depth-First Search (Recursive)
        Time: O(V + E), Space: O(V)
        """
        if visited is None:
            visited = set()
        if result is None:
            result = []
        
        visited.add(start)
        result.append(start)
        
        for neighbor, _ in graph.get_neighbors(start):
            if neighbor not in visited:
                GraphAlgorithms.dfs_recursive(graph, neighbor, visited, result)
        
        return result
    
    @staticmethod
    def dfs_iterative(graph: Graph, start) -> List:
        """
        Depth-First Search (Iterative)
        Time: O(V + E), Space: O(V)
        """
        visited = set()
        stack = [start]
        result = []
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                # Add neighbors in reverse order to maintain left-to-right traversal
                neighbors = [neighbor for neighbor, _ in graph.get_neighbors(vertex)]
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result
    
    @staticmethod
    def bfs(graph: Graph, start) -> List:
        """
        Breadth-First Search
        Time: O(V + E), Space: O(V)
        """
        visited = set()
        queue = deque([start])
        result = []
        
        visited.add(start)
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    @staticmethod
    def bfs_shortest_path(graph: Graph, start, end) -> Optional[List]:
        """
        Find shortest path using BFS (unweighted graphs)
        Time: O(V + E), Space: O(V)
        """
        if start == end:
            return [start]
        
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)
        
        while queue:
            vertex, path = queue.popleft()
            
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    
                    if neighbor == end:
                        return new_path
                    
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        
        return None  # No path found
    
    @staticmethod
    def dijkstra(graph: Graph, start) -> Tuple[Dict, Dict]:
        """
        Dijkstra's Algorithm for shortest paths
        Time: O((V + E) log V), Space: O(V)
        Returns (distances, previous) dictionaries
        """
        distances = {vertex: float('infinity') for vertex in graph.get_vertices()}
        previous = {vertex: None for vertex in graph.get_vertices()}
        distances[start] = 0
        
        # Priority queue: (distance, vertex)
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            if current_vertex in visited:
                continue
            
            visited.add(current_vertex)
            
            for neighbor, weight in graph.get_neighbors(current_vertex):
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_vertex
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances, previous
    
    @staticmethod
    def reconstruct_path(previous: Dict, start, end) -> Optional[List]:
        """Reconstruct path from Dijkstra's previous dictionary"""
        if previous[end] is None and start != end:
            return None
        
        path = []
        current = end
        
        while current is not None:
            path.append(current)
            current = previous[current]
        
        return path[::-1]
    
    @staticmethod
    def floyd_warshall(graph: Graph) -> Dict[Tuple, float]:
        """
        Floyd-Warshall Algorithm for all-pairs shortest paths
        Time: O(V³), Space: O(V²)
        """
        vertices = list(graph.get_vertices())
        n = len(vertices)
        
        # Initialize distance matrix
        dist = {}
        
        # Initialize with infinity
        for i in vertices:
            for j in vertices:
                if i == j:
                    dist[(i, j)] = 0
                else:
                    dist[(i, j)] = float('infinity')
        
        # Set edge weights
        for vertex in vertices:
            for neighbor, weight in graph.get_neighbors(vertex):
                dist[(vertex, neighbor)] = weight
        
        # Floyd-Warshall algorithm
        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
        
        return dist
    
    @staticmethod
    def topological_sort_dfs(graph: Graph) -> Optional[List]:
        """
        Topological Sort using DFS
        Time: O(V + E), Space: O(V)
        Returns None if cycle is detected
        """
        if not graph.directed:
            raise ValueError("Topological sort only works on directed graphs")
        
        visited = set()
        rec_stack = set()  # For cycle detection
        result = []
        
        def dfs_util(vertex):
            visited.add(vertex)
            rec_stack.add(vertex)
            
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor in rec_stack:  # Back edge found (cycle)
                    return False
                if neighbor not in visited and not dfs_util(neighbor):
                    return False
            
            rec_stack.remove(vertex)
            result.append(vertex)
            return True
        
        # Visit all vertices
        for vertex in graph.get_vertices():
            if vertex not in visited:
                if not dfs_util(vertex):
                    return None  # Cycle detected
        
        return result[::-1]  # Reverse to get topological order
    
    @staticmethod
    def topological_sort_kahn(graph: Graph) -> Optional[List]:
        """
        Topological Sort using Kahn's Algorithm
        Time: O(V + E), Space: O(V)
        """
        if not graph.directed:
            raise ValueError("Topological sort only works on directed graphs")
        
        # Calculate in-degrees
        in_degree = {vertex: 0 for vertex in graph.get_vertices()}
        
        for vertex in graph.get_vertices():
            for neighbor, _ in graph.get_neighbors(vertex):
                in_degree[neighbor] += 1
        
        # Find vertices with no incoming edges
        queue = deque([vertex for vertex in in_degree if in_degree[vertex] == 0])
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            # Remove this vertex and update in-degrees
            for neighbor, _ in graph.get_neighbors(vertex):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(graph.get_vertices()):
            return None  # Cycle detected
        
        return result
    
    @staticmethod
    def detect_cycle_directed(graph: Graph) -> bool:
        """
        Detect cycle in directed graph using DFS
        Time: O(V + E), Space: O(V)
        """
        if not graph.directed:
            raise ValueError("This method is for directed graphs")
        
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(vertex):
            visited.add(vertex)
            rec_stack.add(vertex)
            
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor in rec_stack:
                    return True
                if neighbor not in visited and has_cycle_util(neighbor):
                    return True
            
            rec_stack.remove(vertex)
            return False
        
        for vertex in graph.get_vertices():
            if vertex not in visited:
                if has_cycle_util(vertex):
                    return True
        
        return False
    
    @staticmethod
    def detect_cycle_undirected(graph: Graph) -> bool:
        """
        Detect cycle in undirected graph using DFS
        Time: O(V + E), Space: O(V)
        """
        if graph.directed:
            raise ValueError("This method is for undirected graphs")
        
        visited = set()
        
        def has_cycle_util(vertex, parent):
            visited.add(vertex)
            
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    if has_cycle_util(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    return True
            
            return False
        
        for vertex in graph.get_vertices():
            if vertex not in visited:
                if has_cycle_util(vertex, None):
                    return True
        
        return False
    
    @staticmethod
    def kruskal_mst(graph: Graph) -> List[Tuple]:
        """
        Kruskal's Algorithm for Minimum Spanning Tree
        Time: O(E log E), Space: O(V)
        """
        if graph.directed:
            raise ValueError("MST algorithms work on undirected graphs")
        
        # Get all edges
        edges = []
        for vertex in graph.get_vertices():
            for neighbor, weight in graph.get_neighbors(vertex):
                if vertex < neighbor:  # Avoid duplicate edges
                    edges.append((weight, vertex, neighbor))
        
        # Sort edges by weight
        edges.sort()
        
        # Union-Find data structure
        parent = {vertex: vertex for vertex in graph.get_vertices()}
        rank = {vertex: 0 for vertex in graph.get_vertices()}
        
        def find(vertex):
            if parent[vertex] != vertex:
                parent[vertex] = find(parent[vertex])  # Path compression
            return parent[vertex]
        
        def union(u, v):
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                # Union by rank
                if rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                elif rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1
                return True
            return False
        
        mst_edges = []
        mst_weight = 0
        
        for weight, u, v in edges:
            if union(u, v):
                mst_edges.append((u, v, weight))
                mst_weight += weight
                
                # MST has V-1 edges
                if len(mst_edges) == len(graph.get_vertices()) - 1:
                    break
        
        return mst_edges
    
    @staticmethod
    def prim_mst(graph: Graph, start=None) -> List[Tuple]:
        """
        Prim's Algorithm for Minimum Spanning Tree
        Time: O(E log V), Space: O(V)
        """
        if graph.directed:
            raise ValueError("MST algorithms work on undirected graphs")
        
        vertices = graph.get_vertices()
        if not vertices:
            return []
        
        if start is None:
            start = next(iter(vertices))
        
        mst_edges = []
        visited = {start}
        
        # Priority queue: (weight, vertex1, vertex2)
        edges = []
        for neighbor, weight in graph.get_neighbors(start):
            heapq.heappush(edges, (weight, start, neighbor))
        
        while edges and len(visited) < len(vertices):
            weight, u, v = heapq.heappop(edges)
            
            if v in visited:
                continue
            
            # Add edge to MST
            mst_edges.append((u, v, weight))
            visited.add(v)
            
            # Add new edges from v
            for neighbor, edge_weight in graph.get_neighbors(v):
                if neighbor not in visited:
                    heapq.heappush(edges, (edge_weight, v, neighbor))
        
        return mst_edges
    
    @staticmethod
    def connected_components(graph: Graph) -> List[List]:
        """
        Find connected components in undirected graph
        Time: O(V + E), Space: O(V)
        """
        if graph.directed:
            raise ValueError("This method is for undirected graphs")
        
        visited = set()
        components = []
        
        def dfs_component(vertex, component):
            visited.add(vertex)
            component.append(vertex)
            
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs_component(neighbor, component)
        
        for vertex in graph.get_vertices():
            if vertex not in visited:
                component = []
                dfs_component(vertex, component)
                components.append(component)
        
        return components
    
    @staticmethod
    def strongly_connected_components(graph: Graph) -> List[List]:
        """
        Find strongly connected components using Kosaraju's algorithm
        Time: O(V + E), Space: O(V)
        """
        if not graph.directed:
            raise ValueError("This method is for directed graphs")
        
        # Step 1: Fill vertices in stack according to their finishing times
        visited = set()
        stack = []
        
        def dfs1(vertex):
            visited.add(vertex)
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs1(neighbor)
            stack.append(vertex)
        
        for vertex in graph.get_vertices():
            if vertex not in visited:
                dfs1(vertex)
        
        # Step 2: Create transpose graph
        transpose = Graph(directed=True)
        for vertex in graph.get_vertices():
            transpose.add_vertex(vertex)
        
        for vertex in graph.get_vertices():
            for neighbor, weight in graph.get_neighbors(vertex):
                transpose.add_edge(neighbor, vertex, weight)
        
        # Step 3: Do DFS according to order defined by stack
        visited.clear()
        sccs = []
        
        def dfs2(vertex, scc):
            visited.add(vertex)
            scc.append(vertex)
            for neighbor, _ in transpose.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs2(neighbor, scc)
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                scc = []
                dfs2(vertex, scc)
                sccs.append(scc)
        
        return sccs
    
    @staticmethod
    def is_bipartite(graph: Graph) -> Tuple[bool, Dict]:
        """
        Check if graph is bipartite using BFS coloring
        Time: O(V + E), Space: O(V)
        Returns (is_bipartite, coloring)
        """
        color = {}
        
        def bfs_color(start):
            queue = deque([start])
            color[start] = 0
            
            while queue:
                vertex = queue.popleft()
                
                for neighbor, _ in graph.get_neighbors(vertex):
                    if neighbor not in color:
                        color[neighbor] = 1 - color[vertex]
                        queue.append(neighbor)
                    elif color[neighbor] == color[vertex]:
                        return False
            return True
        
        # Check all components
        for vertex in graph.get_vertices():
            if vertex not in color:
                if not bfs_color(vertex):
                    return False, {}
        
        return True, color


# Example usage and demonstrations
if __name__ == "__main__":
    # Create a sample graph
    g = Graph(directed=False)
    
    # Add edges
    edges = [
        ('A', 'B', 4), ('A', 'C', 2),
        ('B', 'C', 1), ('B', 'D', 5),
        ('C', 'D', 8), ('C', 'E', 10),
        ('D', 'E', 2)
    ]
    
    for u, v, w in edges:
        g.add_edge(u, v, w)
    
    print("Graph Algorithms Demonstration")
    print("=" * 40)
    
    # DFS
    print(f"DFS from A (recursive): {GraphAlgorithms.dfs_recursive(g, 'A')}")
    print(f"DFS from A (iterative): {GraphAlgorithms.dfs_iterative(g, 'A')}")
    
    # BFS
    print(f"BFS from A: {GraphAlgorithms.bfs(g, 'A')}")
    
    # Shortest path (unweighted)
    path = GraphAlgorithms.bfs_shortest_path(g, 'A', 'E')
    print(f"Shortest path A to E (unweighted): {path}")
    
    # Dijkstra's algorithm
    distances, previous = GraphAlgorithms.dijkstra(g, 'A')
    print(f"Dijkstra distances from A: {distances}")
    
    dijkstra_path = GraphAlgorithms.reconstruct_path(previous, 'A', 'E')
    print(f"Shortest path A to E (weighted): {dijkstra_path}")
    
    # MST algorithms
    mst_kruskal = GraphAlgorithms.kruskal_mst(g)
    print(f"MST (Kruskal): {mst_kruskal}")
    
    mst_prim = GraphAlgorithms.prim_mst(g, 'A')
    print(f"MST (Prim): {mst_prim}")
    
    # Connected components
    components = GraphAlgorithms.connected_components(g)
    print(f"Connected components: {components}")
    
    # Bipartite check
    is_bip, coloring = GraphAlgorithms.is_bipartite(g)
    print(f"Is bipartite: {is_bip}")
    if is_bip:
        print(f"Coloring: {coloring}")
    
    # Cycle detection
    has_cycle = GraphAlgorithms.detect_cycle_undirected(g)
    print(f"Has cycle (undirected): {has_cycle}")
    
    # Test directed graph
    print("\nDirected Graph Tests:")
    print("-" * 30)
    
    dg = Graph(directed=True)
    directed_edges = [
        ('1', '2'), ('2', '3'), ('3', '4'),
        ('4', '2'), ('1', '5'), ('5', '6')
    ]
    
    for u, v in directed_edges:
        dg.add_edge(u, v)
    
    # Topological sort
    topo_dfs = GraphAlgorithms.topological_sort_dfs(dg)
    print(f"Topological sort (DFS): {topo_dfs}")
    
    topo_kahn = GraphAlgorithms.topological_sort_kahn(dg)
    print(f"Topological sort (Kahn): {topo_kahn}")
    
    # Cycle detection in directed graph
    has_cycle_directed = GraphAlgorithms.detect_cycle_directed(dg)
    print(f"Has cycle (directed): {has_cycle_directed}")
    
    # Strongly connected components
    sccs = GraphAlgorithms.strongly_connected_components(dg)
    print(f"Strongly connected components: {sccs}")
    
    # Floyd-Warshall
    all_pairs = GraphAlgorithms.floyd_warshall(g)
    print(f"All pairs shortest paths (sample): {dict(list(all_pairs.items())[:5])}")
