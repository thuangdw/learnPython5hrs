"""
Graphs Implementation
Comprehensive implementation of graph data structures and algorithms
"""

from collections import defaultdict, deque
import heapq


class Graph:
    """Graph implementation using adjacency list"""
    
    def __init__(self, directed=False):
        self.graph = defaultdict(list)
        self.directed = directed
        self.vertices = set()
    
    def add_vertex(self, vertex):
        """Add a vertex to the graph"""
        self.vertices.add(vertex)
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def add_edge(self, u, v, weight=1):
        """Add an edge between vertices u and v"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append((v, weight))
        
        if not self.directed:
            self.graph[v].append((u, weight))
    
    def remove_edge(self, u, v):
        """Remove edge between vertices u and v"""
        self.graph[u] = [(vertex, weight) for vertex, weight in self.graph[u] if vertex != v]
        
        if not self.directed:
            self.graph[v] = [(vertex, weight) for vertex, weight in self.graph[v] if vertex != u]
    
    def get_neighbors(self, vertex):
        """Get neighbors of a vertex"""
        return self.graph[vertex]
    
    def has_edge(self, u, v):
        """Check if edge exists between u and v"""
        return any(vertex == v for vertex, _ in self.graph[u])
    
    def get_vertices(self):
        """Get all vertices in the graph"""
        return list(self.vertices)
    
    def get_edges(self):
        """Get all edges in the graph"""
        edges = []
        for u in self.graph:
            for v, weight in self.graph[u]:
                if self.directed or u <= v:  # Avoid duplicates in undirected graph
                    edges.append((u, v, weight))
        return edges
    
    def __str__(self):
        result = []
        for vertex in sorted(self.vertices):
            neighbors = [f"{v}({w})" for v, w in self.graph[vertex]]
            result.append(f"{vertex}: {neighbors}")
        return "\n".join(result)


class WeightedGraph(Graph):
    """Weighted graph with additional utilities"""
    
    def __init__(self, directed=False):
        super().__init__(directed)
    
    def dijkstra(self, start):
        """
        Dijkstra's shortest path algorithm
        Time: O((V + E) log V), Space: O(V)
        """
        distances = {vertex: float('inf') for vertex in self.vertices}
        distances[start] = 0
        previous = {vertex: None for vertex in self.vertices}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor, weight in self.graph[current]:
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances, previous
    
    def get_shortest_path(self, start, end):
        """Get shortest path between start and end vertices"""
        distances, previous = self.dijkstra(start)
        
        if distances[end] == float('inf'):
            return None, float('inf')
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        
        return path[::-1], distances[end]


def dfs_recursive(graph, start, visited=None):
    """
    Depth-First Search (recursive)
    Time: O(V + E), Space: O(V)
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    result = [start]
    
    for neighbor, _ in graph.get_neighbors(start):
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    
    return result


def dfs_iterative(graph, start):
    """
    Depth-First Search (iterative)
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


def bfs(graph, start):
    """
    Breadth-First Search
    Time: O(V + E), Space: O(V)
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor, _ in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result


def has_cycle_undirected(graph):
    """
    Detect cycle in undirected graph using DFS
    Time: O(V + E), Space: O(V)
    """
    visited = set()
    
    def dfs_cycle(vertex, parent):
        visited.add(vertex)
        
        for neighbor, _ in graph.get_neighbors(vertex):
            if neighbor not in visited:
                if dfs_cycle(neighbor, vertex):
                    return True
            elif neighbor != parent:
                return True
        
        return False
    
    for vertex in graph.get_vertices():
        if vertex not in visited:
            if dfs_cycle(vertex, None):
                return True
    
    return False


def has_cycle_directed(graph):
    """
    Detect cycle in directed graph using DFS
    Time: O(V + E), Space: O(V)
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {vertex: WHITE for vertex in graph.get_vertices()}
    
    def dfs_cycle(vertex):
        colors[vertex] = GRAY
        
        for neighbor, _ in graph.get_neighbors(vertex):
            if colors[neighbor] == GRAY:
                return True
            if colors[neighbor] == WHITE and dfs_cycle(neighbor):
                return True
        
        colors[vertex] = BLACK
        return False
    
    for vertex in graph.get_vertices():
        if colors[vertex] == WHITE:
            if dfs_cycle(vertex):
                return True
    
    return False


def topological_sort(graph):
    """
    Topological sort using DFS
    Time: O(V + E), Space: O(V)
    """
    if not graph.directed:
        raise ValueError("Topological sort only works on directed graphs")
    
    visited = set()
    stack = []
    
    def dfs_topo(vertex):
        visited.add(vertex)
        
        for neighbor, _ in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs_topo(neighbor)
        
        stack.append(vertex)
    
    for vertex in graph.get_vertices():
        if vertex not in visited:
            dfs_topo(vertex)
    
    return stack[::-1]


def find_connected_components(graph):
    """
    Find all connected components in undirected graph
    Time: O(V + E), Space: O(V)
    """
    if graph.directed:
        raise ValueError("Connected components only work on undirected graphs")
    
    visited = set()
    components = []
    
    for vertex in graph.get_vertices():
        if vertex not in visited:
            component = dfs_recursive(graph, vertex, visited)
            components.append(component)
    
    return components


def kruskal_mst(graph):
    """
    Kruskal's Minimum Spanning Tree algorithm
    Time: O(E log E), Space: O(V)
    """
    if graph.directed:
        raise ValueError("MST only works on undirected graphs")
    
    # Union-Find data structure
    parent = {}
    rank = {}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    # Initialize Union-Find
    for vertex in graph.get_vertices():
        parent[vertex] = vertex
        rank[vertex] = 0
    
    # Sort edges by weight
    edges = sorted(graph.get_edges(), key=lambda x: x[2])
    mst = []
    total_weight = 0
    
    for u, v, weight in edges:
        if union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
            if len(mst) == len(graph.get_vertices()) - 1:
                break
    
    return mst, total_weight


def floyd_warshall(graph):
    """
    Floyd-Warshall all-pairs shortest path algorithm
    Time: O(V³), Space: O(V²)
    """
    vertices = list(graph.get_vertices())
    n = len(vertices)
    vertex_to_index = {v: i for i, v in enumerate(vertices)}
    
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Distance from vertex to itself is 0
    for i in range(n):
        dist[i][i] = 0
    
    # Fill in edge weights
    for u, v, weight in graph.get_edges():
        i, j = vertex_to_index[u], vertex_to_index[v]
        dist[i][j] = weight
        if not graph.directed:
            dist[j][i] = weight
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # Convert back to vertex names
    result = {}
    for i, u in enumerate(vertices):
        result[u] = {}
        for j, v in enumerate(vertices):
            result[u][v] = dist[i][j]
    
    return result


class BipartiteGraph:
    """Bipartite graph implementation"""
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
    
    def add_edge(self, u, v):
        """Add edge between vertices u and v"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def is_bipartite(self):
        """
        Check if graph is bipartite using BFS coloring
        Time: O(V + E), Space: O(V)
        """
        color = {}
        
        for start in self.vertices:
            if start not in color:
                queue = deque([start])
                color[start] = 0
                
                while queue:
                    vertex = queue.popleft()
                    
                    for neighbor in self.graph[vertex]:
                        if neighbor not in color:
                            color[neighbor] = 1 - color[vertex]
                            queue.append(neighbor)
                        elif color[neighbor] == color[vertex]:
                            return False
        
        return True
    
    def get_bipartition(self):
        """Get the two sets of vertices if bipartite"""
        if not self.is_bipartite():
            return None, None
        
        color = {}
        set1, set2 = [], []
        
        for start in self.vertices:
            if start not in color:
                queue = deque([start])
                color[start] = 0
                
                while queue:
                    vertex = queue.popleft()
                    
                    for neighbor in self.graph[vertex]:
                        if neighbor not in color:
                            color[neighbor] = 1 - color[vertex]
                            queue.append(neighbor)
        
        for vertex, c in color.items():
            if c == 0:
                set1.append(vertex)
            else:
                set2.append(vertex)
        
        return set1, set2


if __name__ == "__main__":
    # Demo usage
    g = Graph()
    
    # Add edges
    g.add_edge('A', 'B')
    g.add_edge('A', 'C')
    g.add_edge('B', 'D')
    g.add_edge('C', 'D')
    
    print("Graph:")
    print(g)
    
    print(f"\nDFS from A: {dfs_recursive(g, 'A')}")
    print(f"BFS from A: {bfs(g, 'A')}")
    print(f"Has cycle: {has_cycle_undirected(g)}")
    
    # Weighted graph example
    wg = WeightedGraph()
    wg.add_edge('A', 'B', 4)
    wg.add_edge('A', 'C', 2)
    wg.add_edge('B', 'C', 1)
    wg.add_edge('B', 'D', 5)
    wg.add_edge('C', 'D', 8)
    
    distances, _ = wg.dijkstra('A')
    print(f"\nShortest distances from A: {distances}")
    
    path, distance = wg.get_shortest_path('A', 'D')
    print(f"Shortest path A->D: {path}, distance: {distance}") 