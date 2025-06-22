"""
Greedy Algorithms Implementation
Senior Python Developer Guide

This module contains implementations of classic greedy algorithms
with detailed time and space complexity analysis.
"""

from typing import List, Tuple, Dict
import heapq
from collections import defaultdict, Counter


class GreedyAlgorithms:
    """Collection of greedy algorithms"""
    
    @staticmethod
    def activity_selection(start_times: List[int], end_times: List[int]) -> List[int]:
        """
        Activity Selection Problem
        Time: O(n log n), Space: O(n)
        Select maximum number of non-overlapping activities
        """
        n = len(start_times)
        activities = [(end_times[i], start_times[i], i) for i in range(n)]
        activities.sort()  # Sort by end time
        
        selected = []
        last_end_time = 0
        
        for end_time, start_time, index in activities:
            if start_time >= last_end_time:
                selected.append(index)
                last_end_time = end_time
        
        return selected
    
    @staticmethod
    def fractional_knapsack(weights: List[int], values: List[int], capacity: int) -> float:
        """
        Fractional Knapsack Problem
        Time: O(n log n), Space: O(n)
        Items can be taken fractionally
        """
        n = len(weights)
        items = [(values[i] / weights[i], weights[i], values[i], i) for i in range(n)]
        items.sort(reverse=True)  # Sort by value-to-weight ratio
        
        total_value = 0.0
        remaining_capacity = capacity
        
        for ratio, weight, value, index in items:
            if remaining_capacity >= weight:
                # Take the whole item
                total_value += value
                remaining_capacity -= weight
            else:
                # Take fraction of the item
                total_value += ratio * remaining_capacity
                break
        
        return total_value
    
    @staticmethod
    def job_scheduling(jobs: List[Tuple[int, int, int]]) -> Tuple[List[int], int]:
        """
        Job Scheduling with Deadlines
        Time: O(n²), Space: O(n)
        jobs: [(job_id, deadline, profit)]
        """
        jobs.sort(key=lambda x: x[2], reverse=True)  # Sort by profit
        
        max_deadline = max(job[1] for job in jobs)
        schedule = [-1] * max_deadline
        total_profit = 0
        selected_jobs = []
        
        for job_id, deadline, profit in jobs:
            # Find a free slot for this job (starting from deadline-1)
            for slot in range(min(deadline - 1, max_deadline - 1), -1, -1):
                if schedule[slot] == -1:
                    schedule[slot] = job_id
                    total_profit += profit
                    selected_jobs.append(job_id)
                    break
        
        return selected_jobs, total_profit
    
    @staticmethod
    def huffman_coding(text: str) -> Tuple[Dict[str, str], str]:
        """
        Huffman Coding for text compression
        Time: O(n log n), Space: O(n)
        Returns (codes_dict, encoded_text)
        """
        if not text:
            return {}, ""
        
        # Count frequency of each character
        frequency = Counter(text)
        
        # Special case: single character
        if len(frequency) == 1:
            char = list(frequency.keys())[0]
            return {char: '0'}, '0' * len(text)
        
        # Create priority queue with (frequency, unique_id, node)
        heap = []
        for i, (char, freq) in enumerate(frequency.items()):
            heapq.heappush(heap, (freq, i, char))
        
        # Build Huffman tree
        next_id = len(frequency)
        while len(heap) > 1:
            freq1, _, node1 = heapq.heappop(heap)
            freq2, _, node2 = heapq.heappop(heap)
            
            merged_freq = freq1 + freq2
            merged_node = (node1, node2)
            heapq.heappush(heap, (merged_freq, next_id, merged_node))
            next_id += 1
        
        # Generate codes
        _, _, root = heap[0]
        codes = {}
        
        def generate_codes(node, code=""):
            if isinstance(node, str):  # Leaf node
                codes[node] = code or "0"  # Handle single character case
            else:
                left, right = node
                generate_codes(left, code + "0")
                generate_codes(right, code + "1")
        
        generate_codes(root)
        
        # Encode text
        encoded_text = ''.join(codes[char] for char in text)
        
        return codes, encoded_text
    
    @staticmethod
    def coin_change_greedy(coins: List[int], amount: int) -> List[int]:
        """
        Greedy Coin Change (works for canonical coin systems)
        Time: O(n), Space: O(1)
        Returns list of coins used
        """
        coins.sort(reverse=True)  # Sort in descending order
        result = []
        
        for coin in coins:
            while amount >= coin:
                result.append(coin)
                amount -= coin
        
        return result if amount == 0 else []  # Return empty if not possible
    
    @staticmethod
    def minimum_spanning_tree_kruskal(edges: List[Tuple[int, int, int]], n: int) -> List[Tuple[int, int, int]]:
        """
        Kruskal's Algorithm for MST
        Time: O(E log E), Space: O(V)
        edges: [(u, v, weight)]
        """
        # Union-Find data structure
        parent = list(range(n))
        rank = [0] * n
        
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
        
        # Sort edges by weight
        edges.sort(key=lambda x: x[2])
        
        mst = []
        for u, v, weight in edges:
            if union(u, v):
                mst.append((u, v, weight))
                if len(mst) == n - 1:
                    break
        
        return mst
    
    @staticmethod
    def minimum_spanning_tree_prim(graph: Dict[int, List[Tuple[int, int]]], start: int = 0) -> List[Tuple[int, int, int]]:
        """
        Prim's Algorithm for MST
        Time: O(E log V), Space: O(V)
        graph: {node: [(neighbor, weight)]}
        """
        if not graph:
            return []
        
        visited = set()
        mst = []
        # Priority queue: (weight, from_node, to_node)
        pq = [(0, start, start)]
        
        while pq and len(visited) < len(graph):
            weight, from_node, to_node = heapq.heappop(pq)
            
            if to_node in visited:
                continue
            
            visited.add(to_node)
            if from_node != to_node:  # Skip the initial dummy edge
                mst.append((from_node, to_node, weight))
            
            # Add all edges from the new node
            if to_node in graph:
                for neighbor, edge_weight in graph[to_node]:
                    if neighbor not in visited:
                        heapq.heappush(pq, (edge_weight, to_node, neighbor))
        
        return mst
    
    @staticmethod
    def gas_station(gas: List[int], cost: List[int]) -> int:
        """
        Gas Station Problem
        Time: O(n), Space: O(1)
        Find starting station to complete circular tour
        """
        total_gas = sum(gas)
        total_cost = sum(cost)
        
        if total_gas < total_cost:
            return -1
        
        current_gas = 0
        start_station = 0
        
        for i in range(len(gas)):
            current_gas += gas[i] - cost[i]
            
            if current_gas < 0:
                start_station = i + 1
                current_gas = 0
        
        return start_station
    
    @staticmethod
    def jump_game(nums: List[int]) -> bool:
        """
        Jump Game - Can reach the end?
        Time: O(n), Space: O(1)
        """
        max_reach = 0
        
        for i in range(len(nums)):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i + nums[i])
            if max_reach >= len(nums) - 1:
                return True
        
        return True
    
    @staticmethod
    def jump_game_min_jumps(nums: List[int]) -> int:
        """
        Minimum jumps to reach the end
        Time: O(n), Space: O(1)
        """
        if len(nums) <= 1:
            return 0
        
        jumps = 0
        current_end = 0
        farthest = 0
        
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            
            if i == current_end:
                jumps += 1
                current_end = farthest
                
                if current_end >= len(nums) - 1:
                    break
        
        return jumps
    
    @staticmethod
    def meeting_rooms(intervals: List[List[int]]) -> int:
        """
        Minimum meeting rooms required
        Time: O(n log n), Space: O(n)
        """
        if not intervals:
            return 0
        
        # Separate start and end times
        starts = sorted([interval[0] for interval in intervals])
        ends = sorted([interval[1] for interval in intervals])
        
        rooms = 0
        max_rooms = 0
        start_ptr = end_ptr = 0
        
        while start_ptr < len(starts):
            if starts[start_ptr] < ends[end_ptr]:
                rooms += 1
                max_rooms = max(max_rooms, rooms)
                start_ptr += 1
            else:
                rooms -= 1
                end_ptr += 1
        
        return max_rooms


# Example usage
if __name__ == "__main__":
    greedy = GreedyAlgorithms()
    
    # Activity Selection
    start_times = [1, 3, 0, 5, 8, 5]
    end_times = [2, 4, 6, 7, 9, 9]
    selected = greedy.activity_selection(start_times, end_times)
    print(f"Selected activities: {selected}")
    
    # Fractional Knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    max_value = greedy.fractional_knapsack(weights, values, capacity)
    print(f"Maximum value: {max_value}")
    
    # Job Scheduling
    jobs = [(1, 4, 20), (2, 1, 10), (3, 1, 40), (4, 1, 30)]
    selected_jobs, profit = greedy.job_scheduling(jobs)
    print(f"Selected jobs: {selected_jobs}, Total profit: {profit}")
    
    # Huffman Coding
    text = "hello world"
    codes, encoded = greedy.huffman_coding(text)
    print(f"Huffman codes: {codes}")
    print(f"Encoded text: {encoded}")
    print(f"Compression ratio: {len(encoded)}/{len(text)*8} = {len(encoded)/(len(text)*8):.2f}")
    
    # Coin Change (Greedy)
    coins = [25, 10, 5, 1]
    amount = 67
    result = greedy.coin_change_greedy(coins, amount)
    print(f"Coins used for {amount}: {result}")
    
    # MST - Kruskal's
    edges = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
    mst_kruskal = greedy.minimum_spanning_tree_kruskal(edges, 4)
    print(f"MST (Kruskal): {mst_kruskal}")
    
    # MST - Prim's
    graph = {
        0: [(1, 10), (2, 6), (3, 5)],
        1: [(0, 10), (3, 15)],
        2: [(0, 6), (3, 4)],
        3: [(0, 5), (1, 15), (2, 4)]
    }
    mst_prim = greedy.minimum_spanning_tree_prim(graph)
    print(f"MST (Prim): {mst_prim}")
    
    # Gas Station
    gas = [1, 2, 3, 4, 5]
    cost = [3, 4, 5, 1, 2]
    start = greedy.gas_station(gas, cost)
    print(f"Starting gas station: {start}")
    
    # Jump Game
    nums1 = [2, 3, 1, 1, 4]
    nums2 = [3, 2, 1, 0, 4]
    print(f"Can reach end {nums1}: {greedy.jump_game(nums1)}")
    print(f"Can reach end {nums2}: {greedy.jump_game(nums2)}")
    print(f"Min jumps for {nums1}: {greedy.jump_game_min_jumps(nums1)}")
    
    # Meeting Rooms
    intervals = [[0, 30], [5, 10], [15, 20]]
    rooms_needed = greedy.meeting_rooms(intervals)
    print(f"Minimum meeting rooms needed: {rooms_needed}") 