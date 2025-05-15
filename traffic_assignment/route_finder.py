"""
Route finder implementation for TBRGS.
"""

#105106819 Suman Sutparai
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
import heapq

class RouteFinder:
    """Implements various pathfinding algorithms for TBRGS."""
    
    def __init__(self, config):
        """Initialize the route finder.
        
        Args:
            config (dict): Configuration settings.
        """
        self.config = config
        self.graph = defaultdict(list)
        self.coordinates = {}
        
    def build_graph(self, edges, coordinates):
        """Build the graph from edges and coordinates.
        
        Args:
            edges (list): List of (from, to, cost) tuples.
            coordinates (dict): Dictionary of node coordinates.
        """
        self.graph.clear()
        self.coordinates = coordinates
        
        for from_node, to_node, cost in edges:
            self.graph[from_node].append((to_node, cost))
            self.graph[to_node].append((from_node, cost))
            
    def _heuristic(self, node1, node2):
        """Calculate heuristic (Euclidean distance) between nodes.
        
        Args:
            node1: First node.
            node2: Second node.
            
        Returns:
            float: Heuristic value.
        """
        if node1 not in self.coordinates or node2 not in self.coordinates:
            return float('inf')
            
        x1, y1 = self.coordinates[node1]
        x2, y2 = self.coordinates[node2]
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
    def bfs(self, start, goal):
        """Breadth-first search implementation.
        
        Args:
            start: Start node.
            goal: Goal node.
            
        Returns:
            tuple: (path, cost) if path exists, (None, None) otherwise.
        """
        from collections import deque
        queue = deque([(start, [start], 0)])
        visited = set()
        
        while queue:
            node, path, cost = queue.popleft()
            
            if node == goal:
                return path, cost
                
            if node in visited:
                continue
                
            visited.add(node)
            
            for neighbor, edge_cost in self.graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], cost + edge_cost))
                    
        return None, None
        
    def dfs(self, start, goal):
        """Depth-first search implementation.
        
        Args:
            start: Start node.
            goal: Goal node.
            
        Returns:
            tuple: (path, cost) if path exists, (None, None) otherwise.
        """
        stack = [(start, [start], 0)]
        visited = set()
        
        while stack:
            node, path, cost = stack.pop()
            
            if node == goal:
                return path, cost
                
            if node in visited:
                continue
                
            visited.add(node)
            
            for neighbor, edge_cost in self.graph[node]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], cost + edge_cost))
                    
        return None, None
        
    def a_star(self, start, goal):
        """A* search implementation.
        
        Args:
            start: Start node.
            goal: Goal node.
            
        Returns:
            tuple: (path, cost) if path exists, (None, None) otherwise.
        """
        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start, goal), 0, start, [start]))
        visited = set()
        
        while open_set:
            est_total, cost, node, path = heapq.heappop(open_set)
            
            if node == goal:
                return path, cost
                
            if node in visited:
                continue
                
            visited.add(node)
            
            for neighbor, edge_cost in self.graph[node]:
                if neighbor not in visited:
                    new_cost = cost + edge_cost
                    est = new_cost + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (est, new_cost, neighbor, path + [neighbor]))
                    
        return None, None
        
    def find_top_k_routes(self, start, goal, k=3):
        """Find top k routes between start and goal.
        
        Args:
            start: Start node.
            goal: Goal node.
            k (int): Number of routes to find.
            
        Returns:
            list: List of (path, cost) tuples.
        """
        q = PriorityQueue()
        q.put((0, [start]))
        found_routes = []
        visited_paths = set()
        
        while not q.empty() and len(found_routes) < k:
            cost, path = q.get()
            node = path[-1]
            
            if node == goal:
                found_routes.append((path, cost))
                continue
                
            for neighbor, edge_cost in self.graph[node]:
                if neighbor not in path:
                    new_path = path + [neighbor]
                    path_tuple = tuple(new_path)
                    if path_tuple not in visited_paths:
                        visited_paths.add(path_tuple)
                        q.put((cost + edge_cost, new_path))
        
        found_routes.sort(key=lambda x: x[1])
        return found_routes 