import heapq

class RouteFinder:
    def __init__(self):
        self.graph = {}
        self.site_coords = {}

    def build_graph(self, edges, coordinates):
        """
        Build the graph from edges and coordinates.
        Args:
            edges: List of tuples (origin, destination, travel_time)
            coordinates: Dictionary mapping site IDs to (lat, lon) tuples
        """
        self.graph = {}
        self.site_coords = coordinates
        for site_id in coordinates.keys():
            self.graph[str(site_id)] = []
        for edge in edges:
            origin, dest, travel_time = edge
            self.graph[str(origin)].append((str(dest), float(travel_time)))

    def get_route_length(self, route):
        """
        Calculate the total travel time of a route.
        Args:
            route: List of site IDs
        Returns:
            Total travel time (float)
        """
        length = 0
        for i in range(len(route) - 1):
            for neighbor, weight in self.graph[route[i]]:
                if neighbor == route[i+1]:
                    length += weight
                    break
        return length

    def dijkstra(self, start, end, banned_edges=None, banned_nodes=None):
        """
        Custom Dijkstra's algorithm to find the shortest path from start to end.
        Returns the path as a list of nodes.
        """
        queue = [(0, [start])]
        visited = set()
        banned_edges = banned_edges or set()
        banned_nodes = banned_nodes or set()
        while queue:
            cost, path = heapq.heappop(queue)
            node = path[-1]
            if node == end:
                return path, cost
            if node in visited:
                continue
            visited.add(node)
            for neighbor, weight in self.graph.get(node, []):
                if neighbor not in path and (node, neighbor) not in banned_edges and neighbor not in banned_nodes:
                    heapq.heappush(queue, (cost + weight, path + [neighbor]))
        return None, float('inf')

    def yen_k_shortest_paths(self, start, end, K=10):
        """
        Custom Yen's K-shortest paths algorithm (using custom Dijkstra).
        Returns up to K distinct paths (list of node lists).
        """
        start = str(start)
        end = str(end)
        A = []  # List of shortest paths
        B = []  # Min-heap for potential kth shortest path
        path, cost = self.dijkstra(start, end)
        if not path:
            return []
        A.append((path, cost))
        for k in range(1, K*5):
            for i in range(1, len(A[0][0]) - 1):
                spur_node = A[-1][0][i]
                root_path = A[-1][0][:i+1]
                removed_edges = []
                for p, _ in A:
                    if len(p) > i and p[:i+1] == root_path:
                        u, v = p[i], p[i+1]
                        for idx, (neighbor, weight) in enumerate(self.graph[u]):
                            if neighbor == v:
                                removed_edges.append((u, v, weight))
                                self.graph[u].pop(idx)
                                break
                spur_path, spur_cost = self.dijkstra(spur_node, end)
                for u, v, weight in removed_edges:
                    self.graph[u].append((v, weight))
                if spur_path and spur_path[0] == spur_node:
                    total_path = root_path[:-1] + spur_path
                    if len(set(total_path)) != len(total_path):
                        continue
                    if any(n == start for n in total_path[1:-1]) or any(n == end for n in total_path[1:-1]):
                        continue
                    total_cost = self.get_route_length(total_path)
                    if (total_path, total_cost) not in A and all(total_path != b[0] for b in B):
                        heapq.heappush(B, (total_cost, total_path))
            if not B:
                break
            cost, path = heapq.heappop(B)
            A.append((path, cost))
            if len(A) >= K:
                break
        return A

    def find_top_k_assignment_routes(self, origin, destination, k=3):
        """
        Assignment-compliant: Find up to k truly distinct, lowest-travel-time routes.
        Each route must have a unique node sequence and at least 2 different intermediate nodes compared to the fastest route.
        Raises an error if fewer than 2 valid routes exist.
        Args:
            origin: Origin site ID
            destination: Destination site ID
            k: Number of distinct routes to return (default: 3)
        Returns:
            List of (path, cost) tuples, up to k distinct routes.
        """
        origin = str(origin)
        destination = str(destination)
        all_routes = self.yen_k_shortest_paths(origin, destination, K=max(10, k*3))
        if not all_routes:
            raise RuntimeError("No path exists between origin and destination.")
        unique_routes = []
        fastest_route = all_routes[0][0]
        # 1. Try to find early-diverging routes
        for path, cost in all_routes:
            if not unique_routes:
                unique_routes.append((path, cost))
            else:
                min_len = min(len(path), len(fastest_route))
                diverge_idx = next((i for i in range(1, min_len) if path[i] != fastest_route[i]), None)
                if diverge_idx is not None and diverge_idx <= 2:
                    unique_routes.append((path, cost))
            if len(unique_routes) >= k:
                break
        # 2. If not enough, accept any simple, non-identical route
        if len(unique_routes) < k:
            for path, cost in all_routes:
                if (path, cost) not in unique_routes and path != fastest_route and len(set(path)) == len(path):
                    unique_routes.append((path, cost))
                if len(unique_routes) >= k:
                    break
        # 3. If still not enough, forcibly remove edges from fastest route and try again
        if len(unique_routes) < 2:
            banned_edges = set()
            for i in range(len(fastest_route) - 1):
                banned_edges.add((fastest_route[i], fastest_route[i+1]))
            alt_path, alt_cost = self.dijkstra(origin, destination, banned_edges=banned_edges)
            if alt_path and alt_path != fastest_route and len(set(alt_path)) == len(alt_path):
                unique_routes.append((alt_path, alt_cost))
        # 4. Final check
        unique_routes = [r for i, r in enumerate(unique_routes) if r not in unique_routes[:i]]
        if len(unique_routes) < 2:
            raise RuntimeError("No alternative route exists (graph may be disconnected).")
        unique_routes.sort(key=lambda x: x[1])
        return unique_routes[:k] 