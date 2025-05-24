import heapq

def heuristic(a, b, coordinates=None):
    if coordinates and a in coordinates and b in coordinates:
        lat1, lon1 = coordinates[a]['lat'], coordinates[a]['lon']
        lat2, lon2 = coordinates[b]['lat'], coordinates[b]['lon']
        return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
    return 0

def gbfs(graph, start, goal, coordinates=None):
    """
    Greedy Best-First Search to find a path from start to goal.
    Args:
        graph (dict): {site_id: [(neighbor_id, weight), ...]}
        start (str): Starting site ID
        goal (str): Goal site ID
        coordinates (dict): Optional, for heuristic
    Returns:
        tuple: (path, total_cost) where path is a list of site IDs
    """
    queue = [(heuristic(start, goal, coordinates), start, [start], 0)]
    visited = set()
    while queue:
        _, current, path, cost_so_far = heapq.heappop(queue)
        if current == goal:
            return path, cost_so_far
        if current in visited:
            continue
        visited.add(current)
        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited:
                h = heuristic(neighbor, goal, coordinates)
                heapq.heappush(queue, (h, neighbor, path + [neighbor], cost_so_far + weight))
    return [], float('inf')

def build_graph(edges=None, coordinates=None):
    """
    Build a graph representation from edges list.
    Args:
        edges: list of (origin, dest, weight) tuples
        coordinates: dict of site coordinates (optional)
    Returns:
        dict: Graph representation where each node maps to a list of (neighbor, weight) tuples
    """
    graph = {}
    if edges:
        for origin, dest, weight in edges:
            if origin not in graph:
                graph[origin] = []
            if dest not in graph:
                graph[dest] = []
            graph[origin].append((dest, weight))
            graph[dest].append((origin, weight))  # Add reverse edge for undirected graph
    return graph

# Run GBFS for each destination
if __name__ == '__main__':
    # Example usage
    edges = [
        (1, 2, 4), (1, 3, 5), (1, 4, 6),
        (2, 3, 4), (3, 4, 5), (3, 5, 6),
        (3, 6, 7), (4, 5, 7), (5, 4, 8)
    ]
    coordinates = {
        1: (4, 1), 2: (2, 2), 3: (4, 4),
        4: (6, 3), 5: (5, 6), 6: (7, 5)
    }
    graph = build_graph(edges, coordinates)
    origin = 2
    destinations = [5, 4]
    filename = "gbfs_search.py"
    method = "gbfs"
    for goal in destinations:
        path, cost = gbfs(graph, origin, goal, coordinates)
        print(f"{filename} {method}")
        if path:
            print(f"{goal} {len(path)}")
            print(path)
            print(f"Cost: {cost}")
        else:
            print(f"{goal} 0")
            print([])
