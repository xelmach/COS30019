from collections import defaultdict

# Edge definitions with costs (costs are not used here)
edges = {
    (2,1): 4, (3,1): 5, (1,3): 5, (2,3): 4,
    (3,2): 5, (4,1): 6, (1,4): 6, (4,3): 5,
    (3,5): 6, (5,3): 6, (4,5): 7, (5,4): 8,
    (6,3): 7, (3,6): 7
}

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

def dls(graph, current, goal, limit, path, visited):
    if current == goal:
        return path
    if limit <= 0:
        return None
    visited.add(current)
    for neighbor, _ in graph.get(current, []):
        if neighbor not in visited:
            result = dls(graph, neighbor, goal, limit - 1, path + [neighbor], visited)
            if result:
                return result
    visited.remove(current)
    return None

def cus1_search(graph, start, goal, max_depth=10):
    """
    Custom Search 1: Iterative Deepening Depth-Limited Search (IDDFS)
    Args:
        graph (dict): {site_id: [(neighbor_id, weight), ...]}
        start (str): Starting site ID
        goal (str): Goal site ID
        max_depth (int): Maximum depth to search
    Returns:
        list: Path from start to goal (list of site IDs), or [] if not found
    """
    for depth in range(1, max_depth + 1):
        visited = set()
        result = dls(graph, start, goal, depth, [start], visited)
        if result:
            return result
    return []

def dfs(graph, start, goal, path=None, visited=None):
    """
    Depth-First Search to find a path from start to goal (CUS1).
    Args:
        graph: dict {site_id: [(neighbor_id, cost), ...]}
        start: start node (int or str)
        goal: goal node (int or str)
        path: current path (for recursion)
        visited: set of visited nodes
    Returns:
        path: list of node ids
    """
    if visited is None:
        visited = set()
    if path is None:
        path = []
    visited.add(start)
    path = path + [start]
    if start == goal:
        return path
    for neighbor, _ in graph.get(start, []):
        if neighbor not in visited:
            new_path = dfs(graph, neighbor, goal, path, visited)
            if new_path:
                return new_path
    return []

# Run CUS1 for each destination
if __name__ == "__main__":
    graph = build_graph()
    origin = 2
    destinations = [5, 4]
    filename = "cus1_search.py"
    method = "cus1"
    for goal in destinations:
        goal_node, length, path = cus1_search(graph, origin, goal)
        print(f"{filename} {method}")
        if path:
            print(f"{goal_node} {length}")
            print(path)
        else:
            print(f"{goal} 0")
            print([])
