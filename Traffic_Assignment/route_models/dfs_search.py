from collections import defaultdict

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

# Build the adjacency list (ignore cost for DFS)
def dfs(graph, start, goal):
    """
    Depth-First Search to find a path from start to goal.
    Args:
        graph (dict): {site_id: [(neighbor_id, weight), ...]}
        start (str): Starting site ID
        goal (str): Goal site ID
    Returns:
        list: Path from start to goal (list of site IDs), or [] if not found
    """
    stack = [(start, [start])]
    visited = set()
    while stack:
        current, path = stack.pop()
        if current == goal:
            return path
        if current not in visited:
            visited.add(current)
            for neighbor, _ in graph.get(current, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    return []

# Run DFS for each destination
if __name__ == '__main__':
    origin = 2
    destinations = [5, 4]
    filename = "dfs_search.py"
    method = "dfs"
    for goal in destinations:
        path = dfs(origin, goal)
        print(f"{filename} {method}")
        if path:
            print(f"{goal} {len(path)}")
            print(path)
        else:
            print(f"{goal} 0")
            print([])
