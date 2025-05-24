from collections import deque

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

def bfs(graph, start, goals):
    """
    Breadth-First Search to find the shortest path (by hops) from start to any of the goals.
    Args:
        graph (dict): {site_id: [(neighbor_id, weight), ...]}
        start (str): Starting site ID
        goals (list): List of goal site IDs (as strings)
    Returns:
        list: Path from start to goal (list of site IDs), or [] if not found
    """
    queue = deque([(start, [start])])
    visited = set([start])
    while queue:
        current, path = queue.popleft()
        if current in goals:
            return path
        for neighbor, _ in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []

# Run BFS for each destination goal
if __name__ == '__main__':
    graph = build_graph()
    origin = 2
    destinations = [5, 4]
    filename = "bfs_search.py"
    method = "bfs"
    for goal in destinations:
        path = bfs(graph, origin, [goal])
        print(f"{filename} {method}")
        if path:
            print(f"{goal} {len(path)}")
            print(path)
        else:
            print(f"{goal} 0")
            print([])
