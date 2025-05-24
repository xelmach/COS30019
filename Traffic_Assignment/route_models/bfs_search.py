from collections import deque, defaultdict

# Edge definitions with costs (costs are not used in BFS)
edges = {
    (2,1): 4, (3,1): 5, (1,3): 5, (2,3): 4,
    (3,2): 5, (4,1): 6, (1,4): 6, (4,3): 5,
    (3,5): 6, (5,3): 6, (4,5): 7, (5,4): 8,
    (6,3): 7, (3,6): 7
}

# Build adjacency list from edge definitions
def build_graph():
    graph = defaultdict(list)
    for (u, v), cost in edges.items():
        graph[u].append((v, cost))
    return graph

# Breadth-First Search to find shortest path in terms of number of edges
def bfs(graph, start, goals):
    queue = deque([(start, [start])])  # queue stores (current_node, path)
    visited = set()
    while queue:
        current, path = queue.popleft()
        if current in goals:
            return path  # Return path when any goal is reached
        if current not in visited:
            visited.add(current)
            for neighbor, _ in sorted(graph.get(current, [])):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return []  # No path found

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
