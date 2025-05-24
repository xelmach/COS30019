from collections import defaultdict

# Edge definitions with costs (costs are not used here)
edges = {
    (2,1): 4, (3,1): 5, (1,3): 5, (2,3): 4,
    (3,2): 5, (4,1): 6, (1,4): 6, (4,3): 5,
    (3,5): 6, (5,3): 6, (4,5): 7, (5,4): 8,
    (6,3): 7, (3,6): 7
}

# Convert edges into an adjacency list
def build_graph():
    graph = defaultdict(list)
    for (u, v), cost in edges.items():
        graph[u].append((v, cost))
    return graph

# Recursive Depth-Limited Search (DLS)
def dls(node, goal_set, graph, depth_limit, visited, path):
    if node in goal_set:
        return path  # Goal found
    if depth_limit == 0:
        return None  # Reached depth limit
    for neighbor, _ in sorted(graph.get(node, [])):
        if neighbor not in visited:
            visited.add(neighbor)
            result = dls(neighbor, goal_set, graph, depth_limit - 1, visited, path + [neighbor])
            if result is not None:
                return result
            visited.remove(neighbor)  # Backtrack
    return None  # No path found at this depth

# Iterative Deepening Search Wrapper (CUS1)
def cus1_search(graph, origin, goal):
    max_depth = 100  # Maximum allowed depth
    for depth in range(max_depth):
        visited = set([origin])
        result = dls(origin, {goal}, graph, depth, visited, [origin])
        if result:
            return result[-1], len(result), result
    return None, 0, []  # No path found within depth limit

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
