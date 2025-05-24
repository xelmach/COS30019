from collections import defaultdict

# Build the adjacency list (ignore cost for DFS)
def dfs(graph, start, goal, path=None, visited=None):
    if visited is None: visited = set()  # Track visited nodes
    if path is None: path = []           # Track current path
    visited.add(start)
    path = path + [start]
    if start == goal:                    # Goal reached
        return path
    for neighbor, _ in graph[start]:
        if neighbor not in visited:
            new_path = dfs(graph, neighbor, goal, path, visited)
            if new_path:
                return new_path
    return []                            # No path found

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
