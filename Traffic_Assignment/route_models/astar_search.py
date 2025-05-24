import heapq, math
from collections import defaultdict

# Coordinates of each node, used to calculate heuristic (Euclidean distance)
positions = {1:(4,1), 2:(2,2), 3:(4,4), 4:(6,3), 5:(5,6), 6:(7,5)}

# Edges with associated travel costs
edges = {
    (2,1):4, (3,1):5, (1,3):5, (2,3):4, (3,2):5, (4,1):6, (1,4):6, (4,3):5,
    (3,5):6, (5,3):6, (4,5):7, (5,4):8, (6,3):7, (3,6):7
}

# Build graph as adjacency list
def build_graph():
    g = defaultdict(list)
    for (u, v), cost in edges.items():
        g[u].append((v, cost))
    return g

# Heuristic function: straight-line (Euclidean) distance between nodes
def heuristic(a, b):
    x1, y1 = positions[a]
    x2, y2 = positions[b]
    return math.hypot(x2 - x1, y2 - y1)

# A* Search Algorithm
def astar(graph, start, goal):
    open_set = [(0, 0, start, [])]  # (f = g + h, g, node, path)
    visited = {}
    while open_set:
        f, g, node, path = heapq.heappop(open_set)
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        new_path = path + [node]
        if node == goal:
            return new_path, g  # Return final path and total cost
        for neighbor, cost in graph[node]:
            g_new = g + cost
            f_new = g_new  # No heuristic
            heapq.heappush(open_set, (f_new, g_new, neighbor, new_path))
    return [], 0  # Return empty path if goal not reachable

# Run A* on each destination
if __name__ == '__main__':
    g = build_graph()
    origin = 2
    destinations = [5, 4]
    filename = "astar_search.py"
    method = "astar"
    for goal in destinations:
        path, cost = astar(g, origin, goal)
        print(f"{filename} {method}")
        if path:
            print(f"{goal} {len(path)}")
            print(path)
            print(f"Cost: {cost}")
        else:
            print(f"{goal} 0")
            print([])
