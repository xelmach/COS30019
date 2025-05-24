import heapq, math
from collections import defaultdict

# Coordinates of nodes (used for Euclidean heuristic)
positions = {1:(4,1),2:(2,2),3:(4,4),4:(6,3),5:(5,6),6:(7,5)}

# Edge list with associated costs
edges = {
    (2,1):4,(3,1):5,(1,3):5,(2,3):4,(3,2):5,(4,1):6,(1,4):6,(4,3):5,
    (3,5):6,(5,3):6,(4,5):7,(5,4):8,(6,3):7,(3,6):7
}

# Build graph as adjacency list (with cost)
def build_graph():
    g = defaultdict(list)
    for (u,v),cost in edges.items():
        g[u].append((v,cost))
    return g

# Heuristic function: Euclidean distance between nodes
def heuristic(a,b):
    x1,y1=positions[a]; x2,y2=positions[b]
    return math.hypot(x2-x1, y2-y1)

# Greedy Best-First Search algorithm
def gbfs(graph, start, goal):
    pq = [(0, start, [])]  # Priority queue: (heuristic, node, path)
    visited = set()
    while pq:
        _, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == goal:
            # Calculate total cost of the found path
            cost = sum(dict(graph[path[i]]).get(path[i+1],0) for i in range(len(path)-1))
            return path, cost
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (0, neighbor, path))
    return [], 0  # No path found

# Run GBFS for each destination
if __name__ == '__main__':
    g = build_graph()
    origin = 2
    destinations = [5, 4]
    filename = "gbfs_search.py"
    method = "gbfs"
    for goal in destinations:
        path, cost = gbfs(g, origin, goal)
        print(f"{filename} {method}")
        if path:
            print(f"{goal} {len(path)}")
            print(path)
            print(f"Cost: {cost}")
        else:
            print(f"{goal} 0")
            print([])
