import heapq

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

def cus2_search(graph, start, goal):
    """
    Custom Search 2: Combines cost-so-far and number of moves as a heuristic.
    Args:
        graph (dict): {site_id: [(neighbor_id, weight), ...]}
        start (str): Starting site ID
        goal (str): Goal site ID
    Returns:
        tuple: (goal, total_cost, num_nodes_created, path)
    """
    queue = [(0, 0, start, [start])]  # (priority, cost_so_far, current, path)
    visited = set()
    num_nodes_created = 0
    while queue:
        priority, cost_so_far, current, path = heapq.heappop(queue)
        if current == goal:
            return goal, cost_so_far, num_nodes_created, path
        if current in visited:
            continue
        visited.add(current)
        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited:
                num_nodes_created += 1
                # Custom heuristic: cost_so_far + number of moves
                new_priority = cost_so_far + weight + len(path)
                heapq.heappush(queue, (new_priority, cost_so_far + weight, neighbor, path + [neighbor]))
    return goal, float('inf'), num_nodes_created, []

# Graph defined as adjacency list with edge costs
graph = {
    1: [(2, 4), (3, 5), (4, 6)],
    2: [(1, 4), (3, 4)],
    3: [(1, 5), (2, 4), (4, 5), (5, 6), (6, 7)],
    4: [(1, 6), (3, 5), (5, 7)],
    5: [(3, 6), (4, 8)],
    6: [(3, 7)],
}

origin = 2
destinations = [5, 4]
filename = "cus2_search.py"
method = "cus2"

# Run CUS2 for each goal
if __name__ == '__main__':
    for goal in destinations:
        result, cost, created, path = cus2_search(graph, origin, goal)
        print(f"{filename} {method}")
        if path:
            print(f"{goal} {len(path)}")
            print(path)
            print(f"Cost: {cost}")
        else:
            print(f"{goal} 0")
            print([])
