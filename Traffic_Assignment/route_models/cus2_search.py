import heapq

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

# Custom Search 2: Combines cost-so-far and number of moves as heuristic
def cus2_search(graph, start, goal):
    # Priority queue stores: (heuristic, moves, current_node, cost, path)
    priority_queue = []
    heapq.heappush(priority_queue, (0, 0, start, 0, [start]))
    visited = {}

    while priority_queue:
        heuristic, moves, current, cost, path = heapq.heappop(priority_queue)

        if current == goal:
            return goal, cost, len(path), path  # Return when goal reached

        if current in visited and visited[current] <= (cost, moves):
            continue  # Skip if already visited with a better or equal state

        visited[current] = (cost, moves)

        for neighbor, step_cost in graph[current]:
            new_cost = cost + step_cost
            new_moves = moves + 1
            heuristic = new_cost + new_moves  # Custom heuristic = cost + number of steps
            heapq.heappush(priority_queue, (heuristic, new_moves, neighbor, new_cost, path + [neighbor]))

    return None, None, 0, []  # No path found

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
