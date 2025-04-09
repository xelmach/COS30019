import heapq

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

def cus2_search(graph, start, goal):
    priority_queue = []
    heapq.heappush(priority_queue, (0, 0, start, []))  
    visited = {}

    while priority_queue:
        cost, moves, current, path = heapq.heappop(priority_queue)

        if current == goal:
            print("cus2_search")
            print(f"{goal} {len(visited)}")
            print(" ".join(map(str, path + [current]))) 
            return goal, len(visited), path + [current]

        if current in visited and visited[current] <= (cost, moves):
            continue  

        visited[current] = (cost, moves)

        for neighbor, step_cost in graph[current]:
            new_cost = cost + step_cost
            new_moves = moves + 1
            heuristic = new_cost + new_moves 
            heapq.heappush(priority_queue, (heuristic, new_moves, neighbor, path + [current]))

    print("cus2_search")
    print(f"None {len(visited)}")  
    print("")

    return None, len(visited), []  



for destination in destinations:
    cus2_search(graph, origin, destination)
