import heapq

def cus2_search(graph, start, goal):
    priority_queue = []
    heapq.heappush(priority_queue, (0, 0, start, []))  

    visited = {}

    while priority_queue:
        cost, moves, current, path = heapq.heappop(priority_queue)

        if current == goal:
            return goal, len(visited), path + [current]

        if current in visited and visited[current] <= (cost, moves):
            continue  

        visited[current] = (cost, moves)

        for neighbor, step_cost in graph[current]:
            new_cost = cost + step_cost
            new_moves = moves + 1
            heuristic = new_cost + new_moves  
            heapq.heappush(priority_queue, (heuristic, new_moves, neighbor, path + [current]))

    return None, len(visited), [] 
