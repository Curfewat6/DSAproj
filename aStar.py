"""
Done by: Kane
"""
import heapq
import osmnx as ox
from geopy.distance import geodesic

# Load the graph from OpenStreetMap
#G = ox.graph_from_place('Singapore', network_type='drive')

# Function to find the nearest node in the graph to a given coordinate
def get_nearest_node(graph, point):
    return ox.distance.nearest_nodes(graph, point[1], point[0])

# Heuristic function for A*
def heuristic(node1, node2, graph):
    coords_1 = (graph.nodes[node1]['y'], graph.nodes[node1]['x'])
    coords_2 = (graph.nodes[node2]['y'], graph.nodes[node2]['x'])
    h = geodesic(coords_1, coords_2).meters
    return h

# Custom A* search algorithm with traffic-aware cost calculation
def search(graph, start, goal, avoid_edges=set()):
    node_count = 0
    pq = [(0, start, [])]  # Priority queue as (cost, current_node, path)
    costs = {start: 0}
    visited = set()
    
    while pq:
        node_count += 1
        cost, current, path = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        path = path + [current]
        if current == goal:
            return path, node_count
        
        visited.add(current)
        
        for neighbor in graph.neighbors(current):
            if neighbor in visited or (current, neighbor) in avoid_edges or (neighbor, current) in avoid_edges:
                continue
            
            edge_data = graph[current][neighbor][0]
            travel_time = edge_data.get('travel_time', edge_data['length'] / 5)  # Default to speed 5 m/s if not set
            
            new_cost = costs[current] + travel_time
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal, graph)
                heapq.heappush(pq, (priority, neighbor, path))
    
    print("No path found from {} to {} with avoid_edges: {}".format(start, goal, avoid_edges))
    return None, node_count

def main():
    pass

if __name__ == "__main__":
    main()
