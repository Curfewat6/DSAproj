from flask import Flask, jsonify, render_template
import heapq
import osmnx as ox
from geopy.distance import geodesic
import requests
import dij
import location

app = Flask(__name__)

# Load the graph from OpenStreetMap
graph_name = "singapore.graphml"
G = ox.load_graphml(graph_name)

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
def a_star_search(graph, start, goal):
    pq = [(0, start, [])]  # Priority queue as (cost, current_node, path)
    costs = {start: 0}
    visited = set()
    
    while pq:
        cost, current, path = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        path = path + [current]
        if current == goal:
            return path
        
        visited.add(current)
        
        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
            
            edge_data = graph[current][neighbor][0]
            travel_time = edge_data.get('travel_time', edge_data['length'] / 5)  # Default to speed 5 m/s if not set
            
            new_cost = costs[current] + travel_time
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal, graph)
                heapq.heappush(pq, (priority, neighbor, path))
    
    return None

# Function to fetch real-time traffic data from the Traffic Flow API
def fetch_traffic_flow_data(api_key):
    url = "http://datamall2.mytransport.sg/ltaodataservice/v3/TrafficSpeedBands"
    headers = {
        'AccountKey': api_key,
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        try:
            traffic_data = response.json()
            return traffic_data
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {response.content}")
    else:
        print(f"Failed to fetch data: {response.status_code}")
        print(f"Response content: {response.content}")
    
    return None

# Function to update edge weights based on traffic data
def update_edge_weights(graph, traffic_data):
    for segment in traffic_data.get('value', []):
        try:
            start_lat = float(segment['StartLat'])
            start_lon = float(segment['StartLon'])
            end_lat = float(segment['EndLat'])
            end_lon = float(segment['EndLon'])

            start_coords = (start_lat, start_lon)
            end_coords = (end_lat, end_lon)
            
            start_node = get_nearest_node(graph, start_coords)
            end_node = get_nearest_node(graph, end_coords)
            speed_band = float(segment['SpeedBand'])

            if start_node in graph and end_node in graph[start_node]:
                length = graph[start_node][end_node][0]['length'] / 1000  # Convert to km
                travel_time = length / speed_band  # Time in hours
                graph[start_node][end_node][0]['travel_time'] = travel_time * 60  # Convert to minutes
                graph[start_node][end_node][0]['speed_band'] = speed_band  # Store speed band for coloring
        except KeyError as e:
            print(f"Key error: {e} in segment {segment}")

# Input coordinates (latitude, longitude)
start_coords = location.addr2coord("ubi challenger warehouse")  # [Ubi Challenger warehouse]
destinations = [
    location.addr2coord("great world city"),     # GWC [Furthest]
    location.addr2coord("ion orchard"),     # ION orchard [middle]
    location.addr2coord("bishan mrt")      # Bishan [closest]
]
order_from_dij = dij.main(start_coords, destinations,G)
print("Order of delivery is: ", order_from_dij)
print("order_from_dij  = [(start_lat,start_long,end_lat,endlong),(start_lat,start_long,end_lat,endlong)]")

# Find the nearest nodes in the graph to the given coordinates
start_node = get_nearest_node(G, start_coords)
destination_coords = [(coord[2], coord[3]) for coord in order_from_dij]
destination_nodes = [get_nearest_node(G, coords) for coords in destination_coords]

# Function to calculate the total path distance for a given order of nodes
def calculate_total_distance(order):
    total_distance = 0
    total_time = 0
    current_node = start_node
    for node in order:
        segment = a_star_search(G, current_node, node)
        if segment and len(segment) > 1:
            segment_distance = sum(G[segment[i]][segment[i + 1]][0]['length'] for i in range(len(segment) - 1)) / 1000  # Convert to km
            speed_band = G[segment[0]][segment[1]][0].get('speed_band', 5)
            traffic_factor = 1.0 if speed_band >= 5 else 1.2 if speed_band >= 3 else 1.5
            segment_time = segment_distance / (speed_band * 1000 / 60) * traffic_factor  # Time in minutes
            total_distance += segment_distance
            total_time += segment_time
        current_node = node
    return total_distance, total_time

# Function to find the optimal order of visiting nodes (modified to visit in specified order)
def find_optimal_order(start, destinations):
    return destinations

@app.route('/')
def index():
    return render_template('index.html', start_coords=start_coords, destination_coords=destination_coords)

@app.route('/update_route')
def update_route():
    api_key = 'o2oSSMCJSUOkZQxWvyAjsA=='  # Replace with your actual LTA API key
    traffic_data = fetch_traffic_flow_data(api_key)
    update_edge_weights(G, traffic_data)

    optimal_order = find_optimal_order(start_node, destination_nodes)

    route_nodes = [start_node]
    route_segments = []
    total_distance = 0
    total_time = 0

    for i, dest_node in enumerate(optimal_order):
        segment = a_star_search(G, route_nodes[-1], dest_node)
        if segment and len(segment) > 1:
            route_segments.append(segment)
            route_nodes.extend(segment[1:])  # Avoid duplicating nodes

            segment_distance = sum(G[segment[j]][segment[j + 1]][0]['length'] for j in range(len(segment) - 1)) / 1000  # Convert to km
            speed_band = G[segment[0]][segment[1]][0].get('speed_band', 5)
            traffic_factor = 1.0 if speed_band >= 5 else 1.2 if speed_band >= 3 else 1.5
            segment_time = segment_distance / (speed_band * 1000 / 60) * traffic_factor  # Time in minutes

            total_distance += segment_distance
            total_time += segment_time

    # Prepare the data to send back
    route_data = []
    for segment in route_segments:
        segment_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in segment]
        speed_band = G[segment[0]][segment[1]][0].get('speed_band', 5)
        color = 'green' if speed_band >= 5 else 'yellow' if speed_band >= 3 else 'red'
        route_data.append({'coords': segment_coords, 'color': color})

    return jsonify(new_route_data=route_data, total_distance=total_distance, total_time=total_time)

if __name__ == '__main__':
    app.run(debug=True)
