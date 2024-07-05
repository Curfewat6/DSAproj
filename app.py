import heapq
import osmnx as ox
from geopy.distance import geodesic
import folium
import requests
from flask import Flask, jsonify, render_template
from itertools import permutations

app = Flask(__name__)

# Load the graph from OpenStreetMap
G = ox.graph_from_place('Singapore', network_type='drive')

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
            speed_band = edge_data.get('speed_band', 5)
            traffic_factor = 1.0 if speed_band >= 5 else 1.2 if speed_band >= 3 else 1.5
            travel_time = edge_data['length'] / (speed_band * 1000 / 60) * traffic_factor  # Time in minutes
            
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
            # Convert coordinates to float
            start_lat = float(segment['StartLat'])
            start_lon = float(segment['StartLon'])
            end_lat = float(segment['EndLat'])
            end_lon = float(segment['EndLon'])

            start_coords = (start_lat, start_lon)
            end_coords = (end_lat, end_lon)
            start_node = get_nearest_node(graph, start_coords)
            end_node = get_nearest_node(graph, end_coords)
            speed_band = float(segment['SpeedBand'])  # Convert speed band to float

            if start_node in graph and end_node in graph[start_node]:
                length = graph[start_node][end_node][0]['length'] / 1000  # Convert to km
                travel_time = length / speed_band  # Time in hours
                graph[start_node][end_node][0]['travel_time'] = travel_time * 60  # Convert to minutes
                graph[start_node][end_node][0]['speed_band'] = speed_band  # Store speed band for coloring
        except KeyError as e:
            print(f"Key error: {e} in segment {segment}")

# Function to calculate fuel consumption in liters
def calculate_fuel_consumption(distance_km, fuel_efficiency_l_per_100km, traffic_factor=1.0):
    adjusted_fuel_efficiency = fuel_efficiency_l_per_100km * traffic_factor
    fuel_consumption = (distance_km / 100) * adjusted_fuel_efficiency
    return fuel_consumption

# Function to determine traffic factor based on fetched traffic data
def determine_traffic_factor(traffic_data):
    try:
        average_speed = sum(item['SpeedBand'] for item in traffic_data['value']) / len(traffic_data['value'])
        if average_speed < 2:
            return 1.5  # Heavy traffic
        elif average_speed < 4:
            return 1.2  # Moderate traffic
        else:
            return 1.0  # Normal traffic
    except Exception as e:
        print(f"Error in determining traffic factor: {e}")
        return 1.0  # Default to normal traffic factor if any error occurs

# Input coordinates (latitude, longitude)
start_coords = (1.3321, 103.8934)  # [Ubi Challenger warehouse]
destination_coords = [
    (1.2936, 103.8319),     # GWC [Furthest]
    (1.3507, 103.8488),     # ION orchard [middle]
    (1.3002, 103.8552)      # Bishan [closest]
]

# Find the nearest nodes in the graph to the given coordinates
start_node = get_nearest_node(G, start_coords)
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

# Function to find the optimal order of visiting nodes using all permutations
def find_optimal_order(start, destinations):
    min_distance = float('inf')
    optimal_order = None
    for perm in permutations(destinations):
        distance, _ = calculate_total_distance(perm)
        if distance < min_distance:
            min_distance = distance
            optimal_order = perm
    return optimal_order

@app.route('/')
def index():
    return render_template('index.html', start_coords=start_coords, destination_coords=destination_coords)

@app.route('/update_route')
def update_route():
    api_key = 'o2oSSMCJSUOkZQxWvyAjsA=='  # Replace with your actual LTA API key
    traffic_data = fetch_traffic_flow_data(api_key)
    update_edge_weights(G, traffic_data)
    traffic_factor = determine_traffic_factor(traffic_data) if traffic_data else 1.0

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

    return jsonify(route_data=route_data, total_distance=total_distance, total_time=total_time)

if __name__ == '__main__':
    app.run(debug=True)
