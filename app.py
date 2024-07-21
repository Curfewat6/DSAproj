import heapq
import random
import osmnx as ox
from geopy.distance import geodesic
import requests
from flask import Flask, jsonify, render_template, request,redirect, url_for, session
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
def a_star_search(graph, start, goal, avoid_edges=set()):
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
            if neighbor in visited or (current, neighbor) in avoid_edges or (neighbor, current) in avoid_edges:
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

# Function to simulate high traffic between Ubi and Bishan
def simulate_high_traffic(graph, start_coords, end_coords):
    graph_copy = graph.copy()
    start_node = get_nearest_node(graph_copy, start_coords)
    end_node = get_nearest_node(graph_copy, end_coords)
    for u, v, key, data in graph_copy.edges(keys=True, data=True):
        if (u == start_node and v == end_node) or (u == end_node and v == start_node):
            data['speed_band'] = 1  # Simulate heavy traffic
            data['travel_time'] = data['length'] / (1 * 1000 / 60) * 3  # Triple the travel time
    return graph_copy

# # Input coordinates (latitude, longitude)
# start_coords = location.addr2coord("ubi challenger warehouse")  # [Ubi Challenger warehouse]
# destinations = [
#     location.addr2coord("great world city"),     # GWC [Furthest]
#     location.addr2coord("ion orchard"),     # ION orchard [middle]
#     location.addr2coord("ang mo kio hub")      # Ang Mo Kio Hub
# ]

# order_from_dij = dij.main(start_coords, destinations, G)
# print("Order of delivery is: ", order_from_dij)

# # Find the nearest nodes in the graph to the given coordinates
# start_node = get_nearest_node(G, start_coords)
# destination_coords = [(coord[2], coord[3]) for coord in order_from_dij]
# destination_nodes = [get_nearest_node(G, coords) for coords in destination_coords]

# Function to calculate the total path distance for a given order of nodes
# def calculate_total_distance(order):
#     total_distance = 0
#     total_time = 0
#     current_node = start_node
#     for node in order:
#         segment = a_star_search(G, current_node, node)
#         if segment and len(segment) > 1:
#             segment_distance = sum(G[segment[i]][segment[i + 1]][0]['length'] for i in range(len(segment) - 1)) / 1000  # Convert to km
#             speed_band = G[segment[0]][segment[1]][0].get('speed_band', 5)
#             traffic_factor = 1.0 if speed_band >= 5 else 1.2 if speed_band >= 3 else 1.5
#             segment_time = segment_distance / (speed_band * 1000 / 60) * traffic_factor  # Time in minutes
#             total_distance += segment_distance
#             total_time += segment_time
#         current_node = node
#     return total_distance, total_time

def calculate_route_distance_and_time(route, graph):
    total_distance = 0
    total_time = 0

    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        if u in graph and v in graph[u]:
            edge_data = graph[u][v][0]
            segment_distance = edge_data.get('length', 0) / 1000  # Convert to km
            speed_band = edge_data.get('speed_band', 40)  # Default to 40 km/h if not set
            traffic_factor = 1.0 if speed_band >= 40 else 1.2 if speed_band >= 20 else 1.5
            
            # Calculate segment time in hours and then convert to minutes
            segment_time_hours = segment_distance / speed_band  # Time in hours
            segment_time_minutes = segment_time_hours * 60  # Convert to minutes
            
            adjusted_segment_time = segment_time_minutes * traffic_factor  # Adjusted time with traffic factor

            total_distance += segment_distance
            total_time += adjusted_segment_time

    return total_distance, total_time


def get_nearest_neighbor_node(graph, node, exclude_node):
    nearest_neighbor = None
    min_distance = float('inf')

    for neighbor in graph.neighbors(node):
        if neighbor == exclude_node:
            continue
        distance = geodesic((graph.nodes[node]['y'], graph.nodes[node]['x']),
                            (graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'])).meters
        if distance < min_distance:
            min_distance = distance
            nearest_neighbor = neighbor

    return nearest_neighbor

#Step 1 , user input form
@app.route('/')
def index():
    return render_template('form.html')

# @app.route('/')
# def index():
#     return render_template('index.html', start_coords=start_coords, destination_coords=destination_coords)

#step 2, from user input form to this function , use location.py to check for the coordinates
#results is shown in results.html
@app.route('/checkAddress', methods=['POST'])
def check_address():
    counter = 0
    start_location = request.form['startLocation']
    num_destinations = int(request.form['numDestinations'])
    
    destinations = []
    for i in range(num_destinations):   
        destination = request.form[f'destination{i}']
        destinations.append(destination)
    
    # Convert addresses to coordinates using location.py
    start_data = location.addr2coord(start_location)

    # Store data in session
    session[f'address{counter}'] = start_data['address']
    session[f'lcoords{counter}'] = start_data['coords']
    # VERY IMPORTANT: Add in an ID
    session[f'{counter}'] = counter
    destination_data = [location.addr2coord(dest) for dest in destinations]
    
    for i in destination_data:
        session[f'address{counter+1}'] = i['address']
        session[f'lcoords{counter+1}'] = i['coords']
        # VERY IMPORTANT: Add in an ID
        session[f'{counter + 1}'] = counter + 1
        counter +=1

    session['counter'] = counter
    # Now you have start_data and destination_data
    # You can proceed with the rest of your logic here
    
    # You can redirect to another page or render a template with the results
    return render_template('results.html', start_data=start_data, destination_data=destination_data)
    

#step 3, from result.html, user click generate will come to this function to user dij.py to generate the order
#results is shown in results.html
@app.route('/generate_order', methods=['POST'])
def generate_order():
    counter = session.get('counter')
    # Create an empty dictionary to map ID to coordiante
    mapper = {}

    # Retrieve info from session
    start_coords = session.get('lcoords0')
    destination_coords = [session.get(f'lcoords{data+1}') for data in range(counter+1)]
    destination_coords.pop()    # Just a quick fix 

    # Tag the ID to each coordinate
    for i in range(counter+1):
        identifier = session.get(f'{i}')
        print(identifier)
        mapper[identifier] = session.get(f'lcoords{i}')

    # Call the nearest neighbor function from dij.py
    # I Pass the dict to dij to make sure the coordinate in the dict also change. so now dict will have new coordinate
    order_from_dij, mapped = dij.main(start_coords, destination_coords, mapper, G)
    
    print(order_from_dij)

    # # Accessing individual elements in the first tuple
    # print("First element of the first tuple:", order_from_dij[0][0])
    # print("Second element of the first tuple:", order_from_dij[0][1])
    # print("Third element of the first tuple:", order_from_dij[0][2])
    # print("Fourth element of the first tuple:", order_from_dij[0][3])


    reversed_mapper = {coords: id for id, coords in mapped.items()}

    # Initialize an empty list to store the sorted IDs
    sorted_ids = []
    sorted_ids.append(0)
    # Iterate through the tuples in order_from_dij
    for tup in order_from_dij:
        # Extract the 3rd and 4th elements as a tuple
        coords = (tup[2], tup[3])
        # Check if these coordinates are in the reversed_mapper
        if coords in reversed_mapper:
            # If so, add the corresponding ID to the sorted_ids list
            sorted_ids.append(reversed_mapper[coords])

    print("Sorted IDs:", sorted_ids)
    session['sorted_ids'] = sorted_ids
    # Prepare the data to be displayed in order.html
    # The flow is -> levenstein -> store in session-> put ID in dict-> dij -> change the value in dict but ID remains same -> print using ID to maintain order
    ordered_data = []
    total_distance = 0
    total_time = 0
    for i, coord in enumerate(order_from_dij):
        start_id = sorted_ids[i]
        end_id = sorted_ids[i + 1]
        start_node = get_nearest_node(G, session.get(f'lcoords{start_id}'))
        end_node = get_nearest_node(G, session.get(f'lcoords{end_id}'))

        segment_distance, segment_time = calculate_route_distance_and_time(a_star_search(G, start_node, end_node), G)
        total_distance += segment_distance
        total_time += segment_time
        start_coords = (coord[0], coord[1])
        end_coords = (coord[2], coord[3])
        ordered_data.append({
            'start': session.get(f'lcoords{sorted_ids[i]}'),
            'end': session.get(f'lcoords{sorted_ids[i+1]}'),
            'start_address': session.get(f'address{sorted_ids[i]}'),
            'end_address': session.get(f'address{sorted_ids[i+1]}'),
            'index': i + 1,
            'segment_distance': round(segment_distance, 2),
            'segment_time': round(segment_time, 2)
        })
    
    return render_template('order.html', ordered_data=ordered_data, total_distance=round(total_distance, 2), total_time=round(total_time, 2))

#step 4, based on the order given, plot the route
@app.route('/plot', methods=['POST'])
def plot():
    session['target'] = request.form['step_number']
    counter = session.get('counter')
    start_coords = session.get('lcoords0')
    destination_coords = [session.get(f'lcoords{data+1}') for data in range(counter+1)]
    destination_coords.pop()    # Just a quick fix 
    print("DEBUGGER")
    print(destination_coords)
    start_node = get_nearest_node(G, start_coords)
    destination_nodes = [get_nearest_node(G, coords) for coords in destination_coords]

    route_segments = []
    total_distance = 0
    total_time = 0

    for i in range(len(destination_nodes)):
        if i == 0:
            segment = a_star_search(G, start_node, destination_nodes[i])
        else:
            segment = a_star_search(G, destination_nodes[i-1], destination_nodes[i])

        if segment:
            segment_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in segment]
            route_segments.append(segment_coords)

            segment_distance, segment_time = calculate_route_distance_and_time(segment, G)
            total_distance += segment_distance
            total_time += segment_time

    return render_template('plot.html', start_coords=start_coords, destination_coords=destination_coords, total_distance=total_distance, total_time=total_time)

#step5, if need be, update route
@app.route('/update_route')
def update_route():
    counter = session.get('counter')
    target = int(session.get('target'))  # Get the target step number
    sorted_ids = session.get('sorted_ids')
    print(target)
    print(sorted_ids)
    
    api_key = 'o2oSSMCJSUOkZQxWvyAjsA=='  # Replace with your actual LTA API key
    traffic_data = fetch_traffic_flow_data(api_key)
    update_edge_weights(G, traffic_data)
    
    destination_coords = [session.get(f'lcoords{data+1}') for data in range(counter+1)]
    destination_coords.pop()    # Just a quick fix
    destination_nodes = [get_nearest_node(G, coords) for coords in destination_coords]
    print("cooordies")
    print(destination_coords)
    if target == 1:
        start_coords = session.get('lcoords0')
        start_node = get_nearest_node(G, start_coords)
    else:
        start_node = destination_nodes[sorted_ids[target - 1] - 1]

    
    print(destination_nodes)
    # Select the target destination based on the step number    
    target_node = destination_nodes[sorted_ids[target] - 1]

    route_segments = []
    total_distance = 0
    total_time = 0

    # Calculate the route for the specified step
    segment = a_star_search(G, start_node, target_node)
    if segment and len(segment) > 1:
        segment_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in segment]
        route_segments.append(segment_coords)

        segment_distance, segment_time = calculate_route_distance_and_time(segment, G)
        total_distance += segment_distance
        total_time += segment_time

    # Prepare the data to send back
    new_route_data = [{'coords': segment, 'color': 'green'} for segment in route_segments]

    return jsonify(original_route_data=[], new_route_data=new_route_data, total_distance=total_distance, total_time=total_time)

@app.route('/simulate_traffic')
def simulate_traffic():
    counter = session.get('counter')
    target = int(session.get('target'))  # Get the target step number
    sorted_ids = session.get('sorted_ids')
    destination_coords = [session.get(f'lcoords{data+1}') for data in range(counter+1)]
    destination_coords.pop()  # Just a quick fix
    destination_nodes = [get_nearest_node(G, coords) for coords in destination_coords]

    if target == 1:
        start_coords = session.get('lcoords0')
        start_node = get_nearest_node(G, start_coords)
    else:
        start_coords = destination_coords[sorted_ids[target - 1] - 1]
        start_node = destination_nodes[sorted_ids[target - 1] - 1]

    first_destination = destination_coords[sorted_ids[target] - 1]
    first_destination_node = destination_nodes[sorted_ids[target] - 1]

    # Simulate high traffic using a copied graph
    G_simulated = simulate_high_traffic(G, start_coords, first_destination)

    original_segment = a_star_search(G_simulated, start_node, first_destination_node)
    original_route_data = []

    if original_segment and len(original_segment) > 1:
        segment_coords = [(G_simulated.nodes[node]['y'], G_simulated.nodes[node]['x']) for node in original_segment]
        original_route_data.append({'coords': segment_coords, 'color': 'red'})
        
    # Fetch LTA traffic data and update edge weights
    api_key = 'o2oSSMCJSUOkZQxWvyAjsA=='  # Replace with your actual LTA API key
    traffic_data = fetch_traffic_flow_data(api_key)
    update_edge_weights(G_simulated, traffic_data)

    # Compute an alternative route using the nearest neighbor node of the destination
    avoid_edges = set((original_segment[i], original_segment[i + 1]) for i in range(len(original_segment) - 1))
    neighbor_node = get_nearest_neighbor_node(G, first_destination_node, exclude_node=start_node)
    alternative_segment = a_star_search(G_simulated, start_node, neighbor_node, avoid_edges)
    alternative_route_data = []
    alt_total_distance = 0
    alt_total_time = 0

    if alternative_segment and len(alternative_segment) > 1:
        segment_coords = [(G_simulated.nodes[node]['y'], G_simulated.nodes[node]['x']) for node in alternative_segment]
        alternative_route_data.append({'coords': segment_coords, 'color': 'green'})
        
        # Calculate total distance and time for alternative route with heavy traffic
        alt_total_distance, alt_total_time = calculate_route_distance_and_time(alternative_segment, G_simulated)

    return jsonify(original_route_data=original_route_data, alternative_route_data=alternative_route_data, alt_total_distance=alt_total_distance, alt_total_time=alt_total_time)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
