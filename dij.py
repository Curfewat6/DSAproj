import os
import osmnx as ox
import networkx as nx
import folium
from itertools import permutations
import heapq

def downloadOSMX():
    graph_name = "singapore.graphml"
    if os.path.isfile(graph_name):
        print("Graph already exists")
        graph = ox.load_graphml(graph_name)
        return graph 
    
    print("Downloading Graph")
    # Define the place name or coordinates for the area you want to download
    place_name = "Singapore"

    # Download the road network data for the specified place
    graph = ox.graph_from_place(place_name, network_type='drive', retain_all=True, simplify=True)

    # Save the graph to a GraphML file
    ox.save_graphml(graph, graph_name)

    print("Graph downloaded")

    return graph

def dij(graph, start_node, end_nodes):
     # Initialize arrays
    distTo = {node: float('inf') for node in graph.nodes}
    edgeTo = {node: None for node in graph.nodes}
    marked = {node: False for node in graph.nodes}

    #init start node distance to 0 
    distTo[start_node] = 0

    # Priority queue to process nodes
    # in this case, the distance is the priority , shortest distance highest priority
    pq = [(0, start_node)]  # (distance, node)

    while pq:
        # Get the node with the shortest distance
        current_distance, current_node = heapq.heappop(pq)

        # Check if node is marked as visited
        if marked[current_node]:
            continue
        marked[current_node] = True
        
        #Exit after first end node is found 
        if current_node in end_nodes:
            print("One end node hit, will return array, draw the route and run dij again and treat the current node as the start node again, to find the next closest node")
            return edgeTo, distTo, current_node

        # Get the neighbors of the current node
        for neighbor in graph.neighbors(current_node):
            # Get the edge data
            edge_data = graph.get_edge_data(current_node, neighbor)
            #print(f"Edge data between {current_node} and {neighbor}: {edge_data}")

            # Handle multi-edges
            if isinstance(edge_data, dict):
                # If there are multiple edges, choose the one with shortest distance
                # Ensure 'length' key exists
                edge_data = min(edge_data.values(), key=lambda x: x['length'] if 'length' in x else float('inf'))

            # Get the distance from this neighbor to the current node
            edge_distance = edge_data.get('length', 1)
            distance = current_distance + edge_distance

            if distance < distTo[neighbor]:
                distTo[neighbor] = distance
                edgeTo[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

def extract_path(edgeTo, start_node, end_node):
    path = []
    current_node = end_node
    while current_node != start_node:
        path.append(current_node)
        current_node = edgeTo[current_node]
    path.append(start_node)
    path.reverse()
    return path
        
        
def main():
    #Step1 Get graph
    graph = downloadOSMX()
    if graph == None:
        print("Failed to download the graph. Exiting...")
        return
    
    #Step2 Get locations
    start_location = [
        (1.3521, 103.8198),  # Marina Bay Sands
    ]
    end_locations = [
        (1.2839, 103.8601),  # Singapore Sports Hub
        (1.2806, 103.8500)   # Raffles Place
    ]

    #Step3 Convert coordinates to nodes in osmnx
    start_node = ox.distance.nearest_nodes(graph, start_location[0][1], start_location[0][0])
    end_nodes = []
    for end_location in end_locations:
        end_nodes.append(ox.distance.nearest_nodes(graph, end_location[1], end_location[0]))  
     
    #Step4 Find the shortest path from start_node to end_nodes
    #Use dijstra to find the path from start_node to individual end_nodes
    while end_nodes != []:
        edgeTo, distTo, current_node = dij(graph, start_node, end_nodes)
        path = extract_path(edgeTo, start_node, current_node)
        print(f"Shortest path from {start_node} to {current_node}: {path}")
        start_node = current_node
        end_nodes.remove(current_node)

        #use the path to draw on google maps tbc
        #draw_path(graph, path)

    print("All end nodes have been visited")
if __name__ == "__main__":
    main()