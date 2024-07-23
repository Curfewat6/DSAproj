import os
import osmnx as ox
import networkx as nx
import folium
from itertools import permutations
import heapq
import location

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

def nearest_neighbor(graph, start_node, end_nodes):
     # Initialize arrays
   # print("YOOOOOOOOOOOOOOOOOOOOOOHOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    node_count=0
    path=[]
    distTo = {node: float('inf') for node in graph.nodes}
    edgeTo = {node: None for node in graph.nodes}
    marked = {node: False for node in graph.nodes}

    #init start node distance to 0 
    distTo[start_node] = 0

    # Priority queue to process nodes
    # in this case, the distance is the priority , shortest distance highest priority
    pq = [(0, start_node)]  # (distance, node)

    while pq:
        
        node_count += 1
        # Get the node with the shortest distance
        current_distance, current_node = heapq.heappop(pq)

        # Check if node is marked as visited
        if marked[current_node]:
            continue
        marked[current_node] = True
        # if end_nodes == [1829398209]:
        #     print("You made it")
        #Exit after first end node is found 
        if current_node in end_nodes:
            if current_node == 1829398209:
                print("You made it")
            # print("First end node found, treat this end node as start node again")
            # print(f"YOU ARE WORKING ON THIS NODE {current_node}")
            # #get path to the first end node 
            # trace_node = current_node
            # while trace_node != start_node:
            #     path.append(trace_node)
            #     trace_node = edgeTo[trace_node]
            # path.append(start_node)
            # path.reverse()

            # Convert start_node and current_node to coordinates and add to order tuple
            # start_coords = (graph.nodes[start_node]['y'], graph.nodes[start_node]['x'])
            # end_coords = (graph.nodes[current_node]['y'], graph.nodes[current_node]['x'])
            # print(start_node,current_node)
            order_dij.append((graph.nodes[start_node]['y'],graph.nodes[start_node]['x'],graph.nodes[current_node]['y'],graph.nodes[current_node]['x']))
            # print("DEBUGGIN STATEMENT")
            # print(f"NODE COUNT: {node_count}")
            # print(f"CURRENT_NODE: {current_node}")
            return node_count, current_node

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


        
    
def main(start_coords, destination_coords, mapped_coords ,G):
    
    #To this order to be passed to A* example will be 
    #declare global order tuple
    global order_dij
    
    counting_guy = 1
    order_dij = []
    dij_precomputed_route = []
    nodecount_segment=[]
    graph = G
    # print("COORDS ARE HERE")
    # print(mapped_coords)
    # #Step1 Get graph
    # graph = downloadOSMX()
    # if graph == None:
    #     print("Failed to download the graph. Exiting...")
    #     return
    
    # ###INTEGRATE with location.py
    # start_coords = [location.addr2coord("ubi challenger warehouse")]  # [Ubi Challenger warehouse]
    # destination_coords = [
    # location.addr2coord("great world city"),     # GWC [Furthest]
    # location.addr2coord("ion orchard"),     # ION orchard [middle]
    # location.addr2coord("bishan mrt")      # Bishan [closest]
    # ]
    
    # #Step2 Get locations
    # start_location = [
    #     (1.3521, 103.8198),  # Marina Bay Sands
    # ]
    # end_locations = [
    #     (1.2839, 103.8601),  # Singapore Sports Hub
    #     (1.2806, 103.8500)   # Raffles Place
    # ]

    #Step3 Convert coordinates to nodes in osmnx
    start_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
    end_nodes = []
    for end_location in destination_coords:
        node = ox.distance.nearest_nodes(graph, end_location[1], end_location[0])
        end_nodes.append(node)
        print((graph.nodes[node]['y'],graph.nodes[node]['x']))
        mapped_coords[counting_guy] = (graph.nodes[node]['y'],graph.nodes[node]['x'])
        counting_guy +=1
    # print("END NODES")
    # print(end_nodes)
    #Step4 Find the shortest path from start_node to end_nodes
    #Use dijstra to find the path from start_node to individual end_nodes
    while end_nodes != []:
        print("BEFORE REMOVING")
        print(end_nodes)
        number_of_nodes, current_node = nearest_neighbor(graph, start_node, end_nodes)
        #dij_precomputed_route.append(path)
        nodecount_segment.append(number_of_nodes)
        #path = extract_path(edgeTo, start_node, current_node)
        start_node = current_node
        
        end_nodes.remove(current_node)
        print("AFTER REMOVING")
        print(end_nodes)
    return order_dij, mapped_coords, nodecount_segment

if __name__ == "__main__":
    main()