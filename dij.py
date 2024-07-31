"""
Done by: Weijing
"""
import os
import osmnx as ox
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

def nearest_neighbor(graph, start_node, end_nodes):

    #init things
    node_count=0
    path=[]
    #set to inf distance 
    distTo = {node: float('inf') for node in graph.nodes}
    edgeTo = {node: None for node in graph.nodes}
    marked = {node: False for node in graph.nodes}

    #init start node distance to 0 
    distTo[start_node] = 0

    # Priority queue to process nodes
    # in this case, the distance is the priority , shortest distance highest priority
    pq = [(0, start_node)]  # (distance, node)

    #while priority q not empty 
    while pq:
        
        node_count += 1
        # Get the node with the shortest distance
        current_distance, current_node = heapq.heappop(pq)

        # Check if node is marked as visited
        if marked[current_node]:
            continue
        marked[current_node] = True

        #break if is in end_nodes
        if current_node in end_nodes:
            order_dij.append((graph.nodes[start_node]['y'],graph.nodes[start_node]['x'],graph.nodes[current_node]['y'],graph.nodes[current_node]['x']))
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

            # If the distance to this neighbor is less than the distance to the current node
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
    

    #Step3 Convert coordinates to nodes in osmnx
    start_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
    end_nodes = []
    for end_location in destination_coords:
        node = ox.distance.nearest_nodes(graph, end_location[1], end_location[0])
        end_nodes.append(node)
        print((graph.nodes[node]['y'],graph.nodes[node]['x']))
        mapped_coords[counting_guy] = (graph.nodes[node]['y'],graph.nodes[node]['x'])
        counting_guy +=1

    #Step4 Find the shortest path from start_node to end_nodes
    #Use dijstra to find the path from start_node to individual end_nodes
    while end_nodes != []:

        number_of_nodes, current_node = nearest_neighbor(graph, start_node, end_nodes)
        nodecount_segment.append(number_of_nodes)
        start_node = current_node
        end_nodes.remove(current_node)
      
    return order_dij, mapped_coords, nodecount_segment

if __name__ == "__main__":
    main()