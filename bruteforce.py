import itertools
import networkx as nx
import osmnx as ox
import math

def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def main(start_coords, destination_coords, mapper, G):
    order = []
    sorted_ids = [0]

    # Convert the coordinates to nodes
    start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    #print start node coordinates
    end_nodes = [ox.distance.nearest_nodes(G, end_location[1], end_location[0]) for end_location in destination_coords]

    # Generate all permutations of the end nodes
    permutations = itertools.permutations(end_nodes)

    min_distance = float('inf')
    best_permutation = None

    print("[*] Running Dijkstra - Brute Force")

    for perm in permutations:
        #init 
        total_distance=0
        current_node = start_node

        for next_node in perm:
            # Calculate the distance between the current node and the next node
            distance = nx.shortest_path_length(G, current_node, next_node, weight='length')
            total_distance += distance
            current_node = next_node

        if total_distance < min_distance:
            min_distance = total_distance
            best_permutation = perm

    #make changes to sorted_ids
    # Make changes to sorted_ids
    for node in best_permutation:
        node_coords = (G.nodes[node]['y'], G.nodes[node]['x'])
        nearest_id = None
        nearest_distance = float('inf')
        
        #compare which is the nearest neighbour 
        for id,coords in mapper.items():
            distance = calculate_distance(node_coords, coords)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_id = id
        
        sorted_ids.append(nearest_id)
    

    #convert best_permutation to coordinates
    for node in best_permutation:
        order.append((G.nodes[start_node]['y'],G.nodes[start_node]['x'],G.nodes[node]['y'], G.nodes[node]['x']))
        start_node=node
    


    return order, sorted_ids