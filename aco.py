import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import folium

def get_nearest_node(graph, point):
    return ox.distance.nearest_nodes(graph, point[1], point[0])

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf
    
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []
        
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0
            
            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))
                
                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point, unvisited_point]**alpha / distance(points[current_point], points[unvisited_point])**beta
                
                probabilities /= np.sum(probabilities)
                
                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point] = True
                current_point = next_point
            
            paths.append(path)
            path_lengths.append(path_length)
            
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length
        
        pheromone *= evaporation_rate
        
        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points-1):
                pheromone[path[i], path[i+1]] += Q/path_length
            pheromone[path[-1], path[0]] += Q/path_length
    
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:,0], points[:,1], c='r', marker='o')
    
    for i in range(len(best_path)-1):
        plt.plot([points[best_path[i],0], points[best_path[i+1],0]],
                 [points[best_path[i],1], points[best_path[i+1],1]],
                 c='g', linestyle='-', linewidth=2, marker='o')
        
    plt.plot([points[best_path[0],0], points[best_path[-1],0]],
             [points[best_path[0],1], points[best_path[-1],1]],
             c='g', linestyle='-', linewidth=2, marker='o')
    
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    return best_path, best_path_length

def visualize_path_with_folium(nodes, best_path):
    # Initialize a folium map at the first node's location
    first_node_coords = nodes[best_path[0]]
    m = folium.Map(location=[first_node_coords[0], first_node_coords[1]], zoom_start=14)

    # Add markers for each node in the best path
    for node_index in best_path:
        node_coords = nodes[node_index]
        folium.Marker([node_coords[0], node_coords[1]]).add_to(m)

    # Add lines between nodes in the best path
    for i in range(len(best_path) - 1):
        start_node = nodes[best_path[i]]
        end_node = nodes[best_path[i + 1]]
        folium.PolyLine([start_node, end_node], color="green").add_to(m)

    # Display the map
    return m

def main():
    nodes = {
        0: (1.332, 103.8932),
        1: (1.303472, 103.8314278),
        2: (1.3689602, 103.8493394),
        3: (1.2931961, 103.831299)
    }
    coord = [(1.332, 103.8932)]
    coords = np.array()
    points = np.array(list(nodes.values()))  # Convert nodes to a NumPy array
    print(points)
    # Run the ant colony optimization
    best_path, best_path_length = ant_colony_optimization(points, n_ants=550, n_iterations=100, alpha=1, beta=1, evaporation_rate=0.5, Q=1)

    # Visualize the best path with folium
    map_visualization = visualize_path_with_folium(nodes, best_path)
    map_visualization.save("best_path_map.html")

if __name__ == '__main__':
    main()