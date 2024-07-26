"""
Done by: Lucas
"""
import numpy as np

def get_nearest_node(graph, point):
    return ox.distance.nearest_nodes(graph, point[1], point[0])

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def ant_colony_optimization(points,start_index, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    optimised_order = []
    best_path = None
    best_path_length = np.inf
    
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []
        
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = start_index
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

    return best_path, best_path_length

def ant_colony(points,start_index, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    optimised_order = []
    best_path = None
    best_path_length = np.inf
    
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []
        
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = start_index
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
    first_node_s = nodes[best_path[0]]
    m = folium.Map(location=[first_node_s[0], first_node_s[1]], zoom_start=14)

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

def optimise_results(nodes):
    """
    This function is to be called from app.py

    This function runs ant colony algo to return the optimal paths to visit 
    nodes parameter should be a dictionary. Refer to main()
    """
    print("[*] Running Ant Colony Algorithm")
    points = np.array(list(nodes.values()))  # Convert nodes to a NumPy array
    # Run the ant colony optimization
    best_path, best_path_length = ant_colony_optimization(points,0, n_ants=500, n_iterations=100, alpha=1, beta=1, evaporation_rate=0.5, Q=1)
    optimal_order_in_int = [int(x) for x in best_path]

    return optimal_order_in_int

def main():
    nodes = {
        0: (1.332, 103.8932),   # Ubi
        1: (1.33873066388191, 103.871670008192),    # woodleigh mall
        2: (1.30397974144505, 103.832032328465),    # ion orchard
        3: (1.4360700650803, 103.785981524253),     # CWP
        4: (1.39205314156706, 103.89507054384),     # Compass one
        5: (1.332, 103.8932)                        # return to start
    }

    points = np.array(list(nodes.values()))  # Convert nodes to a NumPy array
    print(points)
    # Run the ant colony optimization
    best_path, best_path_length = ant_colony(points,0, n_ants=550, n_iterations=500, alpha=1, beta=1, evaporation_rate=0.5, Q=1)
    print(best_path)
    # Visualize the best path with folium
    map_visualization = visualize_path_with_folium(nodes, best_path)
    map_visualization.save("best_path_map.html")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import osmnx as ox
    import folium
    main()