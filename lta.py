import osmnx as ox
import networkx as nx
import folium
from itertools import permutations





def download_and_project_graph():
    print("Downloading OSM data for Singapore...")
    try:
        G = ox.graph_from_place('Singapore', network_type='drive')
        print("Data downloaded.")
        return G
    except Exception as e:
        print(f"Error during download: {e}")
        return None
    

def main():
    G = download_and_project_graph()
    if G is None:
        print("Failed to download the graph. Exiting...")
        return

    locations = [
        (1.3521, 103.8198),  # Marina Bay Sands
        (1.2839, 103.8601),  # Singapore Sports Hub
        (1.2806, 103.8500)   # Raffles Place
    ]


    print("hello")

    
if __name__ == "__main__":
    main()