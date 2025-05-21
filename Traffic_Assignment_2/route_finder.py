#105106819 Suman Sutparai

import networkx as nx
import numpy as np
from logger import setup_logger

class RouteFinder:
    def __init__(self, config):
        """Initialize the route finder.
        
        Args:
            config: Configuration object containing route finding parameters.
        """
        self.config = config
        self.graph = nx.DiGraph()
        self.site_coords = {}
        self.logger = setup_logger()
        
    def build_graph(self, edges, coordinates):
        """Build the graph from edges and coordinates.
        
        Args:
            edges: List of tuples (origin, destination, weight)
            coordinates: Dictionary mapping site IDs to (lat, lon) tuples
        """
        try:
            self.graph = nx.DiGraph()
            self.site_coords = coordinates
            
            # Add all nodes first
            for site_id in coordinates.keys():
                self.graph.add_node(str(site_id))
            
            # Then add edges
            for edge in edges:
                origin, dest, weight = edge
                self.graph.add_edge(str(origin), str(dest), weight=float(weight))
                
            self.logger.info(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            
        except Exception as e:
            self.logger.error(f"Error building graph: {str(e)}")
            raise
            
    def find_top_k_routes(self, origin, destination, k=2):
        """Find the top k routes between origin and destination.
        
        Args:
            origin: Origin site ID
            destination: Destination site ID
            k: Number of routes to find
            
        Returns:
            List of tuples (path, cost) where path is a list of site IDs
        """
        try:
            # Convert site IDs to strings
            origin = str(origin)
            destination = str(destination)
            
            print("\n" + "="*50)
            print(f"Starting route calculation from {origin} to {destination}")
            print("="*50)
            
            print(f"\nGraph Statistics:")
            print(f"- Total nodes: {len(self.graph.nodes)}")
            print(f"- Total edges: {len(self.graph.edges)}")
            print(f"- Available nodes: {sorted(self.graph.nodes)}")
            
            # Check if nodes exist in the graph
            if origin not in self.graph:
                print(f"\n❌ Error: Origin node {origin} not found in graph")
                return []
            if destination not in self.graph:
                print(f"\n❌ Error: Destination node {destination} not found in graph")
                return []
            
            print("\n✅ Nodes verified in graph")
            
            # Use k-shortest paths algorithm with progress updates
            print("\nCalculating shortest paths...")
            print("Step 1: Finding initial shortest path")
            
            # First find the shortest path
            try:
                shortest_path = nx.shortest_path(self.graph, origin, destination, weight='weight')
                print(f"Found initial path: {' -> '.join(shortest_path)}")
            except nx.NetworkXNoPath:
                print(f"\n❌ No path exists between {origin} and {destination}")
                return []
            
            print("Step 2: Finding alternative paths")
            paths = [shortest_path]
            
            # Find k-1 more paths
            for i in range(k-1):
                try:
                    # Create a new graph without the previous paths
                    G = self.graph.copy()
                    for path in paths:
                        for j in range(len(path)-1):
                            if G.has_edge(path[j], path[j+1]):
                                G.remove_edge(path[j], path[j+1])
                    
                    # Try to find another path
                    try:
                        new_path = nx.shortest_path(G, origin, destination, weight='weight')
                        paths.append(new_path)
                        print(f"Found alternative path {i+2}: {' -> '.join(new_path)}")
                    except nx.NetworkXNoPath:
                        print(f"No more alternative paths found after {i+1} paths")
                        break
                        
                except Exception as e:
                    print(f"Error finding alternative path: {str(e)}")
                    break
            
            if not paths:
                print(f"\n❌ No paths found between {origin} and {destination}")
                return []
            
            print(f"\n✅ Found {len(paths)} possible paths")
            print("\nCalculating route costs...")
            
            routes = []
            for i, path in enumerate(paths):
                print(f"\nAnalyzing Path {i+1}:")
                print(f"Route: {' -> '.join(path)}")
                cost = self.get_route_length(path)
                print(f"Total travel time: {cost:.2f} minutes")
                routes.append((path, cost))
            
            print("\n" + "="*50)
            print("Route calculation complete!")
            print("="*50 + "\n")
            
            return routes
            
        except Exception as e:
            print(f"\n❌ Error finding routes: {str(e)}")
            return []
            
    def get_route_length(self, route):
        """Calculate the total length of a route.
        
        Args:
            route: List of site IDs
            
        Returns:
            Total length of the route
        """
        try:
            length = 0
            print("\nCalculating segment times:")
            for i in range(len(route) - 1):
                try:
                    segment_time = self.graph[route[i]][route[i+1]]['weight']
                    length += segment_time
                    print(f"- {route[i]} to {route[i+1]}: {segment_time:.2f} minutes")
                except KeyError:
                    print(f"⚠️ Warning: Edge not found between {route[i]} and {route[i+1]}")
                    continue
            return length
        except Exception as e:
            print(f"❌ Error calculating route length: {str(e)}")
            return float('inf')
        
    def get_route_coords(self, route):
        """Get the coordinates for a route.
        
        Args:
            route: List of site IDs
            
        Returns:
            List of (lat, lon) tuples
        """
        try:
            coords = []
            for site in route:
                site = str(site)
                if site in self.site_coords:
                    coord = self.site_coords[site]
                    coords.append((coord['lat'], coord['lon']))
            return coords
        except Exception as e:
            self.logger.error(f"Error getting route coordinates: {str(e)}")
            return [] 