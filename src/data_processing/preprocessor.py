import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import pickle

class TransportDataPreprocessor:
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
    
    def load_raw_data(self):
        """Load all raw data files"""
        self.bus_stops = pd.read_csv(f"{self.raw_data_dir}/bus_stops.csv")
        self.road_network = gpd.read_file(f"{self.raw_data_dir}/road_network.geojson")
        self.bus_routes = pd.read_csv(f"{self.raw_data_dir}/bus_routes.csv")
        self.traffic_data = pd.read_csv(f"{self.raw_data_dir}/traffic_data.csv")
        self.fuel_consumption = pd.read_csv(f"{self.raw_data_dir}/fuel_consumption.csv")
        
        # Convert route stops from string to list
        self.bus_routes['stops'] = self.bus_routes['stops'].apply(
            lambda x: x.strip("[]").replace("'", "").split(", "))
    
    def build_transport_graph(self):
        """Build a graph representation of the transport network"""
        self.graph = nx.Graph()
        
        # Add nodes (bus stops)
        for _, stop in self.bus_stops.iterrows():
            self.graph.add_node(stop['stop_id'], 
                             pos=(stop['longitude'], stop['latitude']),
                             elevation=stop['elevation'],
                             is_terminal=stop['is_terminal'])
        
        # Add edges (road segments)
        for _, road in self.road_network.iterrows():
            # Calculate edge weights
            base_weight = road['length_km']
            
            # Get average traffic for this road
            avg_traffic = self.traffic_data[
                self.traffic_data['road_id'] == road['road_id']
            ]['traffic_level'].mean()
            
            # Adjust weight by traffic (higher traffic = worse)
            traffic_factor = 1 + avg_traffic * 2
            
            # Adjust weight by gradient (steeper = worse)
            gradient_factor = 1 + abs(road['gradient']) * 10
            
            # Combined weight
            weight = base_weight * traffic_factor * gradient_factor
            
            self.graph.add_edge(road['start_stop'], road['end_stop'],
                              road_id=road['road_id'],
                              length=road['length_km'],
                              lanes=road['lanes'],
                              speed_limit=road['speed_limit'],
                              gradient=road['gradient'],
                              weight=weight)
    
    def preprocess_fuel_data(self):
        """Preprocess fuel consumption data"""
        # Calculate fuel efficiency (km/l)
        self.fuel_consumption['fuel_efficiency'] = (
            self.fuel_consumption['distance_km'] / self.fuel_consumption['fuel_liters']
        )
        
        # Group by route and calculate average metrics
        self.route_stats = self.fuel_consumption.groupby('route_id').agg({
            'distance_km': 'mean',
            'fuel_liters': 'mean',
            'fuel_efficiency': 'mean',
            'passenger_count': 'mean'
        }).reset_index()
        
        # Merge with route information
        self.route_stats = pd.merge(
            self.route_stats,
            self.bus_routes[['route_id', 'length_km', 'avg_speed_kmh']],
            on='route_id'
        )
        
        # Calculate additional metrics
        self.route_stats['passengers_per_km'] = (
            self.route_stats['passenger_count'] / self.route_stats['distance_km']
        )
        self.route_stats['fuel_per_passenger_km'] = (
            self.route_stats['fuel_liters'] / 
            (self.route_stats['passenger_count'] * self.route_stats['distance_km'])
        )
    
    def normalize_data(self):
        """Normalize data for ACO algorithm"""
        # Normalize edge weights to [0, 1] range
        weights = nx.get_edge_attributes(self.graph, 'weight').values()
        scaler = MinMaxScaler()
        normalized_weights = scaler.fit_transform(np.array(list(weights)).reshape(-1)
        
        # Update graph with normalized weights
        for i, (u, v) in enumerate(self.graph.edges()):
            self.graph[u][v]['normalized_weight'] = normalized_weights[i]
        
        # Normalize route stats
        features = ['fuel_efficiency', 'passengers_per_km', 'fuel_per_passenger_km']
        self.route_stats[features] = scaler.fit_transform(self.route_stats[features])
    
    def save_processed_data(self):
        """Save processed data to files"""
        import os
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Save network graph
        with open(f"{self.processed_data_dir}/transport_graph.pkl", 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save route stats
        self.route_stats.to_csv(f"{self.processed_data_dir}/route_stats.csv", index=False)
        
        # Save normalized data
        normalized_edges = []
        for u, v, data in self.graph.edges(data=True):
            normalized_edges.append({
                'start': u,
                'end': v,
                'length': data['length'],
                'weight': data['weight'],
                'normalized_weight': data['normalized_weight']
            })
        
        pd.DataFrame(normalized_edges).to_csv(
            f"{self.processed_data_dir}/normalized_edges.csv", index=False)

    def run_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("Loading raw data...")
        self.load_raw_data()
        
        print("Building transport graph...")
        self.build_transport_graph()
        
        print("Preprocessing fuel data...")
        self.preprocess_fuel_data()
        
        print("Normalizing data...")
        self.normalize_data()
        
        print("Saving processed data...")
        self.save_processed_data()
        
        return self.graph, self.route_stats

if __name__ == "__main__":
    preprocessor = TransportDataPreprocessor()
    graph, route_stats = preprocessor.run_pipeline()