import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
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
        
        # Calculate average traffic for each road segment
        avg_traffic = self.traffic_data.groupby('road_id')['traffic_level'].mean().reset_index()
        
        # Add edges with traffic-weighted attributes
        for _, road in self.road_network.iterrows():
            # Get traffic data for this road
            road_traffic = avg_traffic[avg_traffic['road_id'] == road['road_id']]
            traffic_level = road_traffic['traffic_level'].values[0] if not road_traffic.empty else 0.5
            
            # Calculate composite weight considering multiple factors
            base_weight = road['length_km']
            traffic_factor = 1 + traffic_level * 2  # More traffic = worse
            gradient_factor = 1 + abs(road['gradient']) * 10  # Steeper = worse
            
            composite_weight = base_weight * traffic_factor * gradient_factor
            
            self.graph.add_edge(road['start_stop'], road['end_stop'],
                              road_id=road['road_id'],
                              length=road['length_km'],
                              lanes=road['lanes'],
                              speed_limit=road['speed_limit'],
                              gradient=road['gradient'],
                              traffic_level=traffic_level,
                              weight=composite_weight)
    
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
        weights = np.array([data['weight'] for _, _, data in self.graph.edges(data=True)])
        weights = weights.reshape(-1, 1)  # Convert to 2D array
        
        scaler = MinMaxScaler()
        normalized_weights = scaler.fit_transform(weights).flatten()
        
        # Update graph with normalized weights
        for i, (u, v) in enumerate(self.graph.edges()):
            self.graph[u][v]['normalized_weight'] = normalized_weights[i]
    
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

if __name__ == '__main__':
    preprocessor = TransportDataPreprocessor()
    graph, route_stats = preprocessor.run_pipeline()