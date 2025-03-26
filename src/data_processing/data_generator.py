import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import random
from datetime import time
import yaml

class TransportDataGenerator:
    def __init__(self, config_path='config/data_generation.yml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.rng = np.random.default_rng(self.config['seed'])
        
    def generate_bus_stops(self):
        """Generate synthetic bus stop data"""
        num_stops = self.config['num_stops']
        city_center = self.config['city_center']
        radius_km = self.config['city_radius_km']
        
        stops = []
        for i in range(num_stops):
            # Generate points in a circular area around city center
            angle = self.rng.uniform(0, 2*np.pi)
            distance = self.rng.uniform(0, radius_km)
            
            # Convert polar to Cartesian (approximate)
            lat = city_center[0] + (distance/111) * np.cos(angle)
            lon = city_center[1] + (distance/(111*np.cos(np.radians(lat)))) * np.sin(angle)
            
            stops.append({
                'stop_id': f"BS{i:03d}",
                'stop_name': f"Stop {i}",
                'latitude': lat,
                'longitude': lon,
                'is_terminal': self.rng.random() < 0.1,  # 10% chance of being terminal
                'elevation': self.rng.normal(50, 20)  # Mean elevation 50m
            })
        
        return pd.DataFrame(stops)
    
    def generate_road_network(self, stops):
        """Generate a synthetic road network connecting stops"""
        from sklearn.neighbors import NearestNeighbors
        
        # Create a basic road network by connecting nearest neighbors
        coords = stops[['latitude', 'longitude']].values
        n_neighbors = min(5, len(stops)-1)
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        roads = []
        road_id = 0
        
        for i in range(len(stops)):
            for j in indices[i]:
                if i != j and i < j:  # Avoid duplicates
                    start = stops.iloc[i]
                    end = stops.iloc[j]
                    
                    # Create a LineString geometry
                    line = LineString([
                        (start['longitude'], start['latitude']),
                        (end['longitude'], end['latitude'])
                    ])
                    
                    # Calculate road properties
                    length_km = distances[i][list(indices[i]).index(j)] * 111  # Approx km
                    lanes = self.rng.choice([1, 2, 4], p=[0.2, 0.5, 0.3])
                    speed_limit = self.rng.choice([30, 50, 70], p=[0.4, 0.5, 0.1])
                    gradient = (end['elevation'] - start['elevation']) / (length_km * 1000)
                    
                    roads.append({
                        'road_id': f"RD{road_id:03d}",
                        'start_stop': start['stop_id'],
                        'end_stop': end['stop_id'],
                        'length_km': length_km,
                        'lanes': lanes,
                        'speed_limit': speed_limit,
                        'gradient': gradient,
                        'geometry': line
                    })
                    road_id += 1
        
        return gpd.GeoDataFrame(roads, geometry='geometry')
    
    def generate_bus_routes(self, stops, roads):
        """Generate synthetic bus routes"""
        num_routes = self.config['num_routes']
        routes = []
        
        terminals = stops[stops['is_terminal']]['stop_id'].tolist()
        
        for i in range(num_routes):
            # Select 2 distinct terminals
            start, end = self.rng.choice(terminals, 2, replace=False)
            
            # Find path between terminals (simplified)
            path = self.find_path(start, end, stops, roads)
            
            if path:
                # Calculate route length
                length = sum(roads[roads['road_id'].isin(path)]['length_km'].sum()
                
                routes.append({
                    'route_id': f"R{i+1:02d}",
                    'route_name': f"Route {i+1}",
                    'start_stop': start,
                    'end_stop': end,
                    'stops': self.get_stops_along_path(path, stops, roads),
                    'length_km': length,
                    'avg_speed_kmh': self.rng.normal(25, 5)
                })
        
        return pd.DataFrame(routes)
    
    def find_path(self, start, end, stops, roads):
        """Simplified path finding between two stops"""
        # This is a simplified version - in practice you'd use proper graph algorithms
        visited = set()
        path = []
        current = start
        
        while current != end and len(visited) < len(stops):
            # Find all roads from current stop
            available_roads = roads[(roads['start_stop'] == current) | 
                                  (roads['end_stop'] == current)]
            
            if available_roads.empty:
                break
                
            # Choose a random road
            road = available_roads.sample(1).iloc[0]
            path.append(road['road_id'])
            
            # Move to the other end of the road
            current = road['end_stop'] if road['start_stop'] == current else road['start_stop']
            
            if current in visited:
                break
            visited.add(current)
        
        return path if current == end else None
    
    def get_stops_along_path(self, path, stops, roads):
        """Get ordered list of stops along a path"""
        stop_sequence = []
        current_stop = None
        
        for road_id in path:
            road = roads[roads['road_id'] == road_id].iloc[0]
            if not stop_sequence:
                current_stop = road['start_stop']
                stop_sequence.append(current_stop)
            
            next_stop = road['end_stop'] if road['start_stop'] == current_stop else road['start_stop']
            stop_sequence.append(next_stop)
            current_stop = next_stop
        
        return stop_sequence
    
    def generate_traffic_data(self, roads):
        """Generate synthetic traffic patterns"""
        traffic_records = []
        
        time_windows = [
            ('07:00', '09:00'),  # Morning peak
            ('09:00', '11:00'),
            ('11:00', '13:00'),
            ('13:00', '15:00'),
            ('15:00', '17:00'),  # Evening peak
            ('17:00', '19:00'),
            ('19:00', '22:00'),  # Evening
            ('22:00', '07:00')   # Night
        ]
        
        day_types = ['weekday', 'weekend']
        
        for road in roads.itertuples():
            for day in day_types:
                for start, end in time_windows:
                    # Base traffic level
                    if '07:00' <= start < '09:00' or '15:00' <= start < '17:00':
                        base_level = self.rng.uniform(0.7, 0.9)  # High traffic
                    elif '22:00' <= start or start < '07:00':
                        base_level = self.rng.uniform(0.1, 0.3)  # Low traffic
                    else:
                        base_level = self.rng.uniform(0.4, 0.6)  # Medium traffic
                    
                    # Adjust for weekends
                    if day == 'weekend':
                        base_level = base_level * self.rng.uniform(0.7, 1.3)
                    
                    # Adjust for road properties
                    traffic_level = base_level * (4 / road.lanes) * (50 / road.speed_limit)
                    
                    traffic_records.append({
                        'road_id': road.road_id,
                        'time_window': f"{start}-{end}",
                        'day_type': day,
                        'traffic_level': np.clip(traffic_level, 0.1, 0.95),
                        'avg_speed_kmh': road.speed_limit * (1 - traffic_level * 0.7)
                    })
        
        return pd.DataFrame(traffic_records)
    
    def generate_fuel_consumption(self, routes, num_days=30):
        """Generate synthetic fuel consumption data"""
        vehicles = [f"VH{i:03d}" for i in range(1, self.config['num_vehicles']+1)]
        records = []
        
        for day in range(1, num_days+1):
            date = f"2023-01-{day:02d}"
            for route in routes.itertuples():
                for vehicle in self.rng.choice(vehicles, size=min(3, len(vehicles)), replace=False):
                    # Base consumption
                    base_consumption = route.length_km * 0.35  # 0.35 l/km baseline
                    
                    # Adjust for vehicle type
                    vehicle_factor = self.rng.uniform(0.9, 1.2)
                    
                    # Adjust for traffic (random time window)
                    traffic_factor = self.rng.uniform(0.8, 1.5)
                    
                    # Adjust for passenger load
                    passenger_count = self.rng.poisson(30)
                    passenger_factor = 1 + (passenger_count / 50) * 0.2
                    
                    # Calculate total consumption
                    total_consumption = base_consumption * vehicle_factor * traffic_factor * passenger_factor
                    
                    records.append({
                        'vehicle_id': vehicle,
                        'route_id': route.route_id,
                        'date': date,
                        'distance_km': route.length_km,
                        'fuel_liters': total_consumption,
                        'passenger_count': passenger_count
                    })
        
        return pd.DataFrame(records)
    
    def generate_all_data(self):
        """Generate all synthetic datasets"""
        print("Generating bus stops...")
        stops = self.generate_bus_stops()
        
        print("Generating road network...")
        roads = self.generate_road_network(stops)
        
        print("Generating bus routes...")
        routes = self.generate_bus_routes(stops, roads)
        
        print("Generating traffic data...")
        traffic = self.generate_traffic_data(roads)
        
        print("Generating fuel consumption data...")
        fuel = self.generate_fuel_consumption(routes)
        
        return {
            'bus_stops': stops,
            'road_network': roads,
            'bus_routes': routes,
            'traffic_data': traffic,
            'fuel_consumption': fuel
        }
    
    def save_data(self, data, output_dir='data/raw'):
        """Save generated data to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        data['bus_stops'].to_csv(f"{output_dir}/bus_stops.csv", index=False)
        data['road_network'].to_file(f"{output_dir}/road_network.geojson", driver='GeoJSON')
        data['bus_routes'].to_csv(f"{output_dir}/bus_routes.csv", index=False)
        data['traffic_data'].to_csv(f"{output_dir}/traffic_data.csv", index=False)
        data['fuel_consumption'].to_csv(f"{output_dir}/fuel_consumption.csv", index=False)

if __name__ == "__main__":
    generator = TransportDataGenerator()
    data = generator.generate_all_data()
    generator.save_data(data)