import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import random
from datetime import time
import yaml
from collections import defaultdict, deque
from sklearn.neighbors import NearestNeighbors
import os
from tqdm import tqdm

class TransportDataGenerator:
    def __init__(self, config_path='config/data_generation.yml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.rng = np.random.default_rng(self.config['seed'])
        
    def generate_bus_stops(self):
        """Generate synthetic bus stop data with geographic distribution"""
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
                'stop_name': f"Stop {i+1}",
                'latitude': lat,
                'longitude': lon,
                'is_terminal': self.rng.random() < 0.15,  # 15% chance of being terminal
                'elevation': max(0, self.rng.normal(50, 20))  # Mean elevation 50m, no negative
            })
        
        df = pd.DataFrame(stops)
        
        # Ensure at least 2 terminals
        if df['is_terminal'].sum() < 2:
            df.loc[df.sample(2).index, 'is_terminal'] = True
            
        return df
    
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
                        'road_id': f"RD{road_id:04d}",
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
    
    def find_path(self, start, end, stops, roads):
        """Reliable path finding between two stops using BFS"""
        # Create adjacency list
        graph = defaultdict(list)
        for _, road in roads.iterrows():
            graph[road['start_stop']].append(road['end_stop'])
            graph[road['end_stop']].append(road['start_stop'])
        
        # BFS implementation
        queue = deque()
        queue.append((start, [start]))
        
        visited = set()
        visited.add(start)
        
        while queue:
            current, path = queue.popleft()
            
            if current == end:
                return path
                
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
    
    def generate_bus_routes(self, stops, roads):
        """Generate synthetic bus routes with guaranteed valid paths"""
        num_routes = self.config['num_routes']
        routes = []
        
        terminals = stops[stops['is_terminal']]['stop_id'].tolist()
        
        # Ensure we have enough terminals
        if len(terminals) < 2:
            terminals = stops['stop_id'].sample(min(10, len(stops))).tolist()
        
        route_count = 0
        attempts = 0
        max_attempts = num_routes * 3  # Prevent infinite loops
        
        with tqdm(total=num_routes, desc="Generating routes") as pbar:
            while route_count < num_routes and attempts < max_attempts:
                attempts += 1
                
                # Select 2 distinct terminals
                start, end = self.rng.choice(terminals, 2, replace=False)
                
                # Find path between terminals
                path = self.find_path(start, end, stops, roads)
                
                if path and len(path) >= 3:  # Ensure reasonable route length
                    # Get all road segments in this path
                    path_roads = []
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        road = roads[((roads['start_stop'] == u) & (roads['end_stop'] == v)) | 
                                  ((roads['start_stop'] == v) & (roads['end_stop'] == u))]
                        if not road.empty:
                            path_roads.append(road.iloc[0]['road_id'])
                    
                    if path_roads:
                        # Calculate route length
                        length = roads[roads['road_id'].isin(path_roads)]['length_km'].sum()
                        
                        routes.append({
                            'route_id': f"R{route_count+1:02d}",
                            'route_name': f"Route {route_count+1}",
                            'start_stop': start,
                            'end_stop': end,
                            'stops': path,
                            'length_km': length,
                            'avg_speed_kmh': self.rng.normal(25, 5)
                        })
                        route_count += 1
                        pbar.update(1)
        
        return pd.DataFrame(routes)
    
    def generate_traffic_data(self, roads):
        """Generate realistic traffic patterns"""
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
        
        for _, road in roads.iterrows():
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
                    traffic_level = base_level * (4 / road['lanes']) * (50 / road['speed_limit'])
                    traffic_level = np.clip(traffic_level, 0.1, 0.95)
                    
                    avg_speed = road['speed_limit'] * (1 - traffic_level * 0.7)
                    
                    traffic_records.append({
                        'road_id': road['road_id'],
                        'time_window': f"{start}-{end}",
                        'day_type': day,
                        'traffic_level': traffic_level,
                        'avg_speed_kmh': avg_speed
                    })
        
        return pd.DataFrame(traffic_records)
    
    def generate_fuel_consumption(self, routes, num_days=30):
        """Generate realistic fuel consumption data"""
        vehicles = [f"VH{i:03d}" for i in range(1, self.config['num_vehicles']+1)]
        records = []
        
        for day in range(1, num_days+1):
            date = f"2023-01-{day:02d}"
            for _, route in routes.iterrows():
                # Assign 1-3 vehicles per route per day
                for vehicle in self.rng.choice(vehicles, size=min(3, len(vehicles)), replace=False):
                    # Base consumption
                    base_consumption = route['length_km'] * 0.35  # 0.35 l/km baseline
                    
                    # Adjust for vehicle type (older vehicles less efficient)
                    vehicle_factor = 0.8 + 0.4 * (int(vehicle[2:]) / len(vehicles))
                    
                    # Adjust for traffic (random time window)
                    traffic_factor = self.rng.uniform(0.8, 1.5)
                    
                    # Adjust for passenger load
                    passenger_count = int(self.rng.poisson(30))
                    passenger_factor = 1 + (passenger_count / 50) * 0.2
                    
                    # Calculate total consumption with some noise
                    total_consumption = base_consumption * vehicle_factor * traffic_factor * passenger_factor
                    total_consumption *= self.rng.uniform(0.95, 1.05)  # Add small random variation
                    
                    records.append({
                        'vehicle_id': vehicle,
                        'route_id': route['route_id'],
                        'date': date,
                        'distance_km': route['length_km'] * self.rng.uniform(0.98, 1.02),  # Small route variation
                        'fuel_liters': total_consumption,
                        'passenger_count': passenger_count
                    })
        
        return pd.DataFrame(records)
    
    def validate_data(self, data):
        """Validate all generated datasets"""
        required = {
            'bus_stops': 10,       # At least 10 stops
            'road_network': 20,     # At least 20 road segments
            'bus_routes': 3,        # At least 3 routes
            'traffic_data': 100,    # At least 100 traffic records
            'fuel_consumption': 30  # At least 30 fuel records
        }
        
        for name, df in data.items():
            assert not df.empty, f"Empty DataFrame: {name}"
            assert len(df) >= required[name], f"Insufficient data in {name}: {len(df)} < {required[name]}"
            
            # Route-specific validation
            if name == 'bus_routes':
                for _, route in df.iterrows():
                    assert len(route['stops']) >= 3, f"Route {route['route_id']} has too few stops"
                    assert route['length_km'] > 0, f"Route {route['route_id']} has invalid length"
        
        print("All datasets validated successfully")
    
    def save_data(self, data, output_dir='data/raw'):
        """Save generated data to files with verification"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data.items():
            if name == 'road_network':
                path = f"{output_dir}/{name}.geojson"
                df.to_file(path, driver='GeoJSON')
            else:
                path = f"{output_dir}/{name}.csv"
                df.to_csv(path, index=False)
            
            # Verify file was created and has content
            assert os.path.exists(path), f"Failed to create {path}"
            assert os.path.getsize(path) > 100, f"File too small: {path}"
            print(f"Saved {len(df)} records to {path}")
    
    def generate_all_data(self):
        """Generate and validate all synthetic datasets"""
        print("Generating synthetic transportation data...")
        
        data = {}
        
        print("\n1. Generating bus stops...")
        data['bus_stops'] = self.generate_bus_stops()
        
        print("\n2. Generating road network...")
        data['road_network'] = self.generate_road_network(data['bus_stops'])
        
        print("\n3. Generating bus routes...")
        data['bus_routes'] = self.generate_bus_routes(data['bus_stops'], data['road_network'])
        
        print("\n4. Generating traffic data...")
        data['traffic_data'] = self.generate_traffic_data(data['road_network'])
        
        print("\n5. Generating fuel consumption data...")
        data['fuel_consumption'] = self.generate_fuel_consumption(data['bus_routes'])
        
        print("\nValidating data...")
        self.validate_data(data)
        
        print("\nSaving data...")
        self.save_data(data)
        
        print("\nData generation complete!")
        return data

if __name__ == "__main__":
    generator = TransportDataGenerator()
    data = generator.generate_all_data()