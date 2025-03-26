import numpy as np
import networkx as nx
import random
from tqdm import tqdm
import yaml
import pickle
from collections import defaultdict

class ACOTransportOptimizer:
    def __init__(self, graph, params_path='config/aco_params.yml'):
        self.graph = graph
        self.nodes = list(graph.nodes())
        
        # Load parameters
        with open(params_path) as f:
            self.params = yaml.safe_load(f)
        
        # Initialize pheromone matrix
        self.initialize_pheromones()
        
        # Best solution tracking
        self.best_solution = None
        self.best_score = float('inf')
    
    def initialize_pheromones(self):
        """Initialize pheromone matrix with initial value"""
        self.pheromones = defaultdict(dict)
        initial_pheromone = self.params['initial_pheromone']
        
        for u, v in self.graph.edges():
            self.pheromones[u][v] = initial_pheromone
            self.pheromones[v][u] = initial_pheromone
    
    def run(self):
        """Run the ACO optimization"""
        for iteration in tqdm(range(self.params['iterations'])):
            solutions = []
            
            # Let each ant find a solution
            for _ in range(self.params['num_ants']):
                solution = self.construct_solution()
                if solution:
                    solutions.append(solution)
            
            # Update pheromones
            self.update_pheromones(solutions)
            
            # Evaporate pheromones
            self.evaporate_pheromones()
        
        return self.best_solution, self.best_score
    
    def construct_solution(self):
        """Construct a solution (route) for a single ant"""
        # Select random terminals as start and end
        terminals = [n for n, attr in self.graph.nodes(data=True) if attr['is_terminal']]
        if len(terminals) < 2:
            return None
            
        start, end = random.sample(terminals, 2)
        current = start
        visited = set([start])
        path = []
        path_edges = []
        
        while current != end and len(visited) < len(self.nodes):
            # Get possible next nodes
            neighbors = list(self.graph.neighbors(current))
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if not unvisited_neighbors:
                # No unvisited neighbors, backtrack or terminate
                break
            
            # Select next node based on pheromone and heuristic
            next_node = self.select_next_node(current, unvisited_neighbors)
            
            # Add to path
            path.append(next_node)
            path_edges.append((current, next_node))
            visited.add(next_node)
            current = next_node
        
        if current == end:
            # Calculate solution score
            score = self.calculate_solution_score(path_edges)
            
            # Update best solution
            if score < self.best_score:
                self.best_score = score
                self.best_solution = {
                    'path': path,
                    'edges': path_edges,
                    'score': score
                }
            
            return {
                'path': path,
                'edges': path_edges,
                'score': score
            }
        return None
    
    def select_next_node(self, current, neighbors):
        """Select the next node based on pheromone and heuristic"""
        probabilities = []
        total = 0
        
        for neighbor in neighbors:
            # Get pheromone and edge data
            pheromone = self.pheromones[current][neighbor]
            edge_data = self.graph[current][neighbor]
            
            # Calculate heuristic (desirability)
            heuristic = self.calculate_heuristic(edge_data)
            
            # Combined value
            value = (pheromone ** self.params['alpha']) * (heuristic ** self.params['beta'])
            probabilities.append(value)
            total += value
        
        if total == 0:
            return random.choice(neighbors)
        
        # Normalize probabilities
        probabilities = [p/total for p in probabilities]
        
        # Select next node based on probabilities
        return np.random.choice(neighbors, p=probabilities)
    
    def calculate_heuristic(self, edge_data):
        """Calculate heuristic value for an edge"""
        # Inverse of normalized weight (lower weight = more desirable)
        return 1 / (edge_data['normalized_weight'] + 1e-10)
    
    def calculate_solution_score(self, edges):
        """Calculate the total score for a solution"""
        total = 0
        for u, v in edges:
            total += self.graph[u][v]['weight']
        return total
    
    def update_pheromones(self, solutions):
        """Update pheromones based on ant solutions"""
        # First, evaporate pheromones slightly
        for u in self.pheromones:
            for v in self.pheromones[u]:
                self.pheromones[u][v] *= (1 - self.params['evaporation_rate'])
        
        # Then add pheromones based on solutions
        for solution in solutions:
            if solution:
                delta_pheromone = self.params['q'] / solution['score']
                
                for u, v in solution['edges']:
                    self.pheromones[u][v] += delta_pheromone
                    self.pheromones[v][u] += delta_pheromone
    
    def evaporate_pheromones(self):
        """Global pheromone evaporation"""
        min_pheromone = self.params['min_pheromone']
        
        for u in self.pheromones:
            for v in self.pheromones[u]:
                self.pheromones[u][v] = max(
                    self.pheromones[u][v] * (1 - self.params['global_evaporation']),
                    min_pheromone
                )
    
    def save_results(self, output_dir='data/processed'):
        """Save optimization results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'best_solution': self.best_solution,
            'best_score': self.best_score,
            'parameters': self.params
        }
        
        with open(f"{output_dir}/aco_results.pkl", 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    # Load processed data
    with open('data/processed/transport_graph.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    # Run optimization
    optimizer = ACOTransportOptimizer(graph)
    best_solution, best_score = optimizer.run()
    
    print(f"Best solution score: {best_score}")
    print(f"Best path: {best_solution['path']}")
    
    # Save results
    optimizer.save_results()