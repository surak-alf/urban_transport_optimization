import argparse
from src.data_processing.data_generator import TransportDataGenerator
from src.data_processing.preprocessor import TransportDataPreprocessor
from src.aco.optimizer import ACOTransportOptimizer
import pickle

def generate_data():
    print("Generating synthetic data...")
    generator = TransportDataGenerator()
    data = generator.generate_all_data()
    generator.save_data(data)
    print("Data generation complete.")

def preprocess_data():
    print("Preprocessing data...")
    preprocessor = TransportDataPreprocessor()
    graph, route_stats = preprocessor.run_pipeline()
    print("Data preprocessing complete.")

def run_optimization():
    print("Running ACO optimization...")
    with open('data/processed/transport_graph.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    optimizer = ACOTransportOptimizer(graph)
    best_solution, best_score = optimizer.run()
    optimizer.save_results()
    
    print(f"Optimization complete. Best score: {best_score}")
    print(f"Best path: {best_solution['path']}")

def run_dashboard():
    from app.dashboard import app
    print("Starting dashboard...")
    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Urban Transport Optimization Pipeline")
    parser.add_argument('--all', action='store_true', help="Run complete pipeline")
    parser.add_argument('--generate', action='store_true', help="Generate synthetic data")
    parser.add_argument('--preprocess', action='store_true', help="Preprocess data")
    parser.add_argument('--optimize', action='store_true', help="Run ACO optimization")
    parser.add_argument('--dashboard', action='store_true', help="Launch dashboard")
    
    args = parser.parse_args()
    
    if args.all or args.generate:
        generate_data()
    
    if args.all or args.preprocess:
        preprocess_data()
    
    if args.all or args.optimize:
        run_optimization()
    
    if args.all or args.dashboard:
        run_dashboard()