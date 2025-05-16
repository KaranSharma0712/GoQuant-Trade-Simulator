import time
import random
import threading
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from latency_benchmark import global_tracker, measure_latency, generate_latency_report

# Simulated components to benchmark
class MarketDataProcessor:
    @measure_latency("data_processing")
    def process_data(self, data_size):
        """Simulate processing market data with configurable workload."""
        # Simulate data processing with varying complexity
        start = time.perf_counter()
        
        # Simulate work based on data size
        data = np.random.random((data_size, 10))
        
        # Perform some calculations (adjust complexity as needed)
        for _ in range(3):
            data = np.dot(data, data.T)
            data = np.exp(data / data.max())
        
        elapsed = time.perf_counter() - start
        return elapsed * 1000  # Return in ms

class UIUpdater:
    @measure_latency("ui_update")
    def update(self, complexity):
        """Simulate UI updates with configurable complexity."""
        # Simulate UI rendering with varying complexity
        time.sleep(complexity / 1000)  # Convert to seconds
        return complexity

class SimulationLoop:
    @measure_latency("end_to_end")
    def run_cycle(self, data_size, ui_complexity):
        """Simulate a complete end-to-end simulation cycle."""
        # Create component instances
        data_processor = MarketDataProcessor()
        ui_updater = UIUpdater()
        
        # Run the simulated components
        data_processor.process_data(data_size)
        ui_updater.update(ui_complexity)
        
        # Add some randomness to simulate real-world variation
        time.sleep(random.uniform(0, 0.01))

def run_benchmark(cycles, data_sizes, ui_complexities, threads=1):
    """Run a complete benchmark with the specified parameters."""
    print(f"Running benchmark with {cycles} cycles, {threads} threads")
    print(f"Data sizes: {data_sizes}")
    print(f"UI complexities: {ui_complexities}")
    
    # Reset the global tracker
    global_tracker.reset()
    
    def worker():
        """Worker function for benchmark threads."""
        for _ in range(cycles // threads):
            # Randomly select a data size and UI complexity for this cycle
            data_size = random.choice(data_sizes)
            ui_complexity = random.choice(ui_complexities)
            
            # Run a simulation cycle
            sim = SimulationLoop()
            sim.run_cycle(data_size, ui_complexity)
    
    # Create and start the worker threads
    thread_list = []
    for _ in range(threads):
        t = threading.Thread(target=worker)
        thread_list.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in thread_list:
        t.join()
    
    # Print the benchmark report
    print("\n" + generate_latency_report())
    
    # Export the results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_tracker.export_to_csv(f"benchmark_results_{timestamp}.csv")
    
    # Generate visualizations
    generate_visualizations()

def generate_visualizations():
    """Generate visualizations of the benchmark results."""
    stats = global_tracker.get_all_statistics()
    
    # Create a figure with subplots
    fig, axs = plt.subplots(1, len(stats), figsize=(15, 5))
    
    # If there's only one component, axs is not an array
    if len(stats) == 1:
        axs = [axs]
    
    # Plot the data for each component
    for i, (component, metrics) in enumerate(stats.items()):
        # Extract the data for this component
        data = global_tracker.metrics[component]
        
        # Create a histogram
        axs[i].hist(data, bins=30, alpha=0.7, color='blue')
        axs[i].set_title(f"{component} Latency Distribution")
        axs[i].set_xlabel("Latency (ms)")
        axs[i].set_ylabel("Frequency")
        
        # Add vertical lines for key metrics
        axs[i].axvline(metrics['mean'], color='r', linestyle='--', label=f"Mean: {metrics['mean']:.2f}ms")
        axs[i].axvline(metrics['p95'], color='g', linestyle='--', label=f"P95: {metrics['p95']:.2f}ms")
        axs[i].axvline(metrics['p99'], color='orange', linestyle='--', label=f"P99: {metrics['p99']:.2f}ms")
        
        # Add a legend
        axs[i].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"latency_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Create a bar chart comparing the components
    plt.figure(figsize=(10, 6))
    components = list(stats.keys())
    means = [stats[c]['mean'] for c in components]
    p95s = [stats[c]['p95'] for c in components]
    p99s = [stats[c]['p99'] for c in components]
    
    x = np.arange(len(components))
    width = 0.25
    
    plt.bar(x - width, means, width, label='Mean', color='blue')
    plt.bar(x, p95s, width, label='P95', color='green')
    plt.bar(x + width, p99s, width, label='P99', color='orange')
    
    plt.xlabel('Component')
    plt.ylabel('Latency (ms)')
    plt.title('Latency Comparison Across Components')
    plt.xticks(x, components)
    plt.legend()
    
    plt.savefig(f"component_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run latency benchmarks for trading simulator components")
    parser.add_argument("--cycles", type=int, default=1000, help="Number of simulation cycles to run")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("--data-min", type=int, default=100, help="Minimum data size")
    parser.add_argument("--data-max", type=int, default=1000, help="Maximum data size")
    parser.add_argument("--data-steps", type=int, default=5, help="Number of data size steps")
    parser.add_argument("--ui-min", type=float, default=5.0, help="Minimum UI complexity (ms)")
    parser.add_argument("--ui-max", type=float, default=50.0, help="Maximum UI complexity (ms)")
    parser.add_argument("--ui-steps", type=int, default=5, help="Number of UI complexity steps")
    
    args = parser.parse_args()
    
    # Generate the test parameters
    data_sizes = np.linspace(args.data_min, args.data_max, args.data_steps, dtype=int).tolist()
    ui_complexities = np.linspace(args.ui_min, args.ui_max, args.ui_steps).tolist()
    
    # Run the benchmark
    run_benchmark(args.cycles, data_sizes, ui_complexities, args.threads)
