import time
import statistics
import functools
import threading
import queue
import csv
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional


class LatencyTracker:    
    def __init__(self, log_to_file: bool = True, log_path: str = "./latency_logs"):
        self.metrics = {}
        self.log_to_file = log_to_file
        self.log_path = log_path
        self.lock = threading.Lock()
        
        # Create timestamp for this benchmark session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.log_to_file:
            import os
            os.makedirs(self.log_path, exist_ok=True)
    
    def record_latency(self, component: str, latency_ms: float) -> None:
      
        with self.lock:
            if component not in self.metrics:
                self.metrics[component] = []
            
            self.metrics[component].append(latency_ms)
            
            # Log to file if enabled
            if self.log_to_file:
                with open(f"{self.log_path}/{component}_{self.session_id}.csv", "a") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    f.write(f"{timestamp},{latency_ms}\n")
    
    def get_statistics(self, component: str) -> Dict[str, float]:
      
        with self.lock:
            if component not in self.metrics or not self.metrics[component]:
                return {
                    "min": 0,
                    "max": 0,
                    "mean": 0,
                    "median": 0,
                    "p95": 0,
                    "p99": 0,
                    "samples": 0
                }
            
            data = sorted(self.metrics[component])
            
            return {
                "min": min(data),
                "max": max(data),
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "p95": data[int(0.95 * len(data))],
                "p99": data[int(0.99 * len(data))],
                "samples": len(data)
            }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for component in self.metrics:
            result[component] = self.get_statistics(component)
        return result
    
    def reset(self, component: Optional[str] = None) -> None:
        
        with self.lock:
            if component is None:
                self.metrics = {}
            elif component in self.metrics:
                self.metrics[component] = []
    
    def export_to_csv(self, filename: Optional[str] = None) -> None:
       
        if filename is None:
            filename = f"{self.log_path}/latency_summary_{self.session_id}.csv"
        
        all_stats = self.get_all_statistics()
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Component', 'Min (ms)', 'Max (ms)', 'Mean (ms)', 
                           'Median (ms)', 'P95 (ms)', 'P99 (ms)', 'Samples'])
            
            for component, stats in all_stats.items():
                writer.writerow([
                    component,
                    f"{stats['min']:.3f}",
                    f"{stats['max']:.3f}",
                    f"{stats['mean']:.3f}",
                    f"{stats['median']:.3f}",
                    f"{stats['p95']:.3f}",
                    f"{stats['p99']:.3f}",
                    stats['samples']
                ])


class LatencyDecorator:
    
    def __init__(self, tracker: LatencyTracker, component: str):

        self.tracker = tracker
        self.component = component
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Convert to milliseconds
            latency_ms = (end_time - start_time) * 1000
            self.tracker.record_latency(self.component, latency_ms)
            
            return result
        return wrapper


# Global tracker instance for convenience
global_tracker = LatencyTracker()


def measure_latency(component: str):
    
    return LatencyDecorator(global_tracker, component)


class UILatencyMonitor:
    
    def __init__(self, tracker: LatencyTracker):
       
        self.tracker = tracker
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False
        self.thread = None
    
    def start(self) -> None:
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_thread, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _monitor_thread(self) -> None:
        while self.running:
            try:
                # Wait for update requests
                request_id, timestamp = self.request_queue.get(timeout=0.1)
                
                # Wait for the corresponding response
                while self.running:
                    try:
                        response_id, response_time = self.response_queue.get(timeout=0.1)
                        if response_id == request_id:
                            latency_ms = (response_time - timestamp) * 1000
                            self.tracker.record_latency("ui_update", latency_ms)
                            break
                    except queue.Empty:
                        continue
            
            except queue.Empty:
                continue
    
    def request_update(self) -> int:
       
        request_id = id(time.time())
        self.request_queue.put((request_id, time.perf_counter()))
        return request_id
    
    def record_update_complete(self, request_id: int) -> None:
       
        self.response_queue.put((request_id, time.perf_counter()))


class EndToEndLatencyMonitor:
    
    def __init__(self, tracker: LatencyTracker):
        
        self.tracker = tracker
        self.start_times = {}
    
    def start_cycle(self, cycle_id: Any) -> None:
        
        self.start_times[cycle_id] = time.perf_counter()
    
    def end_cycle(self, cycle_id: Any) -> None:
       
        if cycle_id in self.start_times:
            end_time = time.perf_counter()
            latency_ms = (end_time - self.start_times[cycle_id]) * 1000
            self.tracker.record_latency("end_to_end", latency_ms)
            del self.start_times[cycle_id]


def generate_latency_report(tracker: LatencyTracker = global_tracker) -> str:
    
    stats = tracker.get_all_statistics()
    
    report = "=== LATENCY BENCHMARK REPORT ===\n\n"
    
    if not stats:
        report += "No latency data has been collected.\n"
        return report
    
    for component, metrics in stats.items():
        report += f"Component: {component}\n"
        report += f"  Samples: {metrics['samples']}\n"
        report += f"  Min: {metrics['min']:.3f} ms\n"
        report += f"  Max: {metrics['max']:.3f} ms\n"
        report += f"  Mean: {metrics['mean']:.3f} ms\n"
        report += f"  Median: {metrics['median']:.3f} ms\n"
        report += f"  95th percentile: {metrics['p95']:.3f} ms\n"
        report += f"  99th percentile: {metrics['p99']:.3f} ms\n\n"
    
    return report