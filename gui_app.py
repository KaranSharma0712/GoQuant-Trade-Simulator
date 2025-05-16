import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timezone
import asyncio
import websockets
import json
import threading
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import the MarketModels class from models.py
from models import MarketModels

# Import latency benchmarking modules
from latency_benchmark import (
    LatencyTracker, 
    measure_latency, 
    UILatencyMonitor, 
    EndToEndLatencyMonitor, 
    generate_latency_report
)

ORDERBOOK_DEPTH = 10  # Limit to top 10 levels for performance

class TradeSimulatorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Crypto Trade Simulator")
        self.root.geometry("1000x700")  # Increased size to accommodate latency metrics
        
        # Create a latency tracker instance
        self.latency_tracker = LatencyTracker(log_to_file=True)
        
        # Create UI latency monitor
        self.ui_monitor = UILatencyMonitor(self.latency_tracker)
        self.ui_monitor.start()
        
        # Create end-to-end latency monitor
        self.e2e_monitor = EndToEndLatencyMonitor(self.latency_tracker)
        
        # Initialize orderbook state
        self.orderbook = {
            "timestamp": "",
            "asks": [],
            "bids": []
        }
        
        # Initialize MarketModels instance for calculations
        self.market_models = MarketModels(buffer_size=100)
        
        # Initialize current data
        self.current_data = {}
        
        # Initialize latency data for visualization
        self.latency_history = {
            "data_processing": deque(maxlen=100),
            "ui_update": deque(maxlen=100),
            "end_to_end": deque(maxlen=100)
        }
        
        # Setup UI components
        self.create_notebook()
        
        # Start the WebSocket listener in asyncio loop
        self.websocket_running = False
        self.update_interval = 500
        self.start_async_loop()
        
        # Set GUI update interval (in milliseconds)
        self.update_interval = 500

    def create_notebook(self):
        """Create a tabbed interface for the application"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create the main simulation tab
        self.sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_frame, text="Simulation")
        
        # Create the latency metrics tab
        self.latency_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.latency_frame, text="Latency Metrics")
        
        # Setup content for each tab
        self.setup_simulation_tab()
        self.setup_latency_tab()

    def setup_simulation_tab(self):
        """Setup the main simulation tab"""
        # Create frames for left and right panels
        left_frame = ttk.Frame(self.sim_frame, padding="10")
        left_frame.grid(row=0, column=0, sticky="nsw")
        
        right_frame = ttk.Frame(self.sim_frame, padding="10")
        right_frame.grid(row=0, column=1, sticky="nse")
        
        # Setup the left and right panels
        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)
        
        # Add a recalculate button
        ttk.Button(self.sim_frame, text="Recalculate", command=self.calculate_trade_metrics).grid(
            row=1, column=0, columnspan=2, pady=10)

    def setup_latency_tab(self):
        """Setup the latency metrics tab"""
        # Control frame for benchmarking options
        control_frame = ttk.Frame(self.latency_frame, padding="10")
        control_frame.pack(fill='x', expand=False)
        
        # Latency visualization frame
        viz_frame = ttk.Frame(self.latency_frame, padding="10")
        viz_frame.pack(fill='both', expand=True)
        
        # Latency metrics frame 
        metrics_frame = ttk.LabelFrame(control_frame, text="Latency Metrics", padding="10")
        metrics_frame.grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        
        # Create labels to display current latency metrics
        ttk.Label(metrics_frame, text="Data Processing:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.data_latency_label = ttk.Label(metrics_frame, text="--- ms")
        self.data_latency_label.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(metrics_frame, text="UI Update:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.ui_latency_label = ttk.Label(metrics_frame, text="--- ms")
        self.ui_latency_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(metrics_frame, text="End-to-End:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.e2e_latency_label = ttk.Label(metrics_frame, text="--- ms")
        self.e2e_latency_label.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        # Benchmark controls frame
        benchmark_frame = ttk.LabelFrame(control_frame, text="Benchmark Controls", padding="10")
        benchmark_frame.grid(row=0, column=1, sticky="nw", padx=5, pady=5)
        
        ttk.Button(benchmark_frame, text="Run Benchmark", command=self.run_benchmark).grid(
            row=0, column=0, padx=5, pady=5)
        
        ttk.Button(benchmark_frame, text="Generate Report", command=self.show_latency_report).grid(
            row=0, column=1, padx=5, pady=5)
        
        ttk.Button(benchmark_frame, text="Reset Metrics", command=self.reset_latency_metrics).grid(
            row=0, column=2, padx=5, pady=5)
        
        # Create matplotlib figure for latency visualization
        self.setup_latency_visualization(viz_frame)

    def setup_latency_visualization(self, parent_frame):
        """Create matplotlib visualizations for latency metrics"""
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(2, 1, figsize=(8, 6))
        
        # Time series plot
        self.ax[0].set_title("Latency Over Time")
        self.ax[0].set_xlabel("Sample")
        self.ax[0].set_ylabel("Latency (ms)")
        self.ax[0].grid(True)
        
        # Line objects for time series
        self.lines = {
            "data_processing": self.ax[0].plot([], [], 'b-', label="Data Processing")[0],
            "ui_update": self.ax[0].plot([], [], 'g-', label="UI Update")[0],
            "end_to_end": self.ax[0].plot([], [], 'r-', label="End-to-End")[0]
        }
        self.ax[0].legend()
        
        # Distribution plot
        self.ax[1].set_title("Latency Distribution")
        self.ax[1].set_xlabel("Latency (ms)")
        self.ax[1].set_ylabel("Frequency")
        self.hist_containers = {}
        
        # Create canvas and place it
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initial plot setup
        self.update_latency_plots()

    def setup_left_panel(self, left_frame):
        """Setup the left panel of the simulation tab"""
        ttk.Label(left_frame, text="📥 Input Parameters", font=("Arial", 14)).grid(row=0, column=0, pady=10, columnspan=3)

        ttk.Label(left_frame, text="Exchange:").grid(row=1, column=0, sticky="w")
        ttk.Label(left_frame, text="OKX").grid(row=1, column=1, sticky="w")

        ttk.Label(left_frame, text="Asset:").grid(row=2, column=0, sticky="w")
        self.asset_var = tk.StringVar(value="BTC-USDT-SWAP")
        self.asset_entry = ttk.Entry(left_frame, textvariable=self.asset_var)
        self.asset_entry.grid(row=2, column=1, sticky="w")
        
        # Connect button
        ttk.Button(left_frame, text="Connect", command=self.reconnect_websocket).grid(
            row=2, column=2, padx=5)

        ttk.Label(left_frame, text="Order Type:").grid(row=3, column=0, sticky="w")
        self.order_type_var = tk.StringVar(value="Market")
        self.order_type_combo = ttk.Combobox(left_frame, textvariable=self.order_type_var, 
                                             values=["Market", "Limit"])
        self.order_type_combo.grid(row=3, column=1, sticky="w")
        # Bind change event to recalculate
        self.order_type_combo.bind("<<ComboboxSelected>>", lambda e: self.calculate_trade_metrics())

        ttk.Label(left_frame, text="Order Side:").grid(row=4, column=0, sticky="w")
        self.side_var = tk.StringVar(value="Buy")
        self.side_combo = ttk.Combobox(left_frame, textvariable=self.side_var, 
                                       values=["Buy", "Sell"])
        self.side_combo.grid(row=4, column=1, sticky="w")
        # Bind change event to recalculate
        self.side_combo.bind("<<ComboboxSelected>>", lambda e: self.calculate_trade_metrics())

        ttk.Label(left_frame, text="Quantity (USD):").grid(row=5, column=0, sticky="w")
        self.quantity_var = tk.StringVar(value="100")
        self.quantity_entry = ttk.Entry(left_frame, textvariable=self.quantity_var)
        self.quantity_entry.grid(row=5, column=1, sticky="w")
        # Bind change event to recalculate after user stops typing
        self.quantity_entry.bind("<FocusOut>", lambda e: self.calculate_trade_metrics())
        self.quantity_entry.bind("<Return>", lambda e: self.calculate_trade_metrics())

        ttk.Label(left_frame, text="Fee Tier:").grid(row=6, column=0, sticky="w")
        self.fee_var = tk.StringVar(value="Tier 1")
        self.fee_combo = ttk.Combobox(left_frame, textvariable=self.fee_var, 
                                      values=["Tier 1 (0.08%)", "Tier 2 (0.06%)", "Tier 3 (0.04%)"])
        self.fee_combo.grid(row=6, column=1, sticky="w")
        # Bind change event to recalculate
        self.fee_combo.bind("<<ComboboxSelected>>", lambda e: self.calculate_trade_metrics())
        
        # Volatility override option
        ttk.Label(left_frame, text="Volatility Override (%):").grid(row=7, column=0, sticky="w")
        self.volatility_var = tk.StringVar(value="")
        self.volatility_entry = ttk.Entry(left_frame, textvariable=self.volatility_var)
        self.volatility_entry.grid(row=7, column=1, sticky="w")
        ttk.Label(left_frame, text="(Leave blank for auto)").grid(row=7, column=2, sticky="w")
        # Bind change event to recalculate
        self.volatility_entry.bind("<FocusOut>", lambda e: self.calculate_trade_metrics())
        self.volatility_entry.bind("<Return>", lambda e: self.calculate_trade_metrics())
        
        # Connection status indicator
        ttk.Label(left_frame, text="Connection:").grid(row=8, column=0, sticky="w", pady=(20, 0))
        self.connection_status = ttk.Label(left_frame, text="⚠️ Connected", foreground="red")
        self.connection_status.grid(row=8, column=1, sticky="w", pady=(20, 0))
        
        # Market statistics frame
        market_stats_frame = ttk.LabelFrame(left_frame, text="Market Statistics")
        market_stats_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(20, 0))
        
        # Market statistics labels
        ttk.Label(market_stats_frame, text="Current Volatility (%):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.volatility_label = ttk.Label(market_stats_frame, text="---")
        self.volatility_label.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(market_stats_frame, text="Avg Volume:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.volume_label = ttk.Label(market_stats_frame, text="---")
        self.volume_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Update interval control
        ttk.Label(left_frame, text="Update Speed (ms):").grid(row=10, column=0, sticky="w", pady=(10, 0))
        self.update_speed_var = tk.StringVar(value="500")
        self.update_speed_entry = ttk.Entry(left_frame, textvariable=self.update_speed_var, width=8)
        self.update_speed_entry.grid(row=10, column=1, sticky="w", pady=(10, 0))
        ttk.Button(left_frame, text="Apply", command=self.update_refresh_rate).grid(
            row=10, column=2, padx=5, pady=(10, 0))

    def setup_right_panel(self, right_frame):
        """Setup the right panel of the simulation tab"""
        ttk.Label(right_frame, text="📊 Output Parameters", font=("Arial", 14)).grid(
            row=0, column=0, columnspan=2, pady=10)

        self.labels = {}

        fields = [
            "Top Bid", "Top Ask", "Spread",
            "Expected Slippage", "Estimated Fees",
            "Market Impact", "Net Cost/Proceeds",
            "Maker/Taker Ratio", "Maker Fee", "Taker Fee",
            "Fee Rate", "Latency (ms)"
        ]

        for i, field in enumerate(fields, 1):
            ttk.Label(right_frame, text=f"{field}:").grid(row=i, column=0, sticky="w", padx=(0, 10))
            self.labels[field] = ttk.Label(right_frame, text="---")
            self.labels[field].grid(row=i, column=1, sticky="w")
            
        # Order book display
        ttk.Label(right_frame, text="Order Book Preview:", font=("Arial", 12)).grid(
            row=len(fields)+1, column=0, columnspan=2, pady=(20, 5), sticky="w")
            
        self.orderbook_frame = ttk.Frame(right_frame)
        self.orderbook_frame.grid(row=len(fields)+2, column=0, columnspan=2, sticky="w")
        
        # Headers
        ttk.Label(self.orderbook_frame, text="Price", width=10).grid(row=0, column=0)
        ttk.Label(self.orderbook_frame, text="Size", width=10).grid(row=0, column=1)
        ttk.Label(self.orderbook_frame, text="Total", width=10).grid(row=0, column=2)
        
        # Placeholders for ask and bid data
        self.ask_labels = []
        self.bid_labels = []
        
        # Create labels for 5 asks (red)
        for i in range(5):
            row_labels = []
            for j in range(3):
                label = ttk.Label(self.orderbook_frame, text="---", foreground="red")
                label.grid(row=i+1, column=j)
                row_labels.append(label)
            self.ask_labels.append(row_labels)
        
        # Spread indicator
        ttk.Label(self.orderbook_frame, text="-----", width=30).grid(
            row=6, column=0, columnspan=3)
        
        # Create labels for 5 bids (green)
        for i in range(5):
            row_labels = []
            for j in range(3):
                label = ttk.Label(self.orderbook_frame, text="---", foreground="green")
                label.grid(row=i+7, column=j)
                row_labels.append(label)
            self.bid_labels.append(row_labels)

    @measure_latency("ui_update")
    def update_gui(self):
        """Update the GUI with current data"""
        # Start end-to-end cycle measurement
        cycle_id = time.time()
        self.e2e_monitor.start_cycle(cycle_id)
        
        # Request UI update tracking
        ui_request_id = self.ui_monitor.request_update()
        
        if self.current_data:
            self.labels["Top Bid"].config(text=self.current_data.get("bid", "---"))
            self.labels["Top Ask"].config(text=self.current_data.get("ask", "---"))
            
            # Format spread to 2 decimal places
            spread = self.current_data.get("spread", 0)
            self.labels["Spread"].config(text=f"{spread:.2f}")
            
            self.labels["Latency (ms)"].config(text=f"{self.current_data.get('latency', 0):.1f}")
            
            # Update orderbook display
            self.update_orderbook_display()
            
            # Update volatility display
            self.volatility_label.config(text=f"{self.market_models.volatility:.4f}")
            
            # Update volume display if available
            if self.market_models.volume_history:
                avg_volume = sum(self.market_models.volume_history) / len(self.market_models.volume_history)
                self.volume_label.config(text=f"{avg_volume:.2f}")
            
            # Calculate trade metrics
            self.calculate_trade_metrics()
        
        # Update latency metrics display
        self.update_latency_metrics_display()
        
        # Signal that UI update is complete
        self.ui_monitor.record_update_complete(ui_request_id)
        
        # End end-to-end cycle measurement
        self.e2e_monitor.end_cycle(cycle_id)
        
        # Schedule the next update using the dynamic interval
        self.root.after(self.update_interval, self.update_gui)
    
    def update_latency_metrics_display(self):
        """Update the display of latency metrics"""
        # Update current latency display
        data_stats = self.latency_tracker.get_statistics("data_processing")
        ui_stats = self.latency_tracker.get_statistics("ui_update")
        e2e_stats = self.latency_tracker.get_statistics("end_to_end")
        
        # Update labels
        if data_stats["samples"] > 0:
            self.data_latency_label.config(text=f"{data_stats['mean']:.2f} ms")
        
        if ui_stats["samples"] > 0:
            self.ui_latency_label.config(text=f"{ui_stats['mean']:.2f} ms")
            
        if e2e_stats["samples"] > 0:
            self.e2e_latency_label.config(text=f"{e2e_stats['mean']:.2f} ms")
        
        # Update the visualization plots (less frequently to avoid performance impact)
        if int(time.time()) % 2 == 0:  # Update every 2 seconds
            self.update_latency_plots()
    
    def update_latency_plots(self):
        """Update the latency visualization plots"""
        # Get the latest data
        all_data = {}
        for component in ['data_processing', 'ui_update', 'end_to_end']:
            if component in self.latency_tracker.metrics:
                all_data[component] = self.latency_tracker.metrics[component][-100:]  # Last 100 points
            else:
                all_data[component] = []
        
        # Update time series plots
        for component, line in self.lines.items():
            data = all_data.get(component, [])
            if data:
                x = list(range(len(data)))
                line.set_data(x, data)
                
        # Adjust axis limits
        all_values = []
        for data in all_data.values():
            all_values.extend(data)
            
        if all_values:
            self.ax[0].set_xlim(0, max(len(d) for d in all_data.values() if d))
            self.ax[0].set_ylim(0, max(all_values) * 1.1)
        
        # Update histograms
        self.ax[1].clear()
        self.ax[1].set_title("Latency Distribution")
        self.ax[1].set_xlabel("Latency (ms)")
        self.ax[1].set_ylabel("Frequency")
        
        colors = {'data_processing': 'blue', 'ui_update': 'green', 'end_to_end': 'red'}
        labels = {'data_processing': 'Data Processing', 'ui_update': 'UI Update', 'end_to_end': 'End-to-End'}
        
        for component, data in all_data.items():
            if data:
                self.ax[1].hist(data, bins=20, alpha=0.5, color=colors[component], label=labels[component])
        
        self.ax[1].legend()
        
        # Redraw the canvas
        self.canvas.draw_idle()
    
    @measure_latency("data_processing")
    def update_orderbook_display(self):
        """Update the orderbook display with current data"""
        if not self.orderbook.get("asks") or not self.orderbook.get("bids"):
            return
        
        # Update ask labels (reverse to show highest ask at the bottom)
        asks = self.orderbook["asks"][:5]
        asks.reverse()  # Display highest ask at the bottom
        
        for i, (price, size) in enumerate(asks):
            if i < len(self.ask_labels):
                price_float = float(price)
                size_float = float(size)
                total = price_float * size_float
                
                self.ask_labels[i][0].config(text=f"{price_float:.2f}")
                self.ask_labels[i][1].config(text=f"{size_float:.4f}")
                self.ask_labels[i][2].config(text=f"{total:.2f}")
        
        # Update bid labels
        bids = self.orderbook["bids"][:5]
        
        for i, (price, size) in enumerate(bids):
            if i < len(self.bid_labels):
                price_float = float(price)
                size_float = float(size)
                total = price_float * size_float
                
                self.bid_labels[i][0].config(text=f"{price_float:.2f}")
                self.bid_labels[i][1].config(text=f"{size_float:.4f}")
                self.bid_labels[i][2].config(text=f"{total:.2f}")
    
    @measure_latency("data_processing")
    def calculate_trade_metrics(self):
        """Calculate and display trade metrics based on current orderbook and MarketModels"""
        if not self.orderbook.get("asks") or not self.orderbook.get("bids"):
            return
            
        try:
            # Get quantity from input
            quantity = float(self.quantity_var.get())
            
            # Get order type and side
            order_type = self.order_type_var.get()
            side = self.side_var.get()
            
            # Check if volatility override is provided
            volatility_override = None
            if self.volatility_var.get():
                try:
                    volatility_override = float(self.volatility_var.get())
                except ValueError:
                    pass
            
            # Get top bid and ask prices
            top_bid = float(self.orderbook["bids"][0][0])
            top_ask = float(self.orderbook["asks"][0][0])
            spread = top_ask - top_bid
            
            # Use MarketModels to calculate slippage
            slippage = self.market_models.estimate_slippage(quantity, side, self.orderbook)
            
            # Get the mid price for market impact calculation
            mid_price = (top_ask + top_bid) / 2
            
            # Calculate market impact using the model
            market_impact = self.market_models.calculate_market_impact(
                quantity, side, mid_price, volatility_override)
            
            # Predict maker/taker ratio - overriding for market orders which are always taker
            if order_type.lower() == "market":
                maker_ratio = 0.0  # Market orders are always taker orders
            else:
                # Fixed maker/taker prediction to avoid the classification error
                try:
                    maker_ratio = self.market_models.predict_maker_taker_ratio(
                        order_type, quantity, volatility_override)
                except Exception as e:
                    # If model prediction fails, use a default value based on order type
                    print(f"Maker/taker prediction error: {e}")
                    # Use a simple heuristic instead of the model
                    if order_type.lower() == "limit":
                        maker_ratio = 0.7  # Default value for limit orders
                    else:
                        maker_ratio = 0.0  # Default value for market orders
            
            # Calculate fees
            fee_tier = self.fee_var.get()
            fee_amount, fee_rate = self.market_models.calculate_fees(quantity, fee_tier, maker_ratio)
            
            # Calculate net cost/proceeds
            net_cost = self.market_models.calculate_net_cost(
                quantity, slippage, market_impact, fee_amount, side)
            
            # Format maker/taker ratio
            maker_pct = maker_ratio * 100
            taker_pct = (1 - maker_ratio) * 100
            
            # Get fee rates for display
            maker_fee_rate, taker_fee_rate = 0, 0
            if "1" in fee_tier:
                maker_fee_rate, taker_fee_rate = 0.05, 0.08
            elif "2" in fee_tier:
                maker_fee_rate, taker_fee_rate = 0.03, 0.06
            elif "3" in fee_tier:
                maker_fee_rate, taker_fee_rate = 0.01, 0.04
            
            # Update labels
            self.labels["Expected Slippage"].config(text=f"{slippage:.4f}%")
            self.labels["Market Impact"].config(text=f"${market_impact:.4f}")
            self.labels["Maker/Taker Ratio"].config(text=f"{maker_pct:.1f}/{taker_pct:.1f}%")
            self.labels["Maker Fee"].config(text=f"{maker_fee_rate:.3f}%")
            self.labels["Taker Fee"].config(text=f"{taker_fee_rate:.3f}%")
            self.labels["Fee Rate"].config(text=f"{fee_rate*100:.4f}%")
            self.labels["Estimated Fees"].config(text=f"${fee_amount:.4f}")
            
            # Update net cost/proceeds based on side
            if side.lower() == "buy":
                self.labels["Net Cost/Proceeds"].config(text=f"${net_cost:.4f}")
            else:
                self.labels["Net Cost/Proceeds"].config(text=f"${net_cost:.4f}")
                
        except Exception as e:
            print(f"Error calculating metrics: {e}")

    async def listen_orderbook(self):
        """Listen for orderbook updates via WebSocket"""
        asset = self.asset_var.get()
        uri = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{asset}"
        
        self.websocket_running = True
        self.connection_status.config(text="🔄 Connecting...", foreground="orange")
        
        while self.websocket_running:
            try:
                async with websockets.connect(uri) as ws:
                    self.connection_status.config(text="✅ Connected", foreground="green")
                    print(f"✅ Connected to OKX L2 orderbook for {asset}")
                    
                    while self.websocket_running:
                        # Start data processing latency measurement
                        start = datetime.now(timezone.utc)
                        message = await ws.recv()
                        end = datetime.now(timezone.utc)
                        
                        # Calculate latency
                        latency_ms = (end - start).total_seconds() * 1000
                        
                        # Record websocket latency
                        self.latency_tracker.record_latency("websocket", latency_ms)
                        
                        # Start data processing measurement
                        # Parse JSON data
                        data_processing_start = time.perf_counter()
                        data = json.loads(message)
                        
                        # Update the orderbook data
                        self.orderbook["timestamp"] = data.get("timestamp", "")
                        self.orderbook["asks"] = data.get("asks", [])[:ORDERBOOK_DEPTH]
                        self.orderbook["bids"] = data.get("bids", [])[:ORDERBOOK_DEPTH]
                        
                        # Extract top bid and ask
                        if self.orderbook["bids"] and self.orderbook["asks"]:
                            bid = float(self.orderbook["bids"][0][0])
                            ask = float(self.orderbook["asks"][0][0])
                            spread = ask - bid
                            
                            # Update current data dictionary
                            self.current_data = {
                                "bid": bid,
                                "ask": ask,
                                "spread": spread,
                                "latency": latency_ms
                            }
                            
                            # Update market models with new data
                            self.market_models.update(bid, ask, latency_ms)
                        
                        # Record data processing latency
                        data_processing_end = time.perf_counter()
                        processing_latency = (data_processing_end - data_processing_start) * 1000
                        self.latency_tracker.record_latency("data_processing", processing_latency)
                
            except Exception as e:
                print(f"WebSocket error: {e}")
                self.connection_status.config(text="✅ Connected", foreground="green")
                await asyncio.sleep(2)  # Wait before reconnecting
    
    def run_websocket_thread(self):
        """Run the WebSocket listener in a separate thread"""
        asyncio.run(self.listen_orderbook())
    
    def reconnect_websocket(self):
        """Reconnect to the WebSocket"""
        # Stop current connection
        self.websocket_running = False
        time.sleep(0.5)  # Allow time for cleanup
        
        # Start new connection in a thread
        self.websocket_thread = threading.Thread(target=self.run_websocket_thread)
        self.websocket_thread.daemon = True
        self.websocket_thread.start()
    
    def start_async_loop(self):
        """Start the async event loop for WebSocket"""
        # Start WebSocket thread
        self.websocket_thread = threading.Thread(target=self.run_websocket_thread)
        self.websocket_thread.daemon = True
        self.websocket_thread.start()
        
        # Initialize GUI updates
        self.update_gui()
        
    def update_refresh_rate(self):
        """Update the GUI refresh rate based on the user input"""
        try:
            rate = int(self.update_speed_var.get())
            if rate < 100:  # Set minimum rate to avoid freezing
                rate = 100
                self.update_speed_var.set("100")
            
            self.update_interval = rate
            print(f"Update interval set to {self.update_interval}ms")
        except ValueError:
            # Reset to default if invalid input
            self.update_speed_var.set("500")
            self.update_interval = 500
            
    def run_benchmark(self):
        """Run a latency benchmark"""
        # Create a popup with benchmark options
        benchmark_window = tk.Toplevel(self.root)
        benchmark_window.title("Run Latency Benchmark")
        benchmark_window.geometry("400x300")
        
        # Benchmark parameters
        ttk.Label(benchmark_window, text="Benchmark Parameters", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Number of cycles
        ttk.Label(benchmark_window, text="Cycles:").pack(anchor="w", padx=20)
        cycles_var = tk.StringVar(value="100")
        ttk.Entry(benchmark_window, textvariable=cycles_var).pack(fill="x", padx=20, pady=5)
        
        # UI complexity simulation
        ttk.Label(benchmark_window, text="UI Load Simulation:").pack(anchor="w", padx=20)
        load_var = tk.StringVar(value="Normal")
        ttk.Combobox(benchmark_window, textvariable=load_var, 
                    values=["Light", "Normal", "Heavy"]).pack(fill="x", padx=20, pady=5)
        
        # Data processing simulation
        ttk.Label(benchmark_window, text="Data Processing Load:").pack(anchor="w", padx=20)
        data_var = tk.StringVar(value="Normal")
        ttk.Combobox(benchmark_window, textvariable=data_var, 
                    values=["Light", "Normal", "Heavy"]).pack(fill="x", padx=20, pady=5)
        
        # Status label
        status_label = ttk.Label(benchmark_window, text="")
        status_label.pack(pady=10)
        
        # Function to run the benchmark
        def start_benchmark():
            try:
                cycles = int(cycles_var.get())
                if cycles <= 0:
                    status_label.config(text="Cycles must be positive", foreground="red")
                    return
                
                # Reset latency tracker
                self.latency_tracker.reset()
                
                status_label.config(text="Running benchmark...", foreground="blue")
                benchmark_window.update()
                
                # Determine load parameters
                ui_load = {"Light": 5, "Normal": 20, "Heavy": 50}[load_var.get()]
                data_load = {"Light": 100, "Normal": 500, "Heavy": 1000}[data_var.get()]
                
                # Run simple benchmark simulation
                for _ in range(cycles):
                    # Simulate data processing under load
                    @measure_latency("data_processing")
                    def process_data():
                        data = np.random.random((data_load, 10))
                        for _ in range(3):
                            data = np.dot(data, data.T)
                            data = data / data.sum()
                        return data
                    
                    process_data()
                    
                    # Simulate UI update under load
                    @measure_latency("ui_update")
                    def update_ui():
                        time.sleep(ui_load / 1000)  # Convert to seconds
                    
                    update_ui()
                    
                    # Simulate end-to-end cycle
                    cycle_id = time.time()
                    self.e2e_monitor.start_cycle(cycle_id)
                    process_data()
                    update_ui()
                    self.e2e_monitor.end_cycle(cycle_id)
                    
                    # Update progress
                    if _ % 10 == 0:
                        status_label.config(text=f"Running benchmark... {_ + 1}/{cycles}")
                        benchmark_window.update()
                
                status_label.config(text="Benchmark complete!", foreground="green")
                
                # Update the visualization with new data
                self.update_latency_plots()
                
            except Exception as e:
                status_label.config(text=f"Error: {str(e)}", foreground="red")
        
        # Start button
        ttk.Button(benchmark_window, text="Start Benchmark", command=start_benchmark).pack(pady=10)
        
        # Close button
        ttk.Button(benchmark_window, text="Close", command=benchmark_window.destroy).pack()
    
    def show_latency_report(self):
        """Show a detailed latency report"""
        # Generate report
        report = generate_latency_report(self.latency_tracker)
        
        # Create report window
        report_window = tk.Toplevel(self.root)
        report_window.title("Latency Benchmark Report")
        report_window.geometry("600x500")
        
        # Create text widget with scrollbar
        frame = ttk.Frame(report_window)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")
        
        text_widget = tk.Text(frame, wrap="word", yscrollcommand=scrollbar.set)
        text_widget.pack(side="left", fill="both", expand=True)
        
        scrollbar.config(command=text_widget.yview)
        
        # Insert report text
        text_widget.insert("1.0", report)
        text_widget.config(state="disabled")  # Make read-only
        
        # Export button
        def export_report():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"latency_report_{timestamp}.txt"
            
            with open(filename, "w") as f:
                f.write(report)
            
            messagebox.showinfo("Export Successful", f"Report exported to {filename}")
        
        ttk.Button(report_window, text="Export Report", command=export_report).pack(pady=10)
        ttk.Button(report_window, text="Close", command=report_window.destroy).pack()
    
    def reset_latency_metrics(self):
        """Reset all latency metrics"""
        self.latency_tracker.reset()
        
        # Reset visualizations
        for component in ['data_processing', 'ui_update', 'end_to_end']:
            self.latency_history[component].clear()
        
        # Update plots
        self.update_latency_plots()
        
        # Show confirmation
        messagebox.showinfo("Metrics Reset", "All latency metrics have been reset.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TradeSimulatorUI(root)
    root.mainloop()
