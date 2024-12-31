import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Core imports
from src.core.train import TrainParameters
from src.core.track import TrackSegment, RailwayNetwork
from src.core.simulator import EnergySimulator

# ML imports
from src.ml.predictor import EnergyPredictor
from src.ml.optimizer import ScheduleOptimizer
from src.ml.patterns import PatternAnalyzer

# Visualization imports
from src.visualization.dashboards import AdvancedVisualizer
from src.utils.data_handler import save_simulation_results

def run_single_simulation(simulator: EnergySimulator, initial_speed: float, target_speed: float):
    """Run a single simulation with given parameters"""
    print(f"\nRunning simulation with:")
    print(f"Initial speed: {initial_speed} m/s")
    print(f"Target speed: {target_speed} m/s")
    
    total_energy, speeds, positions, energies = simulator.calculate_energy_consumption(
        initial_speed=initial_speed,
        target_speed=target_speed,
        distance=5000,  # 5km section
        gradient=2.0    # 2% gradient
    )
    
    print(f"\nSimulation Results:")
    print(f"Total energy consumption: {total_energy / 3600000:.2f} kWh")
    print(f"Average speed: {np.mean(speeds) * 3.6:.2f} km/h")
    print(f"Distance covered: {positions[-1] / 1000:.2f} km")
    
    return {
        'speeds': speeds,
        'positions': positions,
        'energies': energies,
        'total_energy': total_energy,
        'train_parameters': simulator.train,
        'train_type': simulator.train.train_type,
        'initial_conditions': {
            'initial_speed': initial_speed,
            'target_speed': target_speed,
            'distance': 5000,
            'gradient': 2.0
        }
    }

def simulate_multiple_scenarios(simulator: EnergySimulator, n_scenarios: int = 5):
    """Generate multiple simulation scenarios for ML analysis"""
    scenarios = []
    speed_ranges = np.linspace(0, simulator.train.max_speed * 0.8, n_scenarios)
    
    for init_speed in speed_ranges:
        for target_speed in speed_ranges[speed_ranges > init_speed]:
            result = run_single_simulation(simulator, init_speed, target_speed)
            scenarios.append(result)
    
    return scenarios

def create_quick_visualization(scenarios, output_dir):
    """Create a simple visualization comparing electric vs diesel trains"""
    try:
        # Create basic scatter plot
        fig = go.Figure()
        
        # Convert data to DataFrame for easier plotting
        data = [{
            'avg_speed': np.mean(s['speeds']) * 3.6,  # km/h
            'energy_per_km': s['total_energy'] / s['positions'][-1] / 3600,  # kWh/km
            'train_type': s['train_type']
        } for s in scenarios]
        
        df = pd.DataFrame(data)
        
        # Plot electric trains
        electric_data = df[df['train_type'] == 'electric']
        fig.add_trace(go.Scatter(
            x=electric_data['avg_speed'],
            y=electric_data['energy_per_km'],
            mode='markers',
            name='Electric',
            marker=dict(color='blue', size=8)
        ))
        
        # Plot diesel trains
        diesel_data = df[df['train_type'] == 'diesel']
        fig.add_trace(go.Scatter(
            x=diesel_data['avg_speed'],
            y=diesel_data['energy_per_km'],
            mode='markers',
            name='Diesel',
            marker=dict(color='red', size=8)
        ))
        
        fig.update_layout(
            title="Train Energy Efficiency Comparison",
            xaxis_title="Average Speed (km/h)",
            yaxis_title="Energy Consumption (kWh/km)",
            width=800,
            height=600
        )
        
        # Save visualization
        fig.write_html(output_dir / "energy_comparison.html")
        print("Energy comparison visualization saved successfully")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    print("Railway Energy Efficiency Simulator")
    print("==================================")
    
    # Create network and trains
    print("\nInitializing simulation...")
    segments = [
        TrackSegment(length=5000, gradient=0, speed_limit=30, station_at_end=True),
        TrackSegment(length=10000, gradient=2, speed_limit=25, station_at_end=False),
        TrackSegment(length=7000, gradient=-1, speed_limit=30, station_at_end=True)
    ]
    network = RailwayNetwork(segments)
    
    electric_train = TrainParameters(
        mass=80000, max_power=4000000, efficiency=0.85,
        max_speed=44.4, max_acceleration=1.0,
        regenerative_braking_efficiency=0.7, train_type='electric'
    )
    
    diesel_train = TrainParameters(
        mass=80000, max_power=2500000, efficiency=0.40,
        max_speed=33.3, max_acceleration=0.8,
        regenerative_braking_efficiency=0.0, train_type='diesel'
    )
    
    # Run simulations
    electric_simulator = EnergySimulator(electric_train, network)
    diesel_simulator = EnergySimulator(diesel_train, network)
    
    print("\nRunning simulations...")
    electric_scenarios = simulate_multiple_scenarios(electric_simulator)
    diesel_scenarios = simulate_multiple_scenarios(diesel_simulator)
    all_scenarios = electric_scenarios + diesel_scenarios
    
    # ML Analysis
    print("\nPerforming ML analysis...")
    predictor = EnergyPredictor()
    training_results = predictor.train(all_scenarios)
    print(f"Training Score: {training_results['train_score']:.3f}")
    print(f"Test Score: {training_results['test_score']:.3f}")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create visualization
    print("\nCreating visualization...")
    create_quick_visualization(all_scenarios, output_dir)
    
    print("\nSimulation complete!")
    print(f"\nResults saved in: {output_dir}")
    print("Open energy_comparison.html in a web browser to view the visualization.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nExiting script")