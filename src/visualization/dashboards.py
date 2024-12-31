import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class AdvancedVisualizer:
    def __init__(self, network, simulation_results):
        self.network = network
        self.simulation_results = simulation_results

    def create_interactive_dashboard(self):
        """Creates an interactive dashboard with multiple plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Speed Profile', 
                'Energy Consumption',
                'Track Elevation Profile', 
                'Energy Efficiency Analysis'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                  [{"secondary_y": True}, {"secondary_y": True}]]
        )

        # Speed Profile
        fig.add_trace(
            go.Scatter(
                x=self.simulation_results['positions'],  # Changed from 'position' to 'positions'
                y=[s * 3.6 for s in self.simulation_results['speeds']],  # Changed from 'speed' to 'speeds'
                name="Speed",
                line=dict(color='blue'),
            ),
            row=1, col=1
        )

        # Energy Consumption
        energy_consumption = np.cumsum(self.simulation_results['energies'])  # Calculate cumulative energy
        fig.add_trace(
            go.Scatter(
                x=self.simulation_results['positions'],
                y=[e / 3600000 for e in energy_consumption],  # Convert to kWh
                name="Energy",
                line=dict(color='red'),
            ),
            row=1, col=2
        )

        # Track Elevation Profile
        positions = np.linspace(0, self.network.total_length, 100)
        gradient = self.simulation_results['initial_conditions']['gradient']
        elevations = [p * gradient / 100 for p in positions]  # Simple elevation calculation
        
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=elevations,
                name="Elevation",
                fill='tozeroy',
                line=dict(color='green'),
            ),
            row=2, col=1
        )

        # Energy Efficiency Analysis (Energy per km over distance)
        energy_per_km = [
            (e / p if p > 0 else 0) / 3600000  # Convert to kWh/km
            for e, p in zip(energy_consumption, self.simulation_results['positions'])
        ]
        
        fig.add_trace(
            go.Scatter(
                x=self.simulation_results['positions'],
                y=energy_per_km,
                name="Energy/km",
                line=dict(color='purple'),
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Railway Energy Efficiency Analysis Dashboard",
            showlegend=True
        )

        # Update axes labels
        fig.update_xaxes(title_text="Position (m)", row=1, col=1)
        fig.update_xaxes(title_text="Position (m)", row=1, col=2)
        fig.update_xaxes(title_text="Position (m)", row=2, col=1)
        fig.update_xaxes(title_text="Position (m)", row=2, col=2)

        fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
        fig.update_yaxes(title_text="Energy (kWh)", row=1, col=2)
        fig.update_yaxes(title_text="Elevation (m)", row=2, col=1)
        fig.update_yaxes(title_text="Energy per km (kWh/km)", row=2, col=2)

        return fig

    def generate_efficiency_report(self) -> pd.DataFrame:
        """Generates a detailed efficiency report"""
        
        # Calculate cumulative energy
        energy_consumption = np.cumsum(self.simulation_results['energies'])
        total_energy = energy_consumption[-1] / 3600000  # Convert to kWh
        total_distance = self.simulation_results['positions'][-1] / 1000  # Convert to km
        avg_speed = np.mean(self.simulation_results['speeds']) * 3.6  # Convert to km/h
        
        report = {
            'total_energy_kwh': total_energy,
            'energy_per_km': total_energy / total_distance,
            'average_speed_kmh': avg_speed,
            'maximum_speed_kmh': max(self.simulation_results['speeds']) * 3.6,
            'total_distance_km': total_distance,
            'train_type': self.simulation_results['train_type'],
            'gradient': self.simulation_results['initial_conditions']['gradient']
        }
        
        return pd.DataFrame([report])

    def create_energy_heatmap(self, segment_data: pd.DataFrame):
        """Creates a heatmap showing energy consumption across different segments"""
        fig = go.Figure(data=go.Heatmap(
            z=segment_data['energy_consumption'],
            x=segment_data['positions'],  # Changed from 'position' to 'positions'
            y=segment_data['time'],
            colorscale='Viridis',
            colorbar=dict(title='Energy Consumption (kWh)')
        ))

        fig.update_layout(
            title='Energy Consumption Heatmap',
            xaxis_title='Track Position (m)',
            yaxis_title='Time of Day',
            height=600
        )

        return fig