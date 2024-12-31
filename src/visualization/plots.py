# Basic plotting functions
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from ..core.track import RailwayNetwork

def plot_simulation_results(
    positions: List[float],
    speeds: List[float],
    energies: List[float],
    network: RailwayNetwork
):
    """Plot the simulation results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Speed profile
    ax1.plot(positions, [s * 3.6 for s in speeds])  # Convert to km/h
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Speed (km/h)')
    ax1.grid(True)
    ax1.set_title('Speed Profile')
    
    # Energy consumption
    cumulative_energy = np.cumsum(energies)
    ax2.plot(positions, [e / 3600000 for e in cumulative_energy])  # Convert to kWh
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Cumulative Energy (kWh)')
    ax2.grid(True)
    ax2.set_title('Energy Consumption')
    
    plt.tight_layout()
    plt.show()