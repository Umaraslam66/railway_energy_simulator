# Core simulation engine
import numpy as np
from typing import List, Dict, Tuple, Optional
from .train import TrainParameters
from .track import TrackSegment, RailwayNetwork
class EnergySimulator:
    def __init__(self, train: TrainParameters, network: RailwayNetwork):
        self.train = train
        self.network = network
        self.dt = 1.0  # simulation timestep (seconds)
        
    def calculate_resistance_force(self, speed: float) -> float:
        """Calculate basic rolling and air resistance"""
        # Simple quadratic resistance model: F = A + Bv + Cv²
        A = 2000  # Rolling resistance coefficient
        B = 0.1   # Linear air resistance coefficient
        C = 0.6   # Quadratic air resistance coefficient
        
        return A + B * speed + C * speed * speed
    
    def calculate_energy_consumption(
        self, 
        initial_speed: float,
        target_speed: float,
        distance: float,
        gradient: float
    ) -> tuple[float, List[float], List[float], List[float]]:
        """
        Calculate energy consumption for a section of track
        Returns: (total_energy, speeds, positions, energies)
        """
        current_speed = initial_speed
        position = 0
        
        speeds = [current_speed]
        positions = [position]
        energies = [0]
        total_energy = 0
        
        while position < distance:
            # Calculate forces
            gradient_force = self.train.mass * 9.81 * np.sin(np.arctan(gradient/100))
            resistance_force = self.calculate_resistance_force(current_speed)
            
            # Determine required acceleration
            desired_acceleration = (target_speed - current_speed) / self.dt
            desired_acceleration = np.clip(
                desired_acceleration,
                -self.train.max_acceleration,
                self.train.max_acceleration
            )
            
            # Calculate required force and power
            required_force = (
                self.train.mass * desired_acceleration +
                gradient_force +
                resistance_force
            )
            
            required_power = required_force * current_speed
            
            # Limit power to train's capabilities
            actual_power = np.clip(required_power, -self.train.max_power, self.train.max_power)
            
            # Calculate energy used in this timestep
            if actual_power > 0:
                energy_used = actual_power * self.dt / self.train.efficiency
            else:
                # Regenerative braking
                energy_used = actual_power * self.dt * self.train.regenerative_braking_efficiency
            
            total_energy += energy_used
            
            # Update speed and position
            actual_acceleration = (
                actual_power / current_speed - gradient_force - resistance_force
            ) / self.train.mass if current_speed > 0.1 else desired_acceleration
            
            current_speed += actual_acceleration * self.dt
            current_speed = np.clip(current_speed, 0, self.train.max_speed)
            
            position += current_speed * self.dt
            
            # Store values for plotting
            speeds.append(current_speed)
            positions.append(position)
            energies.append(energy_used)
        
        return total_energy, speeds, positions, energies

def create_sample_network() -> RailwayNetwork:
    """Create a sample railway network for testing"""
    segments = [
        TrackSegment(length=5000, gradient=0, speed_limit=30, station_at_end=True),
        TrackSegment(length=10000, gradient=2, speed_limit=25, station_at_end=False),
        TrackSegment(length=7000, gradient=-1, speed_limit=30, station_at_end=True),
    ]
    return RailwayNetwork(segments)