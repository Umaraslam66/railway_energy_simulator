# Train physics and parameters
from dataclasses import dataclass
from typing import List, Dict, Optional
@dataclass
class TrainParameters:
    """Parameters defining a train's physical characteristics"""
    mass: float  # kg
    max_power: float  # watts
    efficiency: float  # 0-1
    max_speed: float  # m/s
    max_acceleration: float  # m/s^2
    regenerative_braking_efficiency: float  # 0-1
    train_type: str  # 'electric' or 'diesel'
