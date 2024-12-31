# Track and network definitions
from dataclasses import dataclass
from typing import List, Dict, Optional
from dataclasses import dataclass
from typing import List, Dict
from .train import TrainParameters

@dataclass
class TrackSegment:
    """Represents a segment of railway track"""
    length: float  # meters
    gradient: float  # percentage, positive means uphill
    speed_limit: float  # m/s
    station_at_end: bool = False

class RailwayNetwork:
    def __init__(self, segments: List[TrackSegment]):
        self.segments = segments
        self.total_length = sum(segment.length for segment in segments)
    
    def get_gradient_at_position(self, position: float) -> float:
        """Returns the gradient at a given position along the track"""
        current_position = 0
        for segment in self.segments:
            if current_position <= position < current_position + segment.length:
                return segment.gradient
            current_position += segment.length
        return 0.0