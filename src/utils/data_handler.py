# Data processing utilities

import pandas as pd
import json
from typing import Dict, List, Union
from pathlib import Path
from ..core.train import TrainParameters
from ..core.track import RailwayNetwork, TrackSegment
import json
import pandas as pd
from typing import Dict, List, Any
import pickle
from datetime import datetime

def save_simulation_results(results: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save simulation results to a file. Handles different data formats.
    
    Args:
        results: Dictionary containing simulation results
        filepath: Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.csv':
        # Convert simulation results to DataFrame
        df = pd.DataFrame({
            'position': results['positions'],
            'speed': results['speeds'],
            'energy': results['energies'],
            'cumulative_energy': [e / 3600000 for e in results['energies']],  # Convert to kWh
        })
        df.to_csv(filepath, index=False)
    
    elif filepath.suffix == '.json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            key: value.tolist() if hasattr(value, 'tolist') else value
            for key, value in results.items()
        }
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

def load_simulation_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load simulation results from a file.
    
    Args:
        filepath: Path to the file
    
    Returns:
        Dictionary containing the simulation results
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        return {
            'positions': df['position'].values,
            'speeds': df['speed'].values,
            'energies': df['energy'].values,
            'cumulative_energy': df['cumulative_energy'].values * 3600000  # Convert back to joules
        }
    
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

def save_network_config(network: 'RailwayNetwork', filepath: Union[str, Path]) -> None:
    """
    Save railway network configuration to a file.
    
    Args:
        network: RailwayNetwork instance
        filepath: Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert network to serializable format
    network_data = {
        'segments': [
            {
                'length': segment.length,
                'gradient': segment.gradient,
                'speed_limit': segment.speed_limit,
                'station_at_end': segment.station_at_end
            }
            for segment in network.segments
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(network_data, f, indent=4)

def load_network_config(filepath: Union[str, Path]) -> Dict:
    """
    Load railway network configuration from a file.
    
    Args:
        filepath: Path to the file
    
    Returns:
        Dictionary containing network configuration
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def create_results_directory() -> Path:
    """
    Create a timestamped directory for storing simulation results.
    
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def save_full_simulation(
    simulation_results: Dict[str, Any],
    network_config: 'RailwayNetwork',
    train_params: Dict[str, Any],
    base_dir: Union[str, Path] = None
) -> Path:
    """
    Save complete simulation data including results, network config, and train parameters.
    
    Args:
        simulation_results: Dictionary containing simulation results
        network_config: RailwayNetwork instance
        train_params: Dictionary of train parameters
        base_dir: Optional base directory for saving files
    
    Returns:
        Path to the directory containing saved files
    """
    if base_dir is None:
        save_dir = create_results_directory()
    else:
        save_dir = Path(base_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save simulation results
    save_simulation_results(simulation_results, save_dir / "simulation_results.json")
    
    # Save network configuration
    save_network_config(network_config, save_dir / "network_config.json")
    
    # Save train parameters
    with open(save_dir / "train_params.json", 'w') as f:
        json.dump(train_params, f, indent=4)
    
    return save_dir