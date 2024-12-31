# Schedule optimization
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from ..core.track import RailwayNetwork
from .predictor import EnergyPredictor

class ScheduleOptimizer:
    """Optimize train schedules for energy efficiency"""
    
    def __init__(self, network, energy_predictor: EnergyPredictor):
        self.network = network
        self.energy_predictor = energy_predictor

    def optimize_schedule(self, 
                        initial_schedule: List[Dict],
                        constraints: Dict,
                        optimization_params: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """
        Optimize train schedule for energy efficiency
        
        Args:
            initial_schedule: List of service dictionaries
            constraints: Dictionary of constraints
            optimization_params: Optional parameters for optimization
        
        Returns:
            Tuple of (optimized schedule, optimization results)
        """
        if optimization_params is None:
            optimization_params = {
                'max_iter': 100,
                'energy_weight': 1.0,
                'delay_weight': 0.5,
                'separation_weight': 1.0
            }

        def objective(x):
            """Objective function combining energy and constraints"""
            # Reconstruct schedule from optimization variables
            proposed_schedule = self._decode_schedule(x, initial_schedule)
            
            # Calculate total energy consumption
            energy_cost = sum(
                self.energy_predictor.predict(service)
                for service in proposed_schedule
            )
            
            # Calculate delay penalty
            delay_penalty = self._calculate_delay_penalty(
                proposed_schedule,
                initial_schedule,
                weight=optimization_params['delay_weight']
            )
            
            # Calculate separation penalty
            separation_penalty = self._calculate_separation_penalty(
                proposed_schedule,
                constraints,
                weight=optimization_params['separation_weight']
            )
            
            return (optimization_params['energy_weight'] * energy_cost + 
                   delay_penalty + separation_penalty)

        # Encode initial schedule
        x0 = self._encode_schedule(initial_schedule)
        bounds = self._get_optimization_bounds(initial_schedule, constraints)
        
        # Run optimization
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': optimization_params['max_iter']}
        )
        
        # Decode final schedule
        optimized_schedule = self._decode_schedule(result.x, initial_schedule)
        
        # Calculate metrics
        metrics = self._calculate_optimization_metrics(
            initial_schedule,
            optimized_schedule,
            result
        )
        
        return optimized_schedule, metrics

    def _encode_schedule(self, schedule: List[Dict]) -> np.ndarray:
        """Convert schedule into optimization variables"""
        variables = []
        for service in schedule:
            variables.extend([
                service['departure_time'].timestamp(),
                service['target_speed'],
                *[stop['dwell_time'].total_seconds() for stop in service['stops']]
            ])
        return np.array(variables)

    def _decode_schedule(self, x: np.ndarray, template_schedule: List[Dict]) -> List[Dict]:
        """Convert optimization variables back into schedule"""
        new_schedule = []
        var_idx = 0
        
        for service in template_schedule:
            new_service = service.copy()
            new_service['departure_time'] = datetime.fromtimestamp(x[var_idx])
            new_service['target_speed'] = x[var_idx + 1]
            
            # Update stop times
            new_stops = []
            for i, stop in enumerate(service['stops']):
                new_stop = stop.copy()
                new_stop['dwell_time'] = timedelta(seconds=x[var_idx + 2 + i])
                new_stops.append(new_stop)
            
            new_service['stops'] = new_stops
            var_idx += 2 + len(service['stops'])
            new_schedule.append(new_service)
        
        return new_schedule

    def _calculate_delay_penalty(self, 
                               proposed_schedule: List[Dict],
                               initial_schedule: List[Dict],
                               weight: float) -> float:
        """Calculate penalty for schedule delays"""
        total_delay = 0
        for prop, init in zip(proposed_schedule, initial_schedule):
            delay = (prop['departure_time'] - init['departure_time']).total_seconds()
            total_delay += abs(delay)
        return weight * total_delay

    def _calculate_separation_penalty(self,
                                   schedule: List[Dict],
                                   constraints: Dict,
                                   weight: float) -> float:
        """Calculate penalty for insufficient train separation"""
        min_separation = constraints.get('min_separation', 180)  # seconds
        penalty = 0
        
        departures = sorted(
            (s['departure_time'], s['train_id']) 
            for s in schedule
        )
        
        for i in range(len(departures) - 1):
            separation = (departures[i+1][0] - departures[i][0]).total_seconds()
            if separation < min_separation:
                penalty += (min_separation - separation) ** 2
                
        return weight * penalty

    def _get_optimization_bounds(self,
                               schedule: List[Dict],
                               constraints: Dict) -> List[Tuple[float, float]]:
        """Define bounds for optimization variables"""
        bounds = []
        for service in schedule:
            # Departure time bounds (±1 hour)
            original_time = service['departure_time'].timestamp()
            bounds.append((
                original_time - 3600,
                original_time + 3600
            ))
            
            # Speed bounds (50-100% of max speed)
            bounds.append((
                0.5 * service['train_parameters'].max_speed,
                service['train_parameters'].max_speed
            ))
            
            # Dwell time bounds (±5 minutes)
            for stop in service['stops']:
                original_dwell = stop['dwell_time'].total_seconds()
                bounds.append((
                    max(60, original_dwell - 300),  # Minimum 1 minute
                    original_dwell + 300
                ))
        
        return bounds

    def _calculate_optimization_metrics(self,
                                     initial_schedule: List[Dict],
                                     optimized_schedule: List[Dict],
                                     optimization_result: Dict) -> Dict:
        """Calculate metrics comparing initial and optimized schedules"""
        initial_energy = sum(
            self.energy_predictor.predict(service)
            for service in initial_schedule
        )
        
        optimized_energy = sum(
            self.energy_predictor.predict(service)
            for service in optimized_schedule
        )
        
        return {
            'initial_energy': initial_energy,
            'optimized_energy': optimized_energy,
            'energy_savings_percent': 100 * (initial_energy - optimized_energy) / initial_energy,
            'optimization_success': optimization_result.success,
            'optimization_iterations': optimization_result.nit,
            'optimization_message': optimization_result.message
        }