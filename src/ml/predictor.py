import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional
import joblib

class EnergyPredictor:
    """ML model to predict energy consumption based on train and track parameters"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = (RandomForestRegressor(n_estimators=100, random_state=42) 
                     if model_type == 'random_forest' else LinearRegression())
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'avg_speed', 'max_speed', 'total_distance', 'avg_gradient',
            'train_mass', 'max_power', 'is_electric', 'avg_acceleration'
        ]

    def prepare_features(self, simulation_data: List[Dict]) -> pd.DataFrame:
        """Convert simulation data into feature matrix"""
        features = []
        
        for sim in simulation_data:
            # Calculate additional features
            speeds = np.array(sim['speeds'])  # Using 'speeds' instead of 'speed'
            accelerations = np.diff(speeds, prepend=speeds[0])
            
            feature_dict = {
                'avg_speed': np.mean(speeds),
                'max_speed': np.max(speeds),
                'total_distance': sim['positions'][-1],  # Using 'positions' instead of 'position'
                'avg_gradient': sim['initial_conditions']['gradient'],
                'train_mass': sim['train_parameters'].mass,
                'max_power': sim['train_parameters'].max_power,
                'is_electric': 1 if sim['train_type'] == 'electric' else 0,
                'avg_acceleration': np.mean(np.abs(accelerations))
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)

    def train(self, simulation_data: List[Dict], test_size: float = 0.2):
        """Train the energy consumption prediction model"""
        # Prepare features and target
        X = self.prepare_features(simulation_data)
        y = np.array([s['total_energy'] for s in simulation_data])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        if self.model_type == 'random_forest':
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': feature_importance
            }
        
        return {
            'train_score': train_score,
            'test_score': test_score
        }

    def predict(self, simulation_data: Dict) -> float:
        """Predict energy consumption for a given scenario"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
        
        X = self.prepare_features([simulation_data])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]

    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }, filepath)

    def load_model(self, filepath: str):
        """Load a trained model"""
        saved_model = joblib.load(filepath)
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']
        self.is_trained = saved_model['is_trained']
        self.model_type = saved_model['model_type']
        self.feature_names = saved_model['feature_names']