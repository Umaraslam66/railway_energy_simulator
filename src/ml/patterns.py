import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict

class PatternAnalyzer:
    def __init__(self, n_clusters: int = 3):
        # Add max_iter and early stopping parameters to KMeans
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42,
            max_iter=100,  # Limit maximum iterations
            tol=1e-4,      # Increased tolerance for faster convergence
            n_init=10      # Reduce number of initialization attempts
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def analyze_patterns(self, simulation_data: List[Dict]) -> Dict:
        """Analyze patterns with optimized feature extraction"""
        # Simplified feature extraction
        features = pd.DataFrame([{
            'energy_per_km': s['total_energy'] / s['positions'][-1],
            'avg_speed': np.mean(s['speeds']),
            'is_electric': 1 if s['train_type'] == 'electric' else 0
        } for s in simulation_data])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # Quick cluster analysis
        cluster_stats = self._analyze_clusters(features, cluster_labels)
        
        # Generate basic insights
        insights = self._generate_quick_insights(cluster_stats)
        
        self.is_trained = True
        
        return {
            'cluster_assignments': cluster_labels,
            'cluster_statistics': cluster_stats,
            'insights': insights
        }

    def _analyze_clusters(self, features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Simplified cluster analysis"""
        cluster_stats = []
        
        for i in range(self.n_clusters):
            cluster_mask = labels == i
            cluster_data = features[cluster_mask]
            
            stats = {
                'cluster_id': i,
                'size': len(cluster_data),
                'avg_energy_per_km': cluster_data['energy_per_km'].mean() / 3600000,  # Convert to kWh/km
                'avg_speed': cluster_data['avg_speed'].mean() * 3.6,  # Convert to km/h
                'electric_ratio': cluster_data['is_electric'].mean()
            }
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)

    def _generate_quick_insights(self, cluster_stats: pd.DataFrame) -> List[str]:
        """Generate basic insights quickly"""
        insights = []
        
        # Most efficient cluster
        most_efficient = cluster_stats.loc[cluster_stats['avg_energy_per_km'].idxmin()]
        least_efficient = cluster_stats.loc[cluster_stats['avg_energy_per_km'].idxmax()]
        
        insights.append(f"Cluster {most_efficient['cluster_id']} is most efficient: "
                       f"{most_efficient['avg_energy_per_km']:.2f} kWh/km, "
                       f"avg speed {most_efficient['avg_speed']:.1f} km/h")
        
        # Electric vs diesel comparison
        electric_clusters = cluster_stats[cluster_stats['electric_ratio'] > 0.5]
        diesel_clusters = cluster_stats[cluster_stats['electric_ratio'] <= 0.5]
        
        if not electric_clusters.empty and not diesel_clusters.empty:
            e_efficiency = electric_clusters['avg_energy_per_km'].mean()
            d_efficiency = diesel_clusters['avg_energy_per_km'].mean()
            diff = ((d_efficiency - e_efficiency) / d_efficiency * 100)
            
            insights.append(f"Electric trains are {diff:.1f}% more energy-efficient")
        
        return insights

    def generate_cluster_visualization(self) -> Dict:
        """Generate simplified visualization data"""
        if not self.is_trained:
            raise ValueError("Run analyze_patterns first")
            
        return {
            'cluster_centers': self.kmeans.cluster_centers_,
            'labels': self.kmeans.labels_,
            'inertia': self.kmeans.inertia_
        }