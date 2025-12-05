"""
AI Model Observability and Monitoring System

This implementation provides comprehensive observability for AI/ML models in production,
including performance monitoring, drift detection, and real-time metrics collection.

Author: Satbir Singh
Paper: [Paper Title - ID 649]
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    timestamp: datetime
    model_name: str
    accuracy: float
    latency_ms: float
    throughput: float
    prediction_count: int
    error_rate: float
    cpu_usage: float
    memory_usage: float


class DriftDetector:
    """
    Detects data drift and model performance degradation
    using statistical methods and distribution comparison
    """
    
    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        self.reference_data = reference_data
        self.reference_mean = np.mean(reference_data)
        self.reference_std = np.std(reference_data)
        self.threshold = threshold
        self.drift_history = deque(maxlen=100)
    
    def detect_drift(self, current_data: np.ndarray) -> Dict:
        """
        Detect if current data distribution has drifted from reference
        Returns drift status and statistics
        """
        current_mean = np.mean(current_data)
        current_std = np.std(current_data)
        
        # Calculate statistical distance (simplified KL divergence)
        mean_diff = abs(current_mean - self.reference_mean) / (self.reference_std + 1e-8)
        std_diff = abs(current_std - self.reference_std) / (self.reference_std + 1e-8)
        
        drift_score = (mean_diff + std_diff) / 2
        is_drifted = drift_score > self.threshold
        
        drift_info = {
            'is_drifted': is_drifted,
            'drift_score': drift_score,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'reference_mean': self.reference_mean,
            'current_mean': current_mean,
            'timestamp': datetime.now().isoformat()
        }
        
        self.drift_history.append(drift_info)
        return drift_info
    
    def get_drift_trend(self) -> Dict:
        """Analyze drift trend over time"""
        if len(self.drift_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_scores = [d['drift_score'] for d in list(self.drift_history)[-10:]]
        if len(recent_scores) < 2:
            return {'trend': 'stable'}
        
        trend = 'increasing' if recent_scores[-1] > recent_scores[0] else 'decreasing'
        return {
            'trend': trend,
            'average_drift': np.mean(recent_scores),
            'max_drift': np.max(recent_scores)
        }


class ModelMonitor:
    """
    Comprehensive monitoring system for AI models in production
    Tracks performance, latency, errors, and resource usage
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_history = deque(maxlen=1000)
        self.error_log = deque(maxlen=100)
        self.alert_thresholds = {
            'accuracy': 0.8,
            'latency_ms': 1000,
            'error_rate': 0.1,
            'cpu_usage': 0.9,
            'memory_usage': 0.9
        }
    
    def record_prediction(self, 
                         accuracy: float,
                         latency_ms: float,
                         cpu_usage: float = 0.0,
                         memory_usage: float = 0.0,
                         error: Optional[Exception] = None):
        """Record a prediction event with metrics"""
        timestamp = datetime.now()
        
        # Calculate throughput (predictions per second)
        if len(self.metrics_history) > 0:
            time_diff = (timestamp - self.metrics_history[-1].timestamp).total_seconds()
            throughput = 1.0 / time_diff if time_diff > 0 else 0
        else:
            throughput = 0
        
        metrics = ModelMetrics(
            timestamp=timestamp,
            model_name=self.model_name,
            accuracy=accuracy,
            latency_ms=latency_ms,
            throughput=throughput,
            prediction_count=len(self.metrics_history) + 1,
            error_rate=self._calculate_error_rate(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage
        )
        
        self.metrics_history.append(metrics)
        
        if error:
            self.error_log.append({
                'timestamp': timestamp.isoformat(),
                'error': str(error),
                'type': type(error).__name__
            })
        
        # Check for alerts
        alerts = self._check_alerts(metrics)
        return metrics, alerts
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if len(self.error_log) == 0:
            return 0.0
        
        recent_errors = len([e for e in self.error_log 
                           if (datetime.now() - datetime.fromisoformat(e['timestamp'])).seconds < 60])
        total_predictions = len(self.metrics_history)
        
        return recent_errors / total_predictions if total_predictions > 0 else 0.0
    
    def _check_alerts(self, metrics: ModelMetrics) -> List[Dict]:
        """Check if metrics exceed alert thresholds"""
        alerts = []
        
        if metrics.accuracy < self.alert_thresholds['accuracy']:
            alerts.append({
                'level': 'warning',
                'metric': 'accuracy',
                'value': metrics.accuracy,
                'threshold': self.alert_thresholds['accuracy'],
                'message': f'Model accuracy ({metrics.accuracy:.2%}) below threshold'
            })
        
        if metrics.latency_ms > self.alert_thresholds['latency_ms']:
            alerts.append({
                'level': 'warning',
                'metric': 'latency',
                'value': metrics.latency_ms,
                'threshold': self.alert_thresholds['latency_ms'],
                'message': f'Latency ({metrics.latency_ms:.0f}ms) exceeds threshold'
            })
        
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append({
                'level': 'critical',
                'metric': 'error_rate',
                'value': metrics.error_rate,
                'threshold': self.alert_thresholds['error_rate'],
                'message': f'Error rate ({metrics.error_rate:.2%}) above threshold'
            })
        
        return alerts
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict:
        """Get performance summary for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics_history 
                         if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'status': 'no_data'}
        
        return {
            'model_name': self.model_name,
            'window_minutes': window_minutes,
            'total_predictions': len(recent_metrics),
            'average_accuracy': np.mean([m.accuracy for m in recent_metrics]),
            'average_latency_ms': np.mean([m.latency_ms for m in recent_metrics]),
            'average_throughput': np.mean([m.throughput for m in recent_metrics]),
            'current_error_rate': recent_metrics[-1].error_rate if recent_metrics else 0,
            'average_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'average_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        if format == 'json':
            metrics_data = [{
                'timestamp': m.timestamp.isoformat(),
                'model_name': m.model_name,
                'accuracy': m.accuracy,
                'latency_ms': m.latency_ms,
                'throughput': m.throughput,
                'error_rate': m.error_rate,
                'cpu_usage': m.cpu_usage,
                'memory_usage': m.memory_usage
            } for m in self.metrics_history]
            return json.dumps(metrics_data, indent=2)
        else:
            return str(list(self.metrics_history))


class ObservabilityDashboard:
    """
    Central dashboard for monitoring multiple AI models
    Provides unified view of system health and performance
    """
    
    def __init__(self):
        self.monitors: Dict[str, ModelMonitor] = {}
        self.drift_detectors: Dict[str, DriftDetector] = {}
    
    def register_model(self, model_name: str, reference_data: Optional[np.ndarray] = None):
        """Register a new model for monitoring"""
        self.monitors[model_name] = ModelMonitor(model_name)
        
        if reference_data is not None:
            self.drift_detectors[model_name] = DriftDetector(reference_data)
    
    def get_system_health(self) -> Dict:
        """Get overall system health across all models"""
        health_status = {
            'total_models': len(self.monitors),
            'healthy_models': 0,
            'degraded_models': 0,
            'critical_models': 0,
            'models': {}
        }
        
        for model_name, monitor in self.monitors.items():
            summary = monitor.get_performance_summary()
            
            if summary.get('status') == 'no_data':
                continue
            
            # Determine health status
            if (summary['average_accuracy'] >= 0.9 and 
                summary['current_error_rate'] < 0.05):
                status = 'healthy'
                health_status['healthy_models'] += 1
            elif summary['current_error_rate'] > 0.1:
                status = 'critical'
                health_status['critical_models'] += 1
            else:
                status = 'degraded'
                health_status['degraded_models'] += 1
            
            health_status['models'][model_name] = {
                'status': status,
                **summary
            }
        
        return health_status
    
    def check_data_drift(self, model_name: str, current_data: np.ndarray) -> Optional[Dict]:
        """Check for data drift in a specific model"""
        if model_name not in self.drift_detectors:
            return None
        
        return self.drift_detectors[model_name].detect_drift(current_data)


def simulate_model_predictions(monitor: ModelMonitor, n_predictions: int = 100):
    """Simulate model predictions for testing"""
    np.random.seed(42)
    
    for i in range(n_predictions):
        # Simulate varying performance
        accuracy = 0.85 + np.random.rand() * 0.1
        latency = 50 + np.random.rand() * 200
        cpu = 0.3 + np.random.rand() * 0.4
        memory = 0.4 + np.random.rand() * 0.3
        
        # Simulate occasional errors
        error = None
        if np.random.rand() < 0.05:  # 5% error rate
            error = Exception("Simulated prediction error")
        
        monitor.record_prediction(
            accuracy=accuracy,
            latency_ms=latency,
            cpu_usage=cpu,
            memory_usage=memory,
            error=error
        )


def main():
    """Demonstration of AI Observability System"""
    print("=" * 60)
    print("AI Model Observability and Monitoring System")
    print("=" * 60)
    print()
    
    # Initialize dashboard
    dashboard = ObservabilityDashboard()
    
    # Register models
    print("Registering AI models for monitoring...")
    dashboard.register_model("fraud_detection_model")
    dashboard.register_model("recommendation_engine")
    dashboard.register_model("image_classifier")
    print(f"✓ Registered {len(dashboard.monitors)} models")
    print()
    
    # Simulate model operations
    print("Simulating model predictions and collecting metrics...")
    for model_name in dashboard.monitors.keys():
        simulate_model_predictions(dashboard.monitors[model_name], n_predictions=50)
        print(f"  ✓ Collected metrics for {model_name}")
    print()
    
    # Display performance summaries
    print("=" * 60)
    print("Performance Summaries (Last 60 minutes)")
    print("=" * 60)
    for model_name, monitor in dashboard.monitors.items():
        summary = monitor.get_performance_summary()
        if summary.get('status') != 'no_data':
            print(f"\n{model_name}:")
            print(f"  Accuracy: {summary['average_accuracy']:.2%}")
            print(f"  Latency: {summary['average_latency_ms']:.0f} ms")
            print(f"  Throughput: {summary['average_throughput']:.2f} pred/s")
            print(f"  Error Rate: {summary['current_error_rate']:.2%}")
            print(f"  CPU Usage: {summary['average_cpu_usage']:.1%}")
            print(f"  Memory Usage: {summary['average_memory_usage']:.1%}")
    
    # System health overview
    print("\n" + "=" * 60)
    print("System Health Overview")
    print("=" * 60)
    health = dashboard.get_system_health()
    print(f"Total Models: {health['total_models']}")
    print(f"Healthy: {health['healthy_models']}")
    print(f"Degraded: {health['degraded_models']}")
    print(f"Critical: {health['critical_models']}")
    print()


if __name__ == "__main__":
    main()

