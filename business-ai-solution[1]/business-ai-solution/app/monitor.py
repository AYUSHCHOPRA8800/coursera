import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import json
import os
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

class PerformanceMonitor:
    """Performance monitoring system for the AI application"""
    
    def __init__(self, metrics_dir: str = "../logs"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.metrics = {
            'prediction_count': 0,
            'prediction_errors': 0,
            'prediction_times': deque(maxlen=1000),
            'batch_prediction_times': deque(maxlen=100),
            'prediction_accuracies': deque(maxlen=1000),
            'api_requests': 0,
            'api_errors': 0,
            'model_load_time': None,
            'last_prediction_time': None,
            'uptime_start': datetime.now(),
            'country_predictions': defaultdict(int),
            'error_types': defaultdict(int)
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            'max_prediction_time': 5.0,  # seconds
            'min_accuracy': 0.7,
            'max_error_rate': 0.1
        }
        
        # Start background monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitor_thread.start()
    
    def record_prediction(self, country: Optional[str] = None, 
                        prediction_time: float = 0.0, 
                        accuracy: float = 1.0,
                        error: Optional[str] = None):
        """Record a prediction event"""
        with self.lock:
            self.metrics['prediction_count'] += 1
            self.metrics['last_prediction_time'] = datetime.now()
            
            if prediction_time > 0:
                self.metrics['prediction_times'].append(prediction_time)
            
            if accuracy > 0:
                self.metrics['prediction_accuracies'].append(accuracy)
            
            if country:
                self.metrics['country_predictions'][country] += 1
            
            if error:
                self.metrics['prediction_errors'] += 1
                self.metrics['error_types'][error] += 1
    
    def record_prediction_time(self, prediction_time: float):
        """Record prediction processing time"""
        with self.lock:
            self.metrics['prediction_times'].append(prediction_time)
            
            # Check if threshold exceeded
            if prediction_time > self.thresholds['max_prediction_time']:
                logger.warning(f"Prediction time threshold exceeded: {prediction_time:.2f}s")
    
    def record_batch_prediction_time(self, batch_time: float):
        """Record batch prediction processing time"""
        with self.lock:
            self.metrics['batch_prediction_times'].append(batch_time)
    
    def record_prediction_accuracy(self, accuracy: float):
        """Record prediction accuracy"""
        with self.lock:
            self.metrics['prediction_accuracies'].append(accuracy)
            
            # Check if threshold exceeded
            if accuracy < self.thresholds['min_accuracy']:
                logger.warning(f"Accuracy threshold exceeded: {accuracy:.3f}")
    
    def record_error(self, error_type: str = "unknown"):
        """Record an error"""
        with self.lock:
            self.metrics['prediction_errors'] += 1
            self.metrics['error_types'][error_type] += 1
            self.metrics['api_errors'] += 1
    
    def record_api_request(self):
        """Record an API request"""
        with self.lock:
            self.metrics['api_requests'] += 1
    
    def record_model_load_time(self, load_time: float):
        """Record model loading time"""
        with self.lock:
            self.metrics['model_load_time'] = load_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.lock:
            metrics = self.metrics.copy()
            
            # Calculate derived metrics
            if metrics['prediction_times']:
                metrics['avg_prediction_time'] = sum(metrics['prediction_times']) / len(metrics['prediction_times'])
                metrics['max_prediction_time'] = max(metrics['prediction_times'])
                metrics['min_prediction_time'] = min(metrics['prediction_times'])
            else:
                metrics['avg_prediction_time'] = 0
                metrics['max_prediction_time'] = 0
                metrics['min_prediction_time'] = 0
            
            if metrics['batch_prediction_times']:
                metrics['avg_batch_prediction_time'] = sum(metrics['batch_prediction_times']) / len(metrics['batch_prediction_times'])
            else:
                metrics['avg_batch_prediction_time'] = 0
            
            if metrics['prediction_accuracies']:
                metrics['avg_accuracy'] = sum(metrics['prediction_accuracies']) / len(metrics['prediction_accuracies'])
                metrics['min_accuracy'] = min(metrics['prediction_accuracies'])
                metrics['max_accuracy'] = max(metrics['prediction_accuracies'])
            else:
                metrics['avg_accuracy'] = 0
                metrics['min_accuracy'] = 0
                metrics['max_accuracy'] = 0
            
            # Calculate error rate
            if metrics['prediction_count'] > 0:
                metrics['error_rate'] = metrics['prediction_errors'] / metrics['prediction_count']
            else:
                metrics['error_rate'] = 0
            
            # Calculate uptime
            metrics['uptime'] = str(datetime.now() - metrics['uptime_start'])
            
            # Convert deques to lists for JSON serialization
            metrics['prediction_times'] = list(metrics['prediction_times'])
            metrics['batch_prediction_times'] = list(metrics['batch_prediction_times'])
            metrics['prediction_accuracies'] = list(metrics['prediction_accuracies'])
            metrics['country_predictions'] = dict(metrics['country_predictions'])
            metrics['error_types'] = dict(metrics['error_types'])
            
            return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a performance summary"""
        metrics = self.get_metrics()
        
        summary = {
            'total_predictions': metrics['prediction_count'],
            'total_errors': metrics['prediction_errors'],
            'error_rate': f"{metrics['error_rate']:.2%}",
            'avg_prediction_time': f"{metrics['avg_prediction_time']:.3f}s",
            'avg_accuracy': f"{metrics['avg_accuracy']:.3f}",
            'uptime': metrics['uptime'],
            'status': self._get_system_status(metrics)
        }
        
        return summary
    
    def _get_system_status(self, metrics: Dict[str, Any]) -> str:
        """Determine system status based on metrics"""
        if metrics['error_rate'] > self.thresholds['max_error_rate']:
            return "degraded"
        elif metrics['avg_prediction_time'] > self.thresholds['max_prediction_time']:
            return "slow"
        elif metrics['avg_accuracy'] < self.thresholds['min_accuracy']:
            return "low_accuracy"
        else:
            return "healthy"
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            metrics = self.get_metrics()
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Save current metrics
            metrics_file = self.metrics_dir / "current_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save historical metrics
            history_file = self.metrics_dir / "metrics_history.json"
            historical_metrics = []
            
            if history_file.exists():
                with open(history_file, 'r') as f:
                    historical_metrics = json.load(f)
            
            # Keep only last 100 entries
            historical_metrics.append(metrics)
            if len(historical_metrics) > 100:
                historical_metrics = historical_metrics[-100:]
            
            with open(history_file, 'w') as f:
                json.dump(historical_metrics, f, indent=2)
            
            logger.debug("Metrics saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _background_monitoring(self):
        """Background monitoring thread"""
        while self.monitoring_active:
            try:
                # Save metrics every 5 minutes
                time.sleep(300)
                self.save_metrics()
                
                # Check for performance issues
                self._check_performance_alerts()
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
    
    def _check_performance_alerts(self):
        """Check for performance issues and log alerts"""
        metrics = self.get_metrics()
        
        # Check error rate
        if metrics['error_rate'] > self.thresholds['max_error_rate']:
            logger.warning(f"High error rate detected: {metrics['error_rate']:.2%}")
        
        # Check prediction time
        if metrics['avg_prediction_time'] > self.thresholds['max_prediction_time']:
            logger.warning(f"Slow prediction time detected: {metrics['avg_prediction_time']:.3f}s")
        
        # Check accuracy
        if metrics['avg_accuracy'] < self.thresholds['min_accuracy']:
            logger.warning(f"Low accuracy detected: {metrics['avg_accuracy']:.3f}")
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        metrics = self.get_metrics()
        
        prometheus_metrics = []
        
        # Counter metrics
        prometheus_metrics.append(f"# HELP prediction_count Total number of predictions")
        prometheus_metrics.append(f"# TYPE prediction_count counter")
        prometheus_metrics.append(f"prediction_count {metrics['prediction_count']}")
        
        prometheus_metrics.append(f"# HELP prediction_errors Total number of prediction errors")
        prometheus_metrics.append(f"# TYPE prediction_errors counter")
        prometheus_metrics.append(f"prediction_errors {metrics['prediction_errors']}")
        
        prometheus_metrics.append(f"# HELP api_requests Total number of API requests")
        prometheus_metrics.append(f"# TYPE api_requests counter")
        prometheus_metrics.append(f"api_requests {metrics['api_requests']}")
        
        # Gauge metrics
        prometheus_metrics.append(f"# HELP avg_prediction_time Average prediction time in seconds")
        prometheus_metrics.append(f"# TYPE avg_prediction_time gauge")
        prometheus_metrics.append(f"avg_prediction_time {metrics['avg_prediction_time']}")
        
        prometheus_metrics.append(f"# HELP avg_accuracy Average prediction accuracy")
        prometheus_metrics.append(f"# TYPE avg_accuracy gauge")
        prometheus_metrics.append(f"avg_accuracy {metrics['avg_accuracy']}")
        
        prometheus_metrics.append(f"# HELP error_rate Current error rate")
        prometheus_metrics.append(f"# TYPE error_rate gauge")
        prometheus_metrics.append(f"error_rate {metrics['error_rate']}")
        
        # Country-specific metrics
        for country, count in metrics['country_predictions'].items():
            prometheus_metrics.append(f"# HELP predictions_by_country Predictions by country")
            prometheus_metrics.append(f"# TYPE predictions_by_country counter")
            prometheus_metrics.append(f'predictions_by_country{{country="{country}"}} {count}')
        
        return "\n".join(prometheus_metrics)
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics = {
                'prediction_count': 0,
                'prediction_errors': 0,
                'prediction_times': deque(maxlen=1000),
                'batch_prediction_times': deque(maxlen=100),
                'prediction_accuracies': deque(maxlen=1000),
                'api_requests': 0,
                'api_errors': 0,
                'model_load_time': None,
                'last_prediction_time': None,
                'uptime_start': datetime.now(),
                'country_predictions': defaultdict(int),
                'error_types': defaultdict(int)
            }
        
        logger.info("Metrics reset")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Save final metrics
        self.save_metrics()
        logger.info("Performance monitoring stopped")

# Global monitor instance
_monitor_instance = None

def get_monitor() -> PerformanceMonitor:
    """Get the global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance
