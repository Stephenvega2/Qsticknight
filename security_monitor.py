import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

class SecurityBreachDetector:
    def __init__(self, window_size: int = 100):
        """Initialize the security breach detector with sliding window"""
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        self.entropy_window = deque(maxlen=window_size)
        self.security_window = deque(maxlen=window_size)
        self.alert_threshold = 0.3  # Threshold for sudden changes
        self.min_samples = 10  # Minimum samples needed for detection
        
    def add_measurement(self, metrics: Dict) -> Dict:
        """Add new measurement and check for security breaches"""
        try:
            # Extract metrics
            entropy = metrics.get('entropy', 0.0)
            channel_security = metrics.get('channel_security', 0.0)
            
            # Add to sliding windows
            self.entropy_window.append(entropy)
            self.security_window.append(channel_security)
            
            # Analyze for breaches
            breach_status = self._analyze_security_status()
            
            return breach_status
            
        except Exception as e:
            self.logger.error(f"Error in breach detection: {str(e)}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'message': f"Detection error: {str(e)}"
            }
    
    def _analyze_security_status(self) -> Dict:
        """Analyze current security status using sliding windows"""
        if len(self.entropy_window) < self.min_samples:
            return {
                'status': 'insufficient_data',
                'timestamp': datetime.now().isoformat(),
                'message': 'Collecting initial data...'
            }
            
        try:
            # Calculate statistical measures
            entropy_mean = np.mean(list(self.entropy_window))
            entropy_std = np.std(list(self.entropy_window))
            security_mean = np.mean(list(self.security_window))
            
            # Get latest values
            current_entropy = self.entropy_window[-1]
            current_security = self.security_window[-1]
            
            # Check for sudden changes
            entropy_zscore = abs(current_entropy - entropy_mean) / (entropy_std if entropy_std > 0 else 1)
            security_drop = security_mean - current_security
            
            # Determine breach status
            breach_detected = False
            alert_messages = []
            
            if entropy_zscore > self.alert_threshold:
                breach_detected = True
                alert_messages.append(f"Abnormal entropy fluctuation detected (z-score: {entropy_zscore:.2f})")
                
            if security_drop > self.alert_threshold:
                breach_detected = True
                alert_messages.append(f"Security level drop detected ({security_drop:.2f})")
            
            return {
                'status': 'breach_detected' if breach_detected else 'secure',
                'timestamp': datetime.now().isoformat(),
                'message': ' | '.join(alert_messages) if breach_detected else 'No security breaches detected',
                'metrics': {
                    'entropy_zscore': float(entropy_zscore),
                    'security_drop': float(security_drop),
                    'current_entropy': float(current_entropy),
                    'current_security': float(current_security)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in security analysis: {str(e)}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'message': f"Analysis error: {str(e)}"
            }
            
    def get_security_metrics(self) -> Dict:
        """Get current security metrics"""
        if len(self.entropy_window) < self.min_samples:
            return {
                'status': 'initializing',
                'entropy_stability': 0.0,
                'security_level': 0.0
            }
            
        try:
            entropy_stability = 1.0 - min(1.0, np.std(list(self.entropy_window)))
            security_level = np.mean(list(self.security_window))
            
            return {
                'status': 'active',
                'entropy_stability': float(entropy_stability),
                'security_level': float(security_level)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating security metrics: {str(e)}")
            return {
                'status': 'error',
                'entropy_stability': 0.0,
                'security_level': 0.0
            }
