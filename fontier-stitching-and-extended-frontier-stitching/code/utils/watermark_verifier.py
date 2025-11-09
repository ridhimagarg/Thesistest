"""
Watermark verification utilities for statistical model theft detection.

This module provides statistical methods to verify if a suspected model
has been stolen based on watermark accuracy.
"""

import numpy as np
from typing import Dict, Optional
from scipy import stats


class WatermarkVerifier:
    """Statistical watermark verification for model theft detection."""
    
    def __init__(self, victim_acc: float, num_classes: int = 10, 
                 watermark_size: Optional[int] = None):
        """
        Initialize watermark verifier.
        
        Args:
            victim_acc: Accuracy of the victim model on watermark set.
            num_classes: Number of classes in the dataset.
            watermark_size: Size of the watermark set (for statistical tests).
        """
        self.victim_acc = victim_acc
        self.num_classes = num_classes
        self.watermark_size = watermark_size
        self.random_acc = 1.0 / num_classes
    
    def verify_theft(self, suspected_acc: float,
                    threshold_ratio: float = 0.5,
                    confidence: float = 0.99) -> Dict[str, float]:
        """
        Verify if model is stolen based on watermark accuracy.
        
        Uses a statistical threshold: suspicious_acc > random_acc + 
        threshold_ratio * (victim_acc - random_acc)
        
        Args:
            suspected_acc: Accuracy of suspected model on watermark set.
            threshold_ratio: Ratio for threshold calculation (0.0 to 1.0).
                            Higher values = stricter threshold.
            confidence: Confidence level for statistical test (0.0 to 1.0).
        
        Returns:
            Dictionary with:
                - 'is_stolen': bool indicating if model is likely stolen
                - 'confidence': float confidence level (0.0 to 1.0)
                - 'p_value': float p-value from statistical test
                - 'threshold': float threshold used for comparison
                - 'suspected_acc': float suspected model accuracy
                - 'victim_acc': float victim model accuracy
                - 'random_acc': float random baseline accuracy
                - 'margin': float difference between suspected and threshold
        """
        # Calculate threshold: suspicious_acc > random_acc + threshold_ratio * (victim_acc - random_acc)
        threshold = self.random_acc + threshold_ratio * (self.victim_acc - self.random_acc)
        
        # Statistical test (if watermark size is known)
        p_value = self._calculate_p_value(suspected_acc) if self.watermark_size else None
        
        # Determine if stolen: must exceed threshold AND pass statistical test
        # is_stolen = (suspected_acc > threshold) and (p_value < (1 - confidence))
        if p_value is not None:
            is_stolen = (suspected_acc > threshold) and (p_value < (1 - confidence))
        else:
            # If no p-value available, only use threshold
            is_stolen = suspected_acc > threshold
        
        # Calculate confidence: 1 - p_value if stolen, else 0
        if is_stolen and p_value is not None:
            confidence_score = 1 - p_value
        elif is_stolen:
            # If stolen but no p-value, use margin-based confidence
            margin = suspected_acc - threshold
            acc_range = self.victim_acc - self.random_acc
            if acc_range > 0:
                confidence_score = min(1.0, max(0.0, margin / acc_range))
            else:
                confidence_score = 0.0
        else:
            confidence_score = 0.0
        
        return {
            'is_stolen': is_stolen,
            'confidence': confidence_score,
            'p_value': p_value if p_value is not None else 0.0,
            'threshold': threshold,
            'suspected_acc': suspected_acc,
            'victim_acc': self.victim_acc,
            'random_acc': self.random_acc
        }
    
    def _calculate_p_value(self, suspected_acc: float) -> Optional[float]:
        """
        Calculate p-value for statistical test.
        
        Uses binomial test: H0 = model performs at random level,
        H1 = model performs better than random.
        
        Args:
            suspected_acc: Accuracy of suspected model.
            
        Returns:
            p-value or None if watermark_size is not set.
        """
        if self.watermark_size is None:
            return None
        
        # Number of correct predictions
        n_correct = int(suspected_acc * self.watermark_size)
        
        # Binomial test: probability of getting n_correct or more
        # under null hypothesis (random guessing)
        p_value = 1 - stats.binom.cdf(n_correct - 1, self.watermark_size, self.random_acc)
        
        return float(p_value)
    
    def _calculate_confidence(self, margin: float, threshold: float,
                            p_value: Optional[float],
                            requested_confidence: float) -> float:
        """
        Calculate confidence score based on margin and p-value.
        
        Args:
            margin: Difference between suspected_acc and threshold.
            threshold: Threshold used for comparison.
            p_value: p-value from statistical test.
            requested_confidence: Requested confidence level.
            
        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Base confidence from margin (normalized to [0, 1])
        # Margin is normalized by the range [random_acc, victim_acc]
        acc_range = self.victim_acc - self.random_acc
        if acc_range > 0:
            margin_confidence = min(1.0, max(0.0, margin / acc_range))
        else:
            margin_confidence = 0.0
        
        # Combine with p-value if available
        if p_value is not None:
            # Higher confidence if p-value is lower
            p_confidence = 1.0 - min(1.0, p_value)
            # Weighted combination
            confidence = 0.6 * margin_confidence + 0.4 * p_confidence
        else:
            confidence = margin_confidence
        
        # Ensure confidence meets requested level
        if confidence < requested_confidence:
            confidence = 0.0
        
        return float(confidence)
    
    def get_threshold(self, threshold_ratio: float = 0.5) -> float:
        """
        Get the threshold for a given threshold ratio.
        
        Args:
            threshold_ratio: Ratio for threshold calculation.
            
        Returns:
            Threshold value.
        """
        return self.random_acc + threshold_ratio * (self.victim_acc - self.random_acc)
    
    def compare_models(self, suspected_acc: float, 
                      threshold_ratio: float = 0.5) -> Dict[str, float]:
        """
        Compare suspected model against victim model.
        
        Args:
            suspected_acc: Accuracy of suspected model.
            threshold_ratio: Ratio for threshold calculation.
            
        Returns:
            Dictionary with comparison metrics.
        """
        threshold = self.get_threshold(threshold_ratio)
        
        return {
            'suspected_acc': suspected_acc,
            'victim_acc': self.victim_acc,
            'random_acc': self.random_acc,
            'threshold': threshold,
            'margin_from_threshold': suspected_acc - threshold,
            'margin_from_victim': suspected_acc - self.victim_acc,
            'margin_from_random': suspected_acc - self.random_acc,
            'is_above_threshold': suspected_acc > threshold,
            'is_close_to_victim': abs(suspected_acc - self.victim_acc) < 0.05
        }


def create_verifier(victim_acc: float, num_classes: int = 10,
                   watermark_size: Optional[int] = None) -> WatermarkVerifier:
    """
    Convenience function to create a WatermarkVerifier.
    
    Args:
        victim_acc: Accuracy of victim model on watermark set.
        num_classes: Number of classes.
        watermark_size: Size of watermark set.
        
    Returns:
        WatermarkVerifier instance.
    """
    return WatermarkVerifier(victim_acc, num_classes, watermark_size)

