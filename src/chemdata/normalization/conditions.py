"""
Experimental condition normalization.
"""

from typing import Dict, Optional, Any
import pandas as pd
import re


class ConditionNormalizer:
    """
    Normalize experimental conditions.
    
    Handles:
    - pH notation
    - Temperature units
    - Light source descriptions
    - Reactor type
    """
    
    LIGHT_SOURCE_SYNONYMS = {
        'uv': 'UV',
        'u.v.': 'UV',
        'ultraviolet': 'UV',
        'visible': 'Visible',
        'vis': 'Visible',
        'solar': 'Solar',
        'sunlight': 'Solar',
        'led': 'LED',
    }
    
    def __init__(self):
        """Initialize condition normalizer."""
        pass
    
    def normalize_ph(self, ph_value: Any) -> Optional[float]:
        """
        Normalize pH value.
        
        Handles:
        - "pH 7.0" -> 7.0
        - "7" -> 7.0
        - "neutral" -> 7.0
        
        Args:
            ph_value: pH value (string or number)
            
        Returns:
            Normalized pH as float
        """
        if pd.isna(ph_value):
            return None
        
        # Already a number
        if isinstance(ph_value, (int, float)):
            return float(ph_value)
        
        # String processing
        if isinstance(ph_value, str):
            ph_str = ph_value.lower().strip()
            
            # Handle text descriptors
            ph_map = {
                'neutral': 7.0,
                'acidic': 5.0,  # Approximate
                'basic': 9.0,   # Approximate
                'alkaline': 9.0,
            }
            
            if ph_str in ph_map:
                return ph_map[ph_str]
            
            # Extract number from "pH 7.0" or "7.0"
            match = re.search(r'(\d+\.?\d*)', ph_str)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
        
        return None
    
    def normalize_light_source(self, light: str) -> str:
        """
        Normalize light source description.
        
        Args:
            light: Light source description
            
        Returns:
            Normalized light source
        """
        if not isinstance(light, str):
            return light
        
        light_lower = light.lower().strip()
        
        # Check synonyms
        for key, standard in self.LIGHT_SOURCE_SYNONYMS.items():
            if key in light_lower:
                return standard
        
        return light
    
    def normalize_reactor_type(self, reactor: str) -> str:
        """
        Normalize reactor type.
        
        Args:
            reactor: Reactor description
            
        Returns:
            Normalized reactor type
        """
        if not isinstance(reactor, str):
            return reactor
        
        reactor_lower = reactor.lower().strip()
        
        reactor_map = {
            'batch': 'Batch',
            'continuous': 'Continuous',
            'slurry': 'Slurry',
            'fixed bed': 'Fixed Bed',
            'fluidized': 'Fluidized Bed',
        }
        
        for key, standard in reactor_map.items():
            if key in reactor_lower:
                return standard
        
        return reactor
    
    def normalize_boolean_condition(self, value: Any) -> Optional[bool]:
        """
        Normalize boolean conditions (e.g., "air", "oxygen", "dark").
        
        Args:
            value: Boolean value in various formats
            
        Returns:
            True, False, or None
        """
        if pd.isna(value):
            return None
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            value_lower = value.lower().strip()
            
            true_values = {'yes', 'true', '1', 'present', 'with', 'air', 'oxygen'}
            false_values = {'no', 'false', '0', 'absent', 'without', 'dark', 'nitrogen'}
            
            if value_lower in true_values:
                return True
            if value_lower in false_values:
                return False
        
        return None