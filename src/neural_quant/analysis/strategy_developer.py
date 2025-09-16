"""
Strategy Developer LLM Role

This module implements an LLM role that translates selected R1 experiments
into precise change requests: small param grids, overlay toggles, or targeted code changes.

Author: Neural Quant Team
Date: 2024
"""

import json
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class StrategyDeveloper:
    """
    LLM role that translates selected R1 experiments into precise change requests.
    
    This class converts high-level experiment proposals into specific, actionable
    changes that the platform can execute, focusing on parameter grids and overlay
    settings rather than code changes.
    """
    
    def __init__(self):
        self.strategy_catalog = {
            'MovingAverageCrossover': {
                'params': ['ma_fast', 'ma_slow', 'threshold'],
                'param_ranges': {
                    'ma_fast': [5, 10, 15, 20],
                    'ma_slow': [20, 30, 40, 50],
                    'threshold': [0.001, 0.002, 0.005, 0.01]
                }
            },
            'BollingerBands': {
                'params': ['window', 'std_dev', 'threshold'],
                'param_ranges': {
                    'window': [10, 15, 20, 25],
                    'std_dev': [1.5, 2.0, 2.5, 3.0],
                    'threshold': [0.001, 0.002, 0.005]
                }
            },
            'VolatilityBreakout': {
                'params': ['window', 'multiplier', 'threshold'],
                'param_ranges': {
                    'window': [10, 15, 20, 25],
                    'multiplier': [1.5, 2.0, 2.5, 3.0],
                    'threshold': [0.001, 0.002, 0.005]
                }
            },
            'CrossSectionalMomentum': {
                'params': ['lookback', 'rebalance_freq', 'top_n'],
                'param_ranges': {
                    'lookback': [10, 15, 20, 25],
                    'rebalance_freq': ['weekly', 'monthly'],
                    'top_n': [3, 5, 7, 10]
                }
            }
        }
        
        self.overlay_features = {
            'regime': {
                'options': ['bull_SPY_200D', 'bear_SPY_200D', 'none'],
                'default': 'none'
            },
            'vol_target': {
                'options': [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
                'default': 0.10
            },
            'alloc': {
                'options': ['equal', 'vol_weight'],
                'default': 'equal'
            },
            'position_cap': {
                'options': [0.10, 0.12, 0.15, 0.20, 0.25],
                'default': 0.15
            },
            'stops': {
                'atr': {
                    'on': [True, False],
                    'window': [10, 14, 21],
                    'mult': [2.0, 2.5, 3.0]
                },
                'time': {
                    'on': [True, False],
                    'bars': [20, 30, 45, 60]
                }
            }
        }
        
        self.unit_tests = [
            'unit:test_no_lookahead',
            'unit:test_fee_application',
            'unit:test_stop_exit_paths',
            'unit:test_regime_filtering',
            'unit:test_volatility_targeting',
            'unit:test_allocation_methods',
            'unit:test_position_caps',
            'unit:test_basic_exits',
            'unit:test_portfolio_rebalancing',
            'unit:test_signal_generation',
            'unit:test_equity_calculation',
            'unit:test_metrics_calculation'
        ]
    
    def translate_experiments(self, 
                            selected_experiments: List[Dict[str, Any]],
                            max_grid_size: int = 9) -> Dict[str, Any]:
        """
        Translate selected R1 experiments into precise change requests.
        
        Args:
            selected_experiments: List of selected experiments from R1
            max_grid_size: Maximum grid size per sweep (default 9 for 3x3)
            
        Returns:
            Dictionary containing changes, sweeps, and tests
        """
        try:
            changes = []
            sweeps = []
            tests = set()
            
            for experiment in selected_experiments:
                # Generate parameter grid changes
                param_changes = self._generate_param_changes(experiment, max_grid_size)
                changes.extend(param_changes)
                
                # Generate overlay changes
                overlay_changes = self._generate_overlay_changes(experiment)
                changes.extend(overlay_changes)
                
                # Generate code changes if needed
                code_changes = self._generate_code_changes(experiment)
                changes.extend(code_changes)
                
                # Generate sweeps
                sweep = self._generate_sweep(experiment, max_grid_size)
                if sweep:
                    sweeps.append(sweep)
                
                # Add relevant tests
                experiment_tests = self._identify_tests(experiment)
                tests.update(experiment_tests)
            
            # Compile results
            result = {
                'changes': changes,
                'sweeps': sweeps,
                'tests': list(tests)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error translating experiments: {e}")
            return self._generate_error_response(str(e))
    
    def _generate_param_changes(self, 
                              experiment: Dict[str, Any], 
                              max_grid_size: int) -> List[Dict[str, Any]]:
        """Generate parameter grid changes for an experiment."""
        changes = []
        
        strategy_name = experiment.get('strategy', '')
        params = experiment.get('params', {})
        
        if strategy_name not in self.strategy_catalog:
            return changes
        
        strategy_info = self.strategy_catalog[strategy_name]
        available_params = strategy_info['param_ranges']
        
        # Create parameter grid for each parameter
        for param_name, param_value in params.items():
            if param_name in available_params:
                # Generate small grid around the suggested value
                suggested_values = self._generate_param_grid(
                    param_name, param_value, available_params[param_name], max_grid_size
                )
                
                if len(suggested_values) > 1:  # Only create grid if multiple values
                    changes.append({
                        'type': 'param_grid',
                        'strategy': strategy_name,
                        'grid': {param_name: suggested_values}
                    })
        
        return changes
    
    def _generate_param_grid(self, 
                           param_name: str, 
                           suggested_value: Any, 
                           available_values: List[Any], 
                           max_grid_size: int) -> List[Any]:
        """Generate a small parameter grid around the suggested value."""
        if isinstance(suggested_value, (int, float)):
            # For numeric parameters, find values around the suggested one
            available_numeric = [v for v in available_values if isinstance(v, (int, float))]
            if not available_numeric:
                return [suggested_value]
            
            # Find the closest values
            sorted_values = sorted(available_numeric)
            try:
                closest_idx = min(range(len(sorted_values)), 
                                key=lambda i: abs(sorted_values[i] - suggested_value))
            except (ValueError, TypeError):
                return [suggested_value]
            
            # Get values around the closest one
            start_idx = max(0, closest_idx - 1)
            end_idx = min(len(sorted_values), closest_idx + 2)
            grid_values = sorted_values[start_idx:end_idx]
            
            # Limit to max_grid_size
            if len(grid_values) > max_grid_size:
                grid_values = grid_values[:max_grid_size]
            
            return grid_values
        else:
            # For non-numeric parameters, return the suggested value
            return [suggested_value]
    
    def _generate_overlay_changes(self, experiment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate overlay changes for an experiment."""
        changes = []
        overlays = experiment.get('overlays', {})
        
        for overlay_name, overlay_settings in overlays.items():
            if overlay_name == 'stops':
                # Handle stops specially as it has nested structure
                for stop_type, stop_settings in overlay_settings.items():
                    if isinstance(stop_settings, dict) and stop_settings.get('on', False):
                        changes.append({
                            'type': 'overlay_add',
                            'name': f'stops_{stop_type}',
                            'settings': stop_settings
                        })
            else:
                # Handle other overlays
                if overlay_settings is not None and overlay_settings != 'none':
                    changes.append({
                        'type': 'overlay_update',
                        'name': overlay_name,
                        'settings': {overlay_name: overlay_settings}
                    })
        
        return changes
    
    def _generate_code_changes(self, experiment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate code changes if essential (rare)."""
        changes = []
        
        # Only suggest code changes for specific cases
        strategy_name = experiment.get('strategy', '')
        overlays = experiment.get('overlays', {})
        
        # Check if we need to add new strategy
        if strategy_name not in self.strategy_catalog:
            changes.append({
                'type': 'code_change',
                'file': 'src/neural_quant/strategies/',
                'function': 'create_new_strategy',
                'description': f"Add new strategy class {strategy_name} with required parameters and signal generation logic"
            })
        
        # Check if we need to modify existing strategy for new features
        if overlays.get('regime') and overlays['regime'] != 'none':
            changes.append({
                'type': 'code_change',
                'file': 'src/neural_quant/strategies/base/strategy_base.py',
                'function': 'add_regime_filtering',
                'description': "Add regime filtering capability to strategy base class to support market regime-based trading"
            })
        
        return changes
    
    def _generate_sweep(self, 
                       experiment: Dict[str, Any], 
                       max_grid_size: int) -> Optional[Dict[str, Any]]:
        """Generate parameter sweep for an experiment."""
        strategy_name = experiment.get('strategy', '')
        params = experiment.get('params', {})
        
        if strategy_name not in self.strategy_catalog:
            return None
        
        # Create sweep with multiple parameters
        sweep_params = {}
        for param_name, param_value in params.items():
            if param_name in self.strategy_catalog[strategy_name]['param_ranges']:
                available_values = self.strategy_catalog[strategy_name]['param_ranges'][param_name]
                grid_values = self._generate_param_grid(param_name, param_value, available_values, max_grid_size)
                if len(grid_values) > 1:
                    sweep_params[param_name] = grid_values
        
        if not sweep_params:
            return None
        
        # Calculate total grid size
        total_combinations = 1
        for param_values in sweep_params.values():
            total_combinations *= len(param_values)
        
        if total_combinations > max_grid_size:
            # Reduce grid size by taking fewer values per parameter
            for param_name in sweep_params:
                if len(sweep_params[param_name]) > 2:
                    sweep_params[param_name] = sweep_params[param_name][:2]
                    total_combinations = 1
                    for param_values in sweep_params.values():
                        total_combinations *= len(param_values)
                    if total_combinations <= max_grid_size:
                        break
        
        return {
            'strategy': strategy_name,
            'params': sweep_params,
            'max_grid': min(total_combinations, max_grid_size)
        }
    
    def _identify_tests(self, experiment: Dict[str, Any]) -> List[str]:
        """Identify relevant unit tests for an experiment."""
        tests = []
        
        strategy_name = experiment.get('strategy', '')
        overlays = experiment.get('overlays', {})
        
        # Always include basic tests
        tests.extend([
            'unit:test_no_lookahead',
            'unit:test_fee_application'
        ])
        
        # Add strategy-specific tests
        if strategy_name:
            tests.append(f'unit:test_{strategy_name.lower()}_strategy')
        
        # Add overlay-specific tests
        if overlays.get('regime') and overlays['regime'] != 'none':
            tests.append('unit:test_regime_filtering')
        
        if overlays.get('vol_target'):
            tests.append('unit:test_volatility_targeting')
        
        if overlays.get('alloc') == 'vol_weight':
            tests.append('unit:test_allocation_methods')
        
        if overlays.get('position_cap'):
            tests.append('unit:test_position_caps')
        
        if overlays.get('stops', {}).get('atr', {}).get('on'):
            tests.append('unit:test_stop_exit_paths')
        
        if overlays.get('stops', {}).get('time', {}).get('on'):
            tests.append('unit:test_time_stops')
        
        return list(set(tests))  # Remove duplicates
    
    def _generate_error_response(self, error: str) -> Dict[str, Any]:
        """Generate error response when translation fails."""
        return {
            'changes': [],
            'sweeps': [],
            'tests': ['unit:test_error_handling'],
            'error': error
        }
    
    def format_as_json(self, result: Dict[str, Any]) -> str:
        """Format result as strict JSON."""
        try:
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error formatting JSON: {e}")
            return json.dumps({'error': str(e)}, indent=2)
    
    def validate_json(self, json_str: str) -> bool:
        """Validate that the JSON is properly formatted."""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Strategy Developer role."""
        return """
You are a strategy developer. Translate the selected experiments into precise, minimal change requests the platform can execute. Prefer param grids and overlay settings over code edits. If code edit is essential, specify file, function, and concise change description (no code). Include unit tests to add or run. Output strict JSON only matching the schema.

Available Strategies:
- MovingAverageCrossover: ma_fast, ma_slow, threshold
- BollingerBands: window, std_dev, threshold  
- VolatilityBreakout: window, multiplier, threshold
- CrossSectionalMomentum: lookback, rebalance_freq, top_n

Available Overlays:
- regime: bull_SPY_200D, bear_SPY_200D, none
- vol_target: 0.05-0.20
- alloc: equal, vol_weight
- position_cap: 0.10-0.25
- stops: atr (window, mult), time (bars)

Output strict JSON only matching the schema.
"""
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        return list(self.strategy_catalog.keys())
    
    def get_available_overlays(self) -> Dict[str, Any]:
        """Get available overlay options."""
        return self.overlay_features.copy()
    
    def get_available_tests(self) -> List[str]:
        """Get list of available unit tests."""
        return self.unit_tests.copy()
