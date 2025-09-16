"""
Quant Researcher LLM Role

This module implements an LLM role that proposes testable experiments
based on the latest Run Bundle artifacts, focusing on robustness improvements.

Author: Neural Quant Team
Date: 2024
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class QuantResearcher:
    """
    LLM role that proposes testable experiments based on run artifacts.
    
    This class analyzes run artifacts and generates structured experiment
    proposals focused on improving robustness rather than just returns.
    """
    
    def __init__(self):
        self.promotion_rules = {
            'min_sharpe': 1.0,
            'min_sortino': 1.2,
            'max_drawdown': 0.12,
            'min_profit_factor': 1.2,
            'max_mcpt_p_value': 0.05,
            'min_significant_windows': 0.50,
            'min_stress_sharpe': 0.7
        }
        
        self.available_features = {
            'regime_filters': ['bull_SPY_200D', 'bear_SPY_200D', 'none'],
            'allocation_methods': ['equal', 'vol_weight'],
            'vol_targets': [0.05, 0.10, 0.15, 0.20],
            'position_caps': [0.10, 0.15, 0.20, 0.25],
            'stop_types': ['atr', 'time', 'both', 'none']
        }
    
    def generate_experiments(self, 
                           artifacts: Dict[str, Any],
                           universe: List[str],
                           max_experiments: int = 3) -> Dict[str, Any]:
        """
        Generate testable experiments based on run artifacts.
        
        Args:
            artifacts: Dictionary containing run artifacts
            universe: List of available tickers
            max_experiments: Maximum number of experiments to propose
            
        Returns:
            Dictionary containing hypotheses, experiments, and data needs
        """
        try:
            # Extract key metrics from artifacts
            metrics = artifacts.get('metrics', {})
            params = artifacts.get('params', {})
            mcpt_results = artifacts.get('mcpt_results', {})
            bootstrap_results = artifacts.get('bootstrap_results', {})
            walkforward_results = artifacts.get('walkforward_results', {})
            
            # Analyze current performance
            analysis = self._analyze_current_performance(metrics, mcpt_results, bootstrap_results, walkforward_results)
            
            # Generate hypotheses
            hypotheses = self._generate_hypotheses(analysis, universe)
            
            # Generate experiments
            experiments = self._generate_experiments(analysis, universe, max_experiments)
            
            # Identify data needs
            next_data_needs = self._identify_data_needs(analysis, experiments)
            
            # Compile results
            result = {
                'hypotheses': hypotheses,
                'experiments': experiments,
                'next_data_needs': next_data_needs
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating experiments: {e}")
            return self._generate_error_response(str(e))
    
    def _analyze_current_performance(self, 
                                   metrics: Dict[str, Any],
                                   mcpt_results: Dict[str, Any],
                                   bootstrap_results: Dict[str, Any],
                                   walkforward_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current performance to identify improvement areas."""
        analysis = {
            'current_metrics': metrics,
            'performance_gaps': [],
            'statistical_issues': [],
            'risk_concerns': [],
            'opportunities': []
        }
        
        # Check promotion criteria gaps
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < self.promotion_rules['min_sharpe']:
            analysis['performance_gaps'].append(f"Sharpe {sharpe:.2f} below threshold {self.promotion_rules['min_sharpe']}")
        
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd > self.promotion_rules['max_drawdown']:
            analysis['risk_concerns'].append(f"Max drawdown {max_dd:.1%} exceeds threshold {self.promotion_rules['max_drawdown']:.1%}")
        
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor < self.promotion_rules['min_profit_factor']:
            analysis['performance_gaps'].append(f"Profit factor {profit_factor:.2f} below threshold {self.promotion_rules['min_profit_factor']}")
        
        # Check statistical significance
        if mcpt_results and 'results' in mcpt_results:
            sharpe_mcpt = next((r for r in mcpt_results['results'] if r.get('metric_name') == 'sharpe_ratio'), None)
            if sharpe_mcpt:
                p_value = sharpe_mcpt.get('p_value', 1.0)
                if p_value > self.promotion_rules['max_mcpt_p_value']:
                    analysis['statistical_issues'].append(f"MCPT p-value {p_value:.3f} above threshold {self.promotion_rules['max_mcpt_p_value']}")
        
        # Check walk-forward significance
        if walkforward_results and 'rolling_p_values' in walkforward_results:
            p_values = walkforward_results['rolling_p_values']
            significant_windows = sum(1 for p in p_values if p <= 0.05)
            significant_ratio = significant_windows / len(p_values) if p_values else 0
            if significant_ratio < self.promotion_rules['min_significant_windows']:
                analysis['statistical_issues'].append(f"Walk-forward significance {significant_ratio:.1%} below threshold {self.promotion_rules['min_significant_windows']:.1%}")
        
        # Identify opportunities
        if sharpe > 0.5 and sharpe < 1.0:
            analysis['opportunities'].append("Moderate Sharpe suggests room for optimization")
        
        if max_dd < 0.05:
            analysis['opportunities'].append("Low drawdown allows for increased risk-taking")
        
        return analysis
    
    def _generate_hypotheses(self, analysis: Dict[str, Any], universe: List[str]) -> List[Dict[str, Any]]:
        """Generate testable hypotheses based on analysis."""
        hypotheses = []
        
        # Hypothesis 1: Regime-based improvements
        if analysis['performance_gaps'] or analysis['statistical_issues']:
            hypotheses.append({
                'id': 'H1',
                'rationale': f"Current performance gaps: {', '.join(analysis['performance_gaps'][:2])}. Regime filtering may improve signal quality.",
                'regimes': ['bull_SPY_200D', 'bear_SPY_200D'],
                'tickers': universe[:5]  # Top 5 tickers
            })
        
        # Hypothesis 2: Volatility targeting
        if 'Max drawdown' in str(analysis['risk_concerns']):
            hypotheses.append({
                'id': 'H2',
                'rationale': f"Risk concerns: {', '.join(analysis['risk_concerns'][:2])}. Volatility targeting may reduce drawdowns.",
                'regimes': ['none'],
                'tickers': universe
            })
        
        # Hypothesis 3: Allocation optimization
        if analysis['opportunities']:
            hypotheses.append({
                'id': 'H3',
                'rationale': f"Opportunities identified: {', '.join(analysis['opportunities'][:2])}. Volatility-weighted allocation may improve risk-adjusted returns.",
                'regimes': ['none'],
                'tickers': universe
            })
        
        return hypotheses[:3]  # Limit to 3 hypotheses
    
    def _generate_experiments(self, 
                            analysis: Dict[str, Any], 
                            universe: List[str], 
                            max_experiments: int) -> List[Dict[str, Any]]:
        """Generate specific experiments based on analysis."""
        experiments = []
        
        # Experiment 1: Regime filtering
        if analysis['performance_gaps'] or analysis['statistical_issues']:
            experiments.append({
                'id': 'E1',
                'strategy': 'MovingAverageCrossover',
                'params': {
                    'ma_fast': 10,
                    'ma_slow': 30,
                    'threshold': 0.002
                },
                'overlays': {
                    'regime': 'bull_SPY_200D',
                    'vol_target': 0.10,
                    'alloc': 'equal',
                    'position_cap': 0.15,
                    'stops': {
                        'atr': {'on': True, 'window': 14, 'mult': 2.5},
                        'time': {'on': True, 'bars': 30}
                    }
                },
                'success_criteria': {
                    'sharpe_test': '>=1.0',
                    'sortino_test': '>=1.2',
                    'maxdd': '<=0.12',
                    'pval_sharpe': '<=0.05',
                    'wf_windows_sig_share': '>=0.5',
                    'profit_factor': '>=1.2',
                    'sharpe_under_2x_costs': '>=0.7'
                },
                'risks': ['parameter_spike', 'turnover', 'data_fragility']
            })
        
        # Experiment 2: Volatility targeting
        if 'Max drawdown' in str(analysis['risk_concerns']):
            experiments.append({
                'id': 'E2',
                'strategy': 'BollingerBands',
                'params': {
                    'window': 20,
                    'std_dev': 2.0,
                    'threshold': 0.001
                },
                'overlays': {
                    'regime': 'none',
                    'vol_target': 0.08,  # Lower target for risk reduction
                    'alloc': 'vol_weight',
                    'position_cap': 0.12,  # Tighter cap
                    'stops': {
                        'atr': {'on': True, 'window': 10, 'mult': 3.0},
                        'time': {'on': False, 'bars': 30}
                    }
                },
                'success_criteria': {
                    'sharpe_test': '>=1.0',
                    'sortino_test': '>=1.2',
                    'maxdd': '<=0.08',  # Stricter drawdown requirement
                    'pval_sharpe': '<=0.05',
                    'wf_windows_sig_share': '>=0.5',
                    'profit_factor': '>=1.2',
                    'sharpe_under_2x_costs': '>=0.7'
                },
                'risks': ['parameter_spike', 'turnover', 'data_fragility']
            })
        
        # Experiment 3: Allocation optimization
        if analysis['opportunities']:
            experiments.append({
                'id': 'E3',
                'strategy': 'CrossSectionalMomentum',
                'params': {
                    'lookback': 20,
                    'rebalance_freq': 'monthly',
                    'top_n': 5
                },
                'overlays': {
                    'regime': 'none',
                    'vol_target': 0.12,  # Higher target for opportunity capture
                    'alloc': 'vol_weight',
                    'position_cap': 0.20,  # Higher cap for concentration
                    'stops': {
                        'atr': {'on': True, 'window': 21, 'mult': 2.0},
                        'time': {'on': True, 'bars': 45}
                    }
                },
                'success_criteria': {
                    'sharpe_test': '>=1.2',  # Higher Sharpe target
                    'sortino_test': '>=1.5',
                    'maxdd': '<=0.12',
                    'pval_sharpe': '<=0.05',
                    'wf_windows_sig_share': '>=0.6',  # Higher significance requirement
                    'profit_factor': '>=1.5',
                    'sharpe_under_2x_costs': '>=0.8'
                },
                'risks': ['parameter_spike', 'turnover', 'data_fragility']
            })
        
        return experiments[:max_experiments]
    
    def _identify_data_needs(self, analysis: Dict[str, Any], experiments: List[Dict[str, Any]]) -> List[str]:
        """Identify additional data needs for experiments."""
        needs = []
        
        # Check if regime filtering is used
        if any(exp['overlays']['regime'] != 'none' for exp in experiments):
            needs.append("SPY daily data for regime filtering")
        
        # Check if volatility targeting is used
        if any(exp['overlays']['vol_target'] is not None for exp in experiments):
            needs.append("Intraday data for volatility targeting")
        
        # Check if vol-weighted allocation is used
        if any(exp['overlays']['alloc'] == 'vol_weight' for exp in experiments):
            needs.append("Historical volatility data for allocation")
        
        # Check if ATR stops are used
        if any(exp['overlays']['stops']['atr']['on'] for exp in experiments):
            needs.append("OHLC data for ATR calculation")
        
        # General needs
        needs.extend([
            "Extended historical data for walk-forward analysis",
            "Market regime classification data",
            "Sector/industry classification for universe expansion"
        ])
        
        return list(set(needs))  # Remove duplicates
    
    def _generate_error_response(self, error: str) -> Dict[str, Any]:
        """Generate error response when experiment generation fails."""
        return {
            'hypotheses': [],
            'experiments': [],
            'next_data_needs': [f"Fix error: {error}"],
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
        """Get the system prompt for the Quant Researcher role."""
        return f"""
You are a quant researcher. Propose testable experiments that improve robustness, not just returns. Use only the provided artifacts (metrics, p-values, confidence intervals, walk-forward summaries) and the declared universe. Reference exact numbers you see (e.g., "Sharpe 0.32, p=0.41"). Do not write code. No brute-force grids larger than 3×3. Each experiment must include explicit success criteria aligned to our promotion rules. Output strict JSON only matching the schema.

Promotion Rules:
- Sharpe ≥ {self.promotion_rules['min_sharpe']}
- Sortino ≥ {self.promotion_rules['min_sortino']}
- MaxDD ≤ {self.promotion_rules['max_drawdown']:.1%}
- MCPT p(Sharpe) ≤ {self.promotion_rules['max_mcpt_p_value']}
- Walk-forward significant windows ≥ {self.promotion_rules['min_significant_windows']:.1%}
- Profit Factor ≥ {self.promotion_rules['min_profit_factor']}
- Sharpe under 2x costs ≥ {self.promotion_rules['min_stress_sharpe']}

Available Platform Features:
- Regime Filters: {', '.join(self.available_features['regime_filters'])}
- Allocation Methods: {', '.join(self.available_features['allocation_methods'])}
- Volatility Targets: {self.available_features['vol_targets']}
- Position Caps: {self.available_features['position_caps']}
- Stop Types: {', '.join(self.available_features['stop_types'])}

Output strict JSON only matching the schema.
"""
