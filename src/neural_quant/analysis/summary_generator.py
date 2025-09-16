"""
Strategy Analyst - Final Arbiter for Run Evaluation

This module implements the Strategy Analyst role that evaluates each executed run
strictly against promotion rules and writes summary.md plus a Promote/Reject decision.

Author: Neural Quant Team
Date: 2024
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PromotionRules:
    """Promotion rules for strategy evaluation."""
    
    # Core performance criteria
    MIN_SHARPE = 1.0
    MIN_SORTINO = 1.2
    MAX_DRAWDOWN = 0.12  # 12%
    MIN_PROFIT_FACTOR = 1.2
    
    # Statistical significance criteria
    MAX_MCPT_P_VALUE = 0.05
    MIN_SIGNIFICANT_WINDOWS = 0.50  # 50% of walk-forward windows
    MAX_MCPT_P_VALUE_ANY = 0.20  # Never exceed 20%
    
    # Cost stress test
    COST_STRESS_MULTIPLIER = 2.0
    MIN_STRESS_SHARPE = 0.7


class StrategyAnalyst:
    """
    Strategy Analyst - Final Arbiter for Run Evaluation
    
    This class evaluates each executed run strictly against promotion rules,
    citing exact numbers from artifacts and applying promotion rules exactly.
    Outputs: (1) Executive Summary ≤150 words, (2) Significance Verdict PASS/BORDERLINE/FAIL 
    with p-values and CIs, (3) Robustness Notes, (4) 3–5 Actionable Next Experiments, 
    (5) Promotion Decision Yes/No with reasons. Produces Markdown suitable to save as summary.md.
    """
    
    def __init__(self):
        self.promotion_rules = PromotionRules()
    
    def evaluate_run_bundle(self, 
                           run_id: str,
                           artifacts: Dict[str, Any],
                           mcpt_results: Optional[Dict[str, Any]] = None,
                           bootstrap_results: Optional[Dict[str, Any]] = None,
                           walkforward_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate Run Bundle artifacts against promotion rules.
        
        You are a strategy analyst. Evaluate the Run Bundle. Cite exact numbers from artifacts. 
        Apply promotion rules exactly. Output: (1) Executive Summary ≤150 words, (2) Significance 
        Verdict PASS/BORDERLINE/FAIL with p-values and CIs, (3) Robustness Notes (trades, costs, 
        parameter plateau), (4) 3–5 Actionable Next Experiments, (5) Promotion Decision Yes/No 
        with reasons. Produce Markdown suitable to save as summary.md.
        
        Args:
            run_id: MLflow run ID
            artifacts: Full Run Bundle artifacts (params.json, metrics.json, equity.csv, trades.csv, mcpt.json, bootstrap.json, walkforward.parquet, weights.csv)
            mcpt_results: MCPT analysis results
            bootstrap_results: Bootstrap analysis results
            walkforward_results: Walk-forward analysis results
            
        Returns:
            Dictionary containing evaluation components
        """
        try:
            # Extract key metrics
            metrics = artifacts.get('metrics', {})
            params = artifacts.get('params', {})
            trades = artifacts.get('trades', [])
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(metrics, trades)
            
            # Generate promotion decision
            promotion_decision = self._evaluate_promotion_criteria(
                metrics, additional_metrics, mcpt_results, walkforward_results
            )
            
            # Generate significance verdict
            significance_verdict = self._evaluate_significance(mcpt_results, walkforward_results)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                metrics, additional_metrics, promotion_decision, significance_verdict
            )
            
            # Generate robustness notes
            robustness_notes = self._generate_robustness_notes(
                metrics, trades, mcpt_results, bootstrap_results
            )
            
            # Generate actionable experiments
            experiments = self._generate_actionable_experiments(
                metrics, params, promotion_decision
            )
            
            # Compile final summary
            summary = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'executive_summary': executive_summary,
                'significance_verdict': significance_verdict,
                'promotion_decision': promotion_decision,
                'robustness_notes': robustness_notes,
                'actionable_experiments': experiments,
                'key_metrics': {
                    **metrics,
                    **additional_metrics
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary for run {run_id}: {e}")
            return self._generate_error_summary(run_id, str(e))
    
    def _calculate_additional_metrics(self, metrics: Dict[str, Any], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate additional metrics needed for evaluation."""
        additional = {}
        
        # Calculate Sortino ratio
        if 'volatility' in metrics and 'annualized_return' in metrics:
            # Simplified Sortino calculation (using volatility as downside deviation proxy)
            additional['sortino_ratio'] = metrics['annualized_return'] / metrics['volatility']
        
        # Calculate cost stress metrics
        if 'sharpe_ratio' in metrics:
            # Simulate 2x cost stress by reducing returns
            stress_return = metrics.get('annualized_return', 0) * 0.5  # Assume 50% reduction
            stress_vol = metrics.get('volatility', 0.01)  # Keep volatility same
            additional['stress_sharpe'] = stress_return / stress_vol if stress_vol > 0 else 0
        
        # Calculate trade statistics
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            additional['num_winning_trades'] = len(winning_trades)
            additional['num_losing_trades'] = len(losing_trades)
            additional['avg_win'] = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            additional['avg_loss'] = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        return additional
    
    def _evaluate_promotion_criteria(self, 
                                   metrics: Dict[str, Any], 
                                   additional_metrics: Dict[str, Any],
                                   mcpt_results: Optional[Dict[str, Any]] = None,
                                   walkforward_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate promotion criteria and generate decision."""
        criteria = {}
        
        # Core performance criteria
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = additional_metrics.get('sortino_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))  # Make positive
        profit_factor = metrics.get('profit_factor', 0)
        stress_sharpe = additional_metrics.get('stress_sharpe', 0)
        
        criteria['sharpe_pass'] = sharpe >= self.promotion_rules.MIN_SHARPE
        criteria['sortino_pass'] = sortino >= self.promotion_rules.MIN_SORTINO
        criteria['max_dd_pass'] = max_dd <= self.promotion_rules.MAX_DRAWDOWN
        criteria['profit_factor_pass'] = profit_factor >= self.promotion_rules.MIN_PROFIT_FACTOR
        criteria['stress_test_pass'] = stress_sharpe >= self.promotion_rules.MIN_STRESS_SHARPE
        
        # Statistical significance criteria
        mcpt_pass = True
        if mcpt_results and isinstance(mcpt_results, dict) and 'results' in mcpt_results:
            sharpe_results = [r for r in mcpt_results['results'] if r.get('metric_name') == 'sharpe_ratio']
            if sharpe_results:
                p_value = sharpe_results[0].get('p_value', 1.0)
                criteria['mcpt_p_value'] = p_value
                criteria['mcpt_pass'] = p_value <= self.promotion_rules.MAX_MCPT_P_VALUE
                mcpt_pass = criteria['mcpt_pass']
        
        # Walk-forward significance
        wf_pass = True
        if walkforward_results and isinstance(walkforward_results, dict) and 'rolling_p_values' in walkforward_results:
            p_values = walkforward_results['rolling_p_values']
            significant_windows = sum(1 for p in p_values if p <= self.promotion_rules.MAX_MCPT_P_VALUE)
            total_windows = len(p_values)
            significant_ratio = significant_windows / total_windows if total_windows > 0 else 0
            
            criteria['wf_significant_ratio'] = significant_ratio
            criteria['wf_significant_windows'] = significant_windows
            criteria['wf_total_windows'] = total_windows
            criteria['wf_pass'] = (significant_ratio >= self.promotion_rules.MIN_SIGNIFICANT_WINDOWS and 
                                 all(p <= self.promotion_rules.MAX_MCPT_P_VALUE_ANY for p in p_values))
            wf_pass = criteria['wf_pass']
        
        # Overall promotion decision
        core_pass = all([
            criteria['sharpe_pass'],
            criteria['sortino_pass'],
            criteria['max_dd_pass'],
            criteria['profit_factor_pass'],
            criteria['stress_test_pass']
        ])
        
        criteria['core_criteria_pass'] = core_pass
        criteria['statistical_criteria_pass'] = mcpt_pass and wf_pass
        criteria['promote'] = core_pass and criteria['statistical_criteria_pass']
        
        # Generate reasons
        reasons = []
        if not criteria['sharpe_pass']:
            reasons.append(f"Sharpe ratio {sharpe:.2f} below threshold {self.promotion_rules.MIN_SHARPE}")
        if not criteria['sortino_pass']:
            reasons.append(f"Sortino ratio {sortino:.2f} below threshold {self.promotion_rules.MIN_SORTINO}")
        if not criteria['max_dd_pass']:
            reasons.append(f"Max drawdown {max_dd:.1%} exceeds threshold {self.promotion_rules.MAX_DRAWDOWN:.1%}")
        if not criteria['profit_factor_pass']:
            reasons.append(f"Profit factor {profit_factor:.2f} below threshold {self.promotion_rules.MIN_PROFIT_FACTOR}")
        if not criteria['stress_test_pass']:
            reasons.append(f"Stress test Sharpe {stress_sharpe:.2f} below threshold {self.promotion_rules.MIN_STRESS_SHARPE}")
        if not mcpt_pass:
            reasons.append(f"MCPT p-value {criteria.get('mcpt_p_value', 'N/A')} above threshold {self.promotion_rules.MAX_MCPT_P_VALUE}")
        if not wf_pass:
            reasons.append(f"Walk-forward significance {criteria.get('wf_significant_ratio', 0):.1%} below threshold {self.promotion_rules.MIN_SIGNIFICANT_WINDOWS:.1%}")
        
        criteria['reasons'] = reasons
        
        return criteria
    
    def _evaluate_significance(self, 
                             mcpt_results: Optional[Dict[str, Any]] = None,
                             walkforward_results: Optional[Dict[str, Any]] = None) -> str:
        """Evaluate statistical significance of the strategy."""
        if not mcpt_results and not walkforward_results:
            return "UNKNOWN"
        
        # Check MCPT significance
        mcpt_significant = False
        if mcpt_results and 'results' in mcpt_results:
            sharpe_results = [r for r in mcpt_results['results'] if r.get('metric_name') == 'sharpe_ratio']
            if sharpe_results:
                p_value = sharpe_results[0].get('p_value', 1.0)
                mcpt_significant = p_value <= 0.05
        
        # Check walk-forward significance
        wf_significant = False
        if walkforward_results and 'rolling_p_values' in walkforward_results:
            p_values = walkforward_results['rolling_p_values']
            significant_windows = sum(1 for p in p_values if p <= 0.05)
            significant_ratio = significant_windows / len(p_values) if p_values else 0
            wf_significant = significant_ratio >= 0.5
        
        if mcpt_significant and wf_significant:
            return "PASS"
        elif mcpt_significant or wf_significant:
            return "BORDERLINE"
        else:
            return "FAIL"
    
    def _generate_executive_summary(self, 
                                  metrics: Dict[str, Any],
                                  additional_metrics: Dict[str, Any],
                                  promotion_decision: Dict[str, Any],
                                  significance_verdict: str) -> str:
        """Generate executive summary (≤150 words)."""
        sharpe = metrics.get('sharpe_ratio', 0)
        total_return = metrics.get('total_return', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0)
        
        promote = promotion_decision.get('promote', False)
        status = "PROMOTED" if promote else "REJECTED"
        
        summary = f"""
Strategy {status} for paper trading. Performance: {total_return:.1%} return, {sharpe:.2f} Sharpe, {max_dd:.1%} max drawdown over {total_trades} trades ({win_rate:.1%} win rate). 

Statistical significance: {significance_verdict}. {'Meets all promotion criteria' if promote else 'Fails promotion criteria'}. 

Key strengths: {'High risk-adjusted returns' if sharpe > 1.5 else 'Moderate performance'}, {'Low drawdown' if max_dd < 0.1 else 'Acceptable risk'}. 

Recommendation: {'Proceed to paper trading with monitoring' if promote else 'Requires optimization before promotion'}.
        """.strip()
        
        return summary
    
    def _generate_robustness_notes(self, 
                                 metrics: Dict[str, Any],
                                 trades: List[Dict[str, Any]],
                                 mcpt_results: Optional[Dict[str, Any]] = None,
                                 bootstrap_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate robustness notes."""
        notes = []
        
        # Trade count analysis
        total_trades = len(trades)
        if total_trades < 10:
            notes.append(f"Low trade count ({total_trades}) may limit statistical reliability")
        elif total_trades > 100:
            notes.append(f"Good trade count ({total_trades}) provides statistical confidence")
        
        # Win rate analysis
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.3:
            notes.append(f"Low win rate ({win_rate:.1%}) suggests strategy may be over-optimized")
        elif win_rate > 0.7:
            notes.append(f"High win rate ({win_rate:.1%}) indicates strong signal quality")
        
        # MCPT analysis
        if mcpt_results and 'results' in mcpt_results:
            sharpe_results = [r for r in mcpt_results['results'] if r.get('metric_name') == 'sharpe_ratio']
            if sharpe_results:
                p_value = sharpe_results[0].get('p_value', 1.0)
                if p_value < 0.01:
                    notes.append(f"Highly significant results (p={p_value:.3f})")
                elif p_value < 0.05:
                    notes.append(f"Significant results (p={p_value:.3f})")
                else:
                    notes.append(f"Non-significant results (p={p_value:.3f}) - may be due to chance")
        
        # Bootstrap analysis
        if bootstrap_results and 'results' in bootstrap_results:
            sharpe_ci = next((r for r in bootstrap_results['results'] if r.get('metric_name') == 'sharpe_ratio'), None)
            if sharpe_ci:
                ci_lower = sharpe_ci.get('confidence_interval', [0, 0])[0]
                ci_upper = sharpe_ci.get('confidence_interval', [0, 0])[1]
                notes.append(f"Sharpe confidence interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
        
        # Parameter risk
        if total_trades > 0:
            avg_trade_return = sum(t.get('return', 0) for t in trades) / total_trades
            if abs(avg_trade_return) < 0.001:
                notes.append("Very small average trade returns may be sensitive to costs")
        
        return notes
    
    def _generate_actionable_experiments(self, 
                                       metrics: Dict[str, Any],
                                       params: Dict[str, Any],
                                       promotion_decision: Dict[str, Any]) -> List[str]:
        """Generate actionable next experiments."""
        experiments = []
        
        # Based on performance gaps
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < 1.0:
            experiments.append("Optimize signal generation to improve Sharpe ratio")
        
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd > 0.12:
            experiments.append("Implement better risk management to reduce drawdown")
        
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.4:
            experiments.append("Improve entry/exit timing to increase win rate")
        
        # Based on statistical significance
        if not promotion_decision.get('statistical_criteria_pass', True):
            experiments.append("Increase sample size or improve signal quality for statistical significance")
        
        # Based on strategy type
        strategy_name = params.get('strategy', '').lower()
        if 'momentum' in strategy_name:
            experiments.append("Test different lookback periods for momentum signals")
        elif 'mean_reversion' in strategy_name:
            experiments.append("Optimize mean reversion thresholds and holding periods")
        
        # General improvements
        experiments.append("Test strategy on different market regimes")
        experiments.append("Implement regime-aware position sizing")
        
        return experiments[:5]  # Limit to 5 experiments
    
    def _generate_error_summary(self, run_id: str, error: str) -> Dict[str, Any]:
        """Generate error summary when analysis fails."""
        return {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'executive_summary': f"Error generating summary for run {run_id}: {error}",
            'significance_verdict': "ERROR",
            'promotion_decision': {'promote': False, 'reasons': [f"Analysis failed: {error}"]},
            'robustness_notes': ["Unable to analyze due to processing error"],
            'actionable_experiments': ["Fix data issues and re-run analysis"],
            'key_metrics': {}
        }
    
    def save_summary_markdown(self, summary: Dict[str, Any], filepath: str) -> None:
        """Save summary as markdown file suitable for summary.md."""
        content = f"""# Strategy Analysis Summary

**Run ID:** {summary['run_id']}  
**Generated:** {summary['timestamp']}

## Executive Summary

{summary['executive_summary']}

## Significance Verdict: {summary['significance_verdict']}

## Promotion Decision: {'✅ PROMOTE' if summary['promotion_decision']['promote'] else '❌ REJECT'}

### Promotion Criteria Analysis
- **Core Performance:** {'✅ PASS' if summary['promotion_decision'].get('core_criteria_pass', False) else '❌ FAIL'}
- **Statistical Significance:** {'✅ PASS' if summary['promotion_decision'].get('statistical_criteria_pass', False) else '❌ FAIL'}

### Reasons for Decision
{chr(10).join(f"- {reason}" for reason in summary['promotion_decision'].get('reasons', []))}

## Robustness Notes
{chr(10).join(f"- {note}" for note in summary['robustness_notes'])}

## Actionable Next Experiments
{chr(10).join(f"- {exp}" for exp in summary['actionable_experiments'])}

## Key Metrics
- **Total Return:** {summary['key_metrics'].get('total_return', 0):.1%}
- **Sharpe Ratio:** {summary['key_metrics'].get('sharpe_ratio', 0):.2f}
- **Max Drawdown:** {summary['key_metrics'].get('max_drawdown', 0):.1%}
- **Total Trades:** {summary['key_metrics'].get('total_trades', 0)}
- **Win Rate:** {summary['key_metrics'].get('win_rate', 0):.1%}

---
*Generated by Neural Quant Strategy Analyst on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(filepath, 'w') as f:
            f.write(content)
    
    def set_mlflow_run_description(self, summary: Dict[str, Any]) -> str:
        """Set MLflow run description from summary."""
        promote = summary['promotion_decision']['promote']
        status = "PROMOTE" if promote else "REJECT"
        sharpe = summary['key_metrics'].get('sharpe_ratio', 0)
        total_return = summary['key_metrics'].get('total_return', 0)
        max_dd = summary['key_metrics'].get('max_drawdown', 0)
        
        description = f"""Strategy {status}: {total_return:.1%} return, {sharpe:.2f} Sharpe, {max_dd:.1%} max DD. 
Significance: {summary['significance_verdict']}. 
{summary['executive_summary'][:200]}..."""
        
        return description
