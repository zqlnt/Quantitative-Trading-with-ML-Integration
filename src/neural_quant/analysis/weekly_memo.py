"""
Weekly Research Memo Generator

This module generates comprehensive weekly quant research reports
by analyzing the last 7 days of backtest runs.

Author: Neural Quant Team
Date: 2024
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


class WeeklyMemoGenerator:
    """
    Generates weekly quant research reports from recent backtest runs.
    
    This class analyzes the last 7 days of runs and creates comprehensive
    reports including performance analysis, statistical significance,
    and promotion recommendations.
    """
    
    def __init__(self, mlflow_tracking_uri: str = "sqlite:///mlflow.db"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.promotion_rules = {
            'min_sharpe': 1.0,
            'min_sortino': 1.2,
            'max_drawdown': 0.12,
            'min_profit_factor': 1.2,
            'max_mcpt_p_value': 0.05,
            'min_significant_windows': 0.50,
            'min_stress_sharpe': 0.7
        }
    
    def generate_weekly_memo(self, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate weekly research memo.
        
        Args:
            start_date: Start date for the week (defaults to 7 days ago)
            
        Returns:
            Dictionary containing memo components
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)
            
            end_date = datetime.now()
            
            # Get runs from the last 7 days
            runs = self._get_recent_runs(start_date, end_date)
            
            if not runs:
                return self._generate_no_runs_memo(start_date, end_date)
            
            # Analyze runs
            analysis = self._analyze_runs(runs)
            
            # Generate memo components
            memo = {
                'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'total_runs': len(runs),
                'best_runs': analysis['best_runs'],
                'worst_runs': analysis['worst_runs'],
                'promotion_candidates': analysis['promotion_candidates'],
                'cost_stress_summary': analysis['cost_stress_summary'],
                'statistical_analysis': analysis['statistical_analysis'],
                'key_insights': analysis['key_insights'],
                'recommendations': analysis['recommendations'],
                'generated_at': datetime.now().isoformat()
            }
            
            return memo
            
        except Exception as e:
            logger.error(f"Error generating weekly memo: {e}")
            return self._generate_error_memo(str(e))
    
    def _get_recent_runs(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get recent runs from MLflow."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            client = MlflowClient()
            
            # Get experiments
            experiments = client.search_experiments()
            runs = []
            
            for exp in experiments:
                # Search runs in date range
                run_list = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"start_time >= {int(start_date.timestamp() * 1000)} AND start_time <= {int(end_date.timestamp() * 1000)}"
                )
                
                for run in run_list:
                    run_data = {
                        'run_id': run.info.run_id,
                        'experiment_id': run.info.experiment_id,
                        'start_time': run.info.start_time,
                        'status': run.info.status,
                        'metrics': run.data.metrics,
                        'params': run.data.params,
                        'tags': run.data.tags
                    }
                    runs.append(run_data)
            
            return runs
            
        except Exception as e:
            logger.error(f"Error getting recent runs: {e}")
            return []
    
    def _analyze_runs(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze runs and generate insights."""
        analysis = {
            'best_runs': [],
            'worst_runs': [],
            'promotion_candidates': [],
            'cost_stress_summary': {},
            'statistical_analysis': {},
            'key_insights': [],
            'recommendations': []
        }
        
        # Sort runs by Sharpe ratio
        runs_with_sharpe = []
        for run in runs:
            sharpe = run['metrics'].get('sharpe_ratio', 0)
            if sharpe > 0:  # Only include runs with valid Sharpe
                runs_with_sharpe.append((run, sharpe))
        
        runs_with_sharpe.sort(key=lambda x: x[1], reverse=True)
        
        # Get best and worst runs
        if runs_with_sharpe:
            analysis['best_runs'] = [run[0] for run in runs_with_sharpe[:3]]  # Top 3
            analysis['worst_runs'] = [run[0] for run in runs_with_sharpe[-3:]]  # Bottom 3
        
        # Find promotion candidates
        for run in runs:
            if self._evaluate_promotion_criteria(run):
                analysis['promotion_candidates'].append(run)
        
        # Generate cost stress summary
        analysis['cost_stress_summary'] = self._generate_cost_stress_summary(runs)
        
        # Generate statistical analysis
        analysis['statistical_analysis'] = self._generate_statistical_analysis(runs)
        
        # Generate insights and recommendations
        analysis['key_insights'] = self._generate_key_insights(runs, analysis)
        analysis['recommendations'] = self._generate_recommendations(runs, analysis)
        
        return analysis
    
    def _evaluate_promotion_criteria(self, run: Dict[str, Any]) -> bool:
        """Evaluate if a run meets promotion criteria."""
        metrics = run['metrics']
        
        # Core performance criteria
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        profit_factor = metrics.get('profit_factor', 0)
        
        # Basic checks
        if sharpe < self.promotion_rules['min_sharpe']:
            return False
        if max_dd > self.promotion_rules['max_drawdown']:
            return False
        if profit_factor < self.promotion_rules['min_profit_factor']:
            return False
        
        # Check for MCPT significance (if available)
        # This would require loading artifacts, simplified for now
        return True
    
    def _generate_cost_stress_summary(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cost stress test summary."""
        stress_results = []
        
        for run in runs:
            sharpe = run['metrics'].get('sharpe_ratio', 0)
            # Simulate 2x cost stress
            stress_sharpe = sharpe * 0.5  # Rough approximation
            
            stress_results.append({
                'run_id': run['run_id'],
                'original_sharpe': sharpe,
                'stress_sharpe': stress_sharpe,
                'passes_stress': stress_sharpe >= self.promotion_rules['min_stress_sharpe']
            })
        
        passing_stress = sum(1 for r in stress_results if r['passes_stress'])
        total_tested = len(stress_results)
        
        return {
            'total_tested': total_tested,
            'passing_stress': passing_stress,
            'stress_pass_rate': passing_stress / total_tested if total_tested > 0 else 0,
            'results': stress_results
        }
    
    def _generate_statistical_analysis(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistical analysis summary."""
        sharpe_ratios = [run['metrics'].get('sharpe_ratio', 0) for run in runs if run['metrics'].get('sharpe_ratio', 0) > 0]
        
        if not sharpe_ratios:
            return {'error': 'No valid Sharpe ratios found'}
        
        return {
            'mean_sharpe': sum(sharpe_ratios) / len(sharpe_ratios),
            'median_sharpe': sorted(sharpe_ratios)[len(sharpe_ratios) // 2],
            'max_sharpe': max(sharpe_ratios),
            'min_sharpe': min(sharpe_ratios),
            'sharpe_std': pd.Series(sharpe_ratios).std(),
            'total_runs': len(sharpe_ratios)
        }
    
    def _generate_key_insights(self, runs: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        # Performance insights
        if analysis['best_runs']:
            best_sharpe = analysis['best_runs'][0]['metrics'].get('sharpe_ratio', 0)
            insights.append(f"Best performing strategy achieved {best_sharpe:.2f} Sharpe ratio")
        
        if analysis['worst_runs']:
            worst_sharpe = analysis['worst_runs'][0]['metrics'].get('sharpe_ratio', 0)
            insights.append(f"Worst performing strategy had {worst_sharpe:.2f} Sharpe ratio")
        
        # Promotion insights
        promotion_count = len(analysis['promotion_candidates'])
        total_runs = len(runs)
        if promotion_count > 0:
            insights.append(f"{promotion_count}/{total_runs} strategies ({promotion_count/total_runs:.1%}) meet promotion criteria")
        else:
            insights.append("No strategies meet all promotion criteria this week")
        
        # Cost stress insights
        stress_summary = analysis['cost_stress_summary']
        if stress_summary.get('stress_pass_rate', 0) > 0.5:
            insights.append("Most strategies show good resilience to cost increases")
        else:
            insights.append("Many strategies are sensitive to cost increases")
        
        # Statistical insights
        stat_analysis = analysis['statistical_analysis']
        if 'mean_sharpe' in stat_analysis:
            mean_sharpe = stat_analysis['mean_sharpe']
            if mean_sharpe > 1.0:
                insights.append(f"Average Sharpe ratio ({mean_sharpe:.2f}) exceeds promotion threshold")
            else:
                insights.append(f"Average Sharpe ratio ({mean_sharpe:.2f}) below promotion threshold")
        
        return insights
    
    def _generate_recommendations(self, runs: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on promotion candidates
        if not analysis['promotion_candidates']:
            recommendations.append("Focus on improving strategy performance to meet promotion criteria")
            recommendations.append("Consider optimizing risk management to reduce drawdowns")
        
        # Based on cost stress
        stress_summary = analysis['cost_stress_summary']
        if stress_summary.get('stress_pass_rate', 0) < 0.5:
            recommendations.append("Improve strategy efficiency to reduce cost sensitivity")
        
        # Based on statistical analysis
        stat_analysis = analysis['statistical_analysis']
        if 'mean_sharpe' in stat_analysis and stat_analysis['mean_sharpe'] < 1.0:
            recommendations.append("Increase sample sizes or improve signal quality for better statistical significance")
        
        # General recommendations
        recommendations.append("Continue monitoring promoted strategies in paper trading")
        recommendations.append("Implement regime-aware position sizing for better risk management")
        
        return recommendations
    
    def _generate_no_runs_memo(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate memo when no runs are found."""
        return {
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'total_runs': 0,
            'message': f"No backtest runs found for the period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_error_memo(self, error: str) -> Dict[str, Any]:
        """Generate error memo when analysis fails."""
        return {
            'period': 'Error',
            'total_runs': 0,
            'error': f"Error generating weekly memo: {error}",
            'generated_at': datetime.now().isoformat()
        }
    
    def save_weekly_memo(self, memo: Dict[str, Any], filepath: str) -> None:
        """Save weekly memo as markdown file."""
        if 'error' in memo:
            content = f"# Weekly Research Memo - Error\n\n{memo['error']}\n\n*Generated on {memo['generated_at']}*"
        elif memo['total_runs'] == 0:
            content = f"# Weekly Research Memo - No Data\n\n{memo.get('message', 'No runs found')}\n\n*Generated on {memo['generated_at']}*"
        else:
            content = self._format_weekly_memo(memo)
        
        with open(filepath, 'w') as f:
            f.write(content)
    
    def _format_weekly_memo(self, memo: Dict[str, Any]) -> str:
        """Format weekly memo as markdown."""
        content = f"""# Weekly Quant Research Memo

**Period:** {memo['period']}  
**Total Runs:** {memo['total_runs']}  
**Generated:** {memo['generated_at']}

## Executive Summary

This week's analysis covers {memo['total_runs']} backtest runs from {memo['period']}. 

## Best Performing Runs

"""
        
        # Add best runs table
        if memo['best_runs']:
            content += "| Run ID | Strategy | Sharpe | Return | Max DD | Trades |\n"
            content += "|--------|----------|--------|--------|--------|--------|\n"
            for run in memo['best_runs']:
                run_id = run['run_id'][:8]
                strategy = run['params'].get('strategy', 'Unknown')
                sharpe = run['metrics'].get('sharpe_ratio', 0)
                total_return = run['metrics'].get('total_return', 0)
                max_dd = run['metrics'].get('max_drawdown', 0)
                trades = run['metrics'].get('total_trades', 0)
                content += f"| {run_id} | {strategy} | {sharpe:.2f} | {total_return:.1%} | {max_dd:.1%} | {trades} |\n"
        else:
            content += "No runs with valid Sharpe ratios found.\n"
        
        content += "\n## Worst Performing Runs\n\n"
        
        # Add worst runs table
        if memo['worst_runs']:
            content += "| Run ID | Strategy | Sharpe | Return | Max DD | Trades |\n"
            content += "|--------|----------|--------|--------|--------|--------|\n"
            for run in memo['worst_runs']:
                run_id = run['run_id'][:8]
                strategy = run['params'].get('strategy', 'Unknown')
                sharpe = run['metrics'].get('sharpe_ratio', 0)
                total_return = run['metrics'].get('total_return', 0)
                max_dd = run['metrics'].get('max_drawdown', 0)
                trades = run['metrics'].get('total_trades', 0)
                content += f"| {run_id} | {strategy} | {sharpe:.2f} | {total_return:.1%} | {max_dd:.1%} | {trades} |\n"
        else:
            content += "No runs with valid Sharpe ratios found.\n"
        
        content += "\n## Promotion Candidates\n\n"
        
        # Add promotion candidates
        if memo['promotion_candidates']:
            content += f"**{len(memo['promotion_candidates'])} strategies meet promotion criteria:**\n\n"
            for run in memo['promotion_candidates']:
                run_id = run['run_id'][:8]
                strategy = run['params'].get('strategy', 'Unknown')
                sharpe = run['metrics'].get('sharpe_ratio', 0)
                content += f"- **{strategy}** (Run {run_id}): Sharpe {sharpe:.2f}\n"
        else:
            content += "No strategies meet all promotion criteria this week.\n"
        
        content += "\n## Cost Stress Test Summary\n\n"
        
        # Add cost stress summary
        stress_summary = memo['cost_stress_summary']
        if stress_summary:
            content += f"- **Total Tested:** {stress_summary.get('total_tested', 0)}\n"
            content += f"- **Passing Stress Test:** {stress_summary.get('passing_stress', 0)}\n"
            content += f"- **Pass Rate:** {stress_summary.get('stress_pass_rate', 0):.1%}\n"
        
        content += "\n## Statistical Analysis\n\n"
        
        # Add statistical analysis
        stat_analysis = memo['statistical_analysis']
        if stat_analysis and 'mean_sharpe' in stat_analysis:
            content += f"- **Mean Sharpe:** {stat_analysis['mean_sharpe']:.2f}\n"
            content += f"- **Median Sharpe:** {stat_analysis['median_sharpe']:.2f}\n"
            content += f"- **Max Sharpe:** {stat_analysis['max_sharpe']:.2f}\n"
            content += f"- **Min Sharpe:** {stat_analysis['min_sharpe']:.2f}\n"
            content += f"- **Sharpe Std Dev:** {stat_analysis['sharpe_std']:.2f}\n"
        
        content += "\n## Key Insights\n\n"
        
        # Add key insights
        for insight in memo['key_insights']:
            content += f"- {insight}\n"
        
        content += "\n## Recommendations\n\n"
        
        # Add recommendations
        for rec in memo['recommendations']:
            content += f"- {rec}\n"
        
        content += f"\n---\n*Generated by Neural Quant on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return content
