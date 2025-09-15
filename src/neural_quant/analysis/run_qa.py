"""
Run Q&A System

This module provides contextual Q&A functionality for backtest runs,
allowing users to ask questions about specific runs with full context.

Author: Neural Quant Team
Date: 2024
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RunQASystem:
    """
    Q&A system for backtest runs with contextual information.
    
    This class provides a conversational interface for asking questions
    about specific backtest runs, with access to all run artifacts and context.
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
    
    def load_run_context(self, run_id: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load run context for Q&A.
        
        Args:
            run_id: MLflow run ID
            artifacts: Dictionary of run artifacts
            
        Returns:
            Context dictionary for Q&A
        """
        context = {
            'run_id': run_id,
            'strategy': artifacts.get('params', {}).get('strategy', 'Unknown'),
            'tickers': artifacts.get('params', {}).get('tickers', []),
            'period': f"{artifacts.get('params', {}).get('start_date', 'Unknown')} to {artifacts.get('params', {}).get('end_date', 'Unknown')}",
            'metrics': artifacts.get('metrics', {}),
            'trades': artifacts.get('trades', []),
            'mcpt_results': artifacts.get('mcpt_results', {}),
            'bootstrap_results': artifacts.get('bootstrap_results', {}),
            'walkforward_results': artifacts.get('walkforward_results', {}),
            'promotion_rules': self.promotion_rules
        }
        
        return context
    
    def generate_context_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generate context prompt for the AI assistant.
        
        Args:
            context: Run context dictionary
            
        Returns:
            Formatted context prompt
        """
        prompt = f"""
You are analyzing a quantitative trading strategy backtest run. Here is the complete context:

## Run Information
- **Run ID:** {context['run_id']}
- **Strategy:** {context['strategy']}
- **Tickers:** {', '.join(context['tickers'])}
- **Period:** {context['period']}

## Performance Metrics
- **Total Return:** {context['metrics'].get('total_return', 0):.1%}
- **Sharpe Ratio:** {context['metrics'].get('sharpe_ratio', 0):.2f}
- **Max Drawdown:** {context['metrics'].get('max_drawdown', 0):.1%}
- **Total Trades:** {context['metrics'].get('total_trades', 0)}
- **Win Rate:** {context['metrics'].get('win_rate', 0):.1%}
- **Profit Factor:** {context['metrics'].get('profit_factor', 0):.2f}

## Trade Analysis
- **Number of Trades:** {len(context['trades'])}
- **Winning Trades:** {len([t for t in context['trades'] if t.get('pnl', 0) > 0])}
- **Losing Trades:** {len([t for t in context['trades'] if t.get('pnl', 0) < 0])}

## Statistical Analysis
"""
        
        # Add MCPT results
        if context['mcpt_results'] and 'results' in context['mcpt_results']:
            prompt += "\n### Monte Carlo Permutation Test Results\n"
            for result in context['mcpt_results']['results']:
                metric = result.get('metric_name', 'Unknown')
                p_value = result.get('p_value', 0)
                significant = "Yes" if p_value <= 0.05 else "No"
                prompt += f"- **{metric}:** p-value = {p_value:.4f} (Significant: {significant})\n"
        
        # Add Bootstrap results
        if context['bootstrap_results'] and 'results' in context['bootstrap_results']:
            prompt += "\n### Bootstrap Confidence Intervals\n"
            for result in context['bootstrap_results']['results']:
                metric = result.get('metric_name', 'Unknown')
                ci = result.get('confidence_interval', [0, 0])
                prompt += f"- **{metric}:** [{ci[0]:.3f}, {ci[1]:.3f}]\n"
        
        # Add Walk-forward results
        if context['walkforward_results']:
            prompt += "\n### Walk-Forward Analysis\n"
            if 'rolling_sharpe' in context['walkforward_results']:
                rolling_sharpe = context['walkforward_results']['rolling_sharpe']
                prompt += f"- **Rolling Sharpe:** Mean = {rolling_sharpe.mean():.2f}, Std = {rolling_sharpe.std():.2f}\n"
            if 'rolling_p_values' in context['walkforward_results']:
                p_values = context['walkforward_results']['rolling_p_values']
                significant_windows = sum(1 for p in p_values if p <= 0.05)
                prompt += f"- **Significant Windows:** {significant_windows}/{len(p_values)} ({significant_windows/len(p_values):.1%})\n"
        
        # Add promotion rules
        prompt += f"""
## Promotion Rules
- **Min Sharpe:** {self.promotion_rules['min_sharpe']}
- **Min Sortino:** {self.promotion_rules['min_sortino']}
- **Max Drawdown:** {self.promotion_rules['max_drawdown']:.1%}
- **Min Profit Factor:** {self.promotion_rules['min_profit_factor']}
- **Max MCPT P-value:** {self.promotion_rules['max_mcpt_p_value']}
- **Min Significant Windows:** {self.promotion_rules['min_significant_windows']:.1%}
- **Min Stress Sharpe:** {self.promotion_rules['min_stress_sharpe']}

## Instructions
Answer questions about this run with specific references to the data above. Be precise and quantitative in your responses. If asked about specific metrics, provide exact values. If asked about performance issues, identify specific problems and suggest solutions.
"""
        
        return prompt
    
    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        """
        Answer a question about the run using the provided context.
        
        Args:
            question: User's question
            context: Run context dictionary
            
        Returns:
            Answer to the question
        """
        try:
            # Generate context prompt
            context_prompt = self.generate_context_prompt(context)
            
            # For now, provide a simple response based on common questions
            # In a full implementation, this would call an LLM API
            return self._generate_simple_answer(question, context)
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error processing question: {str(e)}"
    
    def _generate_simple_answer(self, question: str, context: Dict[str, Any]) -> str:
        """Generate a simple answer based on common question patterns."""
        question_lower = question.lower()
        
        # Performance questions
        if 'sharpe' in question_lower:
            sharpe = context['metrics'].get('sharpe_ratio', 0)
            return f"The Sharpe ratio is {sharpe:.2f}. {'This meets the promotion threshold of 1.0' if sharpe >= 1.0 else 'This is below the promotion threshold of 1.0'}."
        
        if 'drawdown' in question_lower:
            max_dd = abs(context['metrics'].get('max_drawdown', 0))
            return f"The maximum drawdown is {max_dd:.1%}. {'This is within the acceptable limit of 12%' if max_dd <= 0.12 else 'This exceeds the acceptable limit of 12%'}."
        
        if 'return' in question_lower:
            total_return = context['metrics'].get('total_return', 0)
            return f"The total return is {total_return:.1%} over the backtest period."
        
        if 'trades' in question_lower:
            total_trades = context['metrics'].get('total_trades', 0)
            win_rate = context['metrics'].get('win_rate', 0)
            return f"There were {total_trades} total trades with a {win_rate:.1%} win rate."
        
        if 'profit factor' in question_lower:
            pf = context['metrics'].get('profit_factor', 0)
            return f"The profit factor is {pf:.2f}. {'This meets the promotion threshold of 1.2' if pf >= 1.2 else 'This is below the promotion threshold of 1.2'}."
        
        # Statistical significance questions
        if 'significant' in question_lower or 'p-value' in question_lower:
            if context['mcpt_results'] and 'results' in context['mcpt_results']:
                sharpe_results = [r for r in context['mcpt_results']['results'] if r.get('metric_name') == 'sharpe_ratio']
                if sharpe_results:
                    p_value = sharpe_results[0].get('p_value', 1.0)
                    return f"The Sharpe ratio p-value is {p_value:.4f}. {'This is statistically significant (p < 0.05)' if p_value <= 0.05 else 'This is not statistically significant (p >= 0.05)'}."
            return "No statistical significance analysis available for this run."
        
        # Ticker-specific questions
        if any(ticker.lower() in question_lower for ticker in context['tickers']):
            ticker_mentioned = next((ticker for ticker in context['tickers'] if ticker.lower() in question_lower), None)
            if ticker_mentioned:
                ticker_trades = [t for t in context['trades'] if t.get('symbol') == ticker_mentioned]
                if ticker_trades:
                    ticker_pnl = sum(t.get('pnl', 0) for t in ticker_trades)
                    ticker_return = sum(t.get('return', 0) for t in ticker_trades)
                    return f"{ticker_mentioned} had {len(ticker_trades)} trades with total P&L of ${ticker_pnl:.2f} and total return of {ticker_return:.1%}."
                else:
                    return f"{ticker_mentioned} had no trades in this run."
        
        # Default response
        return f"I can help you analyze this {context['strategy']} strategy run. The strategy achieved {context['metrics'].get('total_return', 0):.1%} return with a {context['metrics'].get('sharpe_ratio', 0):.2f} Sharpe ratio. What specific aspect would you like to know more about?"
    
    def get_common_questions(self) -> List[str]:
        """Get a list of common questions users might ask."""
        return [
            "What is the Sharpe ratio?",
            "What is the maximum drawdown?",
            "How many trades were there?",
            "What is the win rate?",
            "Is the strategy statistically significant?",
            "What is the profit factor?",
            "Which ticker performed best?",
            "Which ticker performed worst?",
            "What are the main risks?",
            "Should this strategy be promoted to paper trading?",
            "What are the key strengths and weaknesses?",
            "How does this compare to the benchmark?",
            "What is the average trade return?",
            "What is the largest winning trade?",
            "What is the largest losing trade?"
        ]
