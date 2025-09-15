"""LLM Assistant for Neural Quant using Anthropic Claude."""

import os
import json
from typing import Dict, Any, List, Optional
import anthropic
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NeuralQuantAssistant:
    """AI Assistant for analyzing trading results and answering questions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM assistant.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def analyze_backtest_results(self, 
                               metrics: Dict[str, Any], 
                               equity_curve: pd.DataFrame,
                               trades: List[Dict],
                               strategy_params: Dict[str, Any]) -> str:
        """
        Analyze backtest results and provide intelligent insights.
        
        Args:
            metrics: Performance metrics from backtest
            equity_curve: Equity curve data
            trades: List of trade records
            strategy_params: Strategy parameters used
            
        Returns:
            AI-generated analysis of the results
        """
        # Prepare data for analysis
        analysis_data = {
            "strategy": strategy_params.get("strategy", "Unknown"),
            "symbol": strategy_params.get("ticker", "Unknown"),
            "period": f"{strategy_params.get('start', '')} to {strategy_params.get('end', '')}",
            "parameters": {
                "fast_ma": strategy_params.get("fast", "Unknown"),
                "slow_ma": strategy_params.get("slow", "Unknown"),
                "threshold": strategy_params.get("threshold_pct", "Unknown"),
                "commission": strategy_params.get("fee_bps", "Unknown"),
                "slippage": strategy_params.get("slippage_bps", "Unknown")
            },
            "performance": {
                "total_return": metrics.get("total_return", 0),
                "annualized_return": metrics.get("annualized_return", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "volatility": metrics.get("volatility", 0),
                "total_trades": metrics.get("total_trades", 0),
                "win_rate": metrics.get("win_rate", 0)
            },
            "equity_stats": {
                "final_value": equity_curve['equity'].iloc[-1] if not equity_curve.empty else 0,
                "max_value": equity_curve['equity'].max() if not equity_curve.empty else 0,
                "min_value": equity_curve['equity'].min() if not equity_curve.empty else 0,
                "total_days": len(equity_curve) if not equity_curve.empty else 0
            },
            "trade_analysis": {
                "total_trades": len(trades),
                "profitable_trades": len([t for t in trades if t.get('pnl', 0) > 0]) if trades else 0,
                "average_pnl": sum(t.get('pnl', 0) for t in trades) / len(trades) if trades else 0,
                "best_trade": max(trades, key=lambda x: x.get('pnl', 0)) if trades else None,
                "worst_trade": min(trades, key=lambda x: x.get('pnl', 0)) if trades else None
            }
        }
        
        prompt = f"""
You are an expert quantitative trading analyst. Analyze the following backtest results and provide:

1. **Executive Summary**: Overall performance assessment (2-3 sentences)
2. **Key Insights**: Most important findings about strategy performance
3. **Risk Analysis**: Assessment of risk metrics and drawdowns
4. **Trade Analysis**: Insights about trade frequency, win rate, and profitability
5. **Recommendations**: Specific suggestions for improving the strategy
6. **Market Context**: How the strategy might perform in different market conditions

Backtest Data:
{json.dumps(analysis_data, indent=2, default=str)}

Please provide a comprehensive but concise analysis (300-500 words) that would be valuable for a quantitative trader.
"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
    
    def answer_question(self, question: str, context: Dict[str, Any] = None) -> str:
        """
        Answer user questions about trading, strategies, or results.
        
        Args:
            question: User's question
            context: Optional context about current experiment
            
        Returns:
            AI-generated answer
        """
        system_prompt = """You are an expert quantitative trading assistant for the Neural Quant platform. 
        You help users understand trading strategies, analyze backtest results, and provide insights about 
        algorithmic trading. Be helpful, accurate, and professional in your responses."""
        
        if context:
            context_str = f"\n\nCurrent Context:\n{json.dumps(context, indent=2, default=str)}"
        else:
            context_str = ""
        
        prompt = f"{question}{context_str}"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                system=system_prompt
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_experiment_summary(self, 
                                  run_id: str,
                                  metrics: Dict[str, Any],
                                  strategy_params: Dict[str, Any]) -> str:
        """
        Generate a summary for MLflow experiment tracking.
        
        Args:
            run_id: MLflow run ID
            metrics: Performance metrics
            strategy_params: Strategy parameters
            
        Returns:
            AI-generated experiment summary
        """
        prompt = f"""
Generate a concise experiment summary for MLflow tracking. Include:

1. Strategy description
2. Key performance highlights
3. Notable insights or concerns
4. Brief recommendation

Run ID: {run_id}
Strategy: {strategy_params.get('strategy', 'Unknown')}
Symbol: {strategy_params.get('ticker', 'Unknown')}
Total Return: {metrics.get('total_return', 0):.2%}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
Total Trades: {metrics.get('total_trades', 0)}

Keep it under 200 words and make it suitable for experiment tracking.
"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def suggest_strategy_improvements(self, 
                                    current_params: Dict[str, Any],
                                    performance_issues: List[str]) -> str:
        """
        Suggest improvements to strategy parameters based on performance issues.
        
        Args:
            current_params: Current strategy parameters
            performance_issues: List of identified performance issues
            
        Returns:
            AI-generated improvement suggestions
        """
        prompt = f"""
As a quantitative trading expert, suggest specific improvements for this strategy:

Current Parameters:
{json.dumps(current_params, indent=2)}

Performance Issues Identified:
{', '.join(performance_issues)}

Provide 3-5 specific, actionable recommendations for improving the strategy's performance.
Focus on parameter tuning, risk management, and strategy logic improvements.
"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                temperature=0.4,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"
