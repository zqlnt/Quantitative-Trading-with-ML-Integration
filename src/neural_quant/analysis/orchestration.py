"""
Orchestration Engine for R1 → R2 → Execution → R3 Loop

This module orchestrates the complete workflow from failed runs to new experiments,
change requests, execution, and re-evaluation.

Author: Neural Quant Team
Date: 2024
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import os
import mlflow
from .quant_researcher import QuantResearcher
from .strategy_developer import StrategyDeveloper
from .summary_generator import StrategyAnalyst

logger = logging.getLogger(__name__)


class OrchestrationEngine:
    """
    Orchestrates the R1 → R2 → Execution → R3 loop for strategy iteration.
    
    This class manages the complete workflow from analyzing failed runs to
    generating new experiments, executing them, and re-evaluating results.
    """
    
    def __init__(self):
        self.quant_researcher = QuantResearcher()
        self.strategy_developer = StrategyDeveloper()
        self.strategy_analyst = StrategyAnalyst()
        
    def iterate_from_run(self, 
                        parent_run_id: str,
                        selected_experiments: Optional[List[int]] = None,
                        max_iterations: int = 3) -> Dict[str, Any]:
        """
        Orchestrate the complete R1 → R2 → Execution → R3 loop from a failed run.
        
        Args:
            parent_run_id: MLflow run ID of the parent run
            selected_experiments: List of experiment indices to execute (if None, auto-select)
            max_iterations: Maximum number of iterations to run
            
        Returns:
            Dictionary containing iteration results and lineage
        """
        try:
            # Step 1: Load parent run artifacts
            parent_artifacts = self._load_run_artifacts(parent_run_id)
            if not parent_artifacts:
                return {'error': f'Could not load artifacts for run {parent_run_id}'}
            
            # Step 2: R3 - Evaluate parent run
            parent_evaluation = self.strategy_analyst.evaluate_run_bundle(
                run_id=parent_run_id,
                artifacts=parent_artifacts
            )
            
            # Check if parent was rejected (only iterate from failed runs)
            if parent_evaluation['promotion_decision']['promote']:
                return {'error': 'Parent run was promoted - no iteration needed'}
            
            # Step 3: R1 - Generate experiments
            r1_plan = self.quant_researcher.generate_experiments(
                artifacts=parent_artifacts,
                universe=parent_artifacts.get('params', {}).get('tickers', []),
                max_experiments=3
            )
            
            # Save R1 plan
            r1_plan_path = self._save_r1_plan(parent_run_id, r1_plan)
            
            # Step 4: Select experiments (manual or heuristic)
            if selected_experiments is None:
                selected_experiments = self._select_experiments_heuristic(r1_plan)
            
            selected_experiment_data = [r1_plan['experiments'][i] for i in selected_experiments]
            
            # Step 5: R2 - Generate changes
            r2_changes = self.strategy_developer.translate_experiments(
                selected_experiments=selected_experiment_data,
                max_grid_size=9
            )
            
            # Save R2 changes
            r2_changes_path = self._save_r2_changes(parent_run_id, r2_changes)
            
            # Step 6: Execute experiments
            iteration_results = []
            for i, experiment in enumerate(selected_experiment_data):
                try:
                    # Execute the experiment
                    new_run_id = self._execute_experiment(
                        parent_run_id=parent_run_id,
                        experiment=experiment,
                        r2_changes=r2_changes,
                        iteration_n=i + 1
                    )
                    
                    if new_run_id:
                        iteration_results.append({
                            'run_id': new_run_id,
                            'experiment_id': experiment['id'],
                            'iteration_n': i + 1,
                            'status': 'completed'
                        })
                    else:
                        iteration_results.append({
                            'run_id': None,
                            'experiment_id': experiment['id'],
                            'iteration_n': i + 1,
                            'status': 'failed'
                        })
                        
                except Exception as e:
                    logger.error(f"Error executing experiment {experiment['id']}: {e}")
                    iteration_results.append({
                        'run_id': None,
                        'experiment_id': experiment['id'],
                        'iteration_n': i + 1,
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Step 7: R3 - Evaluate new runs
            evaluation_results = []
            for result in iteration_results:
                if result['run_id']:
                    try:
                        # Load new run artifacts
                        new_artifacts = self._load_run_artifacts(result['run_id'])
                        if new_artifacts:
                            # R3 evaluation
                            evaluation = self.strategy_analyst.evaluate_run_bundle(
                                run_id=result['run_id'],
                                artifacts=new_artifacts
                            )
                            evaluation_results.append({
                                'run_id': result['run_id'],
                                'experiment_id': result['experiment_id'],
                                'iteration_n': result['iteration_n'],
                                'promotion_decision': evaluation['promotion_decision']['promote'],
                                'significance_verdict': evaluation['significance_verdict'],
                                'sharpe_ratio': evaluation['key_metrics'].get('sharpe_ratio', 0),
                                'max_drawdown': evaluation['key_metrics'].get('max_drawdown', 0)
                            })
                    except Exception as e:
                        logger.error(f"Error evaluating run {result['run_id']}: {e}")
            
            # Step 8: Create iteration graph and lineage
            iteration_graph = self._create_iteration_graph(
                parent_run_id=parent_run_id,
                parent_evaluation=parent_evaluation,
                iteration_results=iteration_results,
                evaluation_results=evaluation_results
            )
            
            # Save experiment lineage
            lineage_path = self._save_experiment_lineage(parent_run_id, iteration_graph)
            
            # Update parent run with iteration info
            self._update_parent_run_tags(parent_run_id, r1_plan_path, lineage_path)
            
            return {
                'parent_run_id': parent_run_id,
                'r1_plan_path': r1_plan_path,
                'r2_changes_path': r2_changes_path,
                'lineage_path': lineage_path,
                'iteration_results': iteration_results,
                'evaluation_results': evaluation_results,
                'iteration_graph': iteration_graph,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in orchestration loop: {e}")
            return {'error': str(e), 'success': False}
    
    def _load_run_artifacts(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load artifacts from a MLflow run."""
        try:
            with mlflow.start_run(run_id=run_id):
                # Load artifacts from MLflow
                artifacts = {}
                
                # Load params.json
                try:
                    params_path = mlflow.artifacts.download_artifacts(artifact_uri="params.json")
                    with open(params_path, 'r') as f:
                        artifacts['params'] = json.load(f)
                except:
                    pass
                
                # Load metrics.json
                try:
                    metrics_path = mlflow.artifacts.download_artifacts(artifact_uri="metrics.json")
                    with open(metrics_path, 'r') as f:
                        artifacts['metrics'] = json.load(f)
                except:
                    pass
                
                # Load trades.csv
                try:
                    trades_path = mlflow.artifacts.download_artifacts(artifact_uri="trades.csv")
                    trades_df = pd.read_csv(trades_path)
                    artifacts['trades'] = trades_df.to_dict('records')
                except:
                    pass
                
                # Load MCPT results
                try:
                    mcpt_path = mlflow.artifacts.download_artifacts(artifact_uri="mcpt.json")
                    with open(mcpt_path, 'r') as f:
                        artifacts['mcpt_results'] = json.load(f)
                except:
                    pass
                
                # Load bootstrap results
                try:
                    bootstrap_path = mlflow.artifacts.download_artifacts(artifact_uri="bootstrap.json")
                    with open(bootstrap_path, 'r') as f:
                        artifacts['bootstrap_results'] = json.load(f)
                except:
                    pass
                
                return artifacts
                
        except Exception as e:
            logger.error(f"Error loading artifacts for run {run_id}: {e}")
            return None
    
    def _save_r1_plan(self, parent_run_id: str, r1_plan: Dict[str, Any]) -> str:
        """Save R1 plan to MLflow artifacts."""
        try:
            with mlflow.start_run(run_id=parent_run_id):
                r1_plan_json = json.dumps(r1_plan, indent=2)
                mlflow.log_text(r1_plan_json, "r1_plan.json")
                return f"runs:/{parent_run_id}/r1_plan.json"
        except Exception as e:
            logger.error(f"Error saving R1 plan: {e}")
            return None
    
    def _save_r2_changes(self, parent_run_id: str, r2_changes: Dict[str, Any]) -> str:
        """Save R2 changes to MLflow artifacts."""
        try:
            with mlflow.start_run(run_id=parent_run_id):
                r2_changes_json = json.dumps(r2_changes, indent=2)
                mlflow.log_text(r2_changes_json, "r2_changes.json")
                return f"runs:/{parent_run_id}/r2_changes.json"
        except Exception as e:
            logger.error(f"Error saving R2 changes: {e}")
            return None
    
    def _select_experiments_heuristic(self, r1_plan: Dict[str, Any]) -> List[int]:
        """Select experiments using simple heuristic."""
        experiments = r1_plan.get('experiments', [])
        if len(experiments) <= 2:
            return list(range(len(experiments)))
        
        # Simple heuristic: select first 2 experiments
        return [0, 1]
    
    def _execute_experiment(self, 
                           parent_run_id: str,
                           experiment: Dict[str, Any],
                           r2_changes: Dict[str, Any],
                           iteration_n: int) -> Optional[str]:
        """Execute a single experiment."""
        try:
            # Dynamic imports to avoid circular dependencies
            from ..core.backtest import Backtester
            from ..core.portfolio_backtest import PortfolioBacktester
            from ..data.yf_loader import load_yf_data
            from ..strategies.strategy_registry import get_strategy_class
            
            # Extract experiment parameters
            strategy_name = experiment['strategy']
            overlays = experiment['overlays']
            
            # Get strategy class
            strategy_class = get_strategy_class(strategy_name)
            if not strategy_class:
                logger.error(f"Strategy {strategy_name} not found")
                return None
            
            # Load data (simplified - would need to extract from parent run)
            tickers = ['AAPL', 'GOOGL', 'MSFT']  # Default for now
            data = load_yf_data(tickers, start_date='2023-01-01', end_date='2024-01-01')
            
            # Create strategy instance
            strategy = strategy_class()
            
            # Determine if portfolio strategy
            is_portfolio = strategy.is_portfolio_strategy()
            
            # Execute backtest
            if is_portfolio:
                backtester = PortfolioBacktester()
                results = backtester.run_portfolio_backtest(
                    strategy=strategy,
                    data_dict={ticker: data for ticker in tickers},
                    start_date='2023-01-01',
                    end_date='2024-01-01'
                )
            else:
                backtester = Backtester()
                results = backtester.run_backtest(
                    strategy=strategy,
                    data=data,
                    start_date='2023-01-01',
                    end_date='2024-01-01'
                )
            
            # Get the new run ID
            if mlflow.active_run():
                new_run_id = mlflow.active_run().info.run_id
                
                # Set parent run tags
                mlflow.set_tag("parent_run_id", parent_run_id)
                mlflow.set_tag("iteration_n", str(iteration_n))
                mlflow.set_tag("experiment_id", experiment['id'])
                
                return new_run_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing experiment: {e}")
            return None
    
    def _create_iteration_graph(self, 
                              parent_run_id: str,
                              parent_evaluation: Dict[str, Any],
                              iteration_results: List[Dict[str, Any]],
                              evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create iteration graph linking runs."""
        graph = {
            'parent_run': {
                'run_id': parent_run_id,
                'promotion_decision': parent_evaluation['promotion_decision']['promote'],
                'significance_verdict': parent_evaluation['significance_verdict'],
                'sharpe_ratio': parent_evaluation['key_metrics'].get('sharpe_ratio', 0),
                'max_drawdown': parent_evaluation['key_metrics'].get('max_drawdown', 0)
            },
            'iterations': evaluation_results,
            'created_at': datetime.now().isoformat(),
            'total_iterations': len(iteration_results),
            'successful_iterations': len([r for r in iteration_results if r['status'] == 'completed']),
            'promoted_iterations': len([r for r in evaluation_results if r['promotion_decision']])
        }
        
        return graph
    
    def _save_experiment_lineage(self, parent_run_id: str, iteration_graph: Dict[str, Any]) -> str:
        """Save experiment lineage to MLflow artifacts."""
        try:
            with mlflow.start_run(run_id=parent_run_id):
                lineage_json = json.dumps(iteration_graph, indent=2)
                mlflow.log_text(lineage_json, "experiment_lineage.json")
                return f"runs:/{parent_run_id}/experiment_lineage.json"
        except Exception as e:
            logger.error(f"Error saving experiment lineage: {e}")
            return None
    
    def _update_parent_run_tags(self, parent_run_id: str, r1_plan_path: str, lineage_path: str):
        """Update parent run with iteration tags."""
        try:
            with mlflow.start_run(run_id=parent_run_id):
                mlflow.set_tag("r1_plan_path", r1_plan_path)
                mlflow.set_tag("r1_time", datetime.now().isoformat())
                mlflow.set_tag("lineage_path", lineage_path)
                mlflow.set_tag("iteration_completed", "true")
        except Exception as e:
            logger.error(f"Error updating parent run tags: {e}")
    
    def get_iteration_status(self, parent_run_id: str) -> Dict[str, Any]:
        """Get status of iteration for a parent run."""
        try:
            with mlflow.start_run(run_id=parent_run_id):
                run = mlflow.get_run(parent_run_id)
                tags = run.data.tags
                
                return {
                    'parent_run_id': parent_run_id,
                    'has_r1_plan': 'r1_plan_path' in tags,
                    'has_lineage': 'lineage_path' in tags,
                    'iteration_completed': tags.get('iteration_completed', 'false') == 'true',
                    'r1_plan_path': tags.get('r1_plan_path'),
                    'lineage_path': tags.get('lineage_path')
                }
        except Exception as e:
            logger.error(f"Error getting iteration status: {e}")
            return {'error': str(e)}
