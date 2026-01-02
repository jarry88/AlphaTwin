"""
Parameter Scanner Module for AlphaTwin

Provides comprehensive parameter optimization and heatmap visualization
for trading strategy parameter tuning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from typing import Dict, List, Tuple, Callable, Optional, Any
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class ParameterScanner:
    """
    Comprehensive parameter scanning and optimization for trading strategies.

    Supports grid search, random search, and visualization of parameter
    combinations with performance heatmaps.
    """

    def __init__(self,
                 strategy_class: Callable,
                 data: pd.DataFrame,
                 param_ranges: Dict[str, List[Any]],
                 metric: str = 'sharpe_ratio',
                 initial_capital: float = 10000,
                 risk_free_rate: float = 0.02):
        """
        Initialize parameter scanner.

        Args:
            strategy_class: Trading strategy class to optimize
            data: Market data for backtesting
            param_ranges: Dictionary of parameter names and their value ranges
            metric: Performance metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown', etc.)
            initial_capital: Starting portfolio value
            risk_free_rate: Risk-free rate for risk-adjusted metrics
        """
        self.strategy_class = strategy_class
        self.data = data
        self.param_ranges = param_ranges
        self.metric = metric
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        # Validate inputs
        self._validate_inputs()

        # Generate parameter combinations
        self.param_combinations = self._generate_param_combinations()

        logger.info(f"Initialized parameter scanner with {len(self.param_combinations)} combinations")

    def _validate_inputs(self):
        """Validate input parameters."""
        if not hasattr(self.strategy_class, 'generate_signals'):
            raise ValueError("Strategy class must have 'generate_signals' method")

        if not isinstance(self.param_ranges, dict):
            raise ValueError("param_ranges must be a dictionary")

        if self.metric not in ['sharpe_ratio', 'sortino_ratio', 'total_return', 'max_drawdown', 'win_rate']:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        param_names = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())

        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def scan_parameters(self,
                       parallel: bool = False,
                       max_workers: int = None) -> pd.DataFrame:
        """
        Perform parameter scanning.

        Args:
            parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers

        Returns:
            DataFrame with parameter combinations and performance metrics
        """
        logger.info(f"Starting parameter scan with {len(self.param_combinations)} combinations")

        results = []

        if parallel and len(self.param_combinations) > 10:
            # Use parallel processing for large parameter spaces
            results = self._scan_parallel(max_workers)
        else:
            # Sequential processing
            for i, params in enumerate(self.param_combinations):
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(self.param_combinations)} combinations")

                result = self._evaluate_params(params)
                results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Add ranking column
        if self.metric in ['sharpe_ratio', 'sortino_ratio', 'total_return', 'win_rate']:
            # Higher is better
            df['rank'] = df[self.metric].rank(ascending=False)
        else:
            # Lower is better (like max_drawdown)
            df['rank'] = df[self.metric].rank(ascending=True)

        logger.info(f"Parameter scan completed. Best {self.metric}: {df[self.metric].max():.4f}")

        return df

    def _scan_parallel(self, max_workers: int = None) -> List[Dict]:
        """Perform parallel parameter scanning."""
        if max_workers is None:
            max_workers = min(4, len(self.param_combinations))

        results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parameter evaluations
            futures = [executor.submit(self._evaluate_params, params)
                      for params in self.param_combinations]

            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parameter evaluation failed: {e}")
                    results.append({'error': str(e)})

        return results

    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single parameter combination.

        Args:
            params: Parameter dictionary

        Returns:
            Dictionary with parameters and performance metrics
        """
        try:
            # Create strategy instance
            strategy = self.strategy_class(**params)

            # Generate signals
            signals = strategy.generate_signals(self.data)

            # Run backtest
            from src.backtest_engine import BacktestEngine
            engine = BacktestEngine(initial_capital=self.initial_capital)

            result = engine.run_backtest(self.data, signals)

            # Extract relevant metrics
            metrics = {
                'total_return': result.performance_metrics['total_return'],
                'annual_return': result.performance_metrics['annual_return'],
                'volatility': result.performance_metrics['volatility'],
                'sharpe_ratio': result.performance_metrics['sharpe_ratio'],
                'max_drawdown': result.performance_metrics['max_drawdown'],
                'final_value': result.performance_metrics['final_value']
            }

            # Add risk metrics if available
            if hasattr(result, 'risk_metrics'):
                metrics.update({
                    'var_95': result.risk_metrics.get('var_95', 0),
                    'sortino_ratio': result.risk_metrics.get('sortino_ratio', 0)
                })

            # Add trading statistics
            if hasattr(result, 'trades') and result.trades is not None and not result.trades.empty:
                metrics.update({
                    'total_trades': len(result.trades),
                    'win_rate': len(result.trades[result.trades['type'] == 'SELL']) / len(result.trades) if len(result.trades) > 0 else 0,
                    'avg_trade_return': result.trades['value'].mean() if 'value' in result.trades.columns else 0
                })

            # Combine parameters and metrics
            result_dict = params.copy()
            result_dict.update(metrics)

            return result_dict

        except Exception as e:
            logger.warning(f"Parameter evaluation failed for {params}: {e}")
            result_dict = params.copy()
            result_dict['error'] = str(e)
            result_dict[self.metric] = -999  # Mark as invalid
            return result_dict

    def create_heatmap(self,
                      results_df: pd.DataFrame,
                      x_param: str,
                      y_param: str,
                      metric: str = None,
                      title: str = None,
                      save_path: str = None) -> plt.Figure:
        """
        Create parameter heatmap visualization.

        Args:
            results_df: Results DataFrame from scan_parameters
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis
            metric: Metric to visualize (defaults to scanner metric)
            title: Plot title
            save_path: Path to save plot

        Returns:
            Matplotlib figure object
        """
        if metric is None:
            metric = self.metric

        # Filter out error results
        valid_results = results_df[results_df[metric] != -999]

        if len(valid_results) == 0:
            raise ValueError("No valid results to plot")

        # Create pivot table for heatmap
        pivot_data = valid_results.pivot_table(
            values=metric,
            index=y_param,
            columns=x_param,
            aggfunc='mean'  # Use mean if multiple values
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        # Custom colormap - red for bad, green for good
        if metric in ['sharpe_ratio', 'sortino_ratio', 'total_return', 'win_rate']:
            cmap = 'RdYlGn'  # Red to Yellow to Green (higher is better)
        else:
            cmap = 'RdYlGn_r'  # Reversed (lower is better)

        sns.heatmap(pivot_data,
                   annot=True,
                   fmt='.3f',
                   cmap=cmap,
                   center=pivot_data.mean().mean(),
                   ax=ax,
                   cbar_kws={'label': metric.replace('_', ' ').title()})

        # Formatting
        if title is None:
            title = f'Parameter Optimization Heatmap\n{metric.replace("_", " ").title()} by {x_param} vs {y_param}'

        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=14)
        ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=14)

        # Rotate x-axis labels if too many
        if len(pivot_data.columns) > 10:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")

        return fig

    def create_surface_plot(self,
                           results_df: pd.DataFrame,
                           x_param: str,
                           y_param: str,
                           metric: str = None,
                           save_path: str = None) -> plt.Figure:
        """
        Create 3D surface plot for parameter visualization.

        Args:
            results_df: Results DataFrame from scan_parameters
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis
            metric: Metric to visualize
            save_path: Path to save plot

        Returns:
            Matplotlib figure object
        """
        from mpl_toolkits.mplot3d import Axes3D

        if metric is None:
            metric = self.metric

        # Filter valid results
        valid_results = results_df[results_df[metric] != -999]

        # Create meshgrid for surface plot
        x_unique = sorted(valid_results[x_param].unique())
        y_unique = sorted(valid_results[y_param].unique())

        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.zeros_like(X, dtype=float)

        # Fill Z values
        for i, y_val in enumerate(y_unique):
            for j, x_val in enumerate(x_unique):
                mask = (valid_results[x_param] == x_val) & (valid_results[y_param] == y_val)
                if mask.any():
                    Z[i, j] = valid_results.loc[mask, metric].mean()

        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(X, Y, Z,
                              cmap='viridis',
                              edgecolor='none',
                              alpha=0.8)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=metric.replace('_', ' ').title())

        # Labels and title
        ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=12)
        ax.set_zlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'3D Parameter Optimization Surface\n{metric.replace("_", " ").title()}', fontsize=14, pad=20)

        # Adjust viewing angle for better visualization
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Surface plot saved to {save_path}")

        return fig

    def find_optimal_parameters(self,
                               results_df: pd.DataFrame,
                               top_n: int = 5,
                               metric: str = None) -> pd.DataFrame:
        """
        Find optimal parameter combinations.

        Args:
            results_df: Results DataFrame from scan_parameters
            top_n: Number of top combinations to return
            metric: Metric to optimize (defaults to scanner metric)

        Returns:
            DataFrame with top parameter combinations
        """
        if metric is None:
            metric = self.metric

        # Filter valid results
        valid_results = results_df[results_df[metric] != -999].copy()

        if len(valid_results) == 0:
            raise ValueError("No valid results to analyze")

        # Sort by metric (handle different optimization directions)
        if metric in ['sharpe_ratio', 'sortino_ratio', 'total_return', 'win_rate']:
            # Higher is better
            valid_results = valid_results.sort_values(metric, ascending=False)
        else:
            # Lower is better (like max_drawdown)
            valid_results = valid_results.sort_values(metric, ascending=True)

        # Return top N combinations
        top_combinations = valid_results.head(top_n)

        # Add parameter summary
        param_cols = [col for col in top_combinations.columns
                     if col not in ['total_return', 'annual_return', 'volatility',
                                  'sharpe_ratio', 'max_drawdown', 'final_value',
                                  'var_95', 'sortino_ratio', 'total_trades',
                                  'win_rate', 'avg_trade_return', 'rank', 'error']]

        summary = top_combinations[param_cols + [metric, 'rank']].copy()

        return summary

    def generate_report(self,
                       results_df: pd.DataFrame,
                       save_path: str = None) -> str:
        """
        Generate comprehensive parameter scanning report.

        Args:
            results_df: Results DataFrame from scan_parameters
            save_path: Path to save report (optional)

        Returns:
            Report as formatted string
        """
        report_lines = []
        report_lines.append("# Parameter Scanning Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Summary statistics
        valid_results = results_df[results_df[self.metric] != -999]
        total_combinations = len(results_df)
        valid_combinations = len(valid_results)
        success_rate = valid_combinations / total_combinations * 100

        report_lines.append("## Summary Statistics")
        report_lines.append(f"- Total parameter combinations tested: {total_combinations}")
        report_lines.append(f"- Valid results: {valid_combinations} ({success_rate:.1f}%)")
        report_lines.append(f"- Optimization metric: {self.metric.replace('_', ' ').title()}")
        report_lines.append("")

        if len(valid_results) > 0:
            # Best result
            if self.metric in ['sharpe_ratio', 'sortino_ratio', 'total_return', 'win_rate']:
                best_result = valid_results.loc[valid_results[self.metric].idxmax()]
            else:
                best_result = valid_results.loc[valid_results[self.metric].idxmin()]

            report_lines.append("## Best Parameter Combination")
            report_lines.append(f"- {self.metric.replace('_', ' ').title()}: {best_result[self.metric]:.4f}")

            # Parameter values
            param_cols = [col for col in valid_results.columns
                         if col not in ['total_return', 'annual_return', 'volatility',
                                      'sharpe_ratio', 'max_drawdown', 'final_value',
                                      'var_95', 'sortino_ratio', 'total_trades',
                                      'win_rate', 'avg_trade_return', 'rank', 'error']]

            for param in param_cols:
                report_lines.append(f"- {param}: {best_result[param]}")

            report_lines.append("")

            # Performance metrics
            report_lines.append("## Performance Metrics (Best Combination)")
            report_lines.append(f"- Total Return: {best_result.get('total_return', 'N/A'):.2%}")
            report_lines.append(f"- Annual Return: {best_result.get('annual_return', 'N/A'):.2%}")
            report_lines.append(f"- Volatility: {best_result.get('volatility', 'N/A'):.2%}")
            report_lines.append(f"- Max Drawdown: {best_result.get('max_drawdown', 'N/A'):.2%}")
            report_lines.append(f"- Sharpe Ratio: {best_result.get('sharpe_ratio', 'N/A'):.2f}")
            report_lines.append("")

            # Statistical summary
            report_lines.append("## Statistical Summary")
            report_lines.append(f"- Mean {self.metric.replace('_', ' ').title()}: {valid_results[self.metric].mean():.4f}")
            report_lines.append(f"- Std {self.metric.replace('_', ' ').title()}: {valid_results[self.metric].std():.4f}")
            report_lines.append(f"- Best {self.metric.replace('_', ' ').title()}: {valid_results[self.metric].max():.4f}")
            report_lines.append(f"- Worst {self.metric.replace('_', ' ').title()}: {valid_results[self.metric].min():.4f}")
            report_lines.append("")

            # Parameter analysis
            if len(param_cols) >= 2:
                report_lines.append("## Parameter Analysis")
                for param in param_cols:
                    best_val = best_result[param]
                    report_lines.append(f"- {param}: Optimal value = {best_val}")
                report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        if len(valid_results) > 0:
            report_lines.append("✅ Parameter scanning completed successfully")
            report_lines.append("✅ Optimal parameters identified")
            if len(param_cols) >= 2:
                report_lines.append("✅ Heatmap visualization recommended for parameter relationships")
        else:
            report_lines.append("❌ No valid results generated - check strategy implementation")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append(f"Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")

        return report


def scan_moving_average_strategy(data: pd.DataFrame,
                               short_range: List[int] = None,
                               long_range: List[int] = None) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Convenience function to scan Moving Average crossover strategy parameters.

    Args:
        data: Market data DataFrame
        short_range: Range of short MA periods
        long_range: Range of long MA periods

    Returns:
        Tuple of (results_df, heatmap_figure)
    """
    from src.signals import MovingAverageCrossover

    # Default parameter ranges
    if short_range is None:
        short_range = [5, 10, 15, 20, 25, 30]
    if long_range is None:
        long_range = [30, 40, 50, 60, 70, 80, 90, 100]

    # Set up parameter scanner
    param_ranges = {
        'short_window': short_range,
        'long_window': long_range
    }

    scanner = ParameterScanner(
        strategy_class=MovingAverageCrossover,
        data=data,
        param_ranges=param_ranges,
        metric='sharpe_ratio'
    )

    # Run parameter scan
    results_df = scanner.scan_parameters()

    # Create heatmap
    heatmap_fig = scanner.create_heatmap(
        results_df=results_df,
        x_param='short_window',
        y_param='long_window',
        title='Moving Average Crossover Parameter Optimization'
    )

    return results_df, heatmap_fig


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append('.')

    # Load sample data
    from src.data_loader import DataLoader
    from src.signals import MovingAverageCrossover

    print("AlphaTwin Parameter Scanner Demo")
    print("=" * 40)

    try:
        # Load sample data
        loader = DataLoader()
        data = loader.download_stock_data('EURUSD=X', '2023-01-01', '2024-01-01')
        processed_data = loader.process_data(data)

        print(f"Loaded {len(processed_data)} days of EUR/USD data")

        # Run parameter scan
        results_df, heatmap = scan_moving_average_strategy(processed_data)

        # Display top results
        print("\nTop 5 Parameter Combinations:")
        top_results = results_df.nlargest(5, 'sharpe_ratio')
        for i, row in top_results.iterrows():
            print(f"{i+1}. Short: {row['short_window']}, Long: {row['long_window']}, Sharpe: {row['sharpe_ratio']:.2f}")

        # Save heatmap
        heatmap.savefig('ma_parameter_heatmap.png', dpi=300, bbox_inches='tight')
        print("\nHeatmap saved as 'ma_parameter_heatmap.png'")

        # Generate report
        scanner = ParameterScanner(MovingAverageCrossover, processed_data,
                                 {'short_window': [5, 10, 15, 20], 'long_window': [30, 50, 70]})
        results = scanner.scan_parameters()
        report = scanner.generate_report(results, save_path='parameter_scan_report.md')

        print("Report saved as 'parameter_scan_report.md'")

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
