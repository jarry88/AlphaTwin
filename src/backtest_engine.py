"""
Backtesting Engine for AlphaTwin

Comprehensive backtesting framework with performance analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class BacktestResult:
    """Container for backtest results."""
    portfolio_value: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    drawdowns: pd.Series


class BacktestEngine:
    """Main backtesting engine class."""

    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting portfolio value
            commission: Trading commission per trade (as fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission

    def run_backtest(self,
                    data: pd.DataFrame,
                    signals: pd.Series,
                    position_size: float = 1.0) -> BacktestResult:
        """
        Run backtest with given signals.

        Args:
            data: Market data DataFrame
            signals: Trading signals (-1, 0, 1)
            position_size: Fraction of capital to invest per trade

        Returns:
            BacktestResult object with all results
        """
        # Initialize portfolio
        portfolio_value = pd.Series(index=data.index, dtype=float)
        portfolio_value.iloc[0] = self.initial_capital

        position = 0  # Current position (-1, 0, 1)
        shares = 0
        cash = self.initial_capital

        trades = []

        for i in range(1, len(data)):
            current_price = data.iloc[i]['Adj Close']
            prev_price = data.iloc[i-1]['Adj Close']
            current_signal = signals.iloc[i]

            # Check for signal change (entry/exit)
            if current_signal != position:
                # Calculate trade size
                trade_value = portfolio_value.iloc[i-1] * position_size

                if current_signal == 1:  # Buy signal
                    shares_to_buy = trade_value / current_price
                    cost = shares_to_buy * current_price * (1 + self.commission)

                    if cash >= cost:
                        shares = shares_to_buy
                        cash -= cost
                        position = 1
                        trades.append({
                            'date': data.index[i],
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'value': trade_value,
                            'commission': cost - (shares * current_price)
                        })

                elif current_signal == -1:  # Sell signal
                    if shares > 0:
                        sale_value = shares * current_price * (1 - self.commission)
                        cash += sale_value
                        trades.append({
                            'date': data.index[i],
                            'type': 'SELL',
                            'price': current_price,
                            'shares': shares,
                            'value': sale_value,
                            'commission': (shares * current_price) - sale_value
                        })
                        shares = 0
                        position = -1

                elif current_signal == 0:  # Exit position
                    if shares > 0:
                        sale_value = shares * current_price * (1 - self.commission)
                        cash += sale_value
                        trades.append({
                            'date': data.index[i],
                            'type': 'SELL',
                            'price': current_price,
                            'shares': shares,
                            'value': sale_value,
                            'commission': (shares * current_price) - sale_value
                        })
                        shares = 0
                        position = 0

            # Update portfolio value
            portfolio_value.iloc[i] = cash + (shares * current_price)

        # Calculate returns
        returns = portfolio_value.pct_change().fillna(0)

        # Create trades DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        # Calculate performance metrics
        perf_metrics = self._calculate_performance_metrics(portfolio_value, returns)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(returns)

        # Calculate drawdowns
        drawdowns = self._calculate_drawdowns(portfolio_value)

        return BacktestResult(
            portfolio_value=portfolio_value,
            returns=returns,
            trades=trades_df,
            performance_metrics=perf_metrics,
            risk_metrics=risk_metrics,
            drawdowns=drawdowns
        )

    def _calculate_performance_metrics(self, portfolio_value: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        annual_return = returns.mean() * 252  # Assuming 252 trading days
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Maximum drawdown
        max_drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_value.iloc[-1]
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics."""
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean()

        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = returns.mean() * 252 / downside_deviation if downside_deviation > 0 else 0

        return {
            'var_95': var_95,
            'es_95': es_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sortino_ratio': sortino_ratio
        }

    def _calculate_drawdowns(self, portfolio_value: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value - peak) / peak
        return drawdown

    def plot_results(self, result: BacktestResult, save_path: Optional[str] = None):
        """Plot backtest results."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Portfolio value
        axes[0].plot(result.portfolio_value.index, result.portfolio_value.values)
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_ylabel('Value ($)')
        axes[0].grid(True)

        # Drawdown
        axes[1].fill_between(result.drawdowns.index, result.drawdowns.values * 100, 0, alpha=0.3, color='red')
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True)

        # Returns distribution
        axes[2].hist(result.returns.values * 100, bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_title('Returns Distribution')
        axes[2].set_xlabel('Daily Return (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def print_summary(self, result: BacktestResult):
        """Print backtest summary."""
        print("=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)

        print("PERFORMANCE METRICS:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")

        print("\nRISK METRICS:")
        print(".4f")
        print(".4f")
        print(".2f")
        print(".2f")
        print(".2f")

        print(f"\nTRADING SUMMARY:")
        if not result.trades.empty:
            print(f"Total Trades: {len(result.trades)}")
            buy_trades = len(result.trades[result.trades['type'] == 'BUY'])
            sell_trades = len(result.trades[result.trades['type'] == 'SELL'])
            print(f"Buy Trades: {buy_trades}")
            print(f"Sell Trades: {sell_trades}")
            print(".2f")
        else:
            print("No trades executed")

        print("=" * 60)


class StrategyComparator:
    """Compare multiple backtest results."""

    def __init__(self):
        self.results = {}

    def add_result(self, name: str, result: BacktestResult):
        """Add a backtest result."""
        self.results[name] = result

    def compare_performance(self) -> pd.DataFrame:
        """Compare performance metrics across strategies."""
        comparison = {}
        for name, result in self.results.items():
            comparison[name] = result.performance_metrics

        return pd.DataFrame(comparison).T

    def plot_comparison(self, metric: str = 'sharpe_ratio', save_path: Optional[str] = None):
        """Plot comparison of a specific metric."""
        if not self.results:
            print("No results to compare")
            return

        data = self.compare_performance()
        if metric not in data.columns:
            print(f"Metric '{metric}' not found")
            return

        plt.figure(figsize=(10, 6))
        data[metric].plot(kind='bar')
        plt.title(f'Comparison of {metric.replace("_", " ").title()}')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data_loader import DataLoader
    from signals import SignalManager, MovingAverageCrossover

    # Load data
    loader = DataLoader()
    try:
        data = loader.load_processed_data("AAPL")
    except FileNotFoundError:
        print("No processed data found. Please run data_loader.py first.")
        sys.exit(1)

    # Generate signals
    manager = SignalManager()
    manager.add_generator("MA_Crossover", MovingAverageCrossover())
    signals = manager.generate_all_signals(data)['MA_Crossover']

    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    result = engine.run_backtest(data, signals)

    # Print summary
    engine.print_summary(result)

    # Plot results (would display if matplotlib backend is available)
    try:
        engine.plot_results(result, save_path="backtest_results.png")
        print("Results plot saved as backtest_results.png")
    except:
        print("Could not generate plot (matplotlib may not be available)")
