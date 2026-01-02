# Performance Metrics Definitions

**Version**: 1.0
**Date**: January 2, 2026
**Status**: Active

## Overview

This document provides comprehensive definitions and explanations of key performance metrics used in quantitative trading strategy evaluation. Understanding these metrics is crucial for making informed decisions about strategy selection, risk management, and portfolio optimization.

## Risk-Adjusted Return Metrics

### Sharpe Ratio

#### Definition
The Sharpe Ratio measures the excess return per unit of risk, quantifying how well a strategy compensates investors for the risk taken.

**Formula**:
```
Sharpe Ratio = (Rp - Rf) / σp
```

Where:
- **Rp**: Portfolio/Strategy return
- **Rf**: Risk-free rate (typically 2-3% annual)
- **σp**: Standard deviation of portfolio returns (volatility)

#### Interpretation
- **Sharpe Ratio > 1**: Good risk-adjusted returns
- **Sharpe Ratio > 2**: Excellent risk-adjusted returns
- **Sharpe Ratio < 0**: Strategy underperforming risk-free assets

#### Example Calculation
```python
# Monthly returns: [2.1%, -1.8%, 3.2%, 1.5%, -0.5%]
returns = [0.021, -0.018, 0.032, 0.015, -0.005]
risk_free_rate = 0.02/12  # 2% annual, monthly

# Calculate Sharpe Ratio
excess_returns = [r - risk_free_rate for r in returns]
avg_excess_return = sum(excess_returns) / len(excess_returns)
volatility = (sum((r - avg_excess_return)**2 for r in excess_returns) / len(excess_returns))**0.5
sharpe_ratio = avg_excess_return / volatility

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
```

#### Annualization
For daily/weekly data, annualize the Sharpe Ratio:
```
Annualized Sharpe = Daily Sharpe × √(trading days per year)
```

#### Limitations
- Assumes normal distribution of returns
- Penalizes upside volatility equally with downside
- Sensitive to outlier returns
- Not suitable for strategies with asymmetric return distributions

### Sortino Ratio

#### Definition
The Sortino Ratio is similar to the Sharpe Ratio but only considers downside volatility, focusing on the risk of losses rather than volatility in general.

**Formula**:
```
Sortino Ratio = (Rp - Rf) / σd
```

Where:
- **Rp**: Portfolio/Strategy return
- **Rf**: Risk-free rate
- **σd**: Standard deviation of negative returns (downside deviation)

#### Interpretation
- **Sortino Ratio > 1**: Good downside risk-adjusted returns
- **Sortino Ratio > 2**: Excellent downside risk-adjusted returns
- Higher values indicate better risk-adjusted performance

#### Example Calculation
```python
def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate Sortino Ratio"""
    # Annualize risk-free rate if needed
    rf_daily = risk_free_rate / 252  # Assuming daily returns

    # Calculate excess returns
    excess_returns = returns - rf_daily

    # Calculate downside deviation (only negative excess returns)
    negative_excess = excess_returns[excess_returns < 0]
    if len(negative_excess) == 0:
        return float('inf')  # No downside risk

    downside_deviation = np.sqrt(np.mean(negative_excess**2))

    # Calculate Sortino Ratio
    if downside_deviation == 0:
        return float('inf')

    sortino_ratio = np.mean(excess_returns) / downside_deviation
    return sortino_ratio

# Example usage
daily_returns = np.array([0.01, -0.005, 0.008, -0.012, 0.015])
sortino = calculate_sortino_ratio(daily_returns)
print(f"Sortino Ratio: {sortino:.2f}")
```

#### Advantages Over Sharpe Ratio
- Focuses on harmful volatility (losses)
- Better for strategies with positive skewness
- More relevant for risk-averse investors
- Penalizes only downside risk

### Information Ratio

#### Definition
The Information Ratio measures the active return per unit of active risk, comparing a portfolio's performance against a benchmark.

**Formula**:
```
Information Ratio = (Rp - Rb) / σ(Rp - Rb)
```

Where:
- **Rp**: Portfolio return
- **Rb**: Benchmark return
- **σ(Rp - Rb)**: Tracking error (volatility of active returns)

#### Interpretation
- **Information Ratio > 0.5**: Good active management
- **Information Ratio > 1.0**: Excellent active management
- Measures consistency of outperformance

## Risk Metrics

### Maximum Drawdown (Max Drawdown)

#### Definition
Maximum Drawdown represents the largest peak-to-trough decline in portfolio value over a specified period.

**Formula**:
```
Max Drawdown = min((Portfolio Value - Peak Value) / Peak Value)
```

#### Calculation Method
```python
def calculate_max_drawdown(portfolio_values):
    """Calculate Maximum Drawdown"""
    portfolio_values = pd.Series(portfolio_values)

    # Calculate cumulative maximum (peaks)
    peaks = portfolio_values.expanding().max()

    # Calculate drawdowns
    drawdowns = (portfolio_values - peaks) / peaks

    # Maximum drawdown (most negative value)
    max_drawdown = drawdowns.min()

    return abs(max_drawdown)  # Return as positive percentage

# Example
portfolio_values = [10000, 10500, 10200, 10800, 9500, 11000]
mdd = calculate_max_drawdown(portfolio_values)
print(f"Max Drawdown: {mdd:.2%}")
```

#### Interpretation
- **Max Drawdown < 10%**: Low risk strategy
- **Max Drawdown 10-20%**: Moderate risk strategy
- **Max Drawdown > 30%**: High risk strategy

#### Recovery Analysis
```python
def analyze_drawdown_recovery(portfolio_values, threshold=0.1):
    """Analyze drawdown periods and recovery times"""
    values = pd.Series(portfolio_values)
    peaks = values.expanding().max()

    # Find drawdowns exceeding threshold
    drawdowns = (values - peaks) / peaks
    significant_drawdowns = drawdowns < -threshold

    # Group consecutive drawdown periods
    drawdown_periods = []
    current_period = []

    for i, is_drawdown in enumerate(significant_drawdowns):
        if is_drawdown:
            current_period.append(i)
        elif current_period:
            drawdown_periods.append(current_period)
            current_period = []

    # Analyze each drawdown period
    for period in drawdown_periods:
        start_idx = period[0]
        end_idx = period[-1]

        start_value = values.iloc[start_idx]
        end_value = values.iloc[end_idx]
        peak_value = peaks.iloc[start_idx]

        drawdown_pct = (start_value - peak_value) / peak_value
        recovery_pct = (end_value - start_value) / start_value

        print(f"Drawdown: {abs(drawdown_pct):.1%}, Recovery: {recovery_pct:.1%}, Duration: {len(period)} periods")

    return drawdown_periods
```

### Value at Risk (VaR)

#### Definition
Value at Risk estimates the maximum potential loss over a specific time period with a given confidence level.

**Methods**:
1. **Historical VaR**: Uses historical data distribution
2. **Parametric VaR**: Assumes normal distribution
3. **Monte Carlo VaR**: Simulates multiple scenarios

#### Historical VaR Calculation
```python
def calculate_historical_var(returns, confidence_level=0.95):
    """Calculate Historical Value at Risk"""
    # Sort returns in ascending order (worst to best)
    sorted_returns = np.sort(returns)

    # Find the return at the confidence level
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[var_index]

    return abs(var)  # Return as positive value

# Example
daily_returns = np.random.normal(0.001, 0.02, 1000)  # Simulated returns
var_95 = calculate_historical_var(daily_returns, 0.95)
var_99 = calculate_historical_var(daily_returns, 0.99)

print(f"95% VaR: {var_95:.2%} (lose this amount or more 5% of the time)")
print(f"99% VaR: {var_99:.2%} (lose this amount or more 1% of the time)")
```

#### Parametric VaR (Normal Distribution)
```python
def calculate_parametric_var(returns, confidence_level=0.95, position_value=100000):
    """Calculate Parametric Value at Risk assuming normal distribution"""
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # Z-score for confidence level
    if confidence_level == 0.95:
        z_score = 1.645
    elif confidence_level == 0.99:
        z_score = 2.326
    else:
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)

    # VaR calculation
    var_return = mean_return - z_score * std_return
    var_amount = abs(var_return) * position_value

    return var_amount, var_return

# Example
var_amount, var_return = calculate_parametric_var(daily_returns, position_value=100000)
print(f"Portfolio VaR: ${var_amount:.0f} ({var_return:.2%})")
```

### Expected Shortfall (CVaR)

#### Definition
Expected Shortfall (Conditional VaR) measures the average loss beyond the VaR threshold, providing a more complete picture of tail risk.

**Formula**:
```
ES = E[Loss | Loss > VaR]
```

#### Calculation
```python
def calculate_expected_shortfall(returns, confidence_level=0.95):
    """Calculate Expected Shortfall (Conditional VaR)"""
    # Calculate VaR first
    sorted_returns = np.sort(returns)
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var_threshold = sorted_returns[var_index]

    # Find all returns worse than VaR
    tail_losses = returns[returns <= var_threshold]

    if len(tail_losses) == 0:
        return 0

    # Expected Shortfall is the average of tail losses
    expected_shortfall = abs(np.mean(tail_losses))

    return expected_shortfall

# Example
es_95 = calculate_expected_shortfall(daily_returns, 0.95)
es_99 = calculate_expected_shortfall(daily_returns, 0.99)

print(f"95% Expected Shortfall: {es_95:.2%}")
print(f"99% Expected Shortfall: {es_99:.2%}")
```

## Performance Attribution Metrics

### Alpha

#### Definition
Alpha measures the excess return of a portfolio relative to its benchmark, representing the value added by active management.

**Formula**:
```
α = Rp - [Rf + β(Rm - Rf)]
```

Where:
- **Rp**: Portfolio return
- **Rf**: Risk-free rate
- **β**: Portfolio beta (market sensitivity)
- **Rm**: Market/benchmark return

#### Interpretation
- **Alpha > 0**: Portfolio outperformed the benchmark (risk-adjusted)
- **Alpha < 0**: Portfolio underperformed the benchmark
- **Alpha = 0**: Portfolio performance matches benchmark expectations

### Beta

#### Definition
Beta measures the volatility of a portfolio relative to the market, indicating systematic risk exposure.

**Formula**:
```
β = Cov(Rp, Rm) / Var(Rm)
```

Where:
- **Rp**: Portfolio returns
- **Rm**: Market/benchmark returns
- **Cov**: Covariance
- **Var**: Variance

#### Interpretation
- **Beta = 1**: Portfolio moves with the market
- **Beta > 1**: Portfolio is more volatile than the market
- **Beta < 1**: Portfolio is less volatile than the market
- **Beta = 0**: Portfolio has no correlation with the market

### R-Squared

#### Definition
R-Squared measures the percentage of portfolio variance explained by market movements, indicating diversification effectiveness.

**Formula**:
```
R² = 1 - (SSE / SST)
```

Where:
- **SSE**: Sum of squared errors (unexplained variance)
- **SST**: Total sum of squares (total variance)

#### Interpretation
- **R² = 1.0**: Portfolio moves perfectly with benchmark (no diversification)
- **R² = 0.0**: Portfolio movements completely independent of benchmark
- **R² = 0.5**: 50% of portfolio variance explained by market movements

## Implementation Examples

### Complete Performance Analysis
```python
import pandas as pd
import numpy as np
from scipy import stats

class PerformanceAnalyzer:
    """Comprehensive performance analysis for trading strategies"""

    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate

    def analyze_strategy(self, returns, benchmark_returns=None):
        """Complete performance analysis"""
        results = {}

        # Basic metrics
        results['total_return'] = (1 + returns).prod() - 1
        results['annual_return'] = results['total_return'] * (252 / len(returns))
        results['volatility'] = returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        excess_returns = returns - self.risk_free_rate/252
        results['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            results['sortino_ratio'] = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            results['sortino_ratio'] = float('inf')

        # Risk metrics
        portfolio_values = (1 + returns).cumprod()
        peaks = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - peaks) / peaks
        results['max_drawdown'] = abs(drawdowns.min())

        # VaR and CVaR
        results['var_95'] = abs(np.percentile(returns, 5))
        tail_losses = returns[returns <= np.percentile(returns, 5)]
        results['cvar_95'] = abs(tail_losses.mean()) if len(tail_losses) > 0 else 0

        # Statistical tests
        results['skewness'] = stats.skew(returns)
        results['kurtosis'] = stats.kurtosis(returns)
        results['is_normal'] = stats.shapiro(returns)[1] > 0.05  # Shapiro-Wilk test

        # Benchmark comparison (if provided)
        if benchmark_returns is not None:
            # Alpha and Beta calculation
            covariance = np.cov(returns, benchmark_returns)[0,1]
            benchmark_variance = np.var(benchmark_returns)
            results['beta'] = covariance / benchmark_variance

            benchmark_annual = (1 + benchmark_returns).prod() ** (252 / len(benchmark_returns)) - 1
            results['alpha'] = results['annual_return'] - (self.risk_free_rate + results['beta'] * (benchmark_annual - self.risk_free_rate))

            # Information Ratio
            active_returns = returns - benchmark_returns
            results['information_ratio'] = active_returns.mean() / active_returns.std() * np.sqrt(252)

        return results

# Example usage
analyzer = PerformanceAnalyzer()
strategy_returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
benchmark_returns = np.random.normal(0.0008, 0.015, 252)  # S&P 500 proxy

analysis = analyzer.analyze_strategy(strategy_returns, benchmark_returns)

print("Performance Analysis Results:")
for metric, value in analysis.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")
```

## Best Practices

### Metric Selection Guidelines
1. **Use Sharpe Ratio** for traditional risk-adjusted performance
2. **Use Sortino Ratio** when downside risk is primary concern
3. **Use Max Drawdown** to understand worst-case scenarios
4. **Use VaR/CVaR** for regulatory compliance and risk limits
5. **Use Alpha/Beta** for benchmark-relative performance

### Interpretation Caveats
1. **Time Period Dependency**: Metrics can vary significantly across different time periods
2. **Market Regime Sensitivity**: Performance metrics may not hold in different market conditions
3. **Benchmark Selection**: Choose appropriate benchmarks for meaningful comparisons
4. **Statistical Significance**: Consider confidence intervals for metric reliability

### Reporting Standards
1. **Always include time period** and market conditions
2. **Specify benchmark** used for relative metrics
3. **Include confidence intervals** where applicable
4. **Document assumptions** (risk-free rate, trading days, etc.)

---

**Understanding and correctly applying performance metrics is essential for systematic trading strategy evaluation. These metrics provide quantitative measures of risk, return, and risk-adjusted performance, enabling data-driven decision making in portfolio management.**
