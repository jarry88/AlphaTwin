"""
Video Demonstration Code for Phase 3 MVP

Contains clean, well-commented code examples for video tutorials:
1. Data Acquisition and Exploration
2. Basic Backtesting (Golden Cross Strategy)

These examples are designed for educational video content.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for professional appearance
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def demo_1_data_acquisition():
    """
    Video 1 Demo: Getting the Data into Python

    Demonstrates:
    - Downloading EUR/USD data with yfinance
    - Basic DataFrame exploration
    - Data quality checks
    - Statistical analysis
    """
    print("🎯 Video 1: Getting the Data into Python")
    print("=" * 50)

    # Step 1: Download EUR/USD hourly data for 1 year
    print("📊 Step 1: Downloading EUR/USD data...")
    symbol = "EURUSD=X"
    data = yf.download(symbol, period="1y", interval="1h")

    print(f"✅ Downloaded {len(data)} hours of data")
    print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
    print()

    # Step 2: Basic exploration
    print("🔍 Step 2: Exploring the DataFrame structure...")
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print()

    # Show first few rows
    print("📋 First 5 rows (head):")
    print(data.head())
    print()

    # Show last few rows
    print("📋 Last 5 rows (tail):")
    print(data.tail())
    print()

    # Step 3: Data quality checks
    print("🧹 Step 3: Data Quality Checks...")

    # Check for missing values
    print("Missing values per column:")
    missing = data.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  ❌ {col}: {count} missing")
        else:
            print(f"  ✅ {col}: No missing values")
    print()

    # Check date continuity (should be every hour during market hours)
    print("Time series continuity check:")
    time_diffs = data.index.to_series().diff().dropna()
    expected_diff = pd.Timedelta(hours=1)

    gaps = (time_diffs != expected_diff).sum()
    if gaps > 0:
        print(f"  ⚠️  Found {gaps} time gaps (expected hourly data)")
    else:
        print("  ✅ Perfect hourly continuity")
    print()

    # Step 4: Statistical analysis
    print("📈 Step 4: Statistical Analysis...")

    # Price statistics
    print("Price statistics (OHLC):")
    price_stats = data[['Open', 'High', 'Low', 'Close']].describe()
    print(price_stats.round(5))
    print()

    # Daily range analysis
    data['Daily_Range'] = data['High'] - data['Low']
    data['Daily_Return'] = data['Close'].pct_change()

    print("Daily range and return statistics:")
    range_stats = data[['Daily_Range', 'Daily_Return']].describe()
    print(range_stats.round(6))
    print()

    # Step 5: Visualization
    print("📊 Step 5: Basic Visualization...")

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Price chart
    axes[0,0].plot(data.index[-100:], data['Close'].iloc[-100:])  # Last 100 hours
    axes[0,0].set_title('EUR/USD Price (Last 100 Hours)')
    axes[0,0].set_ylabel('Price')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Daily range distribution
    axes[0,1].hist(data['Daily_Range'].dropna() * 10000, bins=50, alpha=0.7)
    axes[0,1].set_title('Daily Range Distribution (in pips)')
    axes[0,1].set_xlabel('Range (pips)')
    axes[0,1].set_ylabel('Frequency')

    # Returns distribution
    axes[1,0].hist(data['Daily_Return'].dropna() * 100, bins=50, alpha=0.7, color='green')
    axes[1,0].set_title('Hourly Returns Distribution')
    axes[1,0].set_xlabel('Return (%)')
    axes[1,0].set_ylabel('Frequency')

    # Volume analysis (if available)
    if 'Volume' in data.columns:
        axes[1,1].plot(data.index[-100:], data['Volume'].iloc[-100:])
        axes[1,1].set_title('Volume (Last 100 Hours)')
        axes[1,1].set_ylabel('Volume')
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].text(0.5, 0.5, 'Volume data not available\nfor forex pairs',
                      transform=axes[1,1].transAxes, ha='center', va='center',
                      fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.tight_layout()
    plt.savefig('data_exploration_charts.png', dpi=300, bbox_inches='tight')
    print("✅ Charts saved as 'data_exploration_charts.png'")
    print()

    # Summary
    print("🎉 Data Acquisition Demo Complete!")
    print(f"Total records: {len(data)}")
    print(f"Data completeness: {((1 - data.isnull().mean().mean()) * 100):.1f}%")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    return data


def demo_2_golden_cross_backtest():
    """
    Video 2 Demo: Golden Cross Strategy Backtest

    Demonstrates:
    - SMA calculation and crossover logic
    - Signal generation
    - Simple backtesting framework
    - Performance visualization
    """
    print("🎯 Video 2: Golden Cross Strategy Backtest")
    print("=" * 50)

    # Step 1: Get data (reuse from demo 1)
    print("📊 Step 1: Loading EUR/USD data...")
    symbol = "EURUSD=X"
    data = yf.download(symbol, period="2y", interval="1d")  # Daily data for longer backtest
    print(f"✅ Loaded {len(data)} days of daily data")
    print()

    # Step 2: Calculate moving averages
    print("📈 Step 2: Calculating Moving Averages...")

    # Golden Cross: 50-day and 200-day SMAs
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    print("SMA calculations complete:")
    print(f"  SMA_50 available: {data['SMA_50'].notna().sum()} days")
    print(f"  SMA_200 available: {data['SMA_200'].notna().sum()} days")
    print()

    # Step 3: Generate trading signals
    print("🎯 Step 3: Generating Trading Signals...")

    # Initialize signal column
    data['Signal'] = 0

    # Golden Cross: SMA_50 crosses above SMA_200 (Buy)
    buy_condition = (data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))
    data.loc[buy_condition, 'Signal'] = 1

    # Death Cross: SMA_50 crosses below SMA_200 (Sell)
    sell_condition = (data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))
    data.loc[sell_condition, 'Signal'] = -1

    # Count signals
    buy_signals = (data['Signal'] == 1).sum()
    sell_signals = (data['Signal'] == -1).sum()

    print(f"Trading signals generated:")
    print(f"  🟢 Buy signals: {buy_signals}")
    print(f"  🔴 Sell signals: {sell_signals}")
    print()

    # Step 4: Implement backtest
    print("💰 Step 4: Running Backtest...")

    def backtest_golden_cross(data, initial_capital=10000):
        """Simple backtest for Golden Cross strategy"""
        capital = initial_capital
        position = 0  # Shares/units held
        trades = []
        portfolio_values = []

        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            signal = data['Signal'].iloc[i]

            # Execute trades based on signals
            if signal == 1 and position == 0:  # Buy signal
                position = capital / current_price
                capital = 0
                trades.append({
                    'type': 'BUY',
                    'date': data.index[i],
                    'price': current_price,
                    'position': position
                })
                print(f"BUY: {position:.2f} units at ${current_price:.5f}")

            elif signal == -1 and position > 0:  # Sell signal
                capital = position * current_price
                position = 0
                trades.append({
                    'type': 'SELL',
                    'date': data.index[i],
                    'price': current_price,
                    'capital': capital
                })
                print(f"SELL: ${capital:.2f} realized at ${current_price:.5f}")

            # Calculate current portfolio value
            current_value = capital + (position * current_price)
            portfolio_values.append(current_value)

        return portfolio_values, trades

    # Run backtest
    portfolio_values, trades = backtest_golden_cross(data)

    # Calculate performance metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - 10000) / 10000 * 100

    # Calculate buy-and-hold return for comparison
    buy_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100

    print("
📊 Backtest Results:")
    print(".2f")
    print(".2f")
    print(".2f")
    print(f"Number of trades: {len(trades)}")
    print()

    # Step 5: Visualization
    print("📊 Step 5: Creating Performance Charts...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Price chart with moving averages and signals
    axes[0,0].plot(data.index, data['Close'], label='EUR/USD', alpha=0.7)
    axes[0,0].plot(data.index, data['SMA_50'], label='SMA 50', color='orange')
    axes[0,0].plot(data.index, data['SMA_200'], label='SMA 200', color='red')

    # Mark buy/sell signals
    buy_dates = data[data['Signal'] == 1].index
    sell_dates = data[data['Signal'] == -1].index
    buy_prices = data.loc[buy_dates, 'Close']
    sell_prices = data.loc[sell_dates, 'Close']

    axes[0,0].scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy Signal')
    axes[0,0].scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell Signal')

    axes[0,0].set_title('EUR/USD with Golden Cross Signals')
    axes[0,0].set_ylabel('Price')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)

    # 2. Portfolio value over time
    axes[0,1].plot(data.index, portfolio_values, label='Strategy', color='blue', linewidth=2)

    # Add buy-and-hold comparison
    buy_hold_values = 10000 * (data['Close'] / data['Close'].iloc[0])
    axes[0,1].plot(data.index, buy_hold_values, label='Buy & Hold', color='gray', alpha=0.7)

    axes[0,1].set_title('Portfolio Value Comparison')
    axes[0,1].set_ylabel('Portfolio Value ($)')
    axes[0,1].legend()
    axes[0,1].tick_params(axis='x', rotation=45)

    # 3. Drawdown analysis
    peak = pd.Series(portfolio_values).expanding().max()
    drawdown = (pd.Series(portfolio_values) - peak) / peak * 100
    max_drawdown = drawdown.min()

    axes[1,0].fill_between(data.index, drawdown, 0, color='red', alpha=0.3)
    axes[1,0].set_title('.1f')
    axes[1,0].set_ylabel('Drawdown (%)')
    axes[1,0].tick_params(axis='x', rotation=45)

    # 4. Trade returns histogram
    if trades:
        trade_returns = []
        buy_trades = [t for t in trades if t['type'] == 'BUY']

        for buy_trade in buy_trades:
            buy_date = buy_trade['date']
            buy_price = buy_trade['price']

            # Find corresponding sell trade
            sell_trades = [t for t in trades if t['type'] == 'SELL' and t['date'] > buy_date]
            if sell_trades:
                sell_price = sell_trades[0]['price']
                trade_return = (sell_price - buy_price) / buy_price * 100
                trade_returns.append(trade_return)

        if trade_returns:
            axes[1,1].hist(trade_returns, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[1,1].set_title('Trade Returns Distribution')
            axes[1,1].set_xlabel('Return (%)')
            axes[1,1].set_ylabel('Frequency')
        else:
            axes[1,1].text(0.5, 0.5, 'No completed trades\nin backtest period',
                          transform=axes[1,1].transAxes, ha='center', va='center',
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.tight_layout()
    plt.savefig('golden_cross_backtest.png', dpi=300, bbox_inches='tight')
    print("✅ Charts saved as 'golden_cross_backtest.png'")
    print()

    # Final summary
    print("🎉 Golden Cross Backtest Complete!")
    print("=" * 50)
    print("Strategy Performance Summary:")
    print(".2f")
    print(".2f")
    print(".1f")
    print(f"Total trades executed: {len(trades)}")
    print(f"Max drawdown: {max_drawdown:.1f}%")

    if total_return > buy_hold_return:
        print("✅ Strategy outperformed buy-and-hold!")
    else:
        print("❌ Strategy underperformed buy-and-hold")

    return data, portfolio_values, trades


def run_all_demos():
    """Run both video demonstrations"""
    print("🚀 AlphaTwin Phase 3: Video Demo Suite")
    print("=" * 60)
    print()

    try:
        # Demo 1: Data Acquisition
        print("VIDEO 1: Data Acquisition & Exploration")
        print("-" * 40)
        demo_data = demo_1_data_acquisition()
        print()

        # Demo 2: Backtesting
        print("VIDEO 2: Golden Cross Backtest")
        print("-" * 40)
        backtest_data, portfolio_values, trades = demo_2_golden_cross_backtest()
        print()

        print("✅ All demonstrations completed successfully!")
        print("📊 Generated charts: data_exploration_charts.png, golden_cross_backtest.png")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_demos()
