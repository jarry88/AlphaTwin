"""
Signal Generation Module for AlphaTwin

Contains various trading signal generation strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class SignalGenerator(ABC):
    """Abstract base class for signal generators."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from market data."""
        pass


class MovingAverageCrossover(SignalGenerator):
    """Moving Average Crossover strategy."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on moving average crossover.

        Returns:
            Series with 1 (buy), -1 (sell), 0 (hold)
        """
        # Calculate moving averages
        short_ma = data['Adj Close'].rolling(window=self.short_window).mean()
        long_ma = data['Adj Close'].rolling(window=self.long_window).mean()

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Buy when short MA crosses above long MA
        signals[short_ma > long_ma] = 1

        # Sell when short MA crosses below long MA
        signals[short_ma < long_ma] = -1

        return signals


class MeanReversion(SignalGenerator):
    """Mean Reversion strategy using Bollinger Bands."""

    def __init__(self, window: int = 20, std_dev: float = 2.0):
        self.window = window
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on Bollinger Bands mean reversion.

        Returns:
            Series with 1 (buy), -1 (sell), 0 (hold)
        """
        # Calculate Bollinger Bands
        sma = data['Adj Close'].rolling(window=self.window).mean()
        std = data['Adj Close'].rolling(window=self.window).std()

        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Buy when price touches lower band (oversold)
        signals[data['Adj Close'] <= lower_band] = 1

        # Sell when price touches upper band (overbought)
        signals[data['Adj Close'] >= upper_band] = -1

        return signals


class Momentum(SignalGenerator):
    """Momentum-based strategy."""

    def __init__(self, lookback_period: int = 20, threshold: float = 0.05):
        self.lookback_period = lookback_period
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on price momentum.

        Returns:
            Series with 1 (buy), -1 (sell), 0 (hold)
        """
        # Calculate momentum (rate of change)
        momentum = (data['Adj Close'] - data['Adj Close'].shift(self.lookback_period)) / data['Adj Close'].shift(self.lookback_period)

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Buy on strong upward momentum
        signals[momentum > self.threshold] = 1

        # Sell on strong downward momentum
        signals[momentum < -self.threshold] = -1

        return signals


class RSI(SignalGenerator):
    """Relative Strength Index (RSI) strategy."""

    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator."""
        delta = data['Adj Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on RSI levels.

        Returns:
            Series with 1 (buy), -1 (sell), 0 (hold)
        """
        rsi = self.calculate_rsi(data)

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Buy when RSI indicates oversold
        signals[rsi < self.oversold] = 1

        # Sell when RSI indicates overbought
        signals[rsi > self.overbought] = -1

        return signals


class SignalManager:
    """Manager class to combine multiple signal generators."""

    def __init__(self):
        self.generators = {}

    def add_generator(self, name: str, generator: SignalGenerator):
        """Add a signal generator."""
        self.generators[name] = generator

    def generate_all_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate signals from all registered generators."""
        signals = {}
        for name, generator in self.generators.items():
            signals[name] = generator.generate_signals(data)
        return signals

    def combine_signals(self, signals: Dict[str, pd.Series], method: str = 'majority') -> pd.Series:
        """
        Combine signals from multiple generators.

        Args:
            signals: Dictionary of signal series
            method: Combination method ('majority', 'weighted', 'consensus')

        Returns:
            Combined signal series
        """
        if not signals:
            return pd.Series()

        signal_df = pd.DataFrame(signals)

        if method == 'majority':
            # Simple majority vote
            combined = signal_df.sum(axis=1)
            combined = combined.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        elif method == 'consensus':
            # All signals must agree
            combined = signal_df.sum(axis=1)
            max_signals = len(signals)
            combined = combined.apply(lambda x: 1 if x == max_signals else (-1 if x == -max_signals else 0))

        else:
            # Default to majority
            combined = signal_df.sum(axis=1)
            combined = combined.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        return combined


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data_loader import DataLoader

    # Load sample data
    loader = DataLoader()
    try:
        data = loader.load_processed_data("AAPL")
    except FileNotFoundError:
        print("No processed data found. Please run data_loader.py first.")
        sys.exit(1)

    # Initialize signal manager
    manager = SignalManager()

    # Add different strategies
    manager.add_generator("MA_Crossover", MovingAverageCrossover())
    manager.add_generator("Mean_Reversion", MeanReversion())
    manager.add_generator("Momentum", Momentum())
    manager.add_generator("RSI", RSI())

    # Generate signals
    all_signals = manager.generate_all_signals(data)
    combined_signals = manager.combine_signals(all_signals)

    print("Generated signals for different strategies:")
    for name, signals in all_signals.items():
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        print(f"{name}: {buy_signals} buy, {sell_signals} sell signals")

    combined_buy = (combined_signals == 1).sum()
    combined_sell = (combined_signals == -1).sum()
    print(f"Combined signals: {combined_buy} buy, {combined_sell} sell signals")
