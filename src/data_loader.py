"""
Data Loader Module for AlphaTwin

Handles data collection, processing, and storage for quantitative trading.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path


class DataLoader:
    """Main data loader class for market data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Download historical stock data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            return data
        except Exception as e:
            raise Exception(f"Failed to download data for {symbol}: {str(e)}")

    def save_raw_data(self, data: pd.DataFrame, symbol: str, filename: str = None) -> str:
        """Save raw data to CSV file."""
        if filename is None:
            filename = f"{symbol}_raw.csv"

        filepath = self.raw_dir / filename
        data.to_csv(filepath)
        return str(filepath)

    def load_raw_data(self, symbol: str, filename: str = None) -> pd.DataFrame:
        """Load raw data from CSV file."""
        if filename is None:
            filename = f"{symbol}_raw.csv"

        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")

        return pd.read_csv(filepath, index_col=0, parse_dates=True)

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data: clean, calculate returns, add technical indicators.

        Args:
            data: Raw OHLCV DataFrame

        Returns:
            Processed DataFrame with additional features
        """
        # Remove rows with missing values
        processed = data.dropna()

        # Calculate returns
        processed['returns'] = processed['Adj Close'].pct_change()

        # Calculate simple moving averages
        processed['SMA_20'] = processed['Adj Close'].rolling(window=20).mean()
        processed['SMA_50'] = processed['Adj Close'].rolling(window=50).mean()

        # Calculate volatility (20-day rolling std of returns)
        processed['volatility'] = processed['returns'].rolling(window=20).std()

        return processed.dropna()

    def save_processed_data(self, data: pd.DataFrame, symbol: str, filename: str = None) -> str:
        """Save processed data to CSV file."""
        if filename is None:
            filename = f"{symbol}_processed.csv"

        filepath = self.processed_dir / filename
        data.to_csv(filepath)
        return str(filepath)

    def load_processed_data(self, symbol: str, filename: str = None) -> pd.DataFrame:
        """Load processed data from CSV file."""
        if filename is None:
            filename = f"{symbol}_processed.csv"

        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Processed data file not found: {filepath}")

        return pd.read_csv(filepath, index_col=0, parse_dates=True)


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()

    # Download and process AAPL data
    symbol = "AAPL"
    raw_data = loader.download_stock_data(symbol)
    print(f"Downloaded {len(raw_data)} rows of raw data for {symbol}")

    processed_data = loader.process_data(raw_data)
    print(f"Processed data shape: {processed_data.shape}")

    # Save data
    loader.save_raw_data(raw_data, symbol)
    loader.save_processed_data(processed_data, symbol)

    print("Data saved successfully!")
