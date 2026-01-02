"""
Alpaca Broker Integration for AlphaTwin

Provides live trading capabilities through Alpaca's commission-free API.
Supports both paper trading (for testing) and live trading.
"""

import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from . import BrokerInterface, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class AlpacaBroker(BrokerInterface):
    """
    Alpaca broker implementation for live trading.

    Supports both paper trading and live trading modes.
    """

    def __init__(self,
                 api_key: str,
                 api_secret: str,
                 base_url: str = 'https://paper-api.alpaca.markets',
                 api_version: str = 'v2'):
        """
        Initialize Alpaca broker connection.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: API base URL (paper for testing, live for real trading)
            api_version: API version to use
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.api_version = api_version

        self.api = None
        self.connected = False

        # Connection settings
        self.max_retries = 3
        self.retry_delay = 1.0

    def connect(self) -> bool:
        """
        Establish connection to Alpaca API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version=self.api_version
            )

            # Test connection by getting account info
            account = self.api.get_account()
            if account.status == 'ACTIVE':
                self.connected = True
                logger.info(f"Successfully connected to Alpaca ({'PAPER' if 'paper' in self.base_url else 'LIVE'})")
                return True
            else:
                logger.error(f"Alpaca account not active. Status: {account.status}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self.connected = False
            return False

    def disconnect(self) -> bool:
        """
        Close connection to Alpaca API.

        Returns:
            True (Alpaca REST API doesn't require explicit disconnection)
        """
        self.connected = False
        self.api = None
        logger.info("Disconnected from Alpaca")
        return True

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account balance and margin information.

        Returns:
            Dictionary with account details
        """
        if not self.connected or self.api is None:
            raise ConnectionError("Not connected to Alpaca")

        try:
            account = self.api.get_account()

            return {
                'account_id': account.id,
                'account_type': 'paper' if 'paper' in self.base_url else 'live',
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'daytrade_count': int(account.daytrade_count),
                'status': account.status,
                'currency': account.currency,
                'last_updated': pd.Timestamp.now(tz='UTC')
            }

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.

        Returns:
            List of position dictionaries
        """
        if not self.connected or self.api is None:
            raise ConnectionError("Not connected to Alpaca")

        try:
            positions = self.api.list_positions()

            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'avg_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side
                })

            return result

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise

    def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time market data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'AAPL')

        Returns:
            Dictionary with live market data
        """
        if not self.connected or self.api is None:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Get latest quote
            quote = self.api.get_latest_quote(symbol)

            # Get latest trade
            trade = self.api.get_latest_trade(symbol)

            # Get current minute bar (if available)
            try:
                bars = self.api.get_barset(symbol, 'minute', limit=1)
                current_bar = bars[symbol][0] if bars and symbol in bars else None
            except:
                current_bar = None

            result = {
                'symbol': symbol,
                'bid_price': float(quote.bidprice) if quote.bidprice else None,
                'ask_price': float(quote.askprice) if quote.askprice else None,
                'bid_size': int(quote.bidsize) if quote.bidsize else None,
                'ask_size': int(quote.asksize) if quote.asksize else None,
                'last_price': float(trade.price) if trade.price else None,
                'last_size': int(trade.size) if trade.size else None,
                'timestamp': pd.Timestamp.now(tz='UTC')
            }

            # Add bar data if available
            if current_bar:
                result.update({
                    'open': float(current_bar.o),
                    'high': float(current_bar.h),
                    'low': float(current_bar.l),
                    'close': float(current_bar.c),
                    'volume': int(current_bar.v),
                    'bar_timestamp': pd.Timestamp(current_bar.t, tz='UTC')
                })

            return result

        except Exception as e:
            logger.error(f"Failed to get live data for {symbol}: {e}")
            raise

    def place_order(self,
                   symbol: str,
                   quantity: int,
                   order_type: str,
                   price: Optional[float] = None,
                   **kwargs) -> str:
        """
        Place a trade order.

        Args:
            symbol: Trading symbol
            quantity: Number of shares (positive for buy, negative for sell)
            order_type: Type of order ('market', 'limit', 'stop', 'stop_limit')
            price: Limit price for limit orders
            **kwargs: Additional order parameters

        Returns:
            Order ID string
        """
        if not self.connected or self.api is None:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Determine order side and quantity
            side = 'buy' if quantity > 0 else 'sell'
            qty = abs(quantity)

            # Map order types
            order_params = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'time_in_force': kwargs.get('time_in_force', 'day')
            }

            # Add order-specific parameters
            if order_type == OrderType.LIMIT and price is not None:
                order_params['limit_price'] = price
            elif order_type == OrderType.STOP and price is not None:
                order_params['stop_price'] = price
            elif order_type == OrderType.STOP_LIMIT:
                order_params['limit_price'] = kwargs.get('limit_price', price)
                order_params['stop_price'] = kwargs.get('stop_price')

            # Submit order
            order = self.api.submit_order(**order_params)

            logger.info(f"Placed {order_type} order for {qty} {symbol} {side} - Order ID: {order.id}")
            return order.id

        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        if not self.connected or self.api is None:
            raise ConnectionError("Not connected to Alpaca")

        try:
            self.api.cancel_order(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of a specific order.

        Args:
            order_id: Order ID to check

        Returns:
            Dictionary with order status details
        """
        if not self.connected or self.api is None:
            raise ConnectionError("Not connected to Alpaca")

        try:
            order = self.api.get_order(order_id)

            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'status': self._map_order_status(order.status),
                'order_type': order.type,
                'side': order.side,
                'quantity': int(order.qty),
                'filled_quantity': int(order.filled_qty) if order.filled_qty else 0,
                'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'submitted_at': pd.Timestamp(order.submitted_at) if order.submitted_at else None,
                'updated_at': pd.Timestamp(order.updated_at) if order.updated_at else None
            }

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            raise

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders.

        Returns:
            List of open order dictionaries
        """
        if not self.connected or self.api is None:
            raise ConnectionError("Not connected to Alpaca")

        try:
            orders = self.api.list_orders(status='open')

            result = []
            for order in orders:
                result.append({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'status': self._map_order_status(order.status),
                    'order_type': order.type,
                    'side': order.side,
                    'quantity': int(order.qty),
                    'filled_quantity': int(order.filled_qty) if order.filled_qty else 0,
                    'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'submitted_at': pd.Timestamp(order.submitted_at) if order.submitted_at else None,
                    'updated_at': pd.Timestamp(order.updated_at) if order.updated_at else None
                })

            return result

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise

    def _map_order_status(self, alpaca_status: str) -> str:
        """
        Map Alpaca order status to standardized status.

        Args:
            alpaca_status: Alpaca-specific status

        Returns:
            Standardized status
        """
        status_mapping = {
            'new': OrderStatus.PENDING,
            'partially_filled': OrderStatus.PARTIAL,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.EXPIRED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'accepted': OrderStatus.PENDING,
            'pending_cancel': OrderStatus.PENDING,
            'stopped': OrderStatus.PENDING,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.PENDING,
            'pending_new': OrderStatus.PENDING
        }

        return status_mapping.get(alpaca_status, OrderStatus.PENDING)

    def get_historical_data(self,
                           symbol: str,
                           start_date: str,
                           end_date: str,
                           timeframe: str = '1D') -> pd.DataFrame:
        """
        Get historical market data.

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data frequency ('1D', '1H', '15Min', etc.)

        Returns:
            DataFrame with historical data
        """
        if not self.connected or self.api is None:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Map timeframe to Alpaca format
            timeframe_mapping = {
                '1D': '1D',
                '1H': '1H',
                '15Min': '15Min',
                '5Min': '5Min',
                '1Min': '1Min'
            }

            tf = timeframe_mapping.get(timeframe, '1D')

            # Get data
            bars = self.api.get_barset(symbol, tf, start=start_date, end=end_date)

            if symbol not in bars or len(bars[symbol]) == 0:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for bar in bars[symbol]:
                data.append({
                    'timestamp': pd.Timestamp(bar.t),
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': int(bar.v)
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_convert('UTC')

            return df

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise
