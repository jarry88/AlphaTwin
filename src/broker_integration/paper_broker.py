"""
Paper Trading Broker for AlphaTwin

Simulates live trading with paper money for testing strategies
without financial risk. Uses real market data but virtual positions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import uuid
from . import BrokerInterface, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class PaperBroker(BrokerInterface):
    """
    Paper trading broker implementation.

    Simulates real trading with virtual money and positions.
    Uses real market data for realistic simulation.
    """

    def __init__(self,
                 initial_balance: float = 100000.0,
                 commission_per_trade: float = 0.0,
                 slippage_model: str = 'fixed',
                 slippage_bps: float = 0.5):
        """
        Initialize paper broker.

        Args:
            initial_balance: Starting account balance
            commission_per_trade: Commission per trade (0 for free)
            slippage_model: Slippage model ('fixed', 'percentage', 'none')
            slippage_bps: Slippage in basis points for fixed model
        """
        self.initial_balance = initial_balance
        self.commission_per_trade = commission_per_trade
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps

        # Account state
        self.cash = initial_balance
        self.portfolio_value = initial_balance
        self.equity = initial_balance
        self.buying_power = initial_balance

        # Positions and orders
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []

        # Market data cache (for slippage calculations)
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}

        self.connected = False

        logger.info(f"Paper broker initialized with ${initial_balance:,.2f} balance")

    def connect(self) -> bool:
        """Establish paper trading connection."""
        self.connected = True
        logger.info("Paper trading session started")
        return True

    def disconnect(self) -> bool:
        """End paper trading session."""
        self.connected = False
        logger.info("Paper trading session ended")
        return True

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        # Update portfolio value based on current positions
        self._update_portfolio_value()

        return {
            'account_id': 'paper_trading',
            'account_type': 'paper',
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'equity': self.equity,
            'buying_power': self.buying_power,
            'daytrade_count': 0,  # Not applicable for paper trading
            'status': 'ACTIVE',
            'currency': 'USD',
            'last_updated': pd.Timestamp.now(tz='UTC')
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        positions = []
        for symbol, pos_data in self.positions.items():
            current_price = self._get_current_price(symbol)
            market_value = pos_data['quantity'] * current_price
            unrealized_pl = market_value - (pos_data['quantity'] * pos_data['avg_price'])
            unrealized_plpc = unrealized_pl / (pos_data['quantity'] * pos_data['avg_price'])

            positions.append({
                'symbol': symbol,
                'quantity': pos_data['quantity'],
                'avg_price': pos_data['avg_price'],
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pl': unrealized_pl,
                'unrealized_plpc': unrealized_plpc,
                'side': 'long' if pos_data['quantity'] > 0 else 'short'
            })

        return positions

    def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """Get simulated live market data."""
        # Generate realistic market data simulation
        base_price = self._get_base_price(symbol)

        # Add some realistic volatility
        volatility = 0.02  # 2% daily volatility
        price_change = np.random.normal(0, volatility * base_price / 16)  # Hourly change

        current_price = base_price + price_change
        spread = current_price * 0.0001  # 1 basis point spread

        return {
            'symbol': symbol,
            'bid_price': current_price - spread/2,
            'ask_price': current_price + spread/2,
            'bid_size': np.random.randint(100, 1000),
            'ask_size': np.random.randint(100, 1000),
            'last_price': current_price,
            'last_size': np.random.randint(10, 100),
            'timestamp': pd.Timestamp.now(tz='UTC')
        }

    def place_order(self,
                   symbol: str,
                   quantity: int,
                   order_type: str,
                   price: Optional[float] = None,
                   **kwargs) -> str:
        """
        Place a simulated trade order.

        For paper trading, orders execute immediately with simulated slippage.
        """
        order_id = str(uuid.uuid4())

        # Get execution price based on order type
        execution_price = self._calculate_execution_price(symbol, order_type, price)

        # Apply commission
        commission = self.commission_per_trade

        # Create order record
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'status': OrderStatus.FILLED,  # Paper orders execute immediately
            'order_type': order_type,
            'side': 'buy' if quantity > 0 else 'sell',
            'quantity': abs(quantity),
            'execution_price': execution_price,
            'commission': commission,
            'submitted_at': pd.Timestamp.now(tz='UTC'),
            'filled_at': pd.Timestamp.now(tz='UTC'),
            'filled_quantity': abs(quantity),
            'avg_fill_price': execution_price
        }

        self.orders[order_id] = order
        self.order_history.append(order)

        # Update positions and cash
        self._update_position(symbol, quantity, execution_price, commission)

        logger.info(f"Paper order executed: {order_type} {abs(quantity)} {symbol} @ ${execution_price:.2f}")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order (not applicable for immediate execution)."""
        if order_id in self.orders:
            if self.orders[order_id]['status'] == OrderStatus.PENDING:
                self.orders[order_id]['status'] = OrderStatus.CANCELLED
                logger.info(f"Cancelled paper order {order_id}")
                return True
        return False

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of a paper order."""
        if order_id in self.orders:
            order = self.orders[order_id].copy()
            # Convert timestamps for consistency
            for key in ['submitted_at', 'filled_at']:
                if key in order:
                    order[key] = pd.Timestamp(order[key])
            return order
        else:
            raise ValueError(f"Order {order_id} not found")

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open paper orders (none for immediate execution)."""
        return [order for order in self.orders.values()
                if order['status'] == OrderStatus.PENDING]

    def _calculate_execution_price(self,
                                  symbol: str,
                                  order_type: str,
                                  limit_price: Optional[float] = None) -> float:
        """Calculate execution price with simulated slippage."""
        base_price = self._get_current_price(symbol)

        if order_type == OrderType.MARKET:
            # Market orders execute at current price with slippage
            slippage = self._calculate_slippage(base_price)
            return base_price + slippage

        elif order_type == OrderType.LIMIT and limit_price is not None:
            # For paper trading, limit orders execute if within reasonable range
            if abs(limit_price - base_price) / base_price < 0.05:  # Within 5%
                return limit_price
            else:
                # Cancel order if too far from market
                raise ValueError("Limit price too far from market price")

        else:
            return base_price

    def _calculate_slippage(self, base_price: float) -> float:
        """Calculate simulated slippage."""
        if self.slippage_model == 'none':
            return 0.0
        elif self.slippage_model == 'fixed':
            # Fixed basis points slippage
            slippage_amount = base_price * (self.slippage_bps / 10000)
            # Random direction
            return slippage_amount * (1 if np.random.random() > 0.5 else -1)
        elif self.slippage_model == 'percentage':
            # Percentage-based slippage
            slippage_pct = np.random.normal(0, 0.001)  # 0.1% std deviation
            return base_price * slippage_pct
        else:
            return 0.0

    def _update_position(self, symbol: str, quantity: int, price: float, commission: float):
        """Update position and cash balances."""
        trade_value = abs(quantity) * price

        if quantity > 0:  # Buy
            if self.cash >= trade_value + commission:
                self.cash -= (trade_value + commission)

                if symbol in self.positions:
                    # Average cost calculation
                    current_qty = self.positions[symbol]['quantity']
                    current_cost = current_qty * self.positions[symbol]['avg_price']
                    new_cost = quantity * price
                    total_qty = current_qty + quantity
                    avg_price = (current_cost + new_cost) / total_qty

                    self.positions[symbol]['quantity'] = total_qty
                    self.positions[symbol]['avg_price'] = avg_price
                else:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': price
                    }
        else:  # Sell
            if symbol in self.positions and self.positions[symbol]['quantity'] >= abs(quantity):
                self.cash += (trade_value - commission)

                self.positions[symbol]['quantity'] += quantity  # quantity is negative

                # Remove position if fully closed
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]

        # Update portfolio value
        self._update_portfolio_value()

    def _update_portfolio_value(self):
        """Update portfolio value based on current positions."""
        portfolio_value = self.cash

        for symbol, pos_data in self.positions.items():
            current_price = self._get_current_price(symbol)
            position_value = pos_data['quantity'] * current_price
            portfolio_value += position_value

        self.portfolio_value = portfolio_value
        self.equity = portfolio_value
        self.buying_power = self.cash  # Simplified

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        # Use cached market data or generate realistic price
        if symbol in self.market_data_cache:
            return self.market_data_cache[symbol]['last_price']
        else:
            return self._get_base_price(symbol)

    def _get_base_price(self, symbol: str) -> float:
        """Get base price for a symbol (simplified)."""
        # Simple price mapping for common symbols
        base_prices = {
            'AAPL': 180.0,
            'MSFT': 380.0,
            'GOOGL': 140.0,
            'AMZN': 155.0,
            'TSLA': 250.0,
            'EURUSD': 1.0850,
            'GBPUSD': 1.2750,
            'USDJPY': 145.50,
            'SPY': 480.0,
            'QQQ': 410.0
        }

        return base_prices.get(symbol, 100.0)

    def reset(self):
        """Reset paper trading account to initial state."""
        self.cash = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.equity = self.initial_balance
        self.buying_power = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.order_history.clear()
        self.market_data_cache.clear()

        logger.info("Paper trading account reset")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for paper trading session."""
        total_trades = len(self.order_history)
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0

        for order in self.order_history:
            if order['side'] == 'sell':  # Completed trades
                entry_price = None
                # Find corresponding buy order (simplified logic)
                # In a real implementation, you'd track round-trip trades

                # For now, just calculate based on commission
                total_pnl -= order['commission']

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pnl': total_pnl,
            'current_balance': self.cash,
            'portfolio_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance
        }
