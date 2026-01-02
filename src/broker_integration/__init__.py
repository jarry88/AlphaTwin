"""
Broker Integration Module for AlphaTwin

Provides unified interfaces for connecting to multiple brokers
and executing live trades.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BrokerInterface(ABC):
    """
    Abstract base class for broker integrations.

    Defines the standard interface that all broker implementations must follow.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker API"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to broker API"""
        pass

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get account balance and margin information"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass

    @abstractmethod
    def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data for a symbol"""
        pass

    @abstractmethod
    def place_order(self, symbol: str, quantity: int, order_type: str,
                   price: Optional[float] = None, **kwargs) -> str:
        """Place a trade order"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of a specific order"""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        pass


class OrderStatus:
    """Standard order status constants"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType:
    """Standard order type constants"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class BrokerFactory:
    """
    Factory class for creating broker instances.

    Supports multiple broker implementations with unified interface.
    """

    @staticmethod
    def create_broker(broker_name: str, **credentials) -> BrokerInterface:
        """
        Create a broker instance by name.

        Args:
            broker_name: Name of the broker ('alpaca', 'interactive_brokers', 'paper')
            **credentials: Broker-specific authentication credentials

        Returns:
            BrokerInterface instance
        """
        if broker_name.lower() == 'alpaca':
            from .alpaca_broker import AlpacaBroker
            return AlpacaBroker(**credentials)
        elif broker_name.lower() == 'interactive_brokers':
            from .ibkr_broker import InteractiveBrokers
            return InteractiveBrokers(**credentials)
        elif broker_name.lower() == 'paper':
            from .paper_broker import PaperBroker
            return PaperBroker(**credentials)
        else:
            raise ValueError(f"Unsupported broker: {broker_name}")


# Import concrete implementations
try:
    from .alpaca_broker import AlpacaBroker
    from .paper_broker import PaperBroker
    # from .ibkr_broker import InteractiveBrokers  # Future implementation
except ImportError:
    logger.warning("Some broker implementations not available. Install required packages.")

__all__ = [
    'BrokerInterface',
    'BrokerFactory',
    'OrderStatus',
    'OrderType',
    'AlpacaBroker',
    'PaperBroker'
]
