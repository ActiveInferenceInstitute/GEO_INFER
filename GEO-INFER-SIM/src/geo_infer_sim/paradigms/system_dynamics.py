"""
System Dynamics modeling for GEO-INFER-SIM.

This module provides system dynamics modeling using stocks, flows,
and feedback loops for aggregate system behavior.
"""

import logging
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Stock:
    """Represents a stock (accumulation) in system dynamics."""

    name: str
    initial_value: float
    current_value: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def __post_init__(self) -> None:
        """Initialize stock value."""
        self.current_value = self.initial_value


@dataclass
class Flow:
    """Represents a flow (rate of change) in system dynamics."""

    name: str
    source_stock: str | None = None
    target_stock: str | None = None
    rate_function: Callable[[dict[str, float]], float] | None = None
    constant_rate: float | None = None


class SystemDynamicsModel:
    """
    System Dynamics model for geospatial simulations.

    Models systems using stocks, flows, and feedback loops to understand
    aggregate behavior over time.
    """

    def __init__(self) -> None:
        """Initialize the system dynamics model."""
        self.stocks: dict[str, Stock] = {}
        self.flows: list[Flow] = []
        self.time = 0.0
        self.history: list[dict[str, float]] = []

    def add_stock(
        self,
        name: str,
        initial_value: float,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """
        Add a stock to the model.

        Args:
            name: Stock name
            initial_value: Initial stock value
            min_value: Optional minimum value
            max_value: Optional maximum value
        """
        stock = Stock(
            name=name,
            initial_value=initial_value,
            min_value=min_value,
            max_value=max_value,
        )
        self.stocks[name] = stock
        logger.debug(f"Added stock: {name}")

    def add_flow(
        self,
        name: str,
        source_stock: str | None = None,
        target_stock: str | None = None,
        rate_function: Callable[[dict[str, float]], float] | None = None,
        constant_rate: float | None = None,
    ) -> None:
        """
        Add a flow to the model.

        Args:
            name: Flow name
            source_stock: Source stock name (if None, external source)
            target_stock: Target stock name (if None, external sink)
            rate_function: Function to calculate flow rate
            constant_rate: Constant flow rate (if rate_function not provided)
        """
        if rate_function is None and constant_rate is None:
            raise ValueError("Either rate_function or constant_rate must be provided")

        flow = Flow(
            name=name,
            source_stock=source_stock,
            target_stock=target_stock,
            rate_function=rate_function,
            constant_rate=constant_rate,
        )
        self.flows.append(flow)
        logger.debug(f"Added flow: {name}")

    def calculate_flow_rate(self, flow: Flow, stock_values: dict[str, float]) -> float:
        """
        Calculate flow rate for a flow.

        Args:
            flow: Flow object
            stock_values: Current stock values

        Returns:
            Flow rate
        """
        if flow.constant_rate is not None:
            return flow.constant_rate
        elif flow.rate_function:
            return flow.rate_function(stock_values)
        else:
            return 0.0

    def step(self, time_step: float) -> None:
        """
        Execute one simulation step.

        Args:
            time_step: Time step duration
        """
        # Get current stock values
        stock_values = {
            name: stock.current_value for name, stock in self.stocks.items()
        }

        # Calculate flow rates
        flow_rates = {}
        for flow in self.flows:
            rate = self.calculate_flow_rate(flow, stock_values)
            flow_rates[flow.name] = rate

        # Update stocks based on flows
        stock_changes = {name: 0.0 for name in self.stocks.keys()}

        for flow in self.flows:
            rate = flow_rates[flow.name]

            if flow.source_stock:
                stock_changes[flow.source_stock] -= rate * time_step

            if flow.target_stock:
                stock_changes[flow.target_stock] += rate * time_step

        # Apply changes to stocks
        for name, change in stock_changes.items():
            stock = self.stocks[name]
            new_value = stock.current_value + change

            # Apply bounds
            if stock.min_value is not None:
                new_value = max(new_value, stock.min_value)
            if stock.max_value is not None:
                new_value = min(new_value, stock.max_value)

            stock.current_value = new_value

        # Update time
        self.time += time_step

        # Record history
        self.history.append(
            {
                "time": self.time,
                **{name: stock.current_value for name, stock in self.stocks.items()},
            }
        )

    def get_state(self) -> dict[str, Any]:
        """
        Get current model state.

        Returns:
            Model state dictionary
        """
        return {
            "time": self.time,
            "stocks": {
                name: stock.current_value for name, stock in self.stocks.items()
            },
            "history": self.history[-100:] if self.history else [],  # Last 100 steps
        }

    def reset(self) -> None:
        """Reset the model to initial state."""
        for stock in self.stocks.values():
            stock.current_value = stock.initial_value

        self.time = 0.0
        self.history = []
        logger.info("System dynamics model reset")
