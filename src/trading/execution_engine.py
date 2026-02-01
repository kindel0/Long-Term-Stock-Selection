"""
Order execution engine with approval workflow.

Handles the semi-automated execution flow:
1. Generate orders
2. Review and display to user
3. Wait for approval
4. Execute orders
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable

import pandas as pd

from .order_generator import ProposedOrder
from .position_manager import PositionManager
from .fee_calculator import FeeCalculator
from ..config import TRADES_DIR

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for the engine."""
    PAPER = "paper"
    LIVE = "live"
    DRY_RUN = "dry_run"


@dataclass
class ExecutedOrder:
    """Record of an executed order."""
    symbol: str
    action: str
    quantity: int
    price: float
    fill_price: float
    fee: float
    status: str  # 'filled', 'partial', 'rejected', 'cancelled'
    order_id: str
    executed_at: str
    notes: str = ""


class ExecutionEngine:
    """
    Manages order execution with approval workflow.

    Supports three modes:
    - dry_run: Show orders but don't execute
    - paper: Execute in IBKR paper trading account
    - live: Execute in live account (requires extra confirmation)

    Example:
        engine = ExecutionEngine(mode=ExecutionMode.PAPER)
        engine.set_orders(proposed_orders)
        engine.review_orders()

        if engine.get_approval():
            results = engine.execute()
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.DRY_RUN,
        ibkr_client=None,  # Type hint omitted to avoid import
        position_manager: Optional[PositionManager] = None,
        trades_dir: Optional[Path] = None,
    ):
        """
        Initialize the execution engine.

        Args:
            mode: Execution mode (dry_run, paper, live)
            ibkr_client: Optional IBKR client for real execution
            position_manager: Optional position manager
            trades_dir: Directory for trade logs
        """
        self.mode = mode
        self.ibkr = ibkr_client
        self.positions = position_manager or PositionManager()
        self.trades_dir = Path(trades_dir) if trades_dir else TRADES_DIR
        self.trades_dir.mkdir(parents=True, exist_ok=True)

        self.pending_orders: List[ProposedOrder] = []
        self.executed_orders: List[ExecutedOrder] = []
        self.approval_callback: Optional[Callable[[], bool]] = None

    def set_orders(self, orders: List[ProposedOrder]) -> None:
        """
        Set orders pending execution.

        Args:
            orders: List of proposed orders
        """
        self.pending_orders = orders
        self.executed_orders = []
        logger.info(f"Set {len(orders)} orders for execution")

    def review_orders(self) -> None:
        """Display orders for user review."""
        if not self.pending_orders:
            print("No orders to review")
            return

        print("\n" + "=" * 80)
        print(f"ORDER REVIEW - Mode: {self.mode.value.upper()}")
        print("=" * 80)

        # Summary stats
        n_sells = sum(1 for o in self.pending_orders if o.action == "SELL")
        n_buys = sum(1 for o in self.pending_orders if o.action == "BUY")
        total_sell = sum(o.estimated_value for o in self.pending_orders if o.action == "SELL")
        total_buy = sum(o.estimated_value for o in self.pending_orders if o.action == "BUY")
        total_fees = sum(o.estimated_fee for o in self.pending_orders)

        print(f"\nTotal Orders: {len(self.pending_orders)}")
        print(f"  Sells: {n_sells} (${total_sell:,.2f})")
        print(f"  Buys:  {n_buys} (${total_buy:,.2f})")
        print(f"  Est. Fees: ${total_fees:.2f}")
        print(f"  Net Cash Impact: ${total_sell - total_buy - total_fees:,.2f}")

        # Detailed orders
        if n_sells > 0:
            print("\n--- SELL ORDERS ---")
            print(f"{'Symbol':<8} {'Qty':>8} {'Price':>10} {'Value':>12} {'Fee':>8} Reason")
            print("-" * 70)
            for o in self.pending_orders:
                if o.action == "SELL":
                    print(
                        f"{o.symbol:<8} {o.quantity:>8} "
                        f"${o.estimated_price:>9.2f} ${o.estimated_value:>11.2f} "
                        f"${o.estimated_fee:>7.2f} {o.reason}"
                    )

        if n_buys > 0:
            print("\n--- BUY ORDERS ---")
            print(f"{'Symbol':<8} {'Qty':>8} {'Price':>10} {'Value':>12} {'Fee':>8} Reason")
            print("-" * 70)
            for o in self.pending_orders:
                if o.action == "BUY":
                    print(
                        f"{o.symbol:<8} {o.quantity:>8} "
                        f"${o.estimated_price:>9.2f} ${o.estimated_value:>11.2f} "
                        f"${o.estimated_fee:>7.2f} {o.reason}"
                    )

        print("\n" + "=" * 80)

        if self.mode == ExecutionMode.LIVE:
            print("\n*** WARNING: LIVE TRADING MODE ***")
            print("Orders will be executed in your LIVE account!")

    def get_approval(self, auto_approve: bool = False) -> bool:
        """
        Get user approval for pending orders.

        Args:
            auto_approve: Skip confirmation prompt

        Returns:
            True if approved, False otherwise
        """
        if not self.pending_orders:
            return False

        if auto_approve or self.mode == ExecutionMode.DRY_RUN:
            return True

        if self.approval_callback:
            return self.approval_callback()

        # Interactive approval
        print("\nApprove execution? [y/N]: ", end="")
        try:
            response = input().strip().lower()
            return response in ("y", "yes")
        except EOFError:
            return False

    def execute(self, dry_run: bool = False) -> List[ExecutedOrder]:
        """
        Execute pending orders.

        Args:
            dry_run: Override mode to dry run

        Returns:
            List of executed orders
        """
        if not self.pending_orders:
            logger.warning("No orders to execute")
            return []

        effective_mode = ExecutionMode.DRY_RUN if dry_run else self.mode

        if effective_mode == ExecutionMode.DRY_RUN:
            return self._simulate_execution()
        elif effective_mode == ExecutionMode.PAPER:
            return self._execute_paper()
        else:
            return self._execute_live()

    def _simulate_execution(self) -> List[ExecutedOrder]:
        """Simulate order execution for dry run."""
        logger.info("Simulating order execution (dry run)")

        for order in self.pending_orders:
            executed = ExecutedOrder(
                symbol=order.symbol,
                action=order.action,
                quantity=order.quantity,
                price=order.estimated_price,
                fill_price=order.estimated_price,
                fee=order.estimated_fee,
                status="filled",
                order_id=f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{order.symbol}",
                executed_at=datetime.now().isoformat(),
                notes="Simulated execution",
            )
            self.executed_orders.append(executed)

            # Update position manager
            if order.action == "BUY":
                self.positions.add_position(
                    order.symbol, order.quantity, order.estimated_price
                )
            else:
                self.positions.reduce_position(
                    order.symbol, order.quantity, order.estimated_price
                )

        self._save_execution_log()
        return self.executed_orders

    def _execute_paper(self) -> List[ExecutedOrder]:
        """Execute orders in paper trading mode."""
        if self.ibkr is None:
            logger.warning("No IBKR client, falling back to simulation")
            return self._simulate_execution()

        logger.info("Executing orders in PAPER trading mode via IBKR")
        return self._execute_via_ibkr()

    def _execute_live(self) -> List[ExecutedOrder]:
        """Execute orders in live trading mode."""
        if self.ibkr is None:
            raise ValueError("IBKR client required for live trading")

        logger.info("Executing orders in LIVE mode via IBKR")
        return self._execute_via_ibkr()

    def _execute_via_ibkr(self) -> List[ExecutedOrder]:
        """Execute orders through IBKR API (synchronous)."""
        results = []

        # Execute sells first to free up cash
        sells = [o for o in self.pending_orders if o.action == "SELL"]
        buys = [o for o in self.pending_orders if o.action == "BUY"]

        for order in sells + buys:
            try:
                logger.info(f"Placing {order.action} order: {order.quantity} {order.symbol}")

                order_id = self.ibkr.place_market_order(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    action=order.action,
                )

                if order_id:
                    # Wait for fill
                    fill_price = self.ibkr.wait_for_fill(order_id, timeout=30)

                    if fill_price:
                        executed = ExecutedOrder(
                            symbol=order.symbol,
                            action=order.action,
                            quantity=order.quantity,
                            price=order.estimated_price,
                            fill_price=fill_price,
                            fee=order.estimated_fee,
                            status="filled",
                            order_id=order_id,
                            executed_at=datetime.now().isoformat(),
                            notes=f"IBKR {self.mode.value} execution",
                        )
                        logger.info(f"  Filled: {order.symbol} @ ${fill_price:.2f}")
                    else:
                        # Order submitted but not filled yet
                        executed = ExecutedOrder(
                            symbol=order.symbol,
                            action=order.action,
                            quantity=order.quantity,
                            price=order.estimated_price,
                            fill_price=order.estimated_price,  # Use estimate
                            fee=order.estimated_fee,
                            status="submitted",
                            order_id=order_id,
                            executed_at=datetime.now().isoformat(),
                            notes="Order submitted, fill pending",
                        )
                        logger.info(f"  Submitted: {order.symbol} (fill pending)")

                    results.append(executed)

                    # Update position manager
                    if order.action == "BUY":
                        self.positions.add_position(
                            order.symbol, order.quantity, fill_price or order.estimated_price
                        )
                    else:
                        self.positions.reduce_position(
                            order.symbol, order.quantity, fill_price or order.estimated_price
                        )
                else:
                    executed = ExecutedOrder(
                        symbol=order.symbol,
                        action=order.action,
                        quantity=order.quantity,
                        price=order.estimated_price,
                        fill_price=0,
                        fee=0,
                        status="rejected",
                        order_id="",
                        executed_at=datetime.now().isoformat(),
                        notes="Order placement failed",
                    )
                    results.append(executed)
                    logger.error(f"  Failed: {order.symbol}")

            except Exception as e:
                logger.error(f"Error executing {order.symbol}: {e}")
                executed = ExecutedOrder(
                    symbol=order.symbol,
                    action=order.action,
                    quantity=order.quantity,
                    price=order.estimated_price,
                    fill_price=0,
                    fee=0,
                    status="rejected",
                    order_id="",
                    executed_at=datetime.now().isoformat(),
                    notes=str(e),
                )
                results.append(executed)

        self.executed_orders = results
        self._save_execution_log()
        return self.executed_orders

    def _save_execution_log(self) -> Path:
        """Save execution log to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.trades_dir / f"execution_{timestamp}.json"

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode.value,
            "orders": [
                {
                    "symbol": o.symbol,
                    "action": o.action,
                    "quantity": o.quantity,
                    "price": o.price,
                    "fill_price": o.fill_price,
                    "fee": o.fee,
                    "status": o.status,
                    "order_id": o.order_id,
                    "executed_at": o.executed_at,
                }
                for o in self.executed_orders
            ],
            "summary": {
                "total_orders": len(self.executed_orders),
                "total_fees": sum(o.fee for o in self.executed_orders),
                "total_value": sum(o.quantity * o.fill_price for o in self.executed_orders),
            },
        }

        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved execution log: {filepath}")
        return filepath

    def get_execution_summary(self) -> Dict:
        """Get summary of executed orders."""
        if not self.executed_orders:
            return {"status": "no_executions"}

        filled = [o for o in self.executed_orders if o.status == "filled"]
        rejected = [o for o in self.executed_orders if o.status == "rejected"]

        return {
            "total_orders": len(self.executed_orders),
            "filled": len(filled),
            "rejected": len(rejected),
            "total_value": sum(o.quantity * o.fill_price for o in filled),
            "total_fees": sum(o.fee for o in filled),
            "buy_value": sum(
                o.quantity * o.fill_price for o in filled if o.action == "BUY"
            ),
            "sell_value": sum(
                o.quantity * o.fill_price for o in filled if o.action == "SELL"
            ),
        }

    def print_execution_report(self) -> None:
        """Print execution report to console."""
        summary = self.get_execution_summary()

        print("\n" + "=" * 60)
        print("EXECUTION REPORT")
        print("=" * 60)

        if summary.get("status") == "no_executions":
            print("No orders executed")
            return

        print(f"Total Orders: {summary['total_orders']}")
        print(f"  Filled:   {summary['filled']}")
        print(f"  Rejected: {summary['rejected']}")
        print(f"\nTotal Value: ${summary['total_value']:,.2f}")
        print(f"  Buys:  ${summary['buy_value']:,.2f}")
        print(f"  Sells: ${summary['sell_value']:,.2f}")
        print(f"  Fees:  ${summary['total_fees']:.2f}")

        print("\n" + "-" * 60)
        print("EXECUTED ORDERS:")
        print(f"{'Symbol':<8} {'Action':<6} {'Qty':>8} {'Fill':>10} {'Fee':>8} Status")
        print("-" * 60)

        for o in self.executed_orders:
            print(
                f"{o.symbol:<8} {o.action:<6} {o.quantity:>8} "
                f"${o.fill_price:>9.2f} ${o.fee:>7.2f} {o.status}"
            )

        print("=" * 60)
