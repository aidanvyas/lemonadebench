"""Pydantic models for OpenAI tool parameters.

These models serve as the single source of truth for tool parameters across
all provider implementations, enabling provider-agnostic tool parameter
definitions with provider-specific schema generation.
"""

from pydantic import BaseModel, Field


class OrderSuppliesParams(BaseModel):
    """Parameters for the order_supplies tool."""

    cups: int = Field(ge=0, description="Number of cups to order")
    lemons: int = Field(ge=0, description="Number of lemons to order")
    sugar: int = Field(ge=0, description="Amount of sugar to order")
    water: int = Field(ge=0, description="Amount of water to order")


class SetOperatingHoursParams(BaseModel):
    """Parameters for the set_operating_hours tool."""

    open_hour: int = Field(ge=0, le=23, description="Opening hour (0-23)")
    close_hour: int = Field(
        ge=1,
        le=24,
        description="Closing hour (1-24, must be > open_hour)",
    )


class SetPriceParams(BaseModel):
    """Parameters for the set_price tool."""

    price: float = Field(ge=0, description="Price per lemonade")


class CheckMorningPricesParams(BaseModel):
    """Parameters for the check_morning_prices tool (no parameters)."""


class CheckInventoryParams(BaseModel):
    """Parameters for the check_inventory tool (no parameters)."""


class GetHistoricalSupplyCostsParams(BaseModel):
    """Parameters for the get_historical_supply_costs tool (no parameters)."""


class OpenForBusinessParams(BaseModel):
    """Parameters for the open_for_business tool (no parameters)."""


class PurchaseAutomationParams(BaseModel):
    """Parameters for the purchase_automation tool (no parameters)."""


class PurchaseAdvertisingParams(BaseModel):
    """Parameters for the purchase_advertising tool."""

    spend: int = Field(
        gt=0,
        description="Dollars to spend on advertising today (positive integer)",
    )


class TakeLoanParams(BaseModel):
    """Parameters for the take_loan tool."""

    amount: int = Field(
        gt=0,
        description="Dollars to borrow (positive integer)",
    )


class RepayLoanParams(BaseModel):
    """Parameters for the repay_loan tool."""

    amount: int = Field(
        gt=0,
        description="Dollars to repay from cash (positive integer)",
    )
