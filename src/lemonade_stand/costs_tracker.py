"""Track actual costs using OpenAI's Costs API."""

import logging
import os
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

class CostsTracker:
    """Track OpenAI API costs using the official Costs API."""

    def __init__(self, api_key: str | None = None):
        """Initialize the costs tracker.

        Note: The Costs API requires an admin API key with appropriate permissions.
        """
        self.api_key = api_key or os.getenv("OPENAI_ADMIN_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_ADMIN_KEY or OPENAI_API_KEY.")

        self.base_url = "https://api.openai.com/v1/organization"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_costs(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        bucket_width: str = "1d",
        group_by: list[str] | None = None,
        project_ids: list[str] | None = None,
        limit: int = 100
    ) -> dict:
        """Get costs from the OpenAI Costs API.

        Args:
            start_time: Start of time range (defaults to 30 days ago)
            end_time: End of time range (defaults to now)
            bucket_width: Time bucket size ("1m", "1h", or "1d")
            group_by: Fields to group by (e.g., ["model", "project_id"])
            project_ids: Filter by specific project IDs
            limit: Maximum number of buckets to return

        Returns:
            Dict containing cost data with buckets of time-based costs
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        if end_time is None:
            end_time = datetime.now()

        params = {
            "start_time": int(start_time.timestamp()),
            "end_time": int(end_time.timestamp()),
            "bucket_width": bucket_width,
            "limit": limit
        }

        if group_by:
            params["group_by"] = group_by
        if project_ids:
            params["project_ids"] = project_ids

        try:
            response = requests.get(
                f"{self.base_url}/costs",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching costs: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return {"error": str(e), "data": []}

    def get_usage(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        bucket_width: str = "1d",
        group_by: list[str] | None = None,
        models: list[str] | None = None,
        limit: int = 100
    ) -> dict:
        """Get token usage from the OpenAI Usage API.

        Args:
            start_time: Start of time range
            end_time: End of time range
            bucket_width: Time bucket size
            group_by: Fields to group by
            models: Filter by specific models
            limit: Maximum number of buckets

        Returns:
            Dict containing usage data with input/output token counts
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        if end_time is None:
            end_time = datetime.now()

        params = {
            "start_time": int(start_time.timestamp()),
            "end_time": int(end_time.timestamp()),
            "bucket_width": bucket_width,
            "limit": limit
        }

        if group_by:
            params["group_by"] = group_by
        if models:
            params["models"] = models

        try:
            response = requests.get(
                f"{self.base_url}/usage/completions",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching usage: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return {"error": str(e), "data": []}

    def get_recent_costs(self, hours: int = 1) -> float:
        """Get total costs for the recent time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Total cost in USD
        """
        start_time = datetime.now() - timedelta(hours=hours)
        data = self.get_costs(
            start_time=start_time,
            bucket_width="1h" if hours <= 24 else "1d"
        )

        total_cost = 0.0
        for bucket in data.get("data", []):
            for result in bucket.get("results", []):
                amount = result.get("amount", {})
                if amount.get("currency") == "usd":
                    total_cost += amount.get("value", 0)

        return total_cost

    def print_cost_summary(self, days: int = 7):
        """Print a formatted cost summary for recent days.

        Args:
            days: Number of days to summarize
        """
        start_time = datetime.now() - timedelta(days=days)

        # Get costs grouped by model
        cost_data = self.get_costs(
            start_time=start_time,
            bucket_width="1d",
            group_by=["model"]
        )

        print(f"\nOpenAI API Costs (Last {days} days)")
        print("=" * 60)

        if "error" in cost_data:
            print(f"Error: {cost_data['error']}")
            return

        daily_totals = {}
        model_totals = {}

        for bucket in cost_data.get("data", []):
            date = datetime.fromtimestamp(bucket["start_time"]).strftime("%Y-%m-%d")

            for result in bucket.get("results", []):
                amount = result.get("amount", {})
                if amount.get("currency") == "usd":
                    cost = amount.get("value", 0)
                    model = result.get("model", "unknown")

                    # Accumulate daily totals
                    if date not in daily_totals:
                        daily_totals[date] = 0
                    daily_totals[date] += cost

                    # Accumulate model totals
                    if model not in model_totals:
                        model_totals[model] = 0
                    model_totals[model] += cost

        # Print daily breakdown
        print("\nDaily Costs:")
        for date in sorted(daily_totals.keys()):
            print(f"  {date}: ${daily_totals[date]:.4f}")

        # Print model breakdown
        print("\nCosts by Model:")
        for model in sorted(model_totals.keys()):
            print(f"  {model}: ${model_totals[model]:.4f}")

        # Print total
        total = sum(daily_totals.values())
        print(f"\nTotal: ${total:.4f}")
        print("=" * 60)
