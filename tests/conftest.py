"""
Shared pytest fixtures for OptionsTitan tests.
Project root is on pythonpath via pyproject.toml [tool.pytest.ini_options].
"""

import pytest


@pytest.fixture
def sample_stock_data():
    """Minimal stock_data dict required by generate_strategies."""
    return {
        "symbol": "SPY",
        "current_price": 450.0,
        "volatility": 0.18,
        "trend": "Bullish",
        "high_52w": 460.0,
        "low_52w": 420.0,
        "price_history": None,
        "returns": None,
    }


@pytest.fixture
def sample_strategies():
    """Minimal strategy dicts matching shape used by UI/workers."""
    return [
        {
            "name": "Covered Call",
            "fit_score": 85,
            "reasoning": ["Item 1", "Item 2"],
            "risk_assessment": {"status": "OK", "message": "Within risk parameters"},
            "type": "Income Strategy",
            "description": "Buy 100 shares and sell 1 call",
            "setup": ["Step 1", "Step 2"],
            "capital_required": 45000.0,
            "max_profit": "$500",
            "max_loss": "$1000",
            "risk_level": "Low-Medium",
            "ideal_market": "Neutral to Slightly Bullish",
        },
        {
            "name": "Cash-Secured Put",
            "fit_score": 80,
            "reasoning": ["Item A"],
            "risk_assessment": {"status": "OK", "message": "Acceptable"},
            "type": "Income Strategy",
            "description": "Sell put with cash reserved",
            "setup": ["Step 1"],
            "capital_required": 45000.0,
            "max_profit": "$200",
            "max_loss": "$45000",
            "risk_level": "Medium",
            "ideal_market": "Bullish",
        },
    ]
