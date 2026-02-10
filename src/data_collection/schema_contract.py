"""
Schema Contract Validator

Enforces schema expectations on philippdubach options data to prevent
silent breakage if the upstream dataset evolves. This is a critical
defensive layer for production-grade data ingestion.

Key validations:
- Required columns present
- Correct dtypes
- Uniqueness constraints
- Schema drift detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Expected schema for philippdubach dataset
PHILIPPDUBACH_SCHEMA = {
    "contract_id": "object",  # string
    "symbol": "object",  # string
    "expiration": "datetime64[ns]",
    "strike": "float64",
    "type": "object",  # string: "call" or "put"
    "last": "float64",
    "mark": "float64",
    "bid": "float64",
    "ask": "float64",
    "bid_size": "int64",
    "ask_size": "int64",
    "volume": "int64",
    "open_interest": "int64",
    "date": "datetime64[ns]",
    "implied_volatility": "float64",
    "delta": "float64",
    "gamma": "float64",
    "theta": "float64",
    "vega": "float64",
    "rho": "float64",
    "in_the_money": "bool",
}

# Expected schema for Training.py compatibility
TRAINING_SCHEMA = {
    "price": "float64",  # underlying close
    "option_price": "float64",  # mark or mid
    "strike_distance": "float64",  # strike - price (dollars)
    "time_to_expiry": "int64",  # days to expiration
    "volume": "int64",
    "implied_volatility": "float64",
    "vix_level": "float64",
    "spy_return_1d": "float64",  # RENAMED from spy_return_5min
    "rsi": "float64",
    "timestamp": "datetime64[ns]",
}

# Critical fields that must not have nulls
CRITICAL_FIELDS = [
    "contract_id", "symbol", "expiration", "strike", "type", "date",
    "volume", "open_interest", "bid", "ask", "implied_volatility"
]


def validate_philippdubach_schema(
    df: pd.DataFrame,
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame matches expected philippdubach schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, fail on any schema violation; if False, only warn
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required columns present
    missing_cols = set(PHILIPPDUBACH_SCHEMA.keys()) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check dtypes for present columns
    for col, expected_dtype in PHILIPPDUBACH_SCHEMA.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            # Handle datetime conversions
            if "datetime" in expected_dtype and "datetime" not in actual_dtype:
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
            # Handle numeric types (allow int32/int64 flexibility)
            elif expected_dtype == "int64" and actual_dtype not in ["int32", "int64"]:
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
            elif expected_dtype == "float64" and actual_dtype not in ["float32", "float64"]:
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
            elif expected_dtype == "object" and actual_dtype != "object":
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
            elif expected_dtype == "bool" and actual_dtype != "bool":
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
    
    # Check for nulls in critical fields
    for field in CRITICAL_FIELDS:
        if field in df.columns:
            null_count = df[field].isna().sum()
            if null_count > 0:
                errors.append(f"Critical field '{field}' has {null_count} null values")
    
    is_valid = len(errors) == 0
    
    if errors:
        for error in errors:
            if strict:
                logger.error(f"Schema validation error: {error}")
            else:
                logger.warning(f"Schema validation warning: {error}")
    else:
        logger.info("✓ philippdubach schema validation passed")
    
    return is_valid, errors


def validate_training_schema(
    df: pd.DataFrame,
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame matches expected Training.py schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, fail on any schema violation; if False, only warn
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required columns present
    missing_cols = set(TRAINING_SCHEMA.keys()) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required Training.py columns: {missing_cols}")
    
    # Check dtypes
    for col, expected_dtype in TRAINING_SCHEMA.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if "datetime" in expected_dtype and "datetime" not in actual_dtype:
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
            elif expected_dtype == "int64" and actual_dtype not in ["int32", "int64"]:
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
            elif expected_dtype == "float64" and actual_dtype not in ["float32", "float64"]:
                errors.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
    
    # Check for nulls in all fields (Training.py expects clean data)
    for col in TRAINING_SCHEMA.keys():
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                errors.append(f"Training field '{col}' has {null_count} null values")
    
    is_valid = len(errors) == 0
    
    if errors:
        for error in errors:
            if strict:
                logger.error(f"Training schema validation error: {error}")
            else:
                logger.warning(f"Training schema validation warning: {error}")
    else:
        logger.info("✓ Training.py schema validation passed")
    
    return is_valid, errors


def assert_uniqueness(
    df: pd.DataFrame,
    keys: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Assert that specified columns form a unique key.
    
    Args:
        df: DataFrame to validate
        keys: List of columns that should be unique together
        
    Returns:
        (is_unique, error_message)
    """
    # Check all keys present
    missing_keys = set(keys) - set(df.columns)
    if missing_keys:
        error_msg = f"Cannot check uniqueness: missing columns {missing_keys}"
        logger.error(error_msg)
        return False, error_msg
    
    # Check for duplicates
    duplicates = df.duplicated(subset=keys, keep=False)
    duplicate_count = duplicates.sum()
    
    if duplicate_count > 0:
        error_msg = f"Found {duplicate_count} duplicate rows on keys {keys}"
        logger.error(error_msg)
        # Show sample of duplicates
        sample_dupes = df[duplicates].head(5)[keys]
        logger.error(f"Sample duplicates:\n{sample_dupes}")
        return False, error_msg
    
    logger.info(f"✓ Uniqueness check passed for keys {keys}")
    return True, None


def detect_schema_drift(
    df: pd.DataFrame,
    expected_schema: Dict[str, str]
) -> List[str]:
    """
    Detect columns that exist in DataFrame but not in expected schema.
    This helps identify when upstream dataset adds new fields.
    
    Args:
        df: DataFrame to check
        expected_schema: Dict of expected {column: dtype}
        
    Returns:
        List of unexpected columns (potential schema drift)
    """
    unexpected_cols = set(df.columns) - set(expected_schema.keys())
    
    if unexpected_cols:
        logger.warning(f"⚠ Schema drift detected: unexpected columns {unexpected_cols}")
        logger.warning("This may indicate upstream dataset has been updated")
    else:
        logger.info("✓ No schema drift detected")
    
    return list(unexpected_cols)


def validate_option_type(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate that 'type' column contains only 'call' or 'put'.
    
    Args:
        df: DataFrame with 'type' column
        
    Returns:
        (is_valid, error_message)
    """
    if 'type' not in df.columns:
        return False, "Missing 'type' column"
    
    valid_types = {'call', 'put'}
    actual_types = set(df['type'].dropna().unique())
    invalid_types = actual_types - valid_types
    
    if invalid_types:
        error_msg = f"Invalid option types found: {invalid_types}. Expected: {valid_types}"
        logger.error(error_msg)
        return False, error_msg
    
    logger.info(f"✓ Option type validation passed: {actual_types}")
    return True, None


def validate_numeric_ranges(df: pd.DataFrame) -> List[str]:
    """
    Sanity check numeric fields are in reasonable ranges.
    
    Returns:
        List of warnings (not hard errors)
    """
    warnings = []
    
    # Strike prices should be positive
    if 'strike' in df.columns:
        if (df['strike'] <= 0).any():
            warnings.append("Found non-positive strike prices")
    
    # Volume and OI should be non-negative
    if 'volume' in df.columns:
        if (df['volume'] < 0).any():
            warnings.append("Found negative volume")
    
    if 'open_interest' in df.columns:
        if (df['open_interest'] < 0).any():
            warnings.append("Found negative open_interest")
    
    # Implied volatility should be reasonable (0-10, or 0-1000%)
    if 'implied_volatility' in df.columns:
        iv_max = df['implied_volatility'].max()
        if iv_max > 10:  # Assume IV > 1000% is suspicious
            warnings.append(f"Suspiciously high IV found: {iv_max}")
    
    # Bid should be <= Ask
    if 'bid' in df.columns and 'ask' in df.columns:
        crossed_market = (df['bid'] > df['ask']).sum()
        if crossed_market > 0:
            warnings.append(f"Found {crossed_market} crossed markets (bid > ask)")
    
    if warnings:
        for warning in warnings:
            logger.warning(f"⚠ Numeric range issue: {warning}")
    
    return warnings


def full_validation_report(
    df: pd.DataFrame,
    schema_type: str = "philippdubach",
    check_uniqueness_keys: Optional[List[str]] = None
) -> Dict:
    """
    Run all validations and return comprehensive report.
    
    Args:
        df: DataFrame to validate
        schema_type: "philippdubach" or "training"
        check_uniqueness_keys: Keys to check for uniqueness (e.g., ["contract_id", "date"])
        
    Returns:
        Dict with validation results and metadata
    """
    logger.info(f"=== Full Validation Report ({schema_type}) ===")
    logger.info(f"DataFrame shape: {df.shape}")
    
    report = {
        "shape": df.shape,
        "schema_valid": False,
        "uniqueness_valid": True,
        "type_valid": True,
        "errors": [],
        "warnings": [],
    }
    
    # Schema validation
    if schema_type == "philippdubach":
        is_valid, errors = validate_philippdubach_schema(df, strict=False)
        report["schema_valid"] = is_valid
        report["errors"].extend(errors)
        
        # Check for schema drift
        drift = detect_schema_drift(df, PHILIPPDUBACH_SCHEMA)
        if drift:
            report["warnings"].append(f"Schema drift: {drift}")
    
    elif schema_type == "training":
        is_valid, errors = validate_training_schema(df, strict=False)
        report["schema_valid"] = is_valid
        report["errors"].extend(errors)
    
    # Uniqueness check
    if check_uniqueness_keys:
        is_unique, error_msg = assert_uniqueness(df, check_uniqueness_keys)
        report["uniqueness_valid"] = is_unique
        if error_msg:
            report["errors"].append(error_msg)
    
    # Option type validation (for philippdubach)
    if schema_type == "philippdubach":
        is_valid, error_msg = validate_option_type(df)
        report["type_valid"] = is_valid
        if error_msg:
            report["errors"].append(error_msg)
    
    # Numeric range checks
    warnings = validate_numeric_ranges(df)
    report["warnings"].extend(warnings)
    
    # Summary
    report["overall_valid"] = (
        report["schema_valid"] and 
        report["uniqueness_valid"] and 
        report["type_valid"] and
        len(report["errors"]) == 0
    )
    
    if report["overall_valid"]:
        logger.info("✓✓✓ All validations passed ✓✓✓")
    else:
        logger.error(f"❌ Validation failed with {len(report['errors'])} errors")
    
    return report
