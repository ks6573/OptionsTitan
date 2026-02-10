"""
Corporate Actions Validator

Validates that stock splits are properly handled in the options data.
Checks strike grid continuity, moneyness consistency, and price scaling
around known split events.

Critical for ensuring strike_distance calculations are correct.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Known stock splits (date, ratio, description)
KNOWN_SPLITS = {
    "aapl": [
        ("2020-08-31", 4.0, "4-for-1 split"),
        ("2014-06-09", 7.0, "7-for-1 split"),  # Pre-dataset but good to know
    ],
    "tsla": [
        ("2022-08-25", 3.0, "3-for-1 split"),
        ("2020-08-31", 5.0, "5-for-1 split"),
    ],
    "nvda": [
        ("2024-06-10", 10.0, "10-for-1 split"),
        ("2021-07-20", 4.0, "4-for-1 split"),
    ],
    "googl": [
        ("2022-07-18", 20.0, "20-for-1 split"),
    ],
    "goog": [
        ("2022-07-18", 20.0, "20-for-1 split"),
    ],
    "amzn": [
        ("2022-06-06", 20.0, "20-for-1 split"),
    ],
    "shop": [
        ("2022-06-29", 10.0, "10-for-1 split"),
    ],
}


def get_splits_for_ticker(ticker: str) -> List[Tuple[str, float, str]]:
    """
    Get known splits for a ticker.
    
    Args:
        ticker: Stock symbol (case-insensitive)
        
    Returns:
        List of (date, ratio, description) tuples
    """
    ticker = ticker.lower()
    return KNOWN_SPLITS.get(ticker, [])


def validate_split_continuity(
    df: pd.DataFrame,
    ticker: str,
    underlying_df: Optional[pd.DataFrame] = None
) -> Tuple[bool, List[Dict]]:
    """
    Validate that strike grids and option prices remain continuous around split dates.
    
    Strategy:
    1. For each known split date, examine data 30 days before and after
    2. Check that moneyness (strike / underlying_price) doesn't jump discontinuously
    3. Check that strike grids scale appropriately with split ratio
    4. Flag anomalies for manual review
    
    Args:
        df: Options DataFrame with columns: date, strike, expiration, bid, ask, etc.
        ticker: Stock symbol
        underlying_df: Optional underlying prices DataFrame
        
    Returns:
        (is_valid, list_of_anomalies)
    """
    ticker = ticker.lower()
    splits = get_splits_for_ticker(ticker)
    
    if not splits:
        logger.info(f"✓ No known splits for {ticker}")
        return True, []
    
    logger.info(f"Validating {len(splits)} known splits for {ticker}")
    
    anomalies = []
    
    for split_date_str, split_ratio, description in splits:
        split_date = pd.to_datetime(split_date_str)
        
        logger.info(f"Checking {description} on {split_date_str}")
        
        # Define pre/post windows (30 days each)
        pre_start = split_date - timedelta(days=30)
        pre_end = split_date - timedelta(days=1)
        post_start = split_date
        post_end = split_date + timedelta(days=30)
        
        # Filter to pre/post data
        df['date'] = pd.to_datetime(df['date'])
        pre_split = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)]
        post_split = df[(df['date'] >= post_start) & (df['date'] <= post_end)]
        
        if len(pre_split) == 0 or len(post_split) == 0:
            logger.warning(f"⚠ Insufficient data around {split_date_str} for validation")
            anomalies.append({
                "split_date": split_date_str,
                "description": description,
                "issue": "Insufficient data",
                "pre_rows": len(pre_split),
                "post_rows": len(post_split),
            })
            continue
        
        # Check 1: Strike grid scaling
        pre_strikes = pre_split['strike'].unique()
        post_strikes = post_split['strike'].unique()
        
        # Expected: post_strikes ≈ pre_strikes / split_ratio
        # Sample a few strikes to check
        sample_pre = np.percentile(pre_strikes, [25, 50, 75])
        expected_post = sample_pre / split_ratio
        
        # Find closest actual post-split strikes
        closest_post = []
        for expected in expected_post:
            if len(post_strikes) > 0:
                closest = post_strikes[np.argmin(np.abs(post_strikes - expected))]
                closest_post.append(closest)
        
        if len(closest_post) > 0:
            ratio_check = np.array(sample_pre) / np.array(closest_post)
            mean_ratio = ratio_check.mean()
            ratio_error = abs(mean_ratio - split_ratio) / split_ratio
            
            if ratio_error > 0.05:  # More than 5% error
                anomalies.append({
                    "split_date": split_date_str,
                    "description": description,
                    "issue": "Strike grid scaling mismatch",
                    "expected_ratio": split_ratio,
                    "actual_ratio": mean_ratio,
                    "error_pct": ratio_error * 100,
                })
                logger.warning(f"⚠ Strike scaling issue: expected {split_ratio}x, got {mean_ratio:.2f}x")
            else:
                logger.info(f"✓ Strike grid scales correctly ({mean_ratio:.2f}x)")
        
        # Check 2: Moneyness continuity (if underlying prices available)
        if underlying_df is not None:
            underlying_df['date'] = pd.to_datetime(underlying_df['date'])
            
            # Get underlying prices around split
            pre_price = underlying_df[
                (underlying_df['date'] >= pre_start) & 
                (underlying_df['date'] <= pre_end)
            ]['close'].median()
            
            post_price = underlying_df[
                (underlying_df['date'] >= post_start) & 
                (underlying_df['date'] <= post_end)
            ]['close'].median()
            
            if pre_price > 0 and post_price > 0:
                price_ratio = pre_price / post_price
                price_error = abs(price_ratio - split_ratio) / split_ratio
                
                if price_error > 0.05:  # More than 5% error
                    anomalies.append({
                        "split_date": split_date_str,
                        "description": description,
                        "issue": "Underlying price scaling mismatch",
                        "expected_ratio": split_ratio,
                        "actual_ratio": price_ratio,
                        "error_pct": price_error * 100,
                        "pre_price": pre_price,
                        "post_price": post_price,
                    })
                    logger.warning(f"⚠ Price scaling issue: expected {split_ratio}x, got {price_ratio:.2f}x")
                else:
                    logger.info(f"✓ Underlying price scales correctly ({price_ratio:.2f}x)")
    
    is_valid = len(anomalies) == 0
    
    if is_valid:
        logger.info(f"✓✓✓ All {len(splits)} splits validated for {ticker} ✓✓✓")
    else:
        logger.error(f"❌ Found {len(anomalies)} anomalies in {ticker} splits")
    
    return is_valid, anomalies


def detect_split_anomalies(
    df: pd.DataFrame,
    window_days: int = 60,
    threshold: float = 1.5
) -> List[Dict]:
    """
    Detect potential undocumented splits or data issues by looking for
    sudden jumps in strike distributions.
    
    Args:
        df: Options DataFrame
        window_days: Rolling window for detecting jumps
        threshold: Ratio threshold to flag as anomaly (e.g., 1.5 = 50% jump)
        
    Returns:
        List of detected anomalies with dates and metrics
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate median strike per day
    daily_median_strike = df.groupby('date')['strike'].median()
    
    # Calculate rolling ratio (today / yesterday)
    strike_ratio = daily_median_strike / daily_median_strike.shift(1)
    
    # Flag large jumps
    anomalies = []
    for date, ratio in strike_ratio.items():
        if ratio > threshold or ratio < (1 / threshold):
            anomalies.append({
                "date": date.strftime("%Y-%m-%d"),
                "median_strike_ratio": ratio,
                "issue": f"Large strike jump detected ({ratio:.2f}x)",
                "suggestion": "Possible undocumented split or data quality issue",
            })
            logger.warning(f"⚠ Strike jump on {date.strftime('%Y-%m-%d')}: {ratio:.2f}x")
    
    if not anomalies:
        logger.info("✓ No undocumented split anomalies detected")
    else:
        logger.warning(f"⚠ Found {len(anomalies)} potential anomalies")
    
    return anomalies


def log_split_validation_report(
    ticker: str,
    is_valid: bool,
    anomalies: List[Dict],
    save_path: Optional[str] = None
) -> str:
    """
    Generate human-readable split validation report.
    
    Args:
        ticker: Stock symbol
        is_valid: Overall validation result
        anomalies: List of detected anomalies
        save_path: Optional path to save report as text file
        
    Returns:
        Report as string
    """
    report_lines = [
        "=" * 70,
        f"Corporate Action Validation Report: {ticker.upper()}",
        "=" * 70,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    splits = get_splits_for_ticker(ticker)
    if splits:
        report_lines.append(f"Known splits: {len(splits)}")
        for split_date, ratio, desc in splits:
            report_lines.append(f"  - {split_date}: {desc} ({ratio:.1f}:1)")
    else:
        report_lines.append("No known splits for this ticker")
    
    report_lines.append("")
    report_lines.append(f"Validation result: {'✓ PASSED' if is_valid else '❌ FAILED'}")
    report_lines.append(f"Anomalies found: {len(anomalies)}")
    report_lines.append("")
    
    if anomalies:
        report_lines.append("Detected Anomalies:")
        report_lines.append("-" * 70)
        for i, anomaly in enumerate(anomalies, 1):
            report_lines.append(f"\n{i}. {anomaly.get('description', 'Anomaly')}")
            for key, value in anomaly.items():
                if key != 'description':
                    report_lines.append(f"   {key}: {value}")
    else:
        report_lines.append("✓ No anomalies detected - splits handled correctly")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    
    report = "\n".join(report_lines)
    
    # Log to console
    logger.info(f"\n{report}")
    
    # Optionally save to file
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {save_path}")
    
    return report


def validate_moneyness_distribution(
    df: pd.DataFrame,
    underlying_df: pd.DataFrame,
    ticker: str
) -> Dict:
    """
    Validate that moneyness (strike / spot) distribution looks reasonable
    and doesn't have discontinuities suggesting split issues.
    
    Args:
        df: Options DataFrame
        underlying_df: Underlying prices DataFrame
        ticker: Stock symbol
        
    Returns:
        Dict with validation metrics
    """
    # Merge options with underlying prices
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    underlying_df['date'] = pd.to_datetime(underlying_df['date'])
    
    merged = df.merge(
        underlying_df[['date', 'close']],
        on='date',
        how='left'
    )
    
    # Calculate moneyness
    merged['moneyness'] = merged['strike'] / merged['close']
    
    # Remove outliers (moneyness should typically be 0.5 to 2.0)
    valid_moneyness = merged[
        (merged['moneyness'] >= 0.5) & 
        (merged['moneyness'] <= 2.0)
    ]['moneyness']
    
    # Calculate statistics
    stats = {
        "ticker": ticker,
        "total_rows": len(merged),
        "valid_moneyness_rows": len(valid_moneyness),
        "invalid_pct": (1 - len(valid_moneyness) / len(merged)) * 100 if len(merged) > 0 else 0,
        "moneyness_mean": valid_moneyness.mean(),
        "moneyness_std": valid_moneyness.std(),
        "moneyness_min": valid_moneyness.min(),
        "moneyness_max": valid_moneyness.max(),
    }
    
    # Flag if too many invalid moneyness values
    if stats["invalid_pct"] > 5.0:
        logger.warning(f"⚠ {ticker}: {stats['invalid_pct']:.1f}% of rows have invalid moneyness")
        stats["warning"] = "High percentage of invalid moneyness values"
    else:
        logger.info(f"✓ {ticker}: Moneyness distribution looks reasonable")
        stats["warning"] = None
    
    return stats


def full_corporate_action_validation(
    options_df: pd.DataFrame,
    underlying_df: pd.DataFrame,
    ticker: str,
    save_report: bool = True
) -> Dict:
    """
    Run full corporate action validation suite.
    
    Args:
        options_df: Options DataFrame
        underlying_df: Underlying prices DataFrame
        ticker: Stock symbol
        save_report: Whether to save validation report
        
    Returns:
        Dict with validation results and metrics
    """
    logger.info(f"=== Full Corporate Action Validation: {ticker} ===")
    
    results = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 1. Known splits validation
    splits_valid, anomalies = validate_split_continuity(
        options_df, ticker, underlying_df
    )
    results["known_splits_valid"] = splits_valid
    results["known_splits_anomalies"] = anomalies
    
    # 2. Detect undocumented anomalies
    detected_anomalies = detect_split_anomalies(options_df)
    results["detected_anomalies"] = detected_anomalies
    
    # 3. Moneyness distribution check
    moneyness_stats = validate_moneyness_distribution(
        options_df, underlying_df, ticker
    )
    results["moneyness_stats"] = moneyness_stats
    
    # 4. Generate report
    all_anomalies = anomalies + detected_anomalies
    overall_valid = splits_valid and len(detected_anomalies) == 0
    
    if save_report:
        report_path = f"data/validation/corporate_actions_{ticker}.txt"
        log_split_validation_report(ticker, overall_valid, all_anomalies, report_path)
    else:
        log_split_validation_report(ticker, overall_valid, all_anomalies)
    
    results["overall_valid"] = overall_valid
    
    return results
