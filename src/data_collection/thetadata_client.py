"""
ThetaData REST API Client

Wrapper for ThetaData Terminal's local REST API (http://127.0.0.1:25510).
Handles connection management, rate limiting, error handling, and data parsing.

API Documentation: https://http-docs.thetadata.us/
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import time
import logging
from io import StringIO

from .config import THETADATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThetaDataError(Exception):
    """Base exception for ThetaData client errors"""
    pass


class ThetaDataConnectionError(ThetaDataError):
    """Raised when cannot connect to ThetaData Terminal"""
    pass


class ThetaDataRateLimitError(ThetaDataError):
    """Raised when rate limit is exceeded"""
    pass


class ThetaDataClient:
    """
    REST API client for ThetaData Terminal
    
    Provides methods to:
    - Check connection status
    - List available expirations and strikes
    - Fetch historical EOD option data
    - Fetch historical stock price data
    - Handle rate limiting and retries
    """
    
    def __init__(self, 
                 base_url: str = None,
                 rate_limit_rps: int = None,
                 max_retries: int = None):
        """
        Initialize ThetaData client
        
        Args:
            base_url: ThetaData Terminal URL (default: http://127.0.0.1:25510)
            rate_limit_rps: Requests per second limit
            max_retries: Maximum retry attempts for failed requests
        """
        self.base_url = base_url or THETADATA_CONFIG['base_url']
        self.api_version = THETADATA_CONFIG['api_version']
        self.rate_limit_rps = rate_limit_rps or THETADATA_CONFIG['rate_limit_rps']
        self.max_retries = max_retries or THETADATA_CONFIG['max_retries']
        self.retry_delay = THETADATA_CONFIG['retry_delay']
        self.backoff_factor = THETADATA_CONFIG['backoff_factor']
        self.request_timeout = THETADATA_CONFIG['request_timeout']
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Rate limiting state
        self.last_request_time = 0
        self.request_count = 0
        
        # Verify connection on init
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Terminal is running and accessible"""
        try:
            response = self.session.get(
                f"{self.base_url}/{self.api_version}/list/expirations",
                params={'root': 'SPY'},
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info("✅ Connected to ThetaData Terminal")
            else:
                raise ThetaDataConnectionError(
                    f"Terminal responded with status {response.status_code}"
                )
                
        except requests.exceptions.ConnectionError:
            raise ThetaDataConnectionError(
                "Cannot connect to ThetaData Terminal at {self.base_url}. "
                "Make sure the Terminal application is running."
            )
        except Exception as e:
            raise ThetaDataConnectionError(f"Connection verification failed: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        if self.rate_limit_rps > 0:
            min_interval = 1.0 / self.rate_limit_rps
            elapsed = time.time() - self.last_request_time
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(self, 
                      endpoint: str, 
                      params: Dict,
                      retries: int = 0) -> requests.Response:
        """
        Make HTTP request with retry logic and rate limiting
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            retries: Current retry attempt
            
        Returns:
            Response object
            
        Raises:
            ThetaDataError: On persistent failures
        """
        self._rate_limit()
        
        url = f"{self.base_url}/{self.api_version}/{endpoint}"
        
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.request_timeout
            )
            
            # Check for rate limit errors
            if response.status_code == 429:
                if retries < self.max_retries:
                    wait_time = self.retry_delay * (self.backoff_factor ** retries)
                    logger.warning(f"Rate limited. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    return self._make_request(endpoint, params, retries + 1)
                else:
                    raise ThetaDataRateLimitError("Rate limit exceeded")
            
            # Check for other errors
            if response.status_code >= 400:
                if retries < self.max_retries:
                    wait_time = self.retry_delay * (self.backoff_factor ** retries)
                    logger.warning(
                        f"Request failed (HTTP {response.status_code}). "
                        f"Retrying in {wait_time:.1f}s... (attempt {retries + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    return self._make_request(endpoint, params, retries + 1)
                else:
                    raise ThetaDataError(
                        f"Request failed after {self.max_retries} retries: "
                        f"HTTP {response.status_code}"
                    )
            
            return response
            
        except requests.exceptions.Timeout:
            if retries < self.max_retries:
                logger.warning(f"Request timeout. Retrying...")
                return self._make_request(endpoint, params, retries + 1)
            else:
                raise ThetaDataError("Request timed out after retries")
        
        except requests.exceptions.ConnectionError:
            raise ThetaDataConnectionError(
                "Lost connection to Terminal. Make sure it's still running."
            )
    
    def check_connection(self) -> bool:
        """
        Check if Terminal is accessible
        
        Returns:
            True if connected, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/{self.api_version}/list/expirations",
                params={'root': 'SPY'},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def list_expirations(self, root: str) -> List[str]:
        """
        List all available expiration dates for a ticker
        
        Args:
            root: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            List of expiration dates in YYYYMMDD format
        """
        response = self._make_request(
            'list/expirations',
            params={'root': root}
        )
        
        data = response.json()
        
        # ThetaData returns expirations as list in 'response' field
        if 'response' in data:
            return data['response']
        else:
            return []
    
    def list_strikes(self, root: str, exp: str) -> List[float]:
        """
        List all available strikes for a ticker and expiration
        
        Args:
            root: Stock ticker symbol
            exp: Expiration date (YYYYMMDD format)
            
        Returns:
            List of strike prices
        """
        response = self._make_request(
            'list/strikes',
            params={'root': root, 'exp': exp}
        )
        
        data = response.json()
        
        # ThetaData returns strikes as integers (strike * 1000)
        if 'response' in data:
            strikes = data['response']
            # Convert from integer format (e.g., 150000 -> 150.0)
            return [s / 1000.0 for s in strikes]
        else:
            return []
    
    def fetch_option_eod(self,
                        root: str,
                        exp: str,
                        strike: float,
                        right: str,
                        start_date: str,
                        end_date: str,
                        use_csv: bool = True) -> pd.DataFrame:
        """
        Fetch historical EOD data for a single option contract
        
        Args:
            root: Stock ticker (e.g., 'AAPL')
            exp: Expiration date (YYYYMMDD)
            strike: Strike price (e.g., 150.0)
            right: 'C' for call, 'P' for put
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            use_csv: Return CSV format (faster parsing)
            
        Returns:
            DataFrame with columns:
            - date: Calendar date
            - ms_of_day, ms_of_day2: Milliseconds of day
            - open, high, low, close: OHLC prices
            - volume: Contract volume
            - count: Trade count
            - bid_size, bid_exchange, bid, bid_condition: Bid data
            - ask_size, ask_exchange, ask, ask_condition: Ask data
        """
        # Convert strike to integer format (strike * 1000)
        strike_int = int(strike * 1000)
        
        params = {
            'root': root,
            'exp': exp,
            'strike': strike_int,
            'right': right.upper(),
            'start_date': start_date,
            'end_date': end_date,
        }
        
        if use_csv:
            params['use_csv'] = 'true'
        
        response = self._make_request('hist/option/eod', params)
        
        if use_csv:
            # Parse CSV response
            csv_text = response.text
            
            if not csv_text or csv_text.strip() == '':
                # No data available for this contract
                return pd.DataFrame()
            
            # ThetaData CSV format: no header, comma-separated
            # Order: ms_of_day, ms_of_day2, open, high, low, close, volume, count,
            #        bid_size, bid_exchange, bid, bid_condition,
            #        ask_size, ask_exchange, ask, ask_condition, date
            
            column_names = [
                'ms_of_day', 'ms_of_day2', 'open', 'high', 'low', 'close',
                'volume', 'count', 'bid_size', 'bid_exchange', 'bid',
                'bid_condition', 'ask_size', 'ask_exchange', 'ask',
                'ask_condition', 'date'
            ]
            
            try:
                df = pd.read_csv(
                    StringIO(csv_text),
                    names=column_names,
                    header=None
                )
                
                # Convert date from integer (YYYYMMDD) to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                
                return df
                
            except Exception as e:
                logger.error(f"Failed to parse CSV response: {e}")
                logger.debug(f"CSV content: {csv_text[:500]}")
                return pd.DataFrame()
        
        else:
            # Parse JSON response
            data = response.json()
            
            if 'response' in data:
                return pd.DataFrame(data['response'])
            else:
                return pd.DataFrame()
    
    def fetch_stock_eod(self,
                       root: str,
                       start_date: str,
                       end_date: str,
                       use_csv: bool = True) -> pd.DataFrame:
        """
        Fetch historical EOD stock price data
        
        Args:
            root: Stock ticker
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            use_csv: Return CSV format
            
        Returns:
            DataFrame with columns:
            - date: Calendar date
            - ms_of_day: Milliseconds of day
            - open, high, low, close: OHLC prices
            - volume: Share volume
            - count: Trade count
        """
        params = {
            'root': root,
            'start_date': start_date,
            'end_date': end_date,
        }
        
        if use_csv:
            params['use_csv'] = 'true'
        
        response = self._make_request('hist/stock/eod', params)
        
        if use_csv:
            csv_text = response.text
            
            if not csv_text or csv_text.strip() == '':
                return pd.DataFrame()
            
            # Stock EOD CSV format
            column_names = [
                'ms_of_day', 'open', 'high', 'low', 'close',
                'volume', 'count', 'date'
            ]
            
            try:
                df = pd.read_csv(
                    StringIO(csv_text),
                    names=column_names,
                    header=None
                )
                
                # Convert date
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                
                return df
                
            except Exception as e:
                logger.error(f"Failed to parse stock CSV: {e}")
                return pd.DataFrame()
        
        else:
            data = response.json()
            if 'response' in data:
                return pd.DataFrame(data['response'])
            else:
                return pd.DataFrame()
    
    def get_stats(self) -> Dict:
        """
        Get client statistics
        
        Returns:
            Dictionary with request count, rate, etc.
        """
        return {
            'total_requests': self.request_count,
            'rate_limit_rps': self.rate_limit_rps,
            'last_request_time': self.last_request_time,
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_date_thetadata(date: Union[str, datetime]) -> str:
    """
    Convert date to ThetaData format (YYYYMMDD)
    
    Args:
        date: Date string (YYYY-MM-DD) or datetime object
        
    Returns:
        Date in YYYYMMDD format
    """
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    
    return date.strftime('%Y%m%d')


def find_closest_expiration(expirations: List[str],
                            target_dte: int,
                            reference_date: datetime) -> Optional[str]:
    """
    Find expiration closest to target DTE
    
    Args:
        expirations: List of expiration dates (YYYYMMDD)
        target_dte: Target days to expiration
        reference_date: Reference date for DTE calculation
        
    Returns:
        Closest expiration date (YYYYMMDD), or None if none found
    """
    if not expirations:
        return None
    
    # Convert to datetime
    exp_dates = [datetime.strptime(exp, '%Y%m%d') for exp in expirations]
    
    # Calculate DTE for each
    dtes = [(exp - reference_date).days for exp in exp_dates]
    
    # Find closest to target
    closest_idx = min(range(len(dtes)), key=lambda i: abs(dtes[i] - target_dte))
    
    return expirations[closest_idx]


def find_atm_strike(strikes: List[float], underlying_price: float) -> float:
    """
    Find at-the-money strike closest to underlying price
    
    Args:
        strikes: List of available strikes
        underlying_price: Current underlying price
        
    Returns:
        Closest ATM strike
    """
    if not strikes:
        return underlying_price
    
    return min(strikes, key=lambda s: abs(s - underlying_price))


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    # Test client functionality
    print("Testing ThetaData Client...")
    
    try:
        client = ThetaDataClient()
        print("✅ Client initialized successfully")
        
        # Test listing expirations
        print("\nFetching SPY expirations...")
        expirations = client.list_expirations('SPY')
        print(f"Found {len(expirations)} expirations")
        print(f"Next 5 expirations: {expirations[:5]}")
        
        # Test listing strikes
        if expirations:
            exp = expirations[0]
            print(f"\nFetching strikes for {exp}...")
            strikes = client.list_strikes('SPY', exp)
            print(f"Found {len(strikes)} strikes")
            print(f"Strike range: {min(strikes):.2f} - {max(strikes):.2f}")
        
        # Test fetching option data
        print("\nFetching sample option contract...")
        today = datetime.now()
        start = (today - timedelta(days=30)).strftime('%Y%m%d')
        end = today.strftime('%Y%m%d')
        
        if expirations and strikes:
            # Pick ATM call
            mid_strike = strikes[len(strikes) // 2]
            
            df = client.fetch_option_eod(
                root='SPY',
                exp=expirations[0],
                strike=mid_strike,
                right='C',
                start_date=start,
                end_date=end
            )
            
            print(f"Fetched {len(df)} days of data")
            if not df.empty:
                print(df.head())
        
        # Print stats
        print("\nClient Statistics:")
        stats = client.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ All tests passed!")
        
    except ThetaDataConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\nMake sure:")
        print("1. ThetaData Terminal is installed")
        print("2. Terminal is running")
        print("3. You're logged in to Terminal")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
