"""
OptionsTitan - Risk Management Module
Phase 3: Advanced risk controls and comprehensive risk metrics

This module provides:
1. AdvancedRiskManager - Institutional-grade risk metrics and controls
2. CorrelationMonitor - Portfolio correlation and diversification tracking
3. Stress testing scenarios
4. Real-time risk monitoring and alerts
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    """
    Comprehensive risk management system for options trading
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        self.risk_metrics = {}
        self.stress_scenarios = {}
        self.position_limits = {
            'max_position_size': 0.02,  # 2% of portfolio per position (reduced from 5%)
            'max_portfolio_risk': 0.10,  # 10% maximum portfolio risk (reduced from 15%)
            'max_correlation': 0.60,     # 60% maximum correlation between positions (reduced from 70%)
            'max_leverage': 2.0,         # 2x maximum leverage
            'max_sector_concentration': 0.30  # 30% max in single sector
        }
        
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95, 
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level (0.95 = 95%)
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaR value (negative number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0
        
        if method == 'historical':
            # Historical simulation method
            var = np.percentile(returns, (1 - confidence_level) * 100)
            
        elif method == 'parametric':
            # Parametric method (assumes normal distribution)
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean = returns.mean()
            std = returns.std()
            simulated_returns = np.random.normal(mean, std, 10000)
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return var
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level
            
        Returns:
            CVaR value (expected loss beyond VaR threshold)
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        # Calculate mean of returns below VaR threshold
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        cvar = tail_returns.mean()
        return cvar
    
    def calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Dict:
        """
        Calculate drawdown metrics
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Dictionary with drawdown metrics
        """
        if len(portfolio_values) == 0:
            return {}
        
        # Calculate running maximum (peak)
        peak = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - peak) / peak
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown duration (periods in drawdown)
        in_drawdown = drawdown < 0
        drawdown_periods = 0
        for i in range(len(in_drawdown) - 1, -1, -1):
            if in_drawdown.iloc[i]:
                drawdown_periods += 1
            else:
                break
        
        # Recovery time (time to recover from max drawdown)
        max_dd_idx = drawdown.idxmin()
        recovery_idx = None
        if max_dd_idx in portfolio_values.index:
            peak_before_dd = peak.loc[max_dd_idx]
            recovery_series = portfolio_values.loc[max_dd_idx:] >= peak_before_dd
            if recovery_series.any():
                recovery_idx = recovery_series.idxmax()
        
        recovery_periods = 0
        if recovery_idx and recovery_idx != max_dd_idx:
            recovery_periods = len(portfolio_values.loc[max_dd_idx:recovery_idx]) - 1
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'drawdown_duration': drawdown_periods,
            'recovery_periods': recovery_periods,
            'time_underwater': drawdown_periods  # Alias for compatibility
        }
    
    def calculate_risk_adjusted_returns(self, returns: pd.Series, 
                                      risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate risk-adjusted return metrics
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        if len(returns) == 0:
            return {}
        
        # Annualized metrics (assuming daily returns)
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino Ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_volatility
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio (annual return / max drawdown)
        portfolio_values = (1 + returns).cumprod()
        dd_metrics = self.calculate_drawdown_metrics(portfolio_values)
        max_drawdown = abs(dd_metrics.get('max_drawdown', 0.01))
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def run_stress_tests(self, portfolio_returns: pd.Series, 
                        positions: Dict = None) -> Dict:
        """
        Run comprehensive stress testing scenarios
        
        Args:
            portfolio_returns: Historical portfolio returns
            positions: Current portfolio positions (optional)
            
        Returns:
            Dictionary with stress test results
        """
        stress_results = {}
        
        if len(portfolio_returns) == 0:
            return stress_results
        
        current_value = 100000  # Assume $100k portfolio for percentage calculations
        
        # Scenario 1: Market crash scenarios
        crash_scenarios = {
            'market_crash_10pct': -0.10,
            'market_crash_20pct': -0.20,
            'market_crash_30pct': -0.30
        }
        
        for scenario, market_shock in crash_scenarios.items():
            # Estimate portfolio impact (simplified beta = 1.2 for options)
            portfolio_beta = 1.2
            estimated_loss = market_shock * portfolio_beta
            stress_results[scenario] = {
                'market_shock': market_shock,
                'estimated_portfolio_loss': estimated_loss,
                'estimated_dollar_loss': current_value * estimated_loss
            }
        
        # Scenario 2: Volatility spike
        current_vol = portfolio_returns.std() * np.sqrt(252)
        vol_spike_scenarios = {
            'volatility_spike_2x': 2.0,
            'volatility_spike_3x': 3.0
        }
        
        for scenario, vol_multiplier in vol_spike_scenarios.items():
            new_vol = current_vol * vol_multiplier
            # Estimate impact on options (higher vol generally increases option values)
            # But also increases risk - simplified model
            vol_impact = (vol_multiplier - 1) * 0.1  # 10% impact per vol doubling
            stress_results[scenario] = {
                'vol_multiplier': vol_multiplier,
                'new_volatility': new_vol,
                'estimated_impact': vol_impact
            }
        
        # Scenario 3: Tail events (based on historical distribution)
        tail_percentiles = [1, 5, 10]  # 1st, 5th, 10th percentile events
        for percentile in tail_percentiles:
            tail_return = np.percentile(portfolio_returns, percentile)
            stress_results[f'tail_event_{percentile}pct'] = {
                'percentile': percentile,
                'historical_return': tail_return,
                'estimated_loss': current_value * tail_return
            }
        
        # Scenario 4: Liquidity crisis (increased spreads)
        liquidity_scenarios = {
            'liquidity_crisis_mild': 0.02,    # 2% additional spread cost
            'liquidity_crisis_severe': 0.05   # 5% additional spread cost
        }
        
        for scenario, spread_cost in liquidity_scenarios.items():
            stress_results[scenario] = {
                'additional_spread_cost': spread_cost,
                'estimated_impact': -spread_cost  # Negative impact on returns
            }
        
        return stress_results
    
    def calculate_comprehensive_risk_metrics(self, returns: pd.Series, 
                                           portfolio_values: pd.Series = None,
                                           positions: Dict = None) -> Dict:
        """
        Calculate comprehensive risk metrics suite
        
        Args:
            returns: Portfolio returns series
            portfolio_values: Portfolio value series (optional)
            positions: Current positions dictionary (optional)
            
        Returns:
            Comprehensive risk metrics dictionary
        """
        if len(returns) == 0:
            return {}
        
        # Generate portfolio values if not provided
        if portfolio_values is None:
            portfolio_values = (1 + returns).cumprod() * 100000
        
        risk_metrics = {}
        
        # Basic volatility metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        risk_metrics['volatility_daily'] = daily_vol
        risk_metrics['volatility_annual'] = annual_vol
        
        # VaR and CVaR for multiple confidence levels
        for confidence in self.confidence_levels:
            var_key = f'var_{int(confidence*100)}'
            cvar_key = f'cvar_{int(confidence*100)}'
            
            risk_metrics[var_key] = self.calculate_var(returns, confidence)
            risk_metrics[cvar_key] = self.calculate_cvar(returns, confidence)
        
        # Drawdown metrics
        dd_metrics = self.calculate_drawdown_metrics(portfolio_values)
        risk_metrics.update(dd_metrics)
        
        # Risk-adjusted returns
        ra_metrics = self.calculate_risk_adjusted_returns(returns)
        risk_metrics.update(ra_metrics)
        
        # Stress test results
        stress_results = self.run_stress_tests(returns, positions)
        risk_metrics['stress_tests'] = stress_results
        
        # Additional risk metrics
        risk_metrics['return_skewness'] = returns.skew()
        risk_metrics['return_kurtosis'] = returns.kurtosis()
        
        # Risk concentration (if positions provided)
        if positions:
            risk_metrics['position_concentration'] = self._calculate_position_concentration(positions)
        
        return risk_metrics
    
    def _calculate_position_concentration(self, positions: Dict) -> Dict:
        """Calculate position concentration metrics"""
        if not positions:
            return {}
        
        position_sizes = [pos.get('size', 0) for pos in positions.values()]
        total_exposure = sum(position_sizes)
        
        if total_exposure == 0:
            return {}
        
        # Calculate Herfindahl-Hirschman Index (concentration measure)
        weights = [size / total_exposure for size in position_sizes]
        hhi = sum(w**2 for w in weights)
        
        # Effective number of positions
        effective_positions = 1 / hhi if hhi > 0 else 0
        
        # Largest position weight
        max_position_weight = max(weights) if weights else 0
        
        return {
            'herfindahl_index': hhi,
            'effective_positions': effective_positions,
            'max_position_weight': max_position_weight,
            'total_positions': len(positions)
        }
    
    def check_risk_limits(self, current_metrics: Dict, positions: Dict = None) -> Dict:
        """
        Check if current risk metrics exceed predefined limits
        
        Args:
            current_metrics: Current risk metrics
            positions: Current positions (optional)
            
        Returns:
            Dictionary with limit violations and alerts
        """
        violations = []
        alerts = []
        
        # Check VaR limits
        var_95 = current_metrics.get('var_95', 0)
        if var_95 < -0.10:  # More than 10% daily VaR
            violations.append(f"Daily VaR exceeds limit: {var_95:.1%} < -10%")
        
        # Check drawdown limits
        current_dd = current_metrics.get('current_drawdown', 0)
        if current_dd < -self.position_limits['max_portfolio_risk']:
            violations.append(f"Current drawdown exceeds limit: {current_dd:.1%}")
        
        # Check volatility
        annual_vol = current_metrics.get('annual_volatility', 0)
        if annual_vol > 0.50:  # More than 50% annual volatility
            alerts.append(f"High volatility detected: {annual_vol:.1%}")
        
        # Check Sharpe ratio
        sharpe = current_metrics.get('sharpe_ratio', 0)
        if sharpe < 0.5:
            alerts.append(f"Low Sharpe ratio: {sharpe:.2f}")
        
        # Check position concentration if positions provided
        if positions:
            concentration = current_metrics.get('position_concentration', {})
            max_weight = concentration.get('max_position_weight', 0)
            if max_weight > self.position_limits['max_position_size']:
                violations.append(f"Position size exceeds limit: {max_weight:.1%}")
        
        return {
            'violations': violations,
            'alerts': alerts,
            'risk_score': len(violations) * 2 + len(alerts),  # Simple risk scoring
            'status': 'VIOLATION' if violations else ('WARNING' if alerts else 'OK')
        }
    
    def generate_risk_report(self, risk_metrics: Dict, limit_check: Dict = None) -> str:
        """Generate comprehensive risk report"""
        
        report = []
        report.append("=== RISK MANAGEMENT REPORT ===\n")
        
        # Risk Metrics Summary
        report.append("Risk Metrics Summary:")
        
        key_metrics = [
            ('volatility_daily', 'Daily Volatility', '.4f'),
            ('volatility_annual', 'Annual Volatility', '.4f'),
            ('var_95', '95% VaR', '.4f'),
            ('var_99', '99% VaR', '.4f'),
            ('cvar_95', '95% CVaR', '.4f'),
            ('cvar_99', '99% CVaR', '.4f'),
            ('max_drawdown', 'Max Drawdown', '.4f'),
            ('sharpe_ratio', 'Sharpe Ratio', '.4f'),
            ('sortino_ratio', 'Sortino Ratio', '.4f'),
            ('skewness', 'Return Skewness', '.4f'),
            ('kurtosis', 'Return Kurtosis', '.4f')
        ]
        
        for key, label, fmt in key_metrics:
            if key in risk_metrics:
                value = risk_metrics[key]
                report.append(f"  {label:<25}: {value:{fmt}}")
        
        # Stress Test Results
        if 'stress_tests' in risk_metrics:
            report.append("\nStress Test Results:")
            stress_tests = risk_metrics['stress_tests']
            
            for scenario, results in stress_tests.items():
                if 'estimated_portfolio_loss' in results:
                    loss = results['estimated_portfolio_loss']
                    report.append(f"  {scenario:<25}: Max Loss {loss:.1%}")
                elif 'estimated_impact' in results:
                    impact = results['estimated_impact']
                    report.append(f"  {scenario:<25}: Impact {impact:.1%}")
        
        # Position Concentration
        if 'position_concentration' in risk_metrics:
            conc = risk_metrics['position_concentration']
            report.append(f"\nDiversification Score: {1/conc.get('herfindahl_index', 1):.4f}")
        
        # Risk Limit Violations
        if limit_check:
            report.append(f"\nRisk Status: {limit_check['status']}")
            
            if limit_check['violations']:
                report.append("VIOLATIONS:")
                for violation in limit_check['violations']:
                    report.append(f"  ⚠️  {violation}")
            
            if limit_check['alerts']:
                report.append("ALERTS:")
                for alert in limit_check['alerts']:
                    report.append(f"  ⚡ {alert}")
        
        return "\n".join(report)


class CorrelationMonitor:
    """
    Monitor portfolio correlations and diversification
    """
    
    def __init__(self, correlation_threshold: float = 0.70):
        self.correlation_threshold = correlation_threshold
        self.correlation_history = []
        
    def calculate_position_correlations(self, position_returns: pd.DataFrame) -> Dict:
        """
        Calculate correlations between positions
        
        Args:
            position_returns: DataFrame with returns for each position
            
        Returns:
            Dictionary with correlation analysis
        """
        if position_returns.empty or len(position_returns.columns) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = position_returns.corr()
        
        # Find high correlations (excluding diagonal)
        high_correlations = []
        n_positions = len(corr_matrix)
        
        for i in range(n_positions):
            for j in range(i + 1, n_positions):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > self.correlation_threshold:
                    high_correlations.append({
                        'position_1': corr_matrix.index[i],
                        'position_2': corr_matrix.index[j],
                        'correlation': corr_value
                    })
        
        # Calculate portfolio diversification metrics
        eigenvalues = np.linalg.eigvals(corr_matrix.values)
        effective_positions = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        
        # Average correlation
        upper_triangle = np.triu(corr_matrix.values, k=1)
        avg_correlation = upper_triangle[upper_triangle != 0].mean()
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_correlations,
            'average_correlation': avg_correlation,
            'effective_positions': effective_positions,
            'diversification_ratio': effective_positions / n_positions,
            'max_correlation': abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]).max()
        }
    
    def monitor_correlation_drift(self, current_correlations: pd.DataFrame, 
                                window_size: int = 30) -> Dict:
        """
        Monitor how correlations change over time
        
        Args:
            current_correlations: Current correlation matrix
            window_size: Rolling window size for trend analysis
            
        Returns:
            Dictionary with correlation drift analysis
        """
        # Store current correlations
        self.correlation_history.append({
            'timestamp': datetime.now(),
            'correlations': current_correlations
        })
        
        # Keep only recent history
        if len(self.correlation_history) > window_size * 2:
            self.correlation_history = self.correlation_history[-window_size * 2:]
        
        if len(self.correlation_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate correlation stability
        recent_corrs = [entry['correlations'] for entry in self.correlation_history[-window_size:]]
        
        if len(recent_corrs) < 2:
            return {'status': 'insufficient_recent_data'}
        
        # Calculate standard deviation of correlations over time
        correlation_stds = {}
        positions = current_correlations.index
        
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i < j:  # Upper triangle only
                    pair_corrs = [corr.loc[pos1, pos2] for corr in recent_corrs if pos1 in corr.index and pos2 in corr.index]
                    if len(pair_corrs) > 1:
                        correlation_stds[f"{pos1}_{pos2}"] = np.std(pair_corrs)
        
        # Identify unstable correlations
        unstable_threshold = 0.2  # Correlation changes more than 20%
        unstable_pairs = {pair: std for pair, std in correlation_stds.items() if std > unstable_threshold}
        
        return {
            'status': 'analyzed',
            'correlation_stability': correlation_stds,
            'unstable_pairs': unstable_pairs,
            'avg_correlation_std': np.mean(list(correlation_stds.values())) if correlation_stds else 0,
            'stability_score': 1 - min(1, len(unstable_pairs) / max(1, len(correlation_stds)))
        }


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Create sample portfolio returns
    n_days = 252  # One year of daily returns
    returns = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual vol, positive drift
    returns_series = pd.Series(returns, index=pd.date_range('2023-01-01', periods=n_days))
    
    # Create portfolio values
    portfolio_values = (1 + returns_series).cumprod() * 100000
    
    print("Testing Advanced Risk Manager...")
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager()
    
    # Calculate comprehensive risk metrics
    risk_metrics = risk_manager.calculate_comprehensive_risk_metrics(
        returns_series, 
        portfolio_values
    )
    
    # Check risk limits
    limit_check = risk_manager.check_risk_limits(risk_metrics)
    
    # Generate risk report
    report = risk_manager.generate_risk_report(risk_metrics, limit_check)
    print(report)
    
    print("\n" + "="*50)
    print("Testing Correlation Monitor...")
    
    # Test correlation monitor
    correlation_monitor = CorrelationMonitor()
    
    # Create sample position returns
    n_positions = 5
    position_returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.001] * n_positions,
            cov=np.eye(n_positions) * 0.0004 + 0.0001,  # Some correlation
            size=n_days
        ),
        columns=[f'Position_{i}' for i in range(n_positions)],
        index=pd.date_range('2023-01-01', periods=n_days)
    )
    
    # Calculate position correlations
    corr_analysis = correlation_monitor.calculate_position_correlations(position_returns)
    
    print(f"Average Correlation: {corr_analysis.get('average_correlation', 0):.3f}")
    print(f"Effective Positions: {corr_analysis.get('effective_positions', 0):.2f}")
    print(f"Diversification Ratio: {corr_analysis.get('diversification_ratio', 0):.3f}")
    
    if corr_analysis.get('high_correlations'):
        print("High Correlations Detected:")
        for corr in corr_analysis['high_correlations']:
            print(f"  {corr['position_1']} - {corr['position_2']}: {corr['correlation']:.3f}")
    else:
        print("No high correlations detected.")
    
    print("\nRisk Management System Ready!")