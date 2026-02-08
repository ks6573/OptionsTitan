"""
OptionsTitan - Model Explainability Module
Phase 4: SHAP-based prediction explanations and feature analysis

This module provides:
1. ModelExplainer - SHAP-based prediction explanations
2. Feature contribution analysis
3. Global feature importance
4. Human-readable trading rationales
5. Model interpretation and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
from datetime import datetime

# SHAP imports with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Model explainability features will be limited.")

warnings.filterwarnings('ignore')

class ModelExplainer:
    """
    Comprehensive model explainability using SHAP and custom interpretations
    """
    
    def __init__(self, model, feature_names: List[str], model_type: str = 'tree'):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type  # 'tree', 'linear', 'ensemble'
        self.explainer = None
        self.shap_values = None
        self.feature_importance = {}
        self.is_initialized = False
        
        # Feature categories for better interpretation
        self.feature_categories = self._categorize_features()
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            self._initialize_shap_explainer()
    
    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features for better interpretation"""
        categories = {
            'price_features': [],
            'volatility_features': [],
            'technical_indicators': [],
            'greeks': [],
            'time_features': [],
            'market_regime': [],
            'other': []
        }
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['price', 'strike', 'option_to_spy']):
                categories['price_features'].append(feature)
            elif any(keyword in feature_lower for keyword in ['volatility', 'vix', 'iv_']):
                categories['volatility_features'].append(feature)
            elif any(keyword in feature_lower for keyword in ['rsi', 'macd', 'momentum', 'bb_', 'stoch']):
                categories['technical_indicators'].append(feature)
            elif any(keyword in feature_lower for keyword in ['delta', 'gamma', 'theta', 'vega']):
                categories['greeks'].append(feature)
            elif any(keyword in feature_lower for keyword in ['hour', 'day_', 'time_', 'market_open']):
                categories['time_features'].append(feature)
            elif any(keyword in feature_lower for keyword in ['regime', 'percentile', 'rank']):
                categories['market_regime'].append(feature)
            else:
                categories['other'].append(feature)
        
        return categories
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer based on model type"""
        if not SHAP_AVAILABLE:
            return
        
        try:
            # Check if it's an EnsembleModelTrainer (has ensemble-specific methods)
            if hasattr(self.model, 'predict_ensemble_proba'):
                # Use general Explainer for ensemble models (more compatible)
                background_data = np.zeros((10, len(self.feature_names)))
                self.explainer = shap.Explainer(self.model.predict_proba, background_data)
                logging.info("Using general Explainer for EnsembleModelTrainer")
            elif self.model_type == 'tree' or hasattr(self.model, 'feature_importances_'):
                # Tree-based models (XGBoost, LightGBM, RandomForest, etc.)
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'linear' or hasattr(self.model, 'coef_'):
                # Linear models
                self.explainer = shap.LinearExplainer(self.model, np.zeros((1, len(self.feature_names))))
            else:
                # General model explainer (slower but works with any model)
                # Use a small background dataset for efficiency
                background_data = np.zeros((10, len(self.feature_names)))
                self.explainer = shap.Explainer(self.model.predict_proba, background_data)
            
            self.is_initialized = True
            logging.info(f"SHAP explainer initialized for {self.model_type} model")
            
        except Exception as e:
            logging.warning(f"Failed to initialize SHAP explainer: {e}")
            self.is_initialized = False
    
    def explain_prediction(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict:
        """
        Explain a single prediction with detailed feature contributions
        
        Args:
            X: Feature matrix
            instance_idx: Index of instance to explain
            
        Returns:
            Dictionary with explanation details
        """
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range")
        
        instance = X.iloc[instance_idx:instance_idx+1]
        
        # Get model prediction
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(instance)[0]
            prediction = prediction_proba[1]  # Probability of positive class
            confidence = max(prediction, 1 - prediction)
        else:
            prediction = self.model.predict(instance)[0]
            prediction_proba = [1-prediction, prediction] if prediction in [0, 1] else [0.5, 0.5]
            confidence = 0.5
        
        explanation = {
            'prediction': prediction,
            'prediction_proba': prediction_proba,
            'confidence': confidence,
            'feature_values': instance.iloc[0].to_dict(),
            'feature_contributions': {},
            'trading_rationale': '',
            'risk_factors': [],
            'opportunity_factors': []
        }
        
        # SHAP-based explanation if available
        if SHAP_AVAILABLE and self.is_initialized:
            try:
                shap_values = self.explainer.shap_values(instance)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # Binary classification - use positive class
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
                if len(shap_values.shape) > 1:
                    shap_values = shap_values[0]  # Take first instance
                
                # Create feature contributions dictionary
                for i, feature in enumerate(self.feature_names):
                    if i < len(shap_values):
                        explanation['feature_contributions'][feature] = float(shap_values[i])
                
            except Exception as e:
                logging.warning(f"SHAP explanation failed: {e}")
                # Fallback to simple feature importance
                explanation['feature_contributions'] = self._get_fallback_contributions(instance)
        else:
            # Fallback explanation without SHAP
            explanation['feature_contributions'] = self._get_fallback_contributions(instance)
        
        # Generate trading rationale
        explanation['trading_rationale'] = self._generate_trading_rationale(
            explanation['feature_contributions'], 
            explanation['feature_values'],
            prediction
        )
        
        # Identify risk and opportunity factors
        explanation['risk_factors'], explanation['opportunity_factors'] = self._identify_risk_opportunity_factors(
            explanation['feature_contributions'],
            explanation['feature_values']
        )
        
        return explanation
    
    def _get_fallback_contributions(self, instance: pd.DataFrame) -> Dict:
        """Fallback feature contribution calculation without SHAP"""
        contributions = {}
        
        # Use feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            # Scale by feature values (simple approximation)
            for i, feature in enumerate(self.feature_names):
                if i < len(importance):
                    feature_value = instance.iloc[0, i]
                    # Normalize feature value and multiply by importance
                    normalized_value = (feature_value - 0) / (1 + abs(feature_value))
                    contributions[feature] = importance[i] * normalized_value
        elif hasattr(self.model, 'coef_'):
            # Linear model coefficients
            coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            for i, feature in enumerate(self.feature_names):
                if i < len(coef):
                    feature_value = instance.iloc[0, i]
                    contributions[feature] = coef[i] * feature_value
        else:
            # No feature importance available - use zero contributions
            contributions = {feature: 0.0 for feature in self.feature_names}
        
        return contributions
    
    def _generate_trading_rationale(self, contributions: Dict, feature_values: Dict, 
                                  prediction: float) -> str:
        """Generate human-readable trading rationale"""
        
        # Sort contributions by absolute value
        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Determine overall sentiment
        sentiment = "BULLISH 📈" if prediction > 0.5 else "BEARISH 📉"
        confidence_pct = max(prediction, 1 - prediction) * 100
        
        rationale = [f"Trading Decision: {sentiment}"]
        rationale.append(f"Confidence: {confidence_pct:.1f}%")
        rationale.append(f"Prediction Score: {prediction:.4f}")
        rationale.append("")
        
        # Top positive contributors (bullish factors)
        positive_factors = [(f, c) for f, c in sorted_contributions if c > 0][:3]
        if positive_factors:
            rationale.append("Bullish Factors:")
            for feature, contribution in positive_factors:
                feature_value = feature_values.get(feature, 0)
                interpretation = self._interpret_feature_contribution(feature, contribution, feature_value, True)
                rationale.append(f"  • {interpretation}")
        
        rationale.append("")
        
        # Top negative contributors (bearish factors)
        negative_factors = [(f, c) for f, c in sorted_contributions if c < 0][:3]
        if negative_factors:
            rationale.append("Bearish Factors:")
            for feature, contribution in negative_factors:
                feature_value = feature_values.get(feature, 0)
                interpretation = self._interpret_feature_contribution(feature, contribution, feature_value, False)
                rationale.append(f"  • {interpretation}")
        
        return "\n".join(rationale)
    
    def _interpret_feature_contribution(self, feature: str, contribution: float, 
                                      feature_value: float, is_positive: bool) -> str:
        """Interpret individual feature contribution in trading terms"""
        
        feature_lower = feature.lower()
        abs_contribution = abs(contribution)
        
        # Feature-specific interpretations
        if 'rsi' in feature_lower:
            if feature_value > 70:
                return f"{feature}: {abs_contribution:+.4f} (Overbought at {feature_value:.1f})"
            elif feature_value < 30:
                return f"{feature}: {abs_contribution:+.4f} (Oversold at {feature_value:.1f})"
            else:
                return f"{feature}: {abs_contribution:+.4f} (Neutral at {feature_value:.1f})"
        
        elif 'vix' in feature_lower:
            if feature_value > 25:
                return f"{feature}: {abs_contribution:+.4f} (High fear at {feature_value:.1f})"
            elif feature_value < 15:
                return f"{feature}: {abs_contribution:+.4f} (Low fear at {feature_value:.1f})"
            else:
                return f"{feature}: {abs_contribution:+.4f} (Moderate fear at {feature_value:.1f})"
        
        elif 'implied_volatility' in feature_lower or 'iv_' in feature_lower:
            return f"{feature}: {abs_contribution:+.4f} (IV at {feature_value:.1%})"
        
        elif 'delta' in feature_lower:
            return f"{feature}: {abs_contribution:+.4f} (Delta {feature_value:.3f})"
        
        elif 'time_to_expiry' in feature_lower:
            return f"{feature}: {abs_contribution:+.4f} ({feature_value:.0f} days to expiry)"
        
        elif 'strike_distance' in feature_lower:
            if feature_value > 0:
                return f"{feature}: {abs_contribution:+.4f} (OTM by ${feature_value:.2f})"
            else:
                return f"{feature}: {abs_contribution:+.4f} (ITM by ${abs(feature_value):.2f})"
        
        elif 'momentum' in feature_lower:
            return f"{feature}: {abs_contribution:+.4f} (Momentum {feature_value:+.1%})"
        
        elif 'volume' in feature_lower:
            return f"{feature}: {abs_contribution:+.4f} (Volume {feature_value:,.0f})"
        
        else:
            # Generic interpretation
            return f"{feature}: {abs_contribution:+.4f} (Value: {feature_value:.4f})"
    
    def _identify_risk_opportunity_factors(self, contributions: Dict, 
                                         feature_values: Dict) -> Tuple[List[str], List[str]]:
        """Identify key risk and opportunity factors"""
        
        risk_factors = []
        opportunity_factors = []
        
        for feature, value in feature_values.items():
            feature_lower = feature.lower()
            
            # Risk factors
            if 'vix' in feature_lower and value > 30:
                risk_factors.append(f"High market volatility (VIX: {value:.1f})")
            
            if 'time_to_expiry' in feature_lower and value < 7:
                risk_factors.append(f"High theta decay risk ({value:.0f} days to expiry)")
            
            if 'implied_volatility' in feature_lower and value > 0.5:
                risk_factors.append(f"Elevated implied volatility ({value:.1%})")
            
            # Opportunity factors
            if 'rsi' in feature_lower:
                if value < 30:
                    opportunity_factors.append(f"Oversold condition (RSI: {value:.1f})")
                elif value > 70:
                    opportunity_factors.append(f"Overbought condition (RSI: {value:.1f})")
            
            if 'momentum' in feature_lower and abs(value) > 0.05:
                direction = "positive" if value > 0 else "negative"
                opportunity_factors.append(f"Strong {direction} momentum ({value:+.1%})")
        
        return risk_factors, opportunity_factors
    
    def calculate_global_feature_importance(self, X: pd.DataFrame, 
                                          sample_size: int = 1000) -> Dict:
        """
        Calculate global feature importance across the dataset
        
        Args:
            X: Feature matrix
            sample_size: Number of samples to use for SHAP calculation
            
        Returns:
            Dictionary with global importance metrics
        """
        # Sample data if too large
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
        
        importance_results = {
            'feature_importance': {},
            'feature_categories': self.feature_categories,
            'category_importance': {},
            'top_features': [],
            'interaction_effects': {}
        }
        
        # SHAP-based global importance
        if SHAP_AVAILABLE and self.is_initialized:
            try:
                shap_values = self.explainer.shap_values(X_sample)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
                # Calculate mean absolute SHAP values
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                
                for i, feature in enumerate(self.feature_names):
                    if i < len(mean_abs_shap):
                        importance_results['feature_importance'][feature] = float(mean_abs_shap[i])
                
            except Exception as e:
                logging.warning(f"Global SHAP calculation failed: {e}")
                # Fallback to model feature importance
                importance_results['feature_importance'] = self._get_model_feature_importance()
        else:
            # Fallback to model feature importance
            importance_results['feature_importance'] = self._get_model_feature_importance()
        
        # Calculate category-level importance
        for category, features in self.feature_categories.items():
            category_importance = sum(
                importance_results['feature_importance'].get(feature, 0) 
                for feature in features
            )
            importance_results['category_importance'][category] = category_importance
        
        # Identify top features
        sorted_features = sorted(
            importance_results['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        importance_results['top_features'] = sorted_features[:10]
        
        return importance_results
    
    def _get_model_feature_importance(self) -> Dict:
        """Get feature importance from the model itself"""
        importance = {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            for i, feature in enumerate(self.feature_names):
                if i < len(self.model.feature_importances_):
                    importance[feature] = float(self.model.feature_importances_[i])
        elif hasattr(self.model, 'coef_'):
            # Linear models
            coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            for i, feature in enumerate(self.feature_names):
                if i < len(coef):
                    importance[feature] = float(abs(coef[i]))
        else:
            # No feature importance available
            importance = {feature: 0.0 for feature in self.feature_names}
        
        return importance
    
    def generate_model_interpretation_report(self, X: pd.DataFrame, 
                                           sample_predictions: int = 5) -> str:
        """Generate comprehensive model interpretation report"""
        
        report = []
        report.append("=== MODEL INTERPRETATION REPORT ===\n")
        
        # Global feature importance
        global_importance = self.calculate_global_feature_importance(X)
        
        report.append("Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(global_importance['top_features'], 1):
            report.append(f"  {i:2d}. {feature:<25}: {importance:.6f}")
        
        report.append("\nFeature Category Importance:")
        for category, importance in sorted(global_importance['category_importance'].items(), 
                                         key=lambda x: x[1], reverse=True):
            if importance > 0:
                report.append(f"  {category:<20}: {importance:.6f}")
        
        # Sample predictions with explanations
        if sample_predictions > 0 and len(X) > 0:
            report.append(f"\n=== SAMPLE PREDICTION EXPLANATIONS ===")
            
            sample_indices = np.random.choice(len(X), min(sample_predictions, len(X)), replace=False)
            
            for i, idx in enumerate(sample_indices, 1):
                report.append(f"\n--- Sample Prediction {i} ---")
                
                try:
                    explanation = self.explain_prediction(X, idx)
                    report.append(explanation['trading_rationale'])
                    
                    if explanation['risk_factors']:
                        report.append("\nRisk Factors:")
                        for risk in explanation['risk_factors']:
                            report.append(f"  ⚠️  {risk}")
                    
                    if explanation['opportunity_factors']:
                        report.append("\nOpportunity Factors:")
                        for opp in explanation['opportunity_factors']:
                            report.append(f"  💡 {opp}")
                
                except Exception as e:
                    report.append(f"Error explaining prediction {i}: {e}")
        
        # Model insights
        report.append(f"\n=== MODEL INSIGHTS ===")
        report.append(f"SHAP Available: {'Yes' if SHAP_AVAILABLE else 'No'}")
        report.append(f"Explainer Initialized: {'Yes' if self.is_initialized else 'No'}")
        report.append(f"Model Type: {self.model_type}")
        report.append(f"Number of Features: {len(self.feature_names)}")
        
        return "\n".join(report)


# Utility functions for model interpretation
def compare_model_explanations(explainer1: ModelExplainer, explainer2: ModelExplainer, 
                             X: pd.DataFrame, instance_idx: int = 0) -> Dict:
    """Compare explanations from two different models"""
    
    explanation1 = explainer1.explain_prediction(X, instance_idx)
    explanation2 = explainer2.explain_prediction(X, instance_idx)
    
    comparison = {
        'model1_prediction': explanation1['prediction'],
        'model2_prediction': explanation2['prediction'],
        'prediction_difference': abs(explanation1['prediction'] - explanation2['prediction']),
        'agreement_level': 1 - abs(explanation1['prediction'] - explanation2['prediction']),
        'feature_agreement': {},
        'conflicting_features': []
    }
    
    # Compare feature contributions
    for feature in explainer1.feature_names:
        if feature in explanation1['feature_contributions'] and feature in explanation2['feature_contributions']:
            contrib1 = explanation1['feature_contributions'][feature]
            contrib2 = explanation2['feature_contributions'][feature]
            
            # Check if contributions have same sign (agreement)
            same_direction = (contrib1 * contrib2) >= 0
            comparison['feature_agreement'][feature] = {
                'model1_contribution': contrib1,
                'model2_contribution': contrib2,
                'same_direction': same_direction,
                'difference': abs(contrib1 - contrib2)
            }
            
            if not same_direction and (abs(contrib1) > 0.01 or abs(contrib2) > 0.01):
                comparison['conflicting_features'].append(feature)
    
    return comparison


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data (requires a trained model)
    print("Model Explainability Module")
    print(f"SHAP Available: {SHAP_AVAILABLE}")
    
    if not SHAP_AVAILABLE:
        print("\nTo enable full explainability features, install SHAP:")
        print("pip install shap")
    
    # Create mock model for testing
    class MockModel:
        def __init__(self):
            self.feature_importances_ = np.random.random(10)
        
        def predict_proba(self, X):
            return np.random.random((len(X), 2))
    
    # Test explainer initialization
    mock_model = MockModel()
    feature_names = [f'feature_{i}' for i in range(10)]
    
    explainer = ModelExplainer(mock_model, feature_names, 'tree')
    
    # Test with sample data
    X_sample = pd.DataFrame(np.random.random((100, 10)), columns=feature_names)
    
    # Test global importance
    global_importance = explainer.calculate_global_feature_importance(X_sample)
    print(f"\nTop 5 Features:")
    for feature, importance in global_importance['top_features'][:5]:
        print(f"  {feature}: {importance:.4f}")
    
    # Test single prediction explanation
    try:
        explanation = explainer.explain_prediction(X_sample, 0)
        print(f"\nSample Explanation:")
        print(f"Prediction: {explanation['prediction']:.3f}")
        print(f"Confidence: {explanation['confidence']:.3f}")
        print("\nTrading Rationale:")
        print(explanation['trading_rationale'])
    except Exception as e:
        print(f"Explanation test failed: {e}")
    
    print("\nModel Explainability System Ready!")