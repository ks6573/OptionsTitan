"""
OptionsTitan - Ensemble Models Module
Phase 1: Multi-model ensemble with drift detection and feature quality monitoring

This module provides:
1. EnsembleModelTrainer - 5-model ensemble with automatic weight optimization
2. FeatureStore - Feature drift detection and quality monitoring
3. Prediction uncertainty quantification
4. Model correlation analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class EnsembleModelTrainer:
    """
    Multi-model ensemble trainer with automatic weight optimization
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.is_fitted = False
        self.feature_importance = {}
        self.model_scores = {}
        
        # Initialize base models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the 5 base models for ensemble"""
        self.models = {
            'xgboost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=-1
            ),
            'gradient_boost': HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'logistic': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            )
        }
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train ensemble models and optimize weights
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with training results and metrics
        """
        logging.info("Training ensemble models...")
        
        # Handle any remaining NaN values
        if X.isnull().any().any():
            logging.warning("NaN values detected in training data. Filling with median values.")
            X = X.fillna(X.median())
        
        # Ensure no infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Train individual models and collect predictions
        model_predictions = {}
        failed_models = []
        
        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X, y)
                
                # Get predictions for weight optimization
                predictions = model.predict_proba(X)[:, 1]
                model_predictions[name] = predictions
                
                # Calculate individual model score
                score = roc_auc_score(y, predictions)
                self.model_scores[name] = score
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = np.abs(model.coef_[0])
                
                logging.info(f"{name} trained successfully. AUC: {score:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {name}: {e}")
                # Mark failed model for removal
                failed_models.append(name)
                continue
        
        # Remove failed models after iteration
        for name in failed_models:
            if name in self.models:
                del self.models[name]
        
        if not self.models:
            raise ValueError("No models trained successfully")
        
        # Optimize ensemble weights
        logging.info("Optimizing ensemble weights...")
        self.weights = self._optimize_weights(model_predictions, y)
        
        self.is_fitted = True
        
        # Calculate ensemble performance
        ensemble_pred = self.predict_ensemble_proba(X)
        ensemble_score = roc_auc_score(y, ensemble_pred)
        
        # Calculate ensemble feature importance
        ensemble_importance = self._calculate_ensemble_feature_importance()
        
        results = {
            'individual_scores': self.model_scores,
            'ensemble_score': ensemble_score,
            'optimized_weights': self.weights,
            'num_models': len(self.models),
            'feature_importance': ensemble_importance
        }
        
        logging.info(f"Ensemble training completed. Final AUC: {ensemble_score:.4f}")
        return results
    
    def _optimize_weights(self, predictions: Dict, y: pd.Series) -> Dict:
        """
        Optimize ensemble weights using risk-aware multi-objective optimization
        """
        model_names = list(predictions.keys())
        pred_matrix = np.column_stack([predictions[name] for name in model_names])
        
        def risk_adjusted_objective(weights):
            # Normalize weights to sum to 1
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.dot(pred_matrix, weights)
            
            try:
                # Primary objective: AUC score
                auc_score = roc_auc_score(y, ensemble_pred)
                
                # Secondary objectives: diversity and stability
                # Diversity penalty: prefer more balanced weights
                diversity_penalty = -np.sum(weights ** 2)  # Entropy-like measure
                
                # Stability penalty: avoid extreme weights
                stability_penalty = -np.max(weights)
                
                # Multi-objective combination
                # 70% AUC, 20% diversity, 10% stability
                combined_score = (auc_score * 0.7 + 
                                diversity_penalty * 0.2 + 
                                stability_penalty * 0.1)
                
                return -combined_score  # Minimize negative = maximize positive
            except:
                return 1.0  # Bad score if calculation fails
        
        # Initial equal weights
        initial_weights = np.ones(len(model_names)) / len(model_names)
        
        # Constraints: weights must be positive and sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(len(model_names))]
        
        # Optimize with risk-aware objective
        result = minimize(
            risk_adjusted_objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimized_weights = result.x / result.x.sum()  # Normalize
            return dict(zip(model_names, optimized_weights))
        else:
            logging.warning("Weight optimization failed, using equal weights")
            return dict(zip(model_names, initial_weights))
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels using ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        proba = self.predict_ensemble_proba(X)
        return (proba > 0.5).astype(int)
    
    def predict_ensemble_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using weighted ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        ensemble_pred = np.zeros(len(X))
        
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]
            weight = self.weights.get(name, 0)
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification
        
        Returns:
            predictions: Ensemble predictions
            uncertainty: Prediction uncertainty (standard deviation across models)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        model_predictions = []
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]
            model_predictions.append(pred)
        
        model_predictions = np.array(model_predictions)
        
        # Weighted ensemble prediction
        ensemble_pred = self.predict_ensemble_proba(X)
        
        # Calculate uncertainty as weighted standard deviation
        weights_array = np.array([self.weights.get(name, 0) for name in self.models.keys()])
        
        # Weighted variance calculation
        weighted_mean = ensemble_pred
        weighted_var = np.average(
            (model_predictions - weighted_mean[np.newaxis, :])**2, 
            weights=weights_array, 
            axis=0
        )
        uncertainty = np.sqrt(weighted_var)
        
        return ensemble_pred, uncertainty
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn-compatible probability prediction method"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        proba_positive = self.predict_ensemble_proba(X)
        proba_negative = 1 - proba_positive
        return np.column_stack([proba_negative, proba_positive])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn-compatible binary prediction method"""
        return self.predict_ensemble(X)
    
    def _calculate_ensemble_feature_importance(self) -> Dict:
        """Calculate weighted ensemble feature importance"""
        if not self.feature_importance:
            return {}
        
        # Get feature names (assuming all models have same features)
        first_model = list(self.feature_importance.keys())[0]
        n_features = len(self.feature_importance[first_model])
        
        ensemble_importance = np.zeros(n_features)
        
        for name, importance in self.feature_importance.items():
            weight = self.weights.get(name, 0)
            ensemble_importance += weight * importance
        
        return ensemble_importance
    
    def get_model_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation between model predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before correlation analysis")
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)[:, 1]
        
        pred_df = pd.DataFrame(predictions)
        return pred_df.corr()


class FeatureStore:
    """
    Feature drift detection and quality monitoring system
    """
    
    def __init__(self):
        self.baseline_stats = {}
        self.is_initialized = False
        self.drift_threshold = 0.10  # KS test p-value threshold (reduced sensitivity from 0.05)
        self.quality_threshold = 0.95  # Minimum quality score
        self.feature_importance_weights = {}  # For feature-specific thresholds
        
    def initialize_baseline(self, X: pd.DataFrame) -> Dict:
        """
        Initialize baseline statistics for drift detection
        
        Args:
            X: Training feature matrix
            
        Returns:
            Dictionary with baseline statistics
        """
        logging.info("Initializing feature baseline statistics...")
        
        self.baseline_stats = {}
        
        for column in X.columns:
            if X[column].dtype in ['int64', 'float64']:
                stats_dict = {
                    'mean': X[column].mean(),
                    'std': X[column].std(),
                    'min': X[column].min(),
                    'max': X[column].max(),
                    'q25': X[column].quantile(0.25),
                    'q50': X[column].quantile(0.50),
                    'q75': X[column].quantile(0.75),
                    'skewness': X[column].skew(),
                    'kurtosis': X[column].kurtosis(),
                    'missing_rate': X[column].isnull().mean()
                }
                self.baseline_stats[column] = stats_dict
        
        self.is_initialized = True
        logging.info(f"Baseline initialized for {len(self.baseline_stats)} features")
        
        return self.baseline_stats
    
    def set_feature_importance_weights(self, importance_dict: Dict[str, float]):
        """Set feature importance weights for drift detection sensitivity"""
        # Normalize importance weights to 0-1 range
        if importance_dict:
            max_importance = max(importance_dict.values())
            self.feature_importance_weights = {
                feature: importance / max_importance 
                for feature, importance in importance_dict.items()
            }
    
    def detect_drift(self, X_new: pd.DataFrame) -> Dict:
        """
        Detect feature drift using statistical tests
        
        Args:
            X_new: New feature data to compare against baseline
            
        Returns:
            Dictionary with drift detection results
        """
        if not self.is_initialized:
            raise ValueError("Baseline must be initialized before drift detection")
        
        drift_results = {
            'drifted_features': [],
            'drift_scores': {},
            'mean_shifts': {},
            'ks_statistics': {},
            'p_values': {},
            'overall_drift_score': 0.0
        }
        
        total_features = 0
        total_drift_score = 0.0
        
        for column in X_new.columns:
            if column not in self.baseline_stats:
                continue
                
            baseline = self.baseline_stats[column]
            current_data = X_new[column].dropna()
            
            if len(current_data) == 0:
                continue
            
            total_features += 1
            
            # Calculate mean shift in standard deviations
            mean_shift = abs(current_data.mean() - baseline['mean']) / baseline['std']
            drift_results['mean_shifts'][column] = mean_shift
            
            # Kolmogorov-Smirnov test for distribution drift
            # Generate baseline sample for comparison
            baseline_sample = np.random.normal(
                baseline['mean'], 
                baseline['std'], 
                size=min(1000, len(current_data))
            )
            
            try:
                ks_stat, p_value = stats.ks_2samp(baseline_sample, current_data)
                drift_results['ks_statistics'][column] = ks_stat
                drift_results['p_values'][column] = p_value
                
                # Feature-level drift score (0-1, higher = more drift)
                drift_score = min(1.0, ks_stat + mean_shift * 0.1)
                drift_results['drift_scores'][column] = drift_score
                total_drift_score += drift_score
                
                # Check if drift is significant with feature-specific thresholds
                # High importance features get stricter thresholds
                importance_weight = self.feature_importance_weights.get(column, 0.5)
                adjusted_threshold = self.drift_threshold * (1 - importance_weight * 0.5)
                adjusted_mean_shift_threshold = 2.0 * (1 - importance_weight * 0.3)
                
                if p_value < adjusted_threshold or mean_shift > adjusted_mean_shift_threshold:
                    drift_results['drifted_features'].append(column)
                    
            except Exception as e:
                logging.warning(f"Drift detection failed for {column}: {e}")
                continue
        
        # Overall drift score (percentage)
        if total_features > 0:
            drift_results['overall_drift_score'] = (total_drift_score / total_features) * 100
        
        logging.info(f"Drift detection completed. {len(drift_results['drifted_features'])} features drifted")
        return drift_results
    
    def validate_feature_quality(self, X: pd.DataFrame) -> Dict:
        """
        Validate feature quality and data integrity
        
        Args:
            X: Feature matrix to validate
            
        Returns:
            Dictionary with quality metrics
        """
        quality_results = {
            'total_features': len(X.columns),
            'missing_values': {},
            'constant_features': [],
            'outlier_rates': {},
            'data_types': {},
            'quality_score': 0.0,
            'issues': []
        }
        
        total_quality_score = 0.0
        
        for column in X.columns:
            # Missing values
            missing_rate = X[column].isnull().mean()
            quality_results['missing_values'][column] = missing_rate
            
            # Data type
            quality_results['data_types'][column] = str(X[column].dtype)
            
            # Constant features
            if X[column].nunique() <= 1:
                quality_results['constant_features'].append(column)
                quality_results['issues'].append(f"{column}: Constant feature")
                continue
            
            # Outlier detection (for numeric features)
            if X[column].dtype in ['int64', 'float64']:
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (X[column] < (Q1 - 1.5 * IQR)) | (X[column] > (Q3 + 1.5 * IQR))
                outlier_rate = outlier_mask.mean()
                quality_results['outlier_rates'][column] = outlier_rate
                
                # Feature quality score (0-1)
                feature_quality = 1.0 - missing_rate - min(0.5, outlier_rate)
                total_quality_score += max(0, feature_quality)
                
                # Quality issues
                if missing_rate > 0.1:
                    quality_results['issues'].append(f"{column}: High missing rate ({missing_rate:.1%})")
                if outlier_rate > 0.2:
                    quality_results['issues'].append(f"{column}: High outlier rate ({outlier_rate:.1%})")
        
        # Overall quality score
        valid_features = len(X.columns) - len(quality_results['constant_features'])
        if valid_features > 0:
            quality_results['quality_score'] = (total_quality_score / valid_features) * 100
        
        logging.info(f"Feature quality validation completed. Quality score: {quality_results['quality_score']:.1f}%")
        return quality_results
    
    def generate_drift_report(self, drift_results: Dict, quality_results: Dict) -> str:
        """Generate human-readable drift and quality report"""
        
        report = []
        report.append("=== FEATURE DRIFT & QUALITY REPORT ===\n")
        
        # Quality Summary
        report.append(f"Feature Validation - Quality Score: {quality_results['quality_score']:.2f}%\n")
        
        if quality_results['issues']:
            report.append("Quality Issues:")
            for issue in quality_results['issues'][:5]:  # Show top 5 issues
                report.append(f"  • {issue}")
            if len(quality_results['issues']) > 5:
                report.append(f"  • ... and {len(quality_results['issues']) - 5} more issues")
            report.append("")
        
        # Drift Summary
        report.append("Drift Detection:")
        report.append(f"  Drifted Features: {len(drift_results['drifted_features'])}")
        report.append(f"  Severity Score: {drift_results['overall_drift_score']:.2f}%")
        report.append("")
        
        if drift_results['drifted_features']:
            report.append("Details:")
            for feature in drift_results['drifted_features'][:10]:  # Show top 10
                mean_shift = drift_results['mean_shifts'].get(feature, 0)
                p_value = drift_results['p_values'].get(feature, 1)
                status = "DRIFT DETECTED" if p_value < self.drift_threshold else "MEAN SHIFT"
                report.append(f"  {feature}: mean_shift = {mean_shift:.2f}σ ({status})")
            
            if len(drift_results['drifted_features']) > 10:
                report.append(f"  ... and {len(drift_results['drifted_features']) - 10} more drifted features")
        else:
            report.append("No significant drift detected.")
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Create sample training data
    n_samples = 1000
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = np.random.binomial(1, 0.3, n_samples)
    
    # Test ensemble trainer
    print("Testing Ensemble Model Trainer...")
    ensemble = EnsembleModelTrainer()
    results = ensemble.fit(X_train, y_train)
    
    print("Ensemble Scores:")
    for model, score in results['individual_scores'].items():
        print(f"  {model:<15}: AUC {score:.4f}")
    
    print(f"\nEnsemble Score: AUC {results['ensemble_score']:.4f}")
    
    print("\nOptimized Weights:")
    for model, weight in results['optimized_weights'].items():
        print(f"  {model:<15}: {weight:.4f}")
    
    # Test predictions with uncertainty
    predictions, uncertainty = ensemble.predict_with_uncertainty(X_train[:10])
    print(f"\nSample predictions with uncertainty:")
    for i in range(5):
        print(f"  Sample {i}: pred={predictions[i]:.3f}, uncertainty={uncertainty[i]:.3f}")
    
    # Test feature store
    print("\n" + "="*50)
    print("Testing Feature Store...")
    
    feature_store = FeatureStore()
    baseline = feature_store.initialize_baseline(X_train)
    
    # Create drifted data
    X_drifted = X_train.copy()
    X_drifted['feature_0'] += 2.0  # Add significant drift
    X_drifted['feature_1'] *= 1.5  # Scale drift
    
    # Test quality validation
    quality_results = feature_store.validate_feature_quality(X_drifted)
    print(f"Quality Score: {quality_results['quality_score']:.1f}%")
    
    # Test drift detection
    drift_results = feature_store.detect_drift(X_drifted)
    
    # Generate report
    report = feature_store.generate_drift_report(drift_results, quality_results)
    print("\n" + report)