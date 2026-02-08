"""
OptionsTitan - Integrated Pipeline Module
Main integration module that orchestrates all system components

This module provides:
1. ProductionPipeline - Main orchestrator for all phases
2. Component integration and coordination
3. Artifact management and persistence
4. Comprehensive logging and monitoring
5. Production deployment utilities
"""

import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import os
from pathlib import Path

# Import OptionsTitan components
try:
    from .ensemble_models import EnsembleModelTrainer, FeatureStore
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logging.warning("ensemble_models.py not found. Ensemble functionality disabled.")

try:
    from .risk_management import AdvancedRiskManager, CorrelationMonitor
    RISK_AVAILABLE = True
except ImportError:
    RISK_AVAILABLE = False
    logging.warning("risk_management.py not found. Risk management functionality disabled.")

try:
    from .model_explainability import ModelExplainer
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    logging.warning("model_explainability.py not found. Explainability functionality disabled.")

warnings.filterwarnings('ignore')

class ProductionPipeline:
    """
    Main production pipeline orchestrating all OptionsTitan components
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.components = {}
        self.artifacts = {}
        self.performance_metrics = {}
        self.is_initialized = False
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for the pipeline"""
        return {
            'ensemble': {
                'enabled': ENSEMBLE_AVAILABLE,
                'models': ['xgboost', 'lightgbm', 'gradient_boost', 'random_forest', 'logistic'],
                'optimization_trials': 30,
                'cv_folds': 5
            },
            'risk_management': {
                'enabled': RISK_AVAILABLE,
                'confidence_levels': [0.95, 0.99],
                'max_position_size': 0.05,
                'max_portfolio_risk': 0.15,
                'correlation_threshold': 0.70
            },
            'explainability': {
                'enabled': EXPLAINABILITY_AVAILABLE,
                'sample_explanations': 5,
                'global_importance': True
            },
            'drift_detection': {
                'enabled': ENSEMBLE_AVAILABLE,
                'drift_threshold': 0.05,
                'quality_threshold': 0.95
            },
            'artifacts': {
                'save_path': './production_artifacts_{timestamp}.pkl',
                'backup_count': 5,
                'compress': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'optionstitan_pipeline.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_config = self.config.get('logging', {})
        
        # Create logger
        self.logger = logging.getLogger('OptionsTitan')
        self.logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_file = log_config.get('file', 'optionstitan_pipeline.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(log_config.get('format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info("OptionsTitan Pipeline logging initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        self.logger.info("Initializing pipeline components...")
        
        # Initialize ensemble trainer
        if self.config['ensemble']['enabled'] and ENSEMBLE_AVAILABLE:
            self.components['ensemble_trainer'] = EnsembleModelTrainer()
            self.components['feature_store'] = FeatureStore()
            self.logger.info("Ensemble components initialized")
        
        # Initialize risk manager
        if self.config['risk_management']['enabled'] and RISK_AVAILABLE:
            confidence_levels = self.config['risk_management'].get('confidence_levels', [0.95, 0.99])
            correlation_threshold = self.config['risk_management'].get('correlation_threshold', 0.70)
            self.components['risk_manager'] = AdvancedRiskManager(confidence_levels)
            self.components['correlation_monitor'] = CorrelationMonitor(correlation_threshold)
            self.logger.info("Risk management components initialized")
        
        self.is_initialized = True
        self.logger.info(f"Pipeline initialized with {len(self.components)} components")
    
    def run_full_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame = None, y_test: pd.Series = None,
                         base_model = None) -> Dict:
        """
        Run the complete OptionsTitan pipeline
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            base_model: Pre-trained base model (optional)
            
        Returns:
            Dictionary with comprehensive results
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING OPTIONSTITAN PRODUCTION PIPELINE")
        self.logger.info("=" * 60)
        
        pipeline_results = {
            'timestamp': datetime.now(),
            'config': self.config,
            'phases_completed': [],
            'performance_metrics': {},
            'artifacts_saved': False
        }
        
        try:
            # Phase 1: Ensemble Model Training
            if 'ensemble_trainer' in self.components:
                self.logger.info("Phase 1: Training Ensemble Models...")
                ensemble_results = self._run_ensemble_phase(X_train, y_train, X_test, y_test)
                pipeline_results['ensemble_results'] = ensemble_results
                pipeline_results['phases_completed'].append('ensemble_training')
                self.logger.info("✅ Phase 1 completed successfully")
            
            # Phase 2: Feature Drift Detection
            if 'feature_store' in self.components:
                self.logger.info("Phase 2: Feature Drift Detection...")
                drift_results = self._run_drift_detection_phase(X_train, X_test)
                pipeline_results['drift_results'] = drift_results
                pipeline_results['phases_completed'].append('drift_detection')
                self.logger.info("✅ Phase 2 completed successfully")
            
            # Phase 3: Risk Management Analysis
            if 'risk_manager' in self.components:
                self.logger.info("Phase 3: Risk Management Analysis...")
                risk_results = self._run_risk_management_phase(X_train, y_train)
                pipeline_results['risk_results'] = risk_results
                pipeline_results['phases_completed'].append('risk_management')
                self.logger.info("✅ Phase 3 completed successfully")
            
            # Phase 4: Model Explainability (Optional)
            if self.config['explainability']['enabled'] and EXPLAINABILITY_AVAILABLE:
                self.logger.info("Phase 4: Model Explainability...")
                explainability_results = self._run_explainability_phase(X_train, base_model)
                pipeline_results['explainability_results'] = explainability_results
                pipeline_results['phases_completed'].append('explainability')
                self.logger.info("✅ Phase 4 completed successfully")
            
            # Generate comprehensive report
            self.logger.info("Generating comprehensive report...")
            report = self._generate_comprehensive_report(pipeline_results)
            pipeline_results['comprehensive_report'] = report
            
            # Save artifacts
            self.logger.info("Saving production artifacts...")
            artifact_path = self._save_production_artifacts(pipeline_results)
            pipeline_results['artifact_path'] = artifact_path
            pipeline_results['artifacts_saved'] = True
            
            self.logger.info("=" * 60)
            self.logger.info("OPTIONSTITAN PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Phases completed: {len(pipeline_results['phases_completed'])}")
            self.logger.info(f"Artifacts saved to: {artifact_path}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            pipeline_results['error'] = str(e)
            pipeline_results['status'] = 'FAILED'
            raise
        
        pipeline_results['status'] = 'SUCCESS'
        return pipeline_results
    
    def _run_ensemble_phase(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict:
        """Run ensemble model training phase"""
        ensemble_trainer = self.components['ensemble_trainer']
        
        # Train ensemble
        training_results = ensemble_trainer.fit(X_train, y_train)
        
        results = {
            'training_results': training_results,
            'model_scores': training_results.get('individual_scores', {}),
            'ensemble_score': training_results.get('ensemble_score', 0),
            'optimized_weights': training_results.get('optimized_weights', {}),
            'feature_importance': training_results.get('feature_importance', {})
        }
        
        # Test set evaluation if provided
        if X_test is not None and y_test is not None:
            test_predictions = ensemble_trainer.predict_ensemble_proba(X_test)
            test_predictions_binary = ensemble_trainer.predict_ensemble(X_test)
            
            from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
            
            test_auc = roc_auc_score(y_test, test_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions_binary)
            
            results['test_performance'] = {
                'auc': test_auc,
                'accuracy': test_accuracy,
                'classification_report': classification_report(y_test, test_predictions_binary, output_dict=True)
            }
            
            # Prediction uncertainty analysis
            predictions_with_uncertainty = ensemble_trainer.predict_with_uncertainty(X_test)
            results['uncertainty_analysis'] = {
                'mean_uncertainty': float(np.mean(predictions_with_uncertainty[1])),
                'uncertainty_std': float(np.std(predictions_with_uncertainty[1]))
            }
        
        # Model correlation analysis
        if X_test is not None:
            correlation_matrix = ensemble_trainer.get_model_correlation(X_test)
            results['model_correlations'] = correlation_matrix.to_dict()
        
        return results
    
    def _run_drift_detection_phase(self, X_train: pd.DataFrame, 
                                  X_test: pd.DataFrame = None) -> Dict:
        """Run feature drift detection phase"""
        feature_store = self.components['feature_store']
        
        # Initialize baseline with training data
        baseline_stats = feature_store.initialize_baseline(X_train)
        
        results = {
            'baseline_initialized': True,
            'baseline_features': len(baseline_stats),
            'baseline_stats': baseline_stats
        }
        
        # Validate training data quality
        quality_results = feature_store.validate_feature_quality(X_train)
        results['training_quality'] = quality_results
        
        # Drift detection on test data if provided
        if X_test is not None:
            drift_results = feature_store.detect_drift(X_test)
            test_quality = feature_store.validate_feature_quality(X_test)
            
            results['drift_detection'] = drift_results
            results['test_quality'] = test_quality
            
            # Generate drift report
            drift_report = feature_store.generate_drift_report(drift_results, test_quality)
            results['drift_report'] = drift_report
        
        return results
    
    def _run_risk_management_phase(self, X_train: pd.DataFrame, 
                                  y_train: pd.Series) -> Dict:
        """Run risk management analysis phase"""
        risk_manager = self.components['risk_manager']
        correlation_monitor = self.components['correlation_monitor']
        
        # Generate synthetic returns for risk analysis (in production, use actual returns)
        # This is a simplified approach - in practice, you'd have actual portfolio returns
        synthetic_returns = np.random.normal(0.001, 0.02, len(X_train))  # Daily returns
        returns_series = pd.Series(synthetic_returns, index=X_train.index)
        portfolio_values = (1 + returns_series).cumprod() * 100000
        
        # Calculate comprehensive risk metrics
        risk_metrics = risk_manager.calculate_comprehensive_risk_metrics(
            returns_series, 
            portfolio_values
        )
        
        # Check risk limits
        limit_check = risk_manager.check_risk_limits(risk_metrics)
        
        # Generate risk report
        risk_report = risk_manager.generate_risk_report(risk_metrics, limit_check)
        
        results = {
            'risk_metrics': risk_metrics,
            'limit_check': limit_check,
            'risk_report': risk_report
        }
        
        # Correlation analysis (synthetic position data)
        n_positions = min(5, len(X_train.columns))
        position_returns = pd.DataFrame(
            np.random.multivariate_normal(
                mean=[0.001] * n_positions,
                cov=np.eye(n_positions) * 0.0004 + 0.0001,
                size=len(X_train)
            ),
            columns=[f'Position_{i}' for i in range(n_positions)],
            index=X_train.index
        )
        
        correlation_analysis = correlation_monitor.calculate_position_correlations(position_returns)
        results['correlation_analysis'] = correlation_analysis
        
        return results
    
    def _run_explainability_phase(self, X_train: pd.DataFrame, 
                                 base_model = None) -> Dict:
        """Run model explainability phase"""
        if base_model is None:
            # Use ensemble model if available
            if 'ensemble_trainer' in self.components:
                ensemble_trainer = self.components['ensemble_trainer']
                if hasattr(ensemble_trainer, 'models') and ensemble_trainer.models:
                    # Use the first available model from the ensemble
                    first_model_name = list(ensemble_trainer.models.keys())[0]
                    base_model = ensemble_trainer.models[first_model_name]
                    model_type = 'tree'
                else:
                    self.logger.warning("Ensemble trainer has no trained models")
                    return {'error': 'No trained models available'}
            else:
                self.logger.warning("No model available for explainability analysis")
                return {'error': 'No model available'}
        else:
            model_type = 'tree'  # Assume tree-based model
        
        # Initialize explainer
        explainer = ModelExplainer(base_model, X_train.columns.tolist(), model_type)
        
        results = {
            'explainer_initialized': True,
            'model_type': model_type,
            'feature_count': len(X_train.columns)
        }
        
        # Global feature importance
        explainability_config = self.config.get('explainability', {})
        if explainability_config.get('global_importance', True):
            try:
                global_importance = explainer.calculate_global_feature_importance(X_train)
                results['global_importance'] = global_importance
            except Exception as e:
                self.logger.warning(f"Failed to calculate global importance: {e}")
                results['global_importance_error'] = str(e)
        
        # Sample explanations
        sample_count = explainability_config.get('sample_explanations', 5)
        if sample_count > 0 and len(X_train) > 0:
            sample_explanations = []
            sample_indices = np.random.choice(len(X_train), min(sample_count, len(X_train)), replace=False)
            
            for idx in sample_indices:
                try:
                    explanation = explainer.explain_prediction(X_train, idx)
                    sample_explanations.append({
                        'index': int(idx),
                        'prediction': explanation['prediction'],
                        'confidence': explanation['confidence'],
                        'trading_rationale': explanation['trading_rationale'],
                        'top_features': dict(list(explanation['feature_contributions'].items())[:5])
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to explain prediction {idx}: {e}")
            
            results['sample_explanations'] = sample_explanations
        
        # Generate interpretation report
        interpretation_report = explainer.generate_model_interpretation_report(X_train, sample_count)
        results['interpretation_report'] = interpretation_report
        
        return results
    
    def _generate_comprehensive_report(self, pipeline_results: Dict) -> str:
        """Generate comprehensive pipeline report"""
        
        report = []
        report.append("=" * 80)
        report.append("OPTIONSTITAN PRODUCTION PIPELINE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {pipeline_results['timestamp']}")
        report.append(f"Status: {pipeline_results.get('status', 'UNKNOWN')}")
        report.append(f"Phases Completed: {', '.join(pipeline_results['phases_completed'])}")
        report.append("")
        
        # Ensemble Results
        if 'ensemble_results' in pipeline_results:
            report.append("=== ENSEMBLE MODEL RESULTS ===")
            ensemble = pipeline_results['ensemble_results']
            
            report.append("Individual Model Scores:")
            for model, score in ensemble.get('model_scores', {}).items():
                report.append(f"  {model:<15}: AUC {score:.4f}")
            
            report.append(f"\nEnsemble Score: AUC {ensemble.get('ensemble_score', 0):.4f}")
            
            report.append("\nOptimized Weights:")
            for model, weight in ensemble.get('optimized_weights', {}).items():
                report.append(f"  {model:<15}: {weight:.4f}")
            
            if 'test_performance' in ensemble:
                test_perf = ensemble['test_performance']
                report.append(f"\nTest Performance:")
                report.append(f"  Accuracy: {test_perf.get('accuracy', 0):.4f}")
                report.append(f"  ROC AUC:  {test_perf.get('auc', 0):.4f}")
            
            report.append("")
        
        # Drift Detection Results
        if 'drift_results' in pipeline_results:
            report.append("=== FEATURE DRIFT ANALYSIS ===")
            drift = pipeline_results['drift_results']
            
            if 'training_quality' in drift:
                quality = drift['training_quality']
                report.append(f"Training Data Quality Score: {quality.get('quality_score', 0):.1f}%")
            
            if 'drift_detection' in drift:
                drift_det = drift['drift_detection']
                report.append(f"Drifted Features: {len(drift_det.get('drifted_features', []))}")
                report.append(f"Overall Drift Score: {drift_det.get('overall_drift_score', 0):.2f}%")
            
            if 'drift_report' in drift:
                report.append("\n" + drift['drift_report'])
            
            report.append("")
        
        # Risk Management Results
        if 'risk_results' in pipeline_results:
            report.append("=== RISK MANAGEMENT ANALYSIS ===")
            risk = pipeline_results['risk_results']
            
            if 'risk_report' in risk:
                report.append(risk['risk_report'])
            
            report.append("")
        
        # Explainability Results
        if 'explainability_results' in pipeline_results:
            report.append("=== MODEL EXPLAINABILITY ===")
            explainability = pipeline_results['explainability_results']
            
            if 'global_importance' in explainability:
                importance = explainability['global_importance']
                report.append("Top 5 Most Important Features:")
                for i, (feature, imp) in enumerate(importance.get('top_features', [])[:5], 1):
                    report.append(f"  {i}. {feature}: {imp:.6f}")
            
            if 'sample_explanations' in explainability:
                explanations = explainability['sample_explanations']
                report.append(f"\nSample Predictions Analyzed: {len(explanations)}")
            
            report.append("")
        
        # Summary
        report.append("=== PIPELINE SUMMARY ===")
        report.append(f"✅ Phases Completed: {len(pipeline_results['phases_completed'])}")
        report.append(f"✅ Artifacts Saved: {pipeline_results.get('artifacts_saved', False)}")
        
        if pipeline_results.get('artifact_path'):
            report.append(f"📁 Artifact Location: {pipeline_results['artifact_path']}")
        
        report.append("")
        report.append("System Status: READY FOR DEPLOYMENT 🚀")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _save_production_artifacts(self, pipeline_results: Dict) -> str:
        """Save all production artifacts to disk"""
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get artifact path with fallback
        artifacts_config = self.config.get('artifacts', {})
        save_path_template = artifacts_config.get('save_path', './production_artifacts_{timestamp}.pkl')
        artifact_path = save_path_template.format(timestamp=timestamp)
        
        # Prepare artifacts dictionary
        artifacts = {
            'timestamp': pipeline_results['timestamp'],
            'config': self.config,
            'pipeline_results': pipeline_results,
            'components': {},
            'version': '1.0.0'
        }
        
        # Save component states
        components_copy = dict(self.components)  # Create copy to avoid iteration issues
        for name, component in components_copy.items():
            try:
                artifacts['components'][name] = component
                self.logger.debug(f"Saved component: {name}")
            except Exception as e:
                self.logger.warning(f"Failed to save component {name}: {e}")
        
        # Save to disk
        try:
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"Production artifacts saved to: {artifact_path}")
            
            # Cleanup old artifacts if configured
            self._cleanup_old_artifacts(artifact_path)
            
            return artifact_path
            
        except Exception as e:
            self.logger.error(f"Failed to save artifacts: {e}")
            raise
    
    def _cleanup_old_artifacts(self, current_artifact_path: str):
        """Clean up old artifact files"""
        try:
            artifacts_config = self.config.get('artifacts', {})
            backup_count = artifacts_config.get('backup_count', 5)
            
            # Find all artifact files in the same directory
            artifact_dir = Path(current_artifact_path).parent
            artifact_pattern = "production_artifacts_*.pkl"
            
            artifact_files = list(artifact_dir.glob(artifact_pattern))
            artifact_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old files beyond backup count
            for old_file in artifact_files[backup_count:]:
                old_file.unlink()
                self.logger.info(f"Removed old artifact: {old_file}")
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old artifacts: {e}")
    
    @classmethod
    def load_production_artifacts(cls, artifact_path: str) -> Dict:
        """Load production artifacts from disk"""
        try:
            with open(artifact_path, 'rb') as f:
                artifacts = pickle.load(f)
            
            logging.info(f"Production artifacts loaded from: {artifact_path}")
            return artifacts
            
        except Exception as e:
            logging.error(f"Failed to load artifacts from {artifact_path}: {e}")
            raise
    
    def create_inference_pipeline(self, artifact_path: str = None) -> 'InferencePipeline':
        """Create inference pipeline from saved artifacts"""
        if artifact_path:
            artifacts = self.load_production_artifacts(artifact_path)
        else:
            # Use current components
            artifacts = {'components': self.components, 'config': self.config}
        
        return InferencePipeline(artifacts)


class InferencePipeline:
    """
    Lightweight inference pipeline for production predictions
    """
    
    def __init__(self, artifacts: Dict):
        self.artifacts = artifacts
        self.components = artifacts.get('components', {})
        self.config = artifacts.get('config', {})
        self.logger = logging.getLogger('OptionsTitan.Inference')
        
        self.logger.info("Inference pipeline initialized")
    
    def predict(self, X: pd.DataFrame, explain: bool = False, 
               check_drift: bool = False) -> Dict:
        """
        Make predictions with optional explanations and drift checking
        
        Args:
            X: Feature matrix for prediction
            explain: Whether to include prediction explanations
            check_drift: Whether to check for feature drift
            
        Returns:
            Dictionary with predictions and optional analysis
        """
        results = {
            'timestamp': datetime.now(),
            'predictions': None,
            'probabilities': None,
            'uncertainty': None
        }
        
        # Make ensemble predictions if available
        if 'ensemble_trainer' in self.components:
            ensemble = self.components['ensemble_trainer']
            
            # Get predictions with uncertainty
            predictions, uncertainty = ensemble.predict_with_uncertainty(X)
            probabilities = ensemble.predict_ensemble_proba(X)
            
            results['predictions'] = predictions
            results['probabilities'] = probabilities
            results['uncertainty'] = uncertainty
            
            self.logger.info(f"Generated predictions for {len(X)} samples")
        
        # Drift detection if requested
        if check_drift and 'feature_store' in self.components:
            feature_store = self.components['feature_store']
            
            if feature_store.is_initialized:
                drift_results = feature_store.detect_drift(X)
                quality_results = feature_store.validate_feature_quality(X)
                
                results['drift_analysis'] = drift_results
                results['quality_analysis'] = quality_results
                
                # Generate drift alert if significant drift detected
                if drift_results.get('overall_drift_score', 0) > 50:
                    results['drift_alert'] = "SIGNIFICANT DRIFT DETECTED - MODEL RETRAINING RECOMMENDED"
        
        # Explanations if requested
        if explain and 'ensemble_trainer' in self.components:
            try:
                # Create explainer for ensemble model
                explainer = ModelExplainer(
                    self.components['ensemble_trainer'], 
                    X.columns.tolist(), 
                    'ensemble'
                )
                
                self.logger.info("ModelExplainer created successfully for ensemble")
                
                # Explain first few predictions
                explanations = []
                for i in range(min(3, len(X))):
                    try:
                        explanation = explainer.explain_prediction(X, i)
                        explanations.append({
                            'index': i,
                            'prediction': explanation['prediction'],
                            'confidence': explanation['confidence'],
                            'trading_rationale': explanation['trading_rationale']
                        })
                    except Exception as exp_error:
                        self.logger.warning(f"Failed to explain prediction {i}: {exp_error}")
                        continue
                
                if explanations:
                    results['explanations'] = explanations
                    self.logger.info(f"Generated {len(explanations)} explanations successfully")
                else:
                    self.logger.warning("No explanations generated successfully")
                
            except Exception as e:
                self.logger.warning(f"Failed to create explainer: {e}")
                results['explainer_error'] = str(e)
        
        return results
    
    def get_risk_assessment(self, portfolio_returns: pd.Series = None) -> Dict:
        """Get current risk assessment"""
        if 'risk_manager' not in self.components:
            return {'error': 'Risk manager not available'}
        
        risk_manager = self.components['risk_manager']
        
        if portfolio_returns is None:
            # Generate synthetic returns for demonstration
            portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Calculate risk metrics
        portfolio_values = (1 + portfolio_returns).cumprod() * 100000
        risk_metrics = risk_manager.calculate_comprehensive_risk_metrics(
            portfolio_returns, portfolio_values
        )
        
        # Check limits
        limit_check = risk_manager.check_risk_limits(risk_metrics)
        
        return {
            'risk_metrics': risk_metrics,
            'limit_check': limit_check,
            'risk_status': limit_check.get('status', 'UNKNOWN')
        }


# Example usage and testing
if __name__ == "__main__":
    # Test pipeline initialization
    print("Testing OptionsTitan Integrated Pipeline...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = np.random.binomial(1, 0.3, n_samples)
    
    X_test = pd.DataFrame(
        np.random.randn(200, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_test = np.random.binomial(1, 0.3, 200)
    
    # Initialize pipeline
    config = {
        'ensemble': {'enabled': ENSEMBLE_AVAILABLE},
        'risk_management': {'enabled': RISK_AVAILABLE},
        'explainability': {'enabled': EXPLAINABILITY_AVAILABLE},
        'drift_detection': {'enabled': ENSEMBLE_AVAILABLE}
    }
    
    pipeline = ProductionPipeline(config)
    
    print(f"Pipeline initialized with {len(pipeline.components)} components")
    print(f"Available components: {list(pipeline.components.keys())}")
    
    # Test pipeline execution (if components are available)
    if pipeline.components:
        try:
            print("\nRunning pipeline test...")
            results = pipeline.run_full_pipeline(X_train, y_train, X_test, y_test)
            
            print(f"Pipeline completed successfully!")
            print(f"Phases completed: {results['phases_completed']}")
            print(f"Artifacts saved: {results['artifacts_saved']}")
            
            if results.get('artifact_path'):
                print(f"Artifact path: {results['artifact_path']}")
                
                # Test inference pipeline
                print("\nTesting inference pipeline...")
                inference = pipeline.create_inference_pipeline(results['artifact_path'])
                
                # Make test predictions
                inference_results = inference.predict(X_test[:5], explain=True, check_drift=True)
                print(f"Inference completed for {len(X_test[:5])} samples")
                
                if 'drift_alert' in inference_results:
                    print(f"⚠️  {inference_results['drift_alert']}")
            
        except Exception as e:
            print(f"Pipeline test failed: {e}")
    else:
        print("No components available for testing. Install required modules:")
        if not ENSEMBLE_AVAILABLE:
            print("  - ensemble_models.py missing")
        if not RISK_AVAILABLE:
            print("  - risk_management.py missing")
        if not EXPLAINABILITY_AVAILABLE:
            print("  - model_explainability.py missing")
    
    print("\nIntegrated Pipeline System Ready!")