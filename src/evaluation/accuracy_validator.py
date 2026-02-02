"""
Модуль валидации точности сжатых ML-моделей для криптотрейдинга.
Специализированные метрики и тесты для финансовых временных рядов.

Domain-specific validation patterns для financial ML systems
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationMetric(Enum):
    """Типы метрик валидации"""
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2_score"
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    INFORMATION_RATIO = "information_ratio"
    VAR_95 = "var_95"
    CVAR_95 = "cvar_95"
    CORRELATION = "correlation"
    VOLATILITY_RATIO = "volatility_ratio"

class ValidationLevel(Enum):
    """Уровни валидации"""
    BASIC = "basic"              # Базовые метрики
    TRADING = "trading"          # Торговые метрики
    RISK = "risk"               # Risk management метрики
    COMPREHENSIVE = "comprehensive"  # Полная валидация

@dataclass
class ValidationThresholds:
    """Пороги для валидации"""
    min_accuracy_retention: float = 0.95
    max_mse_increase: float = 0.1
    min_directional_accuracy: float = 0.52
    min_correlation: float = 0.9
    max_volatility_change: float = 0.2
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.15

@dataclass
class ValidationResult:
    """Результат валидации модели"""
    model_name: str
    validation_level: ValidationLevel
    overall_passed: bool
    
    # Метрики
    metrics: Dict[str, float]
    
    # Результаты тестов
    test_results: Dict[str, bool]
    
    # Детальная информация
    failed_tests: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Статистическая значимость
    statistical_significance: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'model_name': self.model_name,
            'validation_level': self.validation_level.value,
            'overall_passed': self.overall_passed,
            'metrics': self.metrics,
            'test_results': self.test_results,
            'failed_tests': self.failed_tests,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'statistical_significance': self.statistical_significance
        }

class AccuracyValidator:
    """
    Специализированный валидатор точности для crypto trading моделей
    с focus на financial metrics и statistical significance
    """
    
    def __init__(self, 
                 thresholds: Optional[ValidationThresholds] = None,
                 confidence_level: float = 0.95):
        """
        Args:
            thresholds: Пороги валидации
            confidence_level: Уровень доверительного интервала
        """
        self.thresholds = thresholds or ValidationThresholds()
        self.confidence_level = confidence_level
        
        self.logger = logging.getLogger(f"{__name__}.AccuracyValidator")
        self.validation_history = []
    
    def validate_model_accuracy(self,
                               original_model: nn.Module,
                               compressed_model: nn.Module,
                               test_data: torch.utils.data.DataLoader,
                               validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
                               model_name: str = "compressed_model") -> ValidationResult:
        """
        Полная валидация точности сжатой модели
        
        Args:
            original_model: Исходная модель
            compressed_model: Сжатая модель
            test_data: Тестовые данные
            validation_level: Уровень валидации
            model_name: Имя модели для отчетов
            
        Returns:
            Результат валидации
        """
        self.logger.info(f"Начинаем валидацию модели {model_name} на уровне {validation_level.value}")
        
        # Получаем предсказания
        original_predictions, compressed_predictions, targets = self._get_model_predictions(
            original_model, compressed_model, test_data
        )
        
        if len(original_predictions) == 0:
            return ValidationResult(
                model_name=model_name,
                validation_level=validation_level,
                overall_passed=False,
                metrics={},
                test_results={},
                failed_tests=["No predictions obtained"],
                warnings=[],
                recommendations=["Check model compatibility with test data"],
                statistical_significance={}
            )
        
        # Вычисляем метрики
        metrics = self._calculate_validation_metrics(
            original_predictions, compressed_predictions, targets, validation_level
        )
        
        # Выполняем тесты
        test_results = self._run_validation_tests(
            original_predictions, compressed_predictions, targets, metrics, validation_level
        )
        
        # Статистическая значимость
        significance_tests = self._run_statistical_significance_tests(
            original_predictions, compressed_predictions, targets
        )
        
        # Анализ результатов
        overall_passed = all(test_results.values())
        failed_tests = [test for test, passed in test_results.items() if not passed]
        warnings = self._generate_warnings(metrics, test_results)
        recommendations = self._generate_recommendations(metrics, test_results, failed_tests)
        
        result = ValidationResult(
            model_name=model_name,
            validation_level=validation_level,
            overall_passed=overall_passed,
            metrics=metrics,
            test_results=test_results,
            failed_tests=failed_tests,
            warnings=warnings,
            recommendations=recommendations,
            statistical_significance=significance_tests
        )
        
        # Сохраняем в историю
        self.validation_history.append(result)
        
        self.logger.info(f"Валидация завершена. Результат: {'PASSED' if overall_passed else 'FAILED'}")
        if failed_tests:
            self.logger.warning(f"Неудавшиеся тесты: {failed_tests}")
        
        return result
    
    def _get_model_predictions(self,
                              original_model: nn.Module,
                              compressed_model: nn.Module,
                              test_data: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Получение предсказаний обеих моделей"""
        
        original_model.eval()
        compressed_model.eval()
        
        original_predictions = []
        compressed_predictions = []
        targets = []
        
        device = next(original_model.parameters()).device
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 200:  # Ограничиваем для скорости
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, batch_targets = batch
                    inputs = inputs.to(device)
                    batch_targets = batch_targets.to(device)
                else:
                    inputs = batch.to(device)
                    batch_targets = inputs  # Self-prediction task
                
                # Предсказания оригинальной модели
                orig_output = original_model(inputs)
                if isinstance(orig_output, dict):
                    orig_output = orig_output.get('trading_signal', orig_output)
                original_predictions.append(orig_output.cpu().numpy())
                
                # Предсказания сжатой модели
                comp_output = compressed_model(inputs)
                if isinstance(comp_output, dict):
                    comp_output = comp_output.get('trading_signal', comp_output)
                compressed_predictions.append(comp_output.cpu().numpy())
                
                # Targets
                targets.append(batch_targets.cpu().numpy())
        
        if not original_predictions:
            return np.array([]), np.array([]), np.array([])
        
        # Объединяем предсказания
        original_pred = np.concatenate(original_predictions, axis=0)
        compressed_pred = np.concatenate(compressed_predictions, axis=0)
        targets_array = np.concatenate(targets, axis=0)
        
        # Flatten если нужно
        if original_pred.ndim > 1:
            original_pred = original_pred.flatten()
        if compressed_pred.ndim > 1:
            compressed_pred = compressed_pred.flatten()
        if targets_array.ndim > 1:
            targets_array = targets_array.flatten()
        
        self.logger.info(f"Получены предсказания: {len(original_pred)} образцов")
        
        return original_pred, compressed_pred, targets_array
    
    def _calculate_validation_metrics(self,
                                    original_predictions: np.ndarray,
                                    compressed_predictions: np.ndarray,
                                    targets: np.ndarray,
                                    validation_level: ValidationLevel) -> Dict[str, float]:
        """Вычисление метрик валидации"""
        
        metrics = {}
        
        # Базовые метрики (всегда вычисляем)
        metrics.update(self._calculate_basic_metrics(
            original_predictions, compressed_predictions, targets
        ))
        
        # Торговые метрики
        if validation_level in [ValidationLevel.TRADING, ValidationLevel.COMPREHENSIVE]:
            metrics.update(self._calculate_trading_metrics(
                original_predictions, compressed_predictions, targets
            ))
        
        # Risk метрики
        if validation_level in [ValidationLevel.RISK, ValidationLevel.COMPREHENSIVE]:
            metrics.update(self._calculate_risk_metrics(
                original_predictions, compressed_predictions, targets
            ))
        
        return metrics
    
    def _calculate_basic_metrics(self,
                               original_predictions: np.ndarray,
                               compressed_predictions: np.ndarray,
                               targets: np.ndarray) -> Dict[str, float]:
        """Базовые метрики точности"""
        
        metrics = {}
        
        # MSE между оригинальными и сжатыми предсказаниями
        prediction_mse = mean_squared_error(original_predictions, compressed_predictions)
        metrics['prediction_mse'] = float(prediction_mse)
        
        # MSE относительно targets
        original_mse = mean_squared_error(targets, original_predictions)
        compressed_mse = mean_squared_error(targets, compressed_predictions)
        
        metrics['original_mse'] = float(original_mse)
        metrics['compressed_mse'] = float(compressed_mse)
        metrics['mse_ratio'] = float(compressed_mse / (original_mse + 1e-8))
        
        # MAE
        original_mae = mean_absolute_error(targets, original_predictions)
        compressed_mae = mean_absolute_error(targets, compressed_predictions)
        
        metrics['original_mae'] = float(original_mae)
        metrics['compressed_mae'] = float(compressed_mae)
        metrics['mae_ratio'] = float(compressed_mae / (original_mae + 1e-8))
        
        # RMSE
        metrics['original_rmse'] = float(np.sqrt(original_mse))
        metrics['compressed_rmse'] = float(np.sqrt(compressed_mse))
        
        # R² Score
        try:
            original_r2 = r2_score(targets, original_predictions)
            compressed_r2 = r2_score(targets, compressed_predictions)
            
            metrics['original_r2'] = float(original_r2)
            metrics['compressed_r2'] = float(compressed_r2)
            metrics['r2_difference'] = float(compressed_r2 - original_r2)
        except:
            metrics['original_r2'] = 0.0
            metrics['compressed_r2'] = 0.0
            metrics['r2_difference'] = 0.0
        
        # Корреляция между предсказаниями
        correlation = np.corrcoef(original_predictions, compressed_predictions)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        metrics['predictions_correlation'] = float(correlation)
        
        # Accuracy retention (на основе MSE)
        accuracy_retention = original_mse / (compressed_mse + 1e-8)
        if accuracy_retention > 1.0:
            accuracy_retention = 2.0 - accuracy_retention  # Инвертируем если сжатая лучше
        metrics['accuracy_retention'] = float(min(1.0, accuracy_retention))
        
        return metrics
    
    def _calculate_trading_metrics(self,
                                 original_predictions: np.ndarray,
                                 compressed_predictions: np.ndarray,
                                 targets: np.ndarray) -> Dict[str, float]:
        """Торговые метрики"""
        
        metrics = {}
        
        # Directional accuracy
        if len(original_predictions) > 1:
            orig_directions = np.sign(np.diff(original_predictions))
            comp_directions = np.sign(np.diff(compressed_predictions))
            target_directions = np.sign(np.diff(targets))
            
            # Directional accuracy для оригинальной модели
            orig_dir_acc = np.mean(orig_directions == target_directions)
            metrics['original_directional_accuracy'] = float(orig_dir_acc)
            
            # Directional accuracy для сжатой модели
            comp_dir_acc = np.mean(comp_directions == target_directions)
            metrics['compressed_directional_accuracy'] = float(comp_dir_acc)
            
            # Сохранение directional accuracy
            metrics['directional_accuracy_retention'] = float(comp_dir_acc / (orig_dir_acc + 1e-8))
        
        # Sharpe Ratio (упрощенный)
        if len(original_predictions) > 1:
            orig_returns = np.diff(original_predictions)
            comp_returns = np.diff(compressed_predictions)
            target_returns = np.diff(targets)
            
            # Sharpe ratio для оригинальных предсказаний
            orig_sharpe = np.mean(orig_returns) / (np.std(orig_returns) + 1e-8)
            metrics['original_sharpe_ratio'] = float(orig_sharpe)
            
            # Sharpe ratio для сжатых предсказаний
            comp_sharpe = np.mean(comp_returns) / (np.std(comp_returns) + 1e-8)
            metrics['compressed_sharpe_ratio'] = float(comp_sharpe)
            
            # Sharpe retention
            metrics['sharpe_retention'] = float(comp_sharpe / (orig_sharpe + 1e-8))
        
        # Hit ratio (процент правильных направлений)
        if len(original_predictions) > 1:
            orig_hits = np.mean((orig_directions * target_directions) > 0)
            comp_hits = np.mean((comp_directions * target_directions) > 0)
            
            metrics['original_hit_ratio'] = float(orig_hits)
            metrics['compressed_hit_ratio'] = float(comp_hits)
        
        return metrics
    
    def _calculate_risk_metrics(self,
                              original_predictions: np.ndarray,
                              compressed_predictions: np.ndarray,
                              targets: np.ndarray) -> Dict[str, float]:
        """Risk management метрики"""
        
        metrics = {}
        
        # Volatility analysis
        orig_vol = np.std(original_predictions)
        comp_vol = np.std(compressed_predictions)
        target_vol = np.std(targets)
        
        metrics['original_volatility'] = float(orig_vol)
        metrics['compressed_volatility'] = float(comp_vol)
        metrics['target_volatility'] = float(target_vol)
        metrics['volatility_ratio'] = float(comp_vol / (orig_vol + 1e-8))
        
        # VaR (Value at Risk) at 95% confidence level
        orig_var_95 = np.percentile(original_predictions, 5)
        comp_var_95 = np.percentile(compressed_predictions, 5)
        target_var_95 = np.percentile(targets, 5)
        
        metrics['original_var_95'] = float(orig_var_95)
        metrics['compressed_var_95'] = float(comp_var_95)
        metrics['target_var_95'] = float(target_var_95)
        
        # CVaR (Conditional Value at Risk)
        orig_cvar_95 = np.mean(original_predictions[original_predictions <= orig_var_95])
        comp_cvar_95 = np.mean(compressed_predictions[compressed_predictions <= comp_var_95])
        
        metrics['original_cvar_95'] = float(orig_cvar_95)
        metrics['compressed_cvar_95'] = float(comp_cvar_95)
        
        # Maximum Drawdown (упрощенная версия)
        if len(original_predictions) > 1:
            orig_cumulative = np.cumsum(original_predictions)
            comp_cumulative = np.cumsum(compressed_predictions)
            
            orig_running_max = np.maximum.accumulate(orig_cumulative)
            comp_running_max = np.maximum.accumulate(comp_cumulative)
            
            orig_drawdown = (orig_running_max - orig_cumulative) / (orig_running_max + 1e-8)
            comp_drawdown = (comp_running_max - comp_cumulative) / (comp_running_max + 1e-8)
            
            metrics['original_max_drawdown'] = float(np.max(orig_drawdown))
            metrics['compressed_max_drawdown'] = float(np.max(comp_drawdown))
        
        # Sortino Ratio (downside deviation)
        if len(original_predictions) > 1:
            orig_returns = np.diff(original_predictions)
            comp_returns = np.diff(compressed_predictions)
            
            orig_downside = orig_returns[orig_returns < 0]
            comp_downside = comp_returns[comp_returns < 0]
            
            if len(orig_downside) > 0:
                orig_sortino = np.mean(orig_returns) / (np.std(orig_downside) + 1e-8)
                metrics['original_sortino_ratio'] = float(orig_sortino)
            
            if len(comp_downside) > 0:
                comp_sortino = np.mean(comp_returns) / (np.std(comp_downside) + 1e-8)
                metrics['compressed_sortino_ratio'] = float(comp_sortino)
        
        return metrics
    
    def _run_validation_tests(self,
                            original_predictions: np.ndarray,
                            compressed_predictions: np.ndarray,
                            targets: np.ndarray,
                            metrics: Dict[str, float],
                            validation_level: ValidationLevel) -> Dict[str, bool]:
        """Выполнение валидационных тестов"""
        
        tests = {}
        
        # Базовые тесты
        tests.update(self._run_basic_tests(metrics))
        
        # Торговые тесты
        if validation_level in [ValidationLevel.TRADING, ValidationLevel.COMPREHENSIVE]:
            tests.update(self._run_trading_tests(metrics))
        
        # Risk тесты
        if validation_level in [ValidationLevel.RISK, ValidationLevel.COMPREHENSIVE]:
            tests.update(self._run_risk_tests(metrics))
        
        return tests
    
    def _run_basic_tests(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Базовые тесты валидации"""
        
        tests = {}
        
        # Тест accuracy retention
        accuracy_retention = metrics.get('accuracy_retention', 0.0)
        tests['accuracy_retention_test'] = accuracy_retention >= self.thresholds.min_accuracy_retention
        
        # Тест MSE increase
        mse_ratio = metrics.get('mse_ratio', float('inf'))
        tests['mse_increase_test'] = (mse_ratio - 1.0) <= self.thresholds.max_mse_increase
        
        # Тест корреляции предсказаний
        correlation = metrics.get('predictions_correlation', 0.0)
        tests['predictions_correlation_test'] = correlation >= self.thresholds.min_correlation
        
        # Тест R² сохранения
        r2_diff = metrics.get('r2_difference', -float('inf'))
        tests['r2_preservation_test'] = r2_diff >= -0.05  # Не более 5% ухудшения
        
        return tests
    
    def _run_trading_tests(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Торговые тесты валидации"""
        
        tests = {}
        
        # Directional accuracy test
        dir_acc = metrics.get('compressed_directional_accuracy', 0.0)
        tests['directional_accuracy_test'] = dir_acc >= self.thresholds.min_directional_accuracy
        
        # Directional accuracy retention test
        dir_acc_retention = metrics.get('directional_accuracy_retention', 0.0)
        tests['directional_accuracy_retention_test'] = dir_acc_retention >= 0.95
        
        # Sharpe ratio test
        sharpe = metrics.get('compressed_sharpe_ratio', -float('inf'))
        tests['sharpe_ratio_test'] = sharpe >= self.thresholds.min_sharpe_ratio
        
        # Sharpe retention test
        sharpe_retention = metrics.get('sharpe_retention', 0.0)
        tests['sharpe_retention_test'] = sharpe_retention >= 0.8
        
        # Hit ratio test
        hit_ratio = metrics.get('compressed_hit_ratio', 0.0)
        tests['hit_ratio_test'] = hit_ratio >= 0.5
        
        return tests
    
    def _run_risk_tests(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Risk management тесты"""
        
        tests = {}
        
        # Volatility preservation test
        vol_ratio = metrics.get('volatility_ratio', float('inf'))
        vol_change = abs(vol_ratio - 1.0)
        tests['volatility_preservation_test'] = vol_change <= self.thresholds.max_volatility_change
        
        # Maximum drawdown test
        max_drawdown = metrics.get('compressed_max_drawdown', float('inf'))
        tests['max_drawdown_test'] = max_drawdown <= self.thresholds.max_drawdown_threshold
        
        # VaR consistency test
        orig_var = metrics.get('original_var_95', 0.0)
        comp_var = metrics.get('compressed_var_95', 0.0)
        var_diff = abs(comp_var - orig_var) / (abs(orig_var) + 1e-8)
        tests['var_consistency_test'] = var_diff <= 0.2  # Не более 20% изменения
        
        return tests
    
    def _run_statistical_significance_tests(self,
                                          original_predictions: np.ndarray,
                                          compressed_predictions: np.ndarray,
                                          targets: np.ndarray) -> Dict[str, float]:
        """Тесты статистической значимости"""
        
        significance = {}
        
        # Paired t-test для ошибок
        orig_errors = np.abs(targets - original_predictions)
        comp_errors = np.abs(targets - compressed_predictions)
        
        try:
            t_stat, p_value = stats.ttest_rel(orig_errors, comp_errors)
            significance['error_difference_p_value'] = float(p_value)
            significance['error_difference_significant'] = p_value < (1 - self.confidence_level)
        except:
            significance['error_difference_p_value'] = 1.0
            significance['error_difference_significant'] = False
        
        # Kolmogorov-Smirnov test для распределений
        try:
            ks_stat, ks_p_value = stats.ks_2samp(original_predictions, compressed_predictions)
            significance['distribution_ks_p_value'] = float(ks_p_value)
            significance['distributions_similar'] = ks_p_value > 0.05
        except:
            significance['distribution_ks_p_value'] = 0.0
            significance['distributions_similar'] = False
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            wilcox_stat, wilcox_p = stats.wilcoxon(orig_errors, comp_errors)
            significance['wilcoxon_p_value'] = float(wilcox_p)
            significance['error_medians_similar'] = wilcox_p > 0.05
        except:
            significance['wilcoxon_p_value'] = 1.0
            significance['error_medians_similar'] = True
        
        return significance
    
    def _generate_warnings(self,
                          metrics: Dict[str, float],
                          test_results: Dict[str, bool]) -> List[str]:
        """Генерация предупреждений"""
        
        warnings = []
        
        # Предупреждения на основе метрик
        accuracy_retention = metrics.get('accuracy_retention', 1.0)
        if accuracy_retention < 0.98:
            warnings.append(f"Снижение точности: {(1-accuracy_retention)*100:.1f}%")
        
        correlation = metrics.get('predictions_correlation', 1.0)
        if correlation < 0.95:
            warnings.append(f"Низкая корреляция предсказаний: {correlation:.3f}")
        
        dir_acc = metrics.get('compressed_directional_accuracy', 0.0)
        if dir_acc < 0.55:
            warnings.append(f"Низкая directional accuracy: {dir_acc:.3f}")
        
        vol_ratio = metrics.get('volatility_ratio', 1.0)
        if abs(vol_ratio - 1.0) > 0.15:
            warnings.append(f"Значительное изменение волатильности: {(vol_ratio-1)*100:.1f}%")
        
        # Предупреждения на основе неудавших тестов
        failed_critical_tests = [
            test for test in ['accuracy_retention_test', 'predictions_correlation_test']
            if not test_results.get(test, True)
        ]
        
        if failed_critical_tests:
            warnings.append("Критические тесты не прошли валидацию")
        
        return warnings
    
    def _generate_recommendations(self,
                                metrics: Dict[str, float],
                                test_results: Dict[str, bool],
                                failed_tests: List[str]) -> List[str]:
        """Генерация рекомендаций"""
        
        recommendations = []
        
        if not failed_tests:
            recommendations.append("Все тесты пройдены успешно. Модель готова к использованию.")
            return recommendations
        
        # Рекомендации на основе неудавших тестов
        if 'accuracy_retention_test' in failed_tests:
            recommendations.append("Точность значительно снизилась. Рассмотрите менее агрессивное сжатие или fine-tuning.")
        
        if 'predictions_correlation_test' in failed_tests:
            recommendations.append("Низкая корреляция предсказаний. Проверьте корректность сжатия или используйте knowledge distillation.")
        
        if 'directional_accuracy_test' in failed_tests:
            recommendations.append("Низкая directional accuracy. Важно для торговых стратегий - рассмотрите специализированное обучение.")
        
        if 'sharpe_ratio_test' in failed_tests:
            recommendations.append("Низкий Sharpe ratio. Модель может быть менее прибыльной в торговле.")
        
        if 'volatility_preservation_test' in failed_tests:
            recommendations.append("Значительное изменение волатильности. Может повлиять на risk management.")
        
        if 'max_drawdown_test' in failed_tests:
            recommendations.append("Высокий максимальный drawdown. Повышенный риск для торговых стратегий.")
        
        # Общие рекомендации
        if len(failed_tests) > 3:
            recommendations.append("Множественные сбои тестов. Рекомендуется пересмотреть стратегию сжатия.")
        
        # Рекомендации на основе метрик
        mse_ratio = metrics.get('mse_ratio', 1.0)
        if mse_ratio > 1.2:
            recommendations.append("Значительное увеличение MSE. Попробуйте knowledge distillation или менее агрессивную pruning.")
        
        return recommendations
    
    def create_validation_report(self,
                               validation_result: ValidationResult,
                               save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Создание детального отчета валидации"""
        
        report = {
            'summary': validation_result.to_dict(),
            'detailed_analysis': self._create_detailed_analysis(validation_result),
            'visualizations': self._create_visualizations(validation_result),
            'comparison_table': self._create_comparison_table(validation_result)
        }
        
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def _create_detailed_analysis(self, result: ValidationResult) -> Dict[str, Any]:
        """Создание детального анализа"""
        
        analysis = {}
        
        # Анализ по категориям метрик
        metrics = result.metrics
        
        # Базовые метрики
        basic_metrics = {k: v for k, v in metrics.items() 
                        if k in ['accuracy_retention', 'predictions_correlation', 'mse_ratio', 'mae_ratio']}
        analysis['basic_metrics_analysis'] = {
            'metrics': basic_metrics,
            'interpretation': self._interpret_basic_metrics(basic_metrics)
        }
        
        # Торговые метрики
        trading_metrics = {k: v for k, v in metrics.items() 
                          if 'directional' in k or 'sharpe' in k or 'hit' in k}
        if trading_metrics:
            analysis['trading_metrics_analysis'] = {
                'metrics': trading_metrics,
                'interpretation': self._interpret_trading_metrics(trading_metrics)
            }
        
        # Risk метрики
        risk_metrics = {k: v for k, v in metrics.items() 
                       if 'volatility' in k or 'drawdown' in k or 'var' in k}
        if risk_metrics:
            analysis['risk_metrics_analysis'] = {
                'metrics': risk_metrics,
                'interpretation': self._interpret_risk_metrics(risk_metrics)
            }
        
        return analysis
    
    def _interpret_basic_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Интерпретация базовых метрик"""
        
        interpretations = []
        
        accuracy_retention = metrics.get('accuracy_retention', 1.0)
        if accuracy_retention > 0.98:
            interpretations.append("Отличное сохранение точности")
        elif accuracy_retention > 0.95:
            interpretations.append("Хорошее сохранение точности")
        else:
            interpretations.append("Значительная потеря точности")
        
        correlation = metrics.get('predictions_correlation', 1.0)
        if correlation > 0.95:
            interpretations.append("Высокая корреляция предсказаний")
        elif correlation > 0.9:
            interpretations.append("Умеренная корреляция предсказаний")
        else:
            interpretations.append("Низкая корреляция предсказаний - возможны проблемы с моделью")
        
        return interpretations
    
    def _interpret_trading_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Интерпретация торговых метрик"""
        
        interpretations = []
        
        dir_acc = metrics.get('compressed_directional_accuracy', 0.0)
        if dir_acc > 0.6:
            interpretations.append("Высокая directional accuracy - подходит для торговых стратегий")
        elif dir_acc > 0.52:
            interpretations.append("Умеренная directional accuracy")
        else:
            interpretations.append("Низкая directional accuracy - не рекомендуется для торговли")
        
        sharpe = metrics.get('compressed_sharpe_ratio', 0.0)
        if sharpe > 1.0:
            interpretations.append("Отличный Sharpe ratio")
        elif sharpe > 0.5:
            interpretations.append("Приемлемый Sharpe ratio")
        else:
            interpretations.append("Низкий Sharpe ratio")
        
        return interpretations
    
    def _interpret_risk_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Интерпретация risk метрик"""
        
        interpretations = []
        
        vol_ratio = metrics.get('volatility_ratio', 1.0)
        vol_change = abs(vol_ratio - 1.0)
        if vol_change < 0.05:
            interpretations.append("Волатильность хорошо сохранена")
        elif vol_change < 0.15:
            interpretations.append("Умеренное изменение волатильности")
        else:
            interpretations.append("Значительное изменение волатильности")
        
        max_drawdown = metrics.get('compressed_max_drawdown', 0.0)
        if max_drawdown < 0.05:
            interpretations.append("Низкий максимальный drawdown")
        elif max_drawdown < 0.15:
            interpretations.append("Умеренный максимальный drawdown")
        else:
            interpretations.append("Высокий максимальный drawdown - повышенный риск")
        
        return interpretations
    
    def _create_visualizations(self, result: ValidationResult) -> Dict[str, str]:
        """Создание визуализаций (paths к файлам)"""
        
        # В реальной реализации здесь были бы графики
        visualizations = {
            'metrics_comparison': 'path/to/metrics_comparison.png',
            'correlation_plot': 'path/to/correlation_plot.png',
            'error_distribution': 'path/to/error_distribution.png'
        }
        
        return visualizations
    
    def _create_comparison_table(self, result: ValidationResult) -> Dict[str, Any]:
        """Создание таблицы сравнения"""
        
        metrics = result.metrics
        
        comparison_data = []
        
        # Основные метрики для сравнения
        key_metrics = [
            ('MSE', 'original_mse', 'compressed_mse'),
            ('MAE', 'original_mae', 'compressed_mae'),
            ('R²', 'original_r2', 'compressed_r2'),
            ('Directional Accuracy', 'original_directional_accuracy', 'compressed_directional_accuracy'),
            ('Sharpe Ratio', 'original_sharpe_ratio', 'compressed_sharpe_ratio'),
            ('Volatility', 'original_volatility', 'compressed_volatility')
        ]
        
        for metric_name, orig_key, comp_key in key_metrics:
            orig_val = metrics.get(orig_key, 0.0)
            comp_val = metrics.get(comp_key, 0.0)
            
            change = ((comp_val - orig_val) / (orig_val + 1e-8)) * 100
            
            comparison_data.append({
                'metric': metric_name,
                'original': orig_val,
                'compressed': comp_val,
                'change_pct': change
            })
        
        return {
            'comparison_table': comparison_data,
            'summary_stats': {
                'total_metrics': len(comparison_data),
                'improved_metrics': len([d for d in comparison_data if d['change_pct'] > 0]),
                'degraded_metrics': len([d for d in comparison_data if d['change_pct'] < -5])
            }
        }
    
    def _save_report(self, report: Dict[str, Any], save_path: Path) -> None:
        """Сохранение отчета"""
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Отчет валидации сохранен: {save_path}")
    
    def get_validation_summary(self, last_n: int = 5) -> Dict[str, Any]:
        """Получение summary последних валидаций"""
        
        recent_validations = self.validation_history[-last_n:] if self.validation_history else []
        
        if not recent_validations:
            return {'message': 'No validation history available'}
        
        summary = {
            'total_validations': len(self.validation_history),
            'recent_validations': last_n,
            'success_rate': sum(1 for v in recent_validations if v.overall_passed) / len(recent_validations),
            'common_failures': self._analyze_common_failures(recent_validations),
            'average_metrics': self._calculate_average_metrics(recent_validations)
        }
        
        return summary
    
    def _analyze_common_failures(self, validations: List[ValidationResult]) -> Dict[str, int]:
        """Анализ частых неудач тестов"""
        
        failure_counts = {}
        
        for validation in validations:
            for failed_test in validation.failed_tests:
                failure_counts[failed_test] = failure_counts.get(failed_test, 0) + 1
        
        return failure_counts
    
    def _calculate_average_metrics(self, validations: List[ValidationResult]) -> Dict[str, float]:
        """Расчет средних метрик"""
        
        if not validations:
            return {}
        
        all_metrics = {}
        
        for validation in validations:
            for metric, value in validation.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        average_metrics = {}
        for metric, values in all_metrics.items():
            average_metrics[metric] = float(np.mean(values))
        
        return average_metrics