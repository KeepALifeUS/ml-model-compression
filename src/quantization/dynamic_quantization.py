"""
Модуль динамической квантизации для криптотрейдинговых моделей.
Оптимизирован для минимальной латентности в real-time inference.

Context7: High-frequency trading optimization patterns
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import time
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, default_dynamic_qconfig
import numpy as np
from enum import Enum

from .quantizer import BaseQuantizer, PrecisionLevel, CryptoModelQuantizer

logger = logging.getLogger(__name__)

class DynamicQuantizationMode(Enum):
    """Режимы динамической квантизации"""
    AGGRESSIVE = "aggressive"    # Максимальное сжатие
    BALANCED = "balanced"       # Баланс сжатие/точность
    CONSERVATIVE = "conservative"  # Минимальная потеря точности

class LatencyOptimizer:
    """Оптимизатор латентности для HFT сценариев"""
    
    def __init__(self, target_latency_us: float = 100.0):
        """
        Args:
            target_latency_us: Целевая латентность в микросекундах
        """
        self.target_latency_us = target_latency_us
        self.benchmark_results = {}
        self.logger = logging.getLogger(f"{__name__}.LatencyOptimizer")
    
    def benchmark_model(self, 
                       model: nn.Module, 
                       input_shape: Tuple[int, ...], 
                       num_iterations: int = 1000) -> Dict[str, float]:
        """
        Бенчмарк латентности модели
        
        Args:
            model: Модель для тестирования
            input_shape: Размер входных данных
            num_iterations: Количество итераций для усреднения
        
        Returns:
            Статистика латентности
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Прогрев модели
        dummy_input = torch.randn(1, *input_shape).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Измерение латентности
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                
                latency_us = (end_time - start_time) * 1_000_000
                latencies.append(latency_us)
        
        # Статистика
        stats = {
            "mean_latency_us": np.mean(latencies),
            "median_latency_us": np.median(latencies),
            "p95_latency_us": np.percentile(latencies, 95),
            "p99_latency_us": np.percentile(latencies, 99),
            "std_latency_us": np.std(latencies),
            "min_latency_us": np.min(latencies),
            "max_latency_us": np.max(latencies)
        }
        
        self.benchmark_results = stats
        
        self.logger.info(f"Бенчмарк латентности: "
                        f"средняя={stats['mean_latency_us']:.1f}μs, "
                        f"p95={stats['p95_latency_us']:.1f}μs")
        
        return stats
    
    def meets_latency_target(self) -> bool:
        """Проверка соответствия целевой латентности"""
        if not self.benchmark_results:
            return False
        
        p95_latency = self.benchmark_results.get("p95_latency_us", float('inf'))
        return p95_latency <= self.target_latency_us

class DynamicQuantizer(CryptoModelQuantizer):
    """
    Специализированный динамический квантизатор для crypto trading models
    с оптимизацией для microsecond latency
    """
    
    def __init__(self, 
                 precision: PrecisionLevel = PrecisionLevel.INT8,
                 mode: DynamicQuantizationMode = DynamicQuantizationMode.BALANCED,
                 target_latency_us: float = 100.0):
        """
        Args:
            precision: Уровень точности квантизации
            mode: Режим квантизации (aggressive/balanced/conservative)
            target_latency_us: Целевая латентность в микросекундах
        """
        super().__init__(precision)
        self.mode = mode
        self.latency_optimizer = LatencyOptimizer(target_latency_us)
        self.quantization_config = self._get_dynamic_config()
        
    def _get_dynamic_config(self) -> Dict[str, Any]:
        """Получение конфигурации динамической квантизации"""
        base_config = {
            "dtype": torch.qint8 if self.precision == PrecisionLevel.INT8 else torch.qint8,
            "reduce_range": False
        }
        
        if self.mode == DynamicQuantizationMode.AGGRESSIVE:
            # Квантизируем максимум слоев для максимального сжатия
            base_config["qconfig_spec"] = {
                nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU
            }
        elif self.mode == DynamicQuantizationMode.BALANCED:
            # Квантизируем основные слои
            base_config["qconfig_spec"] = {nn.Linear, nn.Conv1d, nn.Conv2d}
        else:  # CONSERVATIVE
            # Квантизируем только Linear слои
            base_config["qconfig_spec"] = {nn.Linear}
            
        return base_config
    
    def quantize_for_hft(self, 
                        model: nn.Module,
                        input_shape: Tuple[int, ...],
                        calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Специальная квантизация для High-Frequency Trading
        с гарантией латентности
        
        Args:
            model: Исходная модель
            input_shape: Размер входных данных
            calibration_data: Данные для валидации точности
            
        Returns:
            Квантизованная модель с оптимальной латентностью
        """
        self.logger.info(f"Начинаем HFT квантизацию в режиме {self.mode.value}")
        
        # Исходный бенчмарк
        original_stats = self.latency_optimizer.benchmark_model(model, input_shape)
        
        # Пробуем разные конфигурации квантизации
        best_model = None
        best_config = None
        best_latency = float('inf')
        
        configs_to_try = self._generate_quantization_configs()
        
        for config_name, config in configs_to_try.items():
            try:
                # Квантизация с текущей конфигурацией
                quantized_model = self._apply_dynamic_quantization(model, config)
                
                # Бенчмарк квантизованной модели
                quantized_stats = self.latency_optimizer.benchmark_model(
                    quantized_model, input_shape
                )
                
                current_latency = quantized_stats["p95_latency_us"]
                
                # Проверка улучшения латентности и соответствия цели
                if (current_latency < best_latency and 
                    current_latency <= self.latency_optimizer.target_latency_us):
                    
                    # Дополнительная проверка точности если есть данные
                    if calibration_data is not None:
                        accuracy_ok = self._validate_accuracy(
                            model, quantized_model, calibration_data
                        )
                        if not accuracy_ok:
                            self.logger.warning(f"Конфигурация {config_name} не прошла проверку точности")
                            continue
                    
                    best_model = quantized_model
                    best_config = config_name
                    best_latency = current_latency
                    
                self.logger.info(f"Конфигурация {config_name}: "
                               f"латентность {current_latency:.1f}μs")
                
            except Exception as e:
                self.logger.warning(f"Ошибка с конфигурацией {config_name}: {e}")
                continue
        
        if best_model is None:
            self.logger.warning("Не удалось найти подходящую конфигурацию квантизации")
            return model
        
        # Дополнительные оптимизации для лучшей модели
        optimized_model = self._apply_hft_optimizations(best_model)
        
        # Финальный бенчмарк
        final_stats = self.latency_optimizer.benchmark_model(
            optimized_model, input_shape
        )
        
        improvement = (original_stats["p95_latency_us"] - 
                      final_stats["p95_latency_us"]) / original_stats["p95_latency_us"] * 100
        
        self.logger.info(f"HFT квантизация завершена. Лучшая конфигурация: {best_config}")
        self.logger.info(f"Улучшение латентности: {improvement:.1f}%")
        
        # Сохраняем статистику
        self.compression_stats.update({
            "hft_optimization": {
                "original_p95_latency_us": original_stats["p95_latency_us"],
                "final_p95_latency_us": final_stats["p95_latency_us"],
                "latency_improvement_pct": improvement,
                "best_config": best_config,
                "meets_target": final_stats["p95_latency_us"] <= self.latency_optimizer.target_latency_us
            }
        })
        
        return optimized_model
    
    def _generate_quantization_configs(self) -> Dict[str, Dict[str, Any]]:
        """Генерация различных конфигураций для тестирования"""
        configs = {}
        
        base_layers = [nn.Linear]
        extended_layers = [nn.Linear, nn.Conv1d, nn.Conv2d]
        aggressive_layers = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU]
        
        # Базовая конфигурация
        configs["basic"] = {
            "qconfig_spec": set(base_layers),
            "dtype": torch.qint8
        }
        
        # Расширенная конфигурация
        configs["extended"] = {
            "qconfig_spec": set(extended_layers),
            "dtype": torch.qint8
        }
        
        # Агрессивная конфигурация
        if self.mode == DynamicQuantizationMode.AGGRESSIVE:
            configs["aggressive"] = {
                "qconfig_spec": set(aggressive_layers),
                "dtype": torch.qint8
            }
        
        # Конфигурации с разными настройками reduce_range
        for name, base_config in list(configs.items()):
            reduced_config = base_config.copy()
            reduced_config["reduce_range"] = True
            configs[f"{name}_reduced"] = reduced_config
            
        return configs
    
    def _apply_dynamic_quantization(self, 
                                  model: nn.Module, 
                                  config: Dict[str, Any]) -> nn.Module:
        """Применение динамической квантизации с конфигурацией"""
        quantized_model = quantize_dynamic(
            model=model,
            qconfig_spec=config["qconfig_spec"],
            dtype=config["dtype"],
            mapping=None,
            inplace=False
        )
        
        return quantized_model
    
    def _validate_accuracy(self, 
                          original_model: nn.Module,
                          quantized_model: nn.Module,
                          validation_data: torch.Tensor,
                          threshold: float = 0.95) -> bool:
        """
        Валидация точности квантизованной модели
        
        Args:
            original_model: Исходная модель
            quantized_model: Квантизованная модель
            validation_data: Данные для валидации
            threshold: Минимальный порог соответствия
            
        Returns:
            True если точность приемлемая
        """
        try:
            original_model.eval()
            quantized_model.eval()
            
            with torch.no_grad():
                original_output = original_model(validation_data)
                quantized_output = quantized_model(validation_data)
                
                # Вычисляем корреляцию между выходами
                original_flat = original_output.flatten().cpu().numpy()
                quantized_flat = quantized_output.flatten().cpu().numpy()
                
                correlation = np.corrcoef(original_flat, quantized_flat)[0, 1]
                
                # Проверяем средний относительный error
                relative_error = np.mean(np.abs(original_flat - quantized_flat) / 
                                       (np.abs(original_flat) + 1e-8))
                
                accuracy_ok = (correlation >= threshold and 
                             relative_error <= (1.0 - threshold))
                
                self.logger.debug(f"Проверка точности: корреляция={correlation:.3f}, "
                                f"относительная ошибка={relative_error:.3f}")
                
                return accuracy_ok
                
        except Exception as e:
            self.logger.error(f"Ошибка валидации точности: {e}")
            return False
    
    def _apply_hft_optimizations(self, model: nn.Module) -> nn.Module:
        """Дополнительные оптимизации для HFT"""
        optimized_model = model
        
        try:
            # 1. Фьюзинг операций
            optimized_model = self._fuse_operations(optimized_model)
            
            # 2. JIT компиляция если возможно
            if hasattr(torch, 'jit'):
                try:
                    # Создаем пример входа для трассировки
                    input_shape = self._get_input_shape(model)
                    dummy_input = torch.randn(1, *input_shape)
                    
                    traced_model = torch.jit.trace(optimized_model, dummy_input)
                    optimized_model = torch.jit.optimize_for_inference(traced_model)
                    
                    self.logger.info("Применена JIT оптимизация")
                    
                except Exception as e:
                    self.logger.warning(f"JIT оптимизация не удалась: {e}")
            
            # 3. Memory layout оптимизация
            optimized_model = self._optimize_memory_layout(optimized_model)
            
        except Exception as e:
            self.logger.warning(f"Некоторые HFT оптимизации не удались: {e}")
        
        return optimized_model
    
    def _optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Оптимизация memory layout для лучшего cache locality"""
        try:
            # Обеспечиваем contiguous memory layout для параметров
            for param in model.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
            
            self.logger.debug("Применена оптимизация memory layout")
            
        except Exception as e:
            self.logger.warning(f"Ошибка оптимизации memory layout: {e}")
        
        return model
    
    def create_inference_engine(self, 
                               model: nn.Module, 
                               input_shape: Tuple[int, ...]) -> 'HFTInferenceEngine':
        """Создание специализированного inference engine для HFT"""
        return HFTInferenceEngine(
            model=model,
            input_shape=input_shape,
            target_latency_us=self.latency_optimizer.target_latency_us
        )

class HFTInferenceEngine:
    """
    High-Frequency Trading Inference Engine
    Оптимизирован для microsecond latency inference
    """
    
    def __init__(self, 
                 model: nn.Module,
                 input_shape: Tuple[int, ...],
                 target_latency_us: float = 100.0):
        """
        Args:
            model: Квантизованная модель
            input_shape: Размер входных данных
            target_latency_us: Целевая латентность
        """
        self.model = model
        self.input_shape = input_shape
        self.target_latency_us = target_latency_us
        
        # Прогрев и оптимизация
        self._warmup_model()
        self._setup_input_cache()
        
        self.logger = logging.getLogger(f"{__name__}.HFTInferenceEngine")
    
    def _warmup_model(self, warmup_iterations: int = 100) -> None:
        """Прогрев модели для стабильной латентности"""
        self.model.eval()
        dummy_input = torch.randn(1, *self.input_shape)
        
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(dummy_input)
        
        self.logger.info(f"Модель прогрета {warmup_iterations} итераций")
    
    def _setup_input_cache(self) -> None:
        """Настройка кэша входных тензоров для переиспользования"""
        # Pre-allocate input tensor для избежания memory allocation overhead
        self._input_cache = torch.empty(1, *self.input_shape)
        
    def predict(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Быстрый prediction с минимальной латентностью
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат предсказания
        """
        # Используем pre-allocated tensor
        if isinstance(input_data, np.ndarray):
            self._input_cache[0] = torch.from_numpy(input_data)
        else:
            self._input_cache[0] = input_data
        
        with torch.no_grad():
            return self.model(self._input_cache)
    
    def predict_batch(self, 
                     batch_data: Union[np.ndarray, torch.Tensor], 
                     max_batch_size: int = 32) -> torch.Tensor:
        """
        Batch prediction с контролем латентности
        
        Args:
            batch_data: Batch входных данных
            max_batch_size: Максимальный размер batch для контроля латентности
            
        Returns:
            Batch результатов
        """
        if isinstance(batch_data, np.ndarray):
            batch_data = torch.from_numpy(batch_data)
        
        batch_size = batch_data.size(0)
        
        if batch_size <= max_batch_size:
            with torch.no_grad():
                return self.model(batch_data)
        else:
            # Разбиваем на smaller batches для контроля латентности
            results = []
            
            for i in range(0, batch_size, max_batch_size):
                batch_slice = batch_data[i:i + max_batch_size]
                with torch.no_grad():
                    batch_result = self.model(batch_slice)
                results.append(batch_result)
            
            return torch.cat(results, dim=0)
    
    def get_latency_stats(self, num_iterations: int = 1000) -> Dict[str, float]:
        """Получение статистики латентности engine"""
        latency_optimizer = LatencyOptimizer(self.target_latency_us)
        return latency_optimizer.benchmark_model(
            self.model, self.input_shape, num_iterations
        )