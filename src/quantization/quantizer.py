"""
Базовый модуль для квантизации моделей в Crypto Trading Bot.
Поддерживает INT8, INT4 и смешанную точность для оптимизации развертывания.

Edge computing deployment patterns для высокочастотной торговли
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, Tuple
import logging
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QConfig, default_qconfig
import torch.quantization.quantize_fx as quantize_fx
from pathlib import Path

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Типы квантизации для оптимизации моделей"""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "quantization_aware_training"  # Квантизация с учетом обучения
    MIXED_PRECISION = "mixed_precision"

class PrecisionLevel(Enum):
    """Уровни точности для квантизации"""
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"

class BaseQuantizer(ABC):
    """Базовый класс для всех квантизаторов с enterprise patterns"""
    
    def __init__(self, 
                 precision: PrecisionLevel = PrecisionLevel.INT8,
                 backend: str = "fbgemm"):
        """
        Args:
            precision: Уровень точности (INT8/INT4/FP16)
            backend: Backend для квантизации (fbgemm для CPU, qnnpack для mobile)
        """
        self.precision = precision
        self.backend = backend
        self.config = self._get_quantization_config()
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Настройка логирования для мониторинга процесса"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _get_quantization_config(self) -> QConfig:
        """Получение конфигурации квантизации"""
        if self.precision == PrecisionLevel.INT8:
            return default_qconfig
        elif self.precision == PrecisionLevel.INT4:
            # Пользовательская конфигурация для INT4
            return QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
                weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8)
            )
        else:
            return default_qconfig
    
    @abstractmethod
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Абстрактный метод для квантизации модели"""
        pass
    
    def validate_model(self, model: nn.Module) -> bool:
        """Валидация модели перед квантизацией"""
        try:
            # Проверка на наличие неподдерживаемых слоев
            unsupported_layers = self._find_unsupported_layers(model)
            if unsupported_layers:
                self.logger.warning(f"Найдены неподдерживаемые слои: {unsupported_layers}")
                return False
            
            # Проверка размера модели
            model_size = self._calculate_model_size(model)
            if model_size > 500:  # MB
                self.logger.warning(f"Модель слишком большая: {model_size} MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации модели: {e}")
            return False
    
    def _find_unsupported_layers(self, model: nn.Module) -> list:
        """Поиск неподдерживаемых для квантизации слоев"""
        unsupported = []
        unsupported_types = [nn.EmbeddingBag, nn.MultiheadAttention]
        
        for name, module in model.named_modules():
            if any(isinstance(module, unsupported_type) for unsupported_type in unsupported_types):
                unsupported.append(name)
        
        return unsupported
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Расчет размера модели в MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

class CryptoModelQuantizer(BaseQuantizer):
    """
    Специализированный квантизатор для моделей криптотрейдинга
    с оптимизацией для real-time inference
    """
    
    def __init__(self, 
                 precision: PrecisionLevel = PrecisionLevel.INT8,
                 latency_target: float = 1.0,  # мс
                 accuracy_threshold: float = 0.95):
        """
        Args:
            precision: Уровень точности
            latency_target: Целевая латентность в мс
            accuracy_threshold: Минимальный порог точности
        """
        super().__init__(precision)
        self.latency_target = latency_target
        self.accuracy_threshold = accuracy_threshold
        self.compression_stats = {}
    
    def quantize_model(self, 
                      model: nn.Module, 
                      calibration_data: Optional[torch.Tensor] = None,
                      quantization_type: QuantizationType = QuantizationType.DYNAMIC) -> nn.Module:
        """
        Квантизация модели с учетом специфики криптотрейдинга
        
        Args:
            model: PyTorch модель для квантизации
            calibration_data: Данные для калибровки (для статической квантизации)
            quantization_type: Тип квантизации
            
        Returns:
            Квантизованная модель
        """
        if not self.validate_model(model):
            raise ValueError("Модель не прошла валидацию")
        
        original_size = self._calculate_model_size(model)
        self.logger.info(f"Начинаем квантизацию модели. Исходный размер: {original_size:.2f} MB")
        
        try:
            if quantization_type == QuantizationType.DYNAMIC:
                quantized_model = self._dynamic_quantization(model)
            elif quantization_type == QuantizationType.STATIC:
                quantized_model = self._static_quantization(model, calibration_data)
            elif quantization_type == QuantizationType.QAT:
                quantized_model = self._quantization_aware_training(model)
            else:
                raise ValueError(f"Неподдерживаемый тип квантизации: {quantization_type}")
            
            # Анализ результатов сжатия
            compressed_size = self._calculate_model_size(quantized_model)
            compression_ratio = original_size / compressed_size
            
            self.compression_stats = {
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": compression_ratio,
                "precision": self.precision.value,
                "quantization_type": quantization_type.value
            }
            
            self.logger.info(f"Квантизация завершена. Коэффициент сжатия: {compression_ratio:.2f}x")
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Ошибка при квантизации модели: {e}")
            raise
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Динамическая квантизация для быстрого inference"""
        # Определяем слои для квантизации (обычно Linear и Conv)
        layers_to_quantize = {nn.Linear, nn.Conv2d, nn.Conv1d}
        
        if self.precision == PrecisionLevel.INT8:
            dtype = torch.qint8
        elif self.precision == PrecisionLevel.INT4:
            # PyTorch пока не поддерживает INT4 нативно, эмулируем через INT8
            dtype = torch.qint8
        else:
            dtype = torch.qint8
        
        quantized_model = quantize_dynamic(
            model,
            qconfig_spec=layers_to_quantize,
            dtype=dtype
        )
        
        return quantized_model
    
    def _static_quantization(self, model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
        """Статическая квантизация с калибровкой"""
        if calibration_data is None:
            raise ValueError("Для статической квантизации нужны данные калибровки")
        
        # Подготовка модели к квантизации
        model.eval()
        model.qconfig = self.config
        
        # Подготовка модели
        prepared_model = torch.quantization.prepare(model)
        
        # Калибровка на представительных данных
        self.logger.info("Выполняем калибровку модели...")
        with torch.no_grad():
            for batch in self._create_calibration_batches(calibration_data):
                prepared_model(batch)
        
        # Конвертация в квантизованную модель
        quantized_model = torch.quantization.convert(prepared_model)
        
        return quantized_model
    
    def _quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """Квантизация с учетом обучения (требует дообучения)"""
        model.train()
        model.qconfig = self.config
        
        # Подготовка к QAT
        prepared_model = torch.quantization.prepare_qat(model)
        
        self.logger.info("Модель подготовлена к QAT. Необходимо дообучение.")
        
        return prepared_model
    
    def _create_calibration_batches(self, calibration_data: torch.Tensor, batch_size: int = 32):
        """Создание батчей для калибровки"""
        for i in range(0, len(calibration_data), batch_size):
            yield calibration_data[i:i + batch_size]
    
    def optimize_for_trading(self, model: nn.Module) -> nn.Module:
        """
        Специальная оптимизация для задач криптотрейдинга
        с фокусом на минимизацию латентности
        """
        # Фьюзинг операций для уменьшения латентности
        fused_model = self._fuse_operations(model)
        
        # Оптимизация для ONNX Runtime (для production deployment)
        if hasattr(torch, 'jit'):
            try:
                traced_model = torch.jit.trace(fused_model, torch.randn(1, *self._get_input_shape(model)))
                optimized_model = torch.jit.optimize_for_inference(traced_model)
                return optimized_model
            except Exception as e:
                self.logger.warning(f"JIT оптимизация не удалась: {e}")
                return fused_model
        
        return fused_model
    
    def _fuse_operations(self, model: nn.Module) -> nn.Module:
        """Фьюзинг операций для ускорения inference"""
        try:
            # Стандартные фьюзинги: Conv+ReLU, Linear+ReLU
            modules_to_fuse = []
            
            # Поиск последовательностей для фьюзинга
            module_list = list(model.named_modules())
            for i, (name, module) in enumerate(module_list[:-1]):
                next_name, next_module = module_list[i + 1]
                
                # Conv2d + ReLU
                if isinstance(module, nn.Conv2d) and isinstance(next_module, nn.ReLU):
                    modules_to_fuse.append([name, next_name])
                
                # Linear + ReLU
                elif isinstance(module, nn.Linear) and isinstance(next_module, nn.ReLU):
                    modules_to_fuse.append([name, next_name])
            
            if modules_to_fuse:
                fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)
                self.logger.info(f"Выполнен фьюзинг {len(modules_to_fuse)} пар модулей")
                return fused_model
            
        except Exception as e:
            self.logger.warning(f"Фьюзинг операций не удался: {e}")
        
        return model
    
    def _get_input_shape(self, model: nn.Module) -> Tuple[int, ...]:
        """Определение размера входных данных модели"""
        # Простая эвристика для определения размера входа
        first_layer = next(iter(model.modules()))
        if isinstance(first_layer, nn.Conv2d):
            # Предполагаем стандартный размер для crypto данных
            return (first_layer.in_channels, 64, 64)  # Пример для 2D данных
        elif isinstance(first_layer, nn.Linear):
            return (first_layer.in_features,)
        else:
            # Дефолтный размер для временных рядов
            return (100,)  # 100 временных шагов
    
    def export_model(self, model: nn.Module, export_path: str, format: str = "onnx") -> bool:
        """
        Экспорт квантизованной модели для deployment
        
        Args:
            model: Квантизованная модель
            export_path: Путь для сохранения
            format: Формат экспорта (onnx, torchscript, tflite)
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "onnx":
                self._export_to_onnx(model, export_path)
            elif format.lower() == "torchscript":
                self._export_to_torchscript(model, export_path)
            else:
                raise ValueError(f"Неподдерживаемый формат экспорта: {format}")
            
            self.logger.info(f"Модель экспортирована в {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка экспорта модели: {e}")
            return False
    
    def _export_to_onnx(self, model: nn.Module, export_path: Path) -> None:
        """Экспорт в ONNX формат для cross-platform deployment"""
        try:
            import torch.onnx
            
            model.eval()
            dummy_input = torch.randn(1, *self._get_input_shape(model))
            
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path.with_suffix('.onnx')),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        except ImportError:
            self.logger.error("ONNX не установлен. Установите: pip install onnx")
            raise
    
    def _export_to_torchscript(self, model: nn.Module, export_path: Path) -> None:
        """Экспорт в TorchScript для production inference"""
        model.eval()
        dummy_input = torch.randn(1, *self._get_input_shape(model))
        
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(export_path.with_suffix('.pt')))
    
    def get_compression_report(self) -> Dict[str, Any]:
        """Получение отчета о сжатии модели"""
        return {
            "compression_stats": self.compression_stats,
            "config": {
                "precision": self.precision.value,
                "backend": self.backend,
                "latency_target_ms": self.latency_target,
                "accuracy_threshold": self.accuracy_threshold
            },
            "optimization_recommendations": self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> Dict[str, str]:
        """Рекомендации по дальнейшей оптимизации"""
        recommendations = {}
        
        if self.compression_stats.get("compression_ratio", 0) < 2:
            recommendations["compression"] = "Попробуйте более агрессивную квантизацию или pruning"
        
        if self.precision == PrecisionLevel.INT8:
            recommendations["precision"] = "Рассмотрите INT4 для большего сжатия"
        
        recommendations["deployment"] = "Используйте ONNX Runtime для production inference"
        
        return recommendations