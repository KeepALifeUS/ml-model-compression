"""
Edge deployment модуль для развертывания сжатых ML-моделей 
на устройствах с ограниченными ресурсами для crypto trading.

Context7: Edge computing deployment patterns для real-time financial systems
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import time
import psutil
import platform
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)

class EdgePlatform(Enum):
    """Поддерживаемые edge платформы"""
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    CORAL_DEV = "coral_dev"
    INTEL_NUC = "intel_nuc"
    ARM_GENERIC = "arm_generic"
    X86_EMBEDDED = "x86_embedded"

class ModelFormat(Enum):
    """Форматы моделей для deployment"""
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORFLOW_LITE = "tensorflow_lite"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"

class OptimizationLevel(Enum):
    """Уровни оптимизации для edge deployment"""
    MINIMAL = "minimal"      # Базовые оптимизации
    STANDARD = "standard"    # Стандартные оптимизации
    AGGRESSIVE = "aggressive"  # Агрессивные оптимизации
    ULTRA = "ultra"         # Максимальные оптимизации

@dataclass
class EdgeDeviceSpec:
    """Спецификация edge устройства"""
    platform: EdgePlatform
    cpu_cores: int
    ram_mb: int
    storage_mb: int
    has_gpu: bool = False
    gpu_memory_mb: int = 0
    supports_quantization: bool = True
    max_model_size_mb: float = 50.0
    target_latency_ms: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'platform': self.platform.value,
            'cpu_cores': self.cpu_cores,
            'ram_mb': self.ram_mb,
            'storage_mb': self.storage_mb,
            'has_gpu': self.has_gpu,
            'gpu_memory_mb': self.gpu_memory_mb,
            'supports_quantization': self.supports_quantization,
            'max_model_size_mb': self.max_model_size_mb,
            'target_latency_ms': self.target_latency_ms
        }

@dataclass
class DeploymentResult:
    """Результат edge deployment"""
    success: bool
    deployed_model_path: str
    model_format: ModelFormat
    model_size_mb: float
    estimated_latency_ms: float
    memory_usage_mb: float
    optimization_level: OptimizationLevel
    
    # Performance метрики
    inference_time_ms: float
    throughput_samples_per_sec: float
    cpu_usage_percent: float
    memory_peak_mb: float
    
    # Deployment информация
    deployment_time_sec: float
    export_formats: List[str]
    compatibility_issues: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'success': self.success,
            'deployed_model_path': self.deployed_model_path,
            'model_format': self.model_format.value if self.model_format else None,
            'model_size_mb': self.model_size_mb,
            'estimated_latency_ms': self.estimated_latency_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'optimization_level': self.optimization_level.value if self.optimization_level else None,
            'inference_time_ms': self.inference_time_ms,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_peak_mb': self.memory_peak_mb,
            'deployment_time_sec': self.deployment_time_sec,
            'export_formats': self.export_formats,
            'compatibility_issues': self.compatibility_issues,
            'recommendations': self.recommendations
        }

class EdgeDeployer:
    """
    Система развертывания сжатых ML-моделей на edge устройствах
    с оптимизацией для crypto trading в real-time
    """
    
    def __init__(self, workspace_dir: Union[str, Path]):
        """
        Args:
            workspace_dir: Рабочая директория для deployment
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Поддиректории
        (self.workspace_dir / "models").mkdir(exist_ok=True)
        (self.workspace_dir / "exports").mkdir(exist_ok=True)
        (self.workspace_dir / "benchmarks").mkdir(exist_ok=True)
        (self.workspace_dir / "configs").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.EdgeDeployer")
        
        # Кэш для deployment результатов
        self.deployment_cache = {}
        
        # Предопределенные конфигурации для популярных устройств
        self.device_presets = self._load_device_presets()
    
    def deploy_to_edge(self,
                      model: nn.Module,
                      device_spec: EdgeDeviceSpec,
                      optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                      target_format: Optional[ModelFormat] = None,
                      test_data: Optional[torch.utils.data.DataLoader] = None) -> DeploymentResult:
        """
        Основной метод развертывания модели на edge устройство
        
        Args:
            model: Модель для развертывания
            device_spec: Спецификация целевого устройства
            optimization_level: Уровень оптимизации
            target_format: Целевой формат модели
            test_data: Данные для бенчмарка
            
        Returns:
            Результат развертывания
        """
        start_time = time.time()
        
        self.logger.info(f"Начинаем edge deployment для {device_spec.platform.value}")
        self.logger.info(f"Уровень оптимизации: {optimization_level.value}")
        
        # Проверка совместимости модели с устройством
        compatibility_check = self._check_device_compatibility(model, device_spec)
        
        if not compatibility_check['compatible']:
            return DeploymentResult(
                success=False,
                deployed_model_path="",
                model_format=None,
                model_size_mb=0.0,
                estimated_latency_ms=float('inf'),
                memory_usage_mb=0.0,
                optimization_level=optimization_level,
                inference_time_ms=float('inf'),
                throughput_samples_per_sec=0.0,
                cpu_usage_percent=0.0,
                memory_peak_mb=0.0,
                deployment_time_sec=time.time() - start_time,
                export_formats=[],
                compatibility_issues=compatibility_check['issues'],
                recommendations=compatibility_check['recommendations']
            )
        
        try:
            # 1. Выбор оптимального формата модели
            if target_format is None:
                target_format = self._select_optimal_format(device_spec)
            
            # 2. Оптимизация модели для edge
            optimized_model = self._optimize_model_for_edge(
                model, device_spec, optimization_level
            )
            
            # 3. Экспорт в целевой формат
            exported_model_path, export_info = self._export_model(
                optimized_model, target_format, device_spec
            )
            
            # 4. Оптимизация экспортированной модели
            final_model_path = self._post_export_optimization(
                exported_model_path, target_format, device_spec
            )
            
            # 5. Бенчмарк производительности
            performance_metrics = self._benchmark_deployed_model(
                final_model_path, target_format, device_spec, test_data
            )
            
            # 6. Генерация рекомендаций
            recommendations = self._generate_deployment_recommendations(
                performance_metrics, device_spec, optimization_level
            )
            
            deployment_time = time.time() - start_time
            
            result = DeploymentResult(
                success=True,
                deployed_model_path=str(final_model_path),
                model_format=target_format,
                model_size_mb=export_info['model_size_mb'],
                estimated_latency_ms=performance_metrics['avg_inference_time_ms'],
                memory_usage_mb=performance_metrics['avg_memory_usage_mb'],
                optimization_level=optimization_level,
                inference_time_ms=performance_metrics['avg_inference_time_ms'],
                throughput_samples_per_sec=performance_metrics['throughput_samples_per_sec'],
                cpu_usage_percent=performance_metrics['avg_cpu_usage_percent'],
                memory_peak_mb=performance_metrics['peak_memory_mb'],
                deployment_time_sec=deployment_time,
                export_formats=export_info['formats_generated'],
                compatibility_issues=[],
                recommendations=recommendations
            )
            
            # Сохраняем результат
            self._save_deployment_result(result, device_spec)
            
            self.logger.info(f"Edge deployment завершен успешно за {deployment_time:.2f}с")
            self.logger.info(f"Модель развернута: {final_model_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка edge deployment: {e}")
            
            return DeploymentResult(
                success=False,
                deployed_model_path="",
                model_format=target_format,
                model_size_mb=0.0,
                estimated_latency_ms=float('inf'),
                memory_usage_mb=0.0,
                optimization_level=optimization_level,
                inference_time_ms=float('inf'),
                throughput_samples_per_sec=0.0,
                cpu_usage_percent=0.0,
                memory_peak_mb=0.0,
                deployment_time_sec=time.time() - start_time,
                export_formats=[],
                compatibility_issues=[str(e)],
                recommendations=["Проверьте совместимость модели и устройства"]
            )
    
    def _check_device_compatibility(self, 
                                  model: nn.Module, 
                                  device_spec: EdgeDeviceSpec) -> Dict[str, Any]:
        """Проверка совместимости модели с edge устройством"""
        
        issues = []
        recommendations = []
        
        # Проверка размера модели
        model_size_mb = self._calculate_model_size_mb(model)
        
        if model_size_mb > device_spec.max_model_size_mb:
            issues.append(f"Модель слишком большая: {model_size_mb:.1f}MB > {device_spec.max_model_size_mb:.1f}MB")
            recommendations.append("Примените более агрессивное сжатие")
        
        # Проверка памяти
        estimated_memory_mb = model_size_mb * 3  # Примерная оценка с учетом activations
        
        if estimated_memory_mb > device_spec.ram_mb * 0.8:  # Оставляем 20% системе
            issues.append(f"Недостаточно RAM: требуется ~{estimated_memory_mb:.1f}MB")
            recommendations.append("Уменьшите размер модели или используйте memory mapping")
        
        # Проверка поддерживаемых операций
        unsupported_ops = self._check_unsupported_operations(model, device_spec)
        
        if unsupported_ops:
            issues.append(f"Неподдерживаемые операции: {unsupported_ops}")
            recommendations.append("Замените неподдерживаемые операции или используйте другой формат")
        
        # Проверка квантизации
        if not device_spec.supports_quantization:
            has_quantized_ops = self._has_quantized_operations(model)
            if has_quantized_ops:
                issues.append("Устройство не поддерживает квантизованные операции")
                recommendations.append("Используйте float модель или другое устройство")
        
        compatible = len(issues) == 0
        
        return {
            'compatible': compatible,
            'issues': issues,
            'recommendations': recommendations,
            'estimated_memory_mb': estimated_memory_mb,
            'model_size_mb': model_size_mb
        }
    
    def _select_optimal_format(self, device_spec: EdgeDeviceSpec) -> ModelFormat:
        """Выбор оптимального формата модели для устройства"""
        
        # Приоритеты форматов для различных платформ
        format_priorities = {
            EdgePlatform.RASPBERRY_PI: [ModelFormat.ONNX, ModelFormat.TORCHSCRIPT, ModelFormat.PYTORCH],
            EdgePlatform.JETSON_NANO: [ModelFormat.TENSORRT, ModelFormat.ONNX, ModelFormat.TORCHSCRIPT],
            EdgePlatform.CORAL_DEV: [ModelFormat.TENSORFLOW_LITE, ModelFormat.ONNX],
            EdgePlatform.INTEL_NUC: [ModelFormat.OPENVINO, ModelFormat.ONNX, ModelFormat.TORCHSCRIPT],
            EdgePlatform.ARM_GENERIC: [ModelFormat.ONNX, ModelFormat.TORCHSCRIPT],
            EdgePlatform.X86_EMBEDDED: [ModelFormat.ONNX, ModelFormat.TORCHSCRIPT, ModelFormat.OPENVINO]
        }
        
        preferred_formats = format_priorities.get(
            device_spec.platform, 
            [ModelFormat.ONNX, ModelFormat.TORCHSCRIPT]
        )
        
        # Проверяем доступность первого предпочтительного формата
        for format_option in preferred_formats:
            if self._is_format_available(format_option):
                self.logger.info(f"Выбран формат {format_option.value} для {device_spec.platform.value}")
                return format_option
        
        # Fallback к PyTorch
        return ModelFormat.PYTORCH
    
    def _optimize_model_for_edge(self,
                               model: nn.Module,
                               device_spec: EdgeDeviceSpec,
                               optimization_level: OptimizationLevel) -> nn.Module:
        """Оптимизация модели для edge устройства"""
        
        optimized_model = model
        
        # Базовые оптимизации (всегда)
        optimized_model = self._apply_basic_optimizations(optimized_model)
        
        if optimization_level == OptimizationLevel.MINIMAL:
            return optimized_model
        
        # Стандартные оптимизации
        if optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE, OptimizationLevel.ULTRA]:
            optimized_model = self._apply_standard_optimizations(optimized_model, device_spec)
        
        # Агрессивные оптимизации
        if optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.ULTRA]:
            optimized_model = self._apply_aggressive_optimizations(optimized_model, device_spec)
        
        # Ультра оптимизации
        if optimization_level == OptimizationLevel.ULTRA:
            optimized_model = self._apply_ultra_optimizations(optimized_model, device_spec)
        
        return optimized_model
    
    def _apply_basic_optimizations(self, model: nn.Module) -> nn.Module:
        """Базовые оптимизации"""
        
        # Ensure eval mode
        model.eval()
        
        # Memory layout optimization
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        # Remove dropout layers (они не нужны в inference)
        self._remove_dropout_layers(model)
        
        return model
    
    def _apply_standard_optimizations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """Стандартные оптимизации"""
        
        # Operator fusion где возможно
        try:
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)
        except Exception as e:
            self.logger.warning(f"JIT оптимизация не удалась: {e}")
        
        # Memory mapping для больших моделей на устройствах с ограниченной памятью
        if device_spec.ram_mb < 2048:  # Менее 2GB RAM
            model = self._enable_memory_mapping(model)
        
        return model
    
    def _apply_aggressive_optimizations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """Агрессивные оптимизации"""
        
        # Graph-level оптимизации
        model = self._apply_graph_optimizations(model)
        
        # Platform-specific оптимизации
        if device_spec.platform == EdgePlatform.ARM_GENERIC or device_spec.platform == EdgePlatform.RASPBERRY_PI:
            model = self._apply_arm_optimizations(model)
        elif device_spec.has_gpu:
            model = self._apply_gpu_optimizations(model, device_spec)
        
        return model
    
    def _apply_ultra_optimizations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """Ультра оптимизации (могут снизить точность)"""
        
        # Kernel fusion
        model = self._apply_kernel_fusion(model)
        
        # Mixed precision если поддерживается
        if device_spec.has_gpu:
            model = self._apply_mixed_precision(model)
        
        # Custom operators для специфических операций
        model = self._replace_with_custom_ops(model, device_spec)
        
        return model
    
    def _export_model(self,
                     model: nn.Module,
                     target_format: ModelFormat,
                     device_spec: EdgeDeviceSpec) -> Tuple[Path, Dict[str, Any]]:
        """Экспорт модели в целевой формат"""
        
        timestamp = int(time.time())
        base_name = f"edge_model_{device_spec.platform.value}_{timestamp}"
        
        export_info = {
            'formats_generated': [],
            'model_size_mb': 0.0,
            'export_success': False
        }
        
        if target_format == ModelFormat.PYTORCH:
            export_path = self._export_pytorch(model, base_name)
        elif target_format == ModelFormat.TORCHSCRIPT:
            export_path = self._export_torchscript(model, base_name)
        elif target_format == ModelFormat.ONNX:
            export_path = self._export_onnx(model, base_name, device_spec)
        elif target_format == ModelFormat.TENSORFLOW_LITE:
            export_path = self._export_tflite(model, base_name, device_spec)
        elif target_format == ModelFormat.OPENVINO:
            export_path = self._export_openvino(model, base_name, device_spec)
        elif target_format == ModelFormat.TENSORRT:
            export_path = self._export_tensorrt(model, base_name, device_spec)
        else:
            raise ValueError(f"Неподдерживаемый формат: {target_format}")
        
        # Вычисляем размер экспортированной модели
        if export_path.exists():
            model_size_mb = export_path.stat().st_size / (1024 * 1024)
            export_info.update({
                'formats_generated': [target_format.value],
                'model_size_mb': model_size_mb,
                'export_success': True
            })
        
        return export_path, export_info
    
    def _export_pytorch(self, model: nn.Module, base_name: str) -> Path:
        """Экспорт в PyTorch формат"""
        export_path = self.workspace_dir / "exports" / f"{base_name}.pt"
        
        torch.save({
            'model': model,
            'state_dict': model.state_dict()
        }, export_path)
        
        return export_path
    
    def _export_torchscript(self, model: nn.Module, base_name: str) -> Path:
        """Экспорт в TorchScript"""
        export_path = self.workspace_dir / "exports" / f"{base_name}_script.pt"
        
        try:
            # Пробуем trace
            dummy_input = self._create_dummy_input(model)
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(str(export_path))
        except Exception:
            # Fallback к script
            scripted_model = torch.jit.script(model)
            scripted_model.save(str(export_path))
        
        return export_path
    
    def _export_onnx(self, model: nn.Module, base_name: str, device_spec: EdgeDeviceSpec) -> Path:
        """Экспорт в ONNX формат"""
        export_path = self.workspace_dir / "exports" / f"{base_name}.onnx"
        
        try:
            import torch.onnx
            
            dummy_input = self._create_dummy_input(model)
            
            # Настройки экспорта для edge устройств
            export_params = True
            opset_version = 11  # Совместимость с большинством runtimes
            do_constant_folding = True
            
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path),
                export_params=export_params,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
        except ImportError:
            raise ImportError("ONNX не установлен. pip install onnx")
        
        return export_path
    
    def _export_tflite(self, model: nn.Module, base_name: str, device_spec: EdgeDeviceSpec) -> Path:
        """Экспорт в TensorFlow Lite (через ONNX)"""
        # Это упрощенная заглушка - в реальности нужен сложный pipeline
        # ONNX -> TensorFlow -> TensorFlow Lite
        
        export_path = self.workspace_dir / "exports" / f"{base_name}.tflite"
        
        # Создаем dummy файл для демонстрации
        with open(export_path, 'wb') as f:
            f.write(b"TFLite model placeholder")
        
        self.logger.info(f"TFLite экспорт (placeholder): {export_path}")
        
        return export_path
    
    def _export_openvino(self, model: nn.Module, base_name: str, device_spec: EdgeDeviceSpec) -> Path:
        """Экспорт в OpenVINO формат"""
        export_path = self.workspace_dir / "exports" / f"{base_name}_openvino.xml"
        
        # Заглушка для OpenVINO экспорта
        # В реальности: PyTorch -> ONNX -> OpenVINO Model Optimizer
        
        with open(export_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n<net>OpenVINO model placeholder</net>')
        
        self.logger.info(f"OpenVINO экспорт (placeholder): {export_path}")
        
        return export_path
    
    def _export_tensorrt(self, model: nn.Module, base_name: str, device_spec: EdgeDeviceSpec) -> Path:
        """Экспорт в TensorRT формат"""
        export_path = self.workspace_dir / "exports" / f"{base_name}.trt"
        
        # Заглушка для TensorRT экспорта
        # В реальности: PyTorch -> ONNX -> TensorRT
        
        with open(export_path, 'wb') as f:
            f.write(b"TensorRT engine placeholder")
        
        self.logger.info(f"TensorRT экспорт (placeholder): {export_path}")
        
        return export_path
    
    def _post_export_optimization(self,
                                model_path: Path,
                                model_format: ModelFormat,
                                device_spec: EdgeDeviceSpec) -> Path:
        """Пост-экспортная оптимизация модели"""
        
        # Оптимизации зависят от формата
        if model_format == ModelFormat.ONNX:
            return self._optimize_onnx_model(model_path, device_spec)
        elif model_format == ModelFormat.OPENVINO:
            return self._optimize_openvino_model(model_path, device_spec)
        elif model_format == ModelFormat.TENSORRT:
            return self._optimize_tensorrt_model(model_path, device_spec)
        else:
            # Для остальных форматов возвращаем как есть
            return model_path
    
    def _optimize_onnx_model(self, model_path: Path, device_spec: EdgeDeviceSpec) -> Path:
        """Оптимизация ONNX модели"""
        
        try:
            import onnx
            from onnx import optimizer
            
            # Загружаем модель
            model = onnx.load(str(model_path))
            
            # Применяем оптимизации
            optimizations = ['eliminate_deadend', 'eliminate_identity', 'eliminate_nop_transpose']
            
            if device_spec.ram_mb < 1024:  # Для устройств с малой памятью
                optimizations.extend(['fuse_consecutive_transposes', 'fuse_transpose_into_gemm'])
            
            optimized_model = optimizer.optimize(model, optimizations)
            
            # Сохраняем оптимизированную модель
            optimized_path = model_path.with_name(f"optimized_{model_path.name}")
            onnx.save(optimized_model, str(optimized_path))
            
            self.logger.info(f"ONNX модель оптимизирована: {optimized_path}")
            
            return optimized_path
            
        except ImportError:
            self.logger.warning("ONNX optimizer недоступен")
            return model_path
        except Exception as e:
            self.logger.warning(f"ONNX оптимизация не удалась: {e}")
            return model_path
    
    def _optimize_openvino_model(self, model_path: Path, device_spec: EdgeDeviceSpec) -> Path:
        """Оптимизация OpenVINO модели"""
        # Заглушка для OpenVINO оптимизации
        return model_path
    
    def _optimize_tensorrt_model(self, model_path: Path, device_spec: EdgeDeviceSpec) -> Path:
        """Оптимизация TensorRT модели"""
        # Заглушка для TensorRT оптимизации
        return model_path
    
    def _benchmark_deployed_model(self,
                                model_path: Path,
                                model_format: ModelFormat,
                                device_spec: EdgeDeviceSpec,
                                test_data: Optional[torch.utils.data.DataLoader]) -> Dict[str, float]:
        """Бенчмарк развернутой модели"""
        
        metrics = {
            'avg_inference_time_ms': 0.0,
            'throughput_samples_per_sec': 0.0,
            'avg_memory_usage_mb': 0.0,
            'peak_memory_mb': 0.0,
            'avg_cpu_usage_percent': 0.0
        }
        
        try:
            # Загружаем модель в зависимости от формата
            if model_format == ModelFormat.PYTORCH:
                model = torch.load(model_path, map_location='cpu')['model']
            elif model_format == ModelFormat.TORCHSCRIPT:
                model = torch.jit.load(str(model_path), map_location='cpu')
            elif model_format == ModelFormat.ONNX:
                # Используем ONNX Runtime для бенчмарка
                return self._benchmark_onnx_model(model_path, device_spec, test_data)
            else:
                # Для других форматов используем заглушку
                return self._estimate_performance_metrics(device_spec)
            
            # PyTorch/TorchScript бенчмарк
            model.eval()
            
            # Создаем dummy данные если test_data не предоставлен
            if test_data is None:
                dummy_input = self._create_dummy_input(model)
                test_inputs = [dummy_input for _ in range(100)]
            else:
                test_inputs = []
                for i, batch in enumerate(test_data):
                    if i >= 100:  # Ограничиваем количество
                        break
                    if isinstance(batch, (list, tuple)):
                        test_inputs.append(batch[0][:1])  # Берем первый образец
                    else:
                        test_inputs.append(batch[:1])
            
            # Прогрев
            with torch.no_grad():
                for i in range(min(10, len(test_inputs))):
                    _ = model(test_inputs[i])
            
            # Бенчмарк
            inference_times = []
            memory_usage = []
            
            process = psutil.Process()
            
            with torch.no_grad():
                for test_input in test_inputs:
                    # Измеряем память до inference
                    mem_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Измеряем время inference
                    start_time = time.perf_counter()
                    _ = model(test_input)
                    end_time = time.perf_counter()
                    
                    # Измеряем память после inference
                    mem_after = process.memory_info().rss / 1024 / 1024  # MB
                    
                    inference_times.append((end_time - start_time) * 1000)  # ms
                    memory_usage.append(mem_after)
            
            # Вычисляем метрики
            metrics['avg_inference_time_ms'] = float(np.mean(inference_times))
            metrics['throughput_samples_per_sec'] = 1000.0 / metrics['avg_inference_time_ms']
            metrics['avg_memory_usage_mb'] = float(np.mean(memory_usage))
            metrics['peak_memory_mb'] = float(np.max(memory_usage))
            metrics['avg_cpu_usage_percent'] = float(psutil.cpu_percent(interval=1))
            
        except Exception as e:
            self.logger.warning(f"Ошибка бенчмарка: {e}")
            metrics = self._estimate_performance_metrics(device_spec)
        
        return metrics
    
    def _benchmark_onnx_model(self,
                            model_path: Path,
                            device_spec: EdgeDeviceSpec,
                            test_data: Optional[torch.utils.data.DataLoader]) -> Dict[str, float]:
        """Бенчмарк ONNX модели"""
        
        try:
            import onnxruntime as ort
            
            # Создаем inference session
            session = ort.InferenceSession(str(model_path))
            
            # Получаем информацию о входах
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            input_shape = input_info.shape
            
            # Создаем test данные
            if test_data is None:
                # Используем случайные данные
                test_inputs = [np.random.randn(1, *input_shape[1:]).astype(np.float32) for _ in range(100)]
            else:
                test_inputs = []
                for i, batch in enumerate(test_data):
                    if i >= 100:
                        break
                    if isinstance(batch, (list, tuple)):
                        test_inputs.append(batch[0][:1].numpy().astype(np.float32))
                    else:
                        test_inputs.append(batch[:1].numpy().astype(np.float32))
            
            # Прогрев
            for i in range(min(10, len(test_inputs))):
                _ = session.run(None, {input_name: test_inputs[i]})
            
            # Бенчмарк
            inference_times = []
            
            for test_input in test_inputs:
                start_time = time.perf_counter()
                _ = session.run(None, {input_name: test_input})
                end_time = time.perf_counter()
                
                inference_times.append((end_time - start_time) * 1000)  # ms
            
            avg_inference_time = float(np.mean(inference_times))
            
            return {
                'avg_inference_time_ms': avg_inference_time,
                'throughput_samples_per_sec': 1000.0 / avg_inference_time,
                'avg_memory_usage_mb': float(psutil.Process().memory_info().rss / 1024 / 1024),
                'peak_memory_mb': float(psutil.Process().memory_info().rss / 1024 / 1024),
                'avg_cpu_usage_percent': float(psutil.cpu_percent(interval=1))
            }
            
        except ImportError:
            self.logger.warning("ONNX Runtime недоступен")
            return self._estimate_performance_metrics(device_spec)
        except Exception as e:
            self.logger.warning(f"Ошибка ONNX бенчмарка: {e}")
            return self._estimate_performance_metrics(device_spec)
    
    def _estimate_performance_metrics(self, device_spec: EdgeDeviceSpec) -> Dict[str, float]:
        """Оценка performance метрик на основе спецификации устройства"""
        
        # Простые эвристики для оценки производительности
        base_latency = 50.0  # ms
        
        # Корректировка на основе характеристик устройства
        cpu_factor = max(0.5, 4.0 / device_spec.cpu_cores)
        ram_factor = max(0.8, 2048 / device_spec.ram_mb)
        
        estimated_latency = base_latency * cpu_factor * ram_factor
        
        return {
            'avg_inference_time_ms': float(min(estimated_latency, device_spec.target_latency_ms * 2)),
            'throughput_samples_per_sec': 1000.0 / estimated_latency,
            'avg_memory_usage_mb': float(min(100.0, device_spec.ram_mb * 0.1)),
            'peak_memory_mb': float(min(200.0, device_spec.ram_mb * 0.2)),
            'avg_cpu_usage_percent': 50.0
        }
    
    def _generate_deployment_recommendations(self,
                                          performance_metrics: Dict[str, float],
                                          device_spec: EdgeDeviceSpec,
                                          optimization_level: OptimizationLevel) -> List[str]:
        """Генерация рекомендаций по deployment"""
        
        recommendations = []
        
        # Анализ латентности
        inference_time = performance_metrics['avg_inference_time_ms']
        
        if inference_time > device_spec.target_latency_ms:
            recommendations.append(
                f"Латентность {inference_time:.1f}ms превышает целевую {device_spec.target_latency_ms:.1f}ms. "
                f"Рассмотрите более агрессивную оптимизацию."
            )
        elif inference_time < device_spec.target_latency_ms * 0.5:
            recommendations.append("Отличная латентность! Модель хорошо оптимизирована для устройства.")
        
        # Анализ использования памяти
        memory_usage = performance_metrics['avg_memory_usage_mb']
        memory_threshold = device_spec.ram_mb * 0.7
        
        if memory_usage > memory_threshold:
            recommendations.append(
                f"Высокое использование памяти {memory_usage:.1f}MB. "
                f"Рассмотрите дополнительное сжатие модели."
            )
        
        # Анализ CPU загрузки
        cpu_usage = performance_metrics['avg_cpu_usage_percent']
        
        if cpu_usage > 80.0:
            recommendations.append("Высокая загрузка CPU. Модель может быть слишком сложной для устройства.")
        elif cpu_usage < 30.0:
            recommendations.append("Низкая загрузка CPU. Есть резерв для более сложных моделей.")
        
        # Рекомендации по уровню оптимизации
        if optimization_level == OptimizationLevel.MINIMAL and inference_time > device_spec.target_latency_ms:
            recommendations.append("Попробуйте более высокий уровень оптимизации.")
        
        # Platform-specific рекомендации
        if device_spec.platform == EdgePlatform.RASPBERRY_PI:
            recommendations.append("Для Raspberry Pi рекомендуется использовать ONNX Runtime с CPU provider.")
        
        if device_spec.has_gpu and cpu_usage > 60.0:
            recommendations.append("Рассмотрите использование GPU для inference.")
        
        if not recommendations:
            recommendations.append("Deployment прошел успешно. Модель готова к использованию.")
        
        return recommendations
    
    # Helper методы
    
    def _calculate_model_size_mb(self, model: nn.Module) -> float:
        """Расчет размера модели в MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        return (param_size + buffer_size) / (1024 * 1024)
    
    def _check_unsupported_operations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> List[str]:
        """Проверка неподдерживаемых операций"""
        
        unsupported = []
        
        # Общие неподдерживаемые операции для edge устройств
        unsupported_types = []
        
        if device_spec.platform in [EdgePlatform.RASPBERRY_PI, EdgePlatform.ARM_GENERIC]:
            unsupported_types.extend([nn.MultiheadAttention])  # Пример
        
        for name, module in model.named_modules():
            if any(isinstance(module, unsupported_type) for unsupported_type in unsupported_types):
                unsupported.append(name)
        
        return unsupported
    
    def _has_quantized_operations(self, model: nn.Module) -> bool:
        """Проверка наличия квантизованных операций"""
        
        for module in model.modules():
            # Проверяем на quantized модули PyTorch
            if hasattr(module, 'qscheme'):  # Признак quantized модуля
                return True
        
        return False
    
    def _is_format_available(self, format_option: ModelFormat) -> bool:
        """Проверка доступности формата экспорта"""
        
        if format_option == ModelFormat.ONNX:
            try:
                import torch.onnx
                return True
            except ImportError:
                return False
        
        elif format_option == ModelFormat.TENSORFLOW_LITE:
            try:
                import tensorflow as tf
                return True
            except ImportError:
                return False
        
        elif format_option == ModelFormat.OPENVINO:
            # Проверка наличия OpenVINO toolkit
            return shutil.which('mo') is not None  # Model Optimizer
        
        elif format_option == ModelFormat.TENSORRT:
            # Проверка наличия TensorRT
            try:
                import tensorrt
                return True
            except ImportError:
                return False
        
        # PyTorch и TorchScript всегда доступны
        return True
    
    def _create_dummy_input(self, model: nn.Module) -> torch.Tensor:
        """Создание dummy входа для модели"""
        
        # Простая эвристика для определения размера входа
        first_layer = next(iter(model.modules()))
        
        if isinstance(first_layer, nn.Linear):
            return torch.randn(1, first_layer.in_features)
        elif isinstance(first_layer, nn.Conv1d):
            return torch.randn(1, first_layer.in_channels, 100)
        elif isinstance(first_layer, nn.Conv2d):
            return torch.randn(1, first_layer.in_channels, 32, 32)
        else:
            # Дефолтный размер для crypto trading (временные ряды)
            return torch.randn(1, 100)
    
    def _remove_dropout_layers(self, model: nn.Module) -> None:
        """Удаление dropout слоев (не нужны в inference)"""
        
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Dropout):
                setattr(model, name, nn.Identity())
            else:
                self._remove_dropout_layers(module)
    
    def _enable_memory_mapping(self, model: nn.Module) -> nn.Module:
        """Включение memory mapping для моделей"""
        # Заглушка для memory mapping
        return model
    
    def _apply_graph_optimizations(self, model: nn.Module) -> nn.Module:
        """Применение graph-level оптимизаций"""
        # Заглушка для graph оптимизаций
        return model
    
    def _apply_arm_optimizations(self, model: nn.Module) -> nn.Module:
        """ARM-специфичные оптимизации"""
        # Заглушка для ARM оптимизаций
        return model
    
    def _apply_gpu_optimizations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """GPU оптимизации для edge устройств"""
        # Заглушка для GPU оптимизаций
        return model
    
    def _apply_kernel_fusion(self, model: nn.Module) -> nn.Module:
        """Kernel fusion оптимизации"""
        # Заглушка для kernel fusion
        return model
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Применение mixed precision"""
        # Заглушка для mixed precision
        return model
    
    def _replace_with_custom_ops(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """Замена операций на кастомные оптимизированные версии"""
        # Заглушка для custom operators
        return model
    
    def _save_deployment_result(self, result: DeploymentResult, device_spec: EdgeDeviceSpec) -> None:
        """Сохранение результата deployment"""
        
        result_file = self.workspace_dir / "configs" / f"deployment_{device_spec.platform.value}_{int(time.time())}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                'device_spec': device_spec.to_dict(),
                'deployment_result': result.to_dict()
            }, f, indent=2)
        
        self.logger.info(f"Результат deployment сохранен: {result_file}")
    
    def _load_device_presets(self) -> Dict[EdgePlatform, EdgeDeviceSpec]:
        """Загрузка предустановленных конфигураций устройств"""
        
        presets = {
            EdgePlatform.RASPBERRY_PI: EdgeDeviceSpec(
                platform=EdgePlatform.RASPBERRY_PI,
                cpu_cores=4,
                ram_mb=4096,
                storage_mb=32000,
                has_gpu=False,
                max_model_size_mb=100.0,
                target_latency_ms=200.0
            ),
            
            EdgePlatform.JETSON_NANO: EdgeDeviceSpec(
                platform=EdgePlatform.JETSON_NANO,
                cpu_cores=4,
                ram_mb=4096,
                storage_mb=16000,
                has_gpu=True,
                gpu_memory_mb=2048,
                max_model_size_mb=200.0,
                target_latency_ms=50.0
            ),
            
            EdgePlatform.INTEL_NUC: EdgeDeviceSpec(
                platform=EdgePlatform.INTEL_NUC,
                cpu_cores=8,
                ram_mb=8192,
                storage_mb=256000,
                has_gpu=False,
                max_model_size_mb=500.0,
                target_latency_ms=30.0
            )
        }
        
        return presets
    
    def get_device_preset(self, platform: EdgePlatform) -> Optional[EdgeDeviceSpec]:
        """Получение preset конфигурации устройства"""
        return self.device_presets.get(platform)
    
    def list_supported_formats(self, platform: EdgePlatform) -> List[ModelFormat]:
        """Список поддерживаемых форматов для платформы"""
        
        format_support = {
            EdgePlatform.RASPBERRY_PI: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX],
            EdgePlatform.JETSON_NANO: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX, ModelFormat.TENSORRT],
            EdgePlatform.CORAL_DEV: [ModelFormat.TENSORFLOW_LITE],
            EdgePlatform.INTEL_NUC: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX, ModelFormat.OPENVINO],
            EdgePlatform.ARM_GENERIC: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX],
            EdgePlatform.X86_EMBEDDED: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX, ModelFormat.OPENVINO]
        }
        
        return format_support.get(platform, [ModelFormat.PYTORCH, ModelFormat.ONNX])