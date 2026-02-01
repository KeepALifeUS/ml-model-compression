"""
Универсальный оптимизатор моделей для криптотрейдинга с комбинированием
техник сжатия и Context7 production optimization patterns.

Context7: Holistic optimization patterns для production ML deployment
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import torch
import torch.nn as nn
import numpy as np
from enum import Enum
import copy
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from ..quantization.quantizer import CryptoModelQuantizer, PrecisionLevel
from ..quantization.dynamic_quantization import DynamicQuantizer, DynamicQuantizationMode
from ..pruning.structured_pruning import CryptoTradingStructuredPruner, StructuredPruningStrategy
from ..pruning.unstructured_pruning import CryptoTradingUnstructuredPruner, UnstructuredPruningStrategy
from ..distillation.knowledge_distiller import ResponseDistiller, FeatureDistiller
from ..distillation.teacher_student import TeacherStudentArchitecture, EnsembleDistiller

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Цели оптимизации модели"""
    LATENCY = "latency"              # Минимизация латентности
    MEMORY = "memory"                # Минимизация памяти
    ACCURACY = "accuracy"            # Максимизация точности
    THROUGHPUT = "throughput"        # Максимизация пропускной способности
    BALANCED = "balanced"            # Баланс всех факторов
    EDGE_DEPLOYMENT = "edge"         # Оптимизация для edge devices

class OptimizationStrategy(Enum):
    """Стратегии оптимизации"""
    SEQUENTIAL = "sequential"        # Последовательное применение техник
    PARALLEL = "parallel"           # Параллельное применение
    ADAPTIVE = "adaptive"           # Адаптивный выбор техник
    MULTI_OBJECTIVE = "multi_obj"   # Многокритериальная оптимизация
    EVOLUTIONARY = "evolutionary"   # Эволюционная оптимизация

@dataclass
class OptimizationConfig:
    """Конфигурация оптимизации"""
    # Общие параметры
    objective: OptimizationObjective = OptimizationObjective.BALANCED
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    target_compression_ratio: float = 4.0
    accuracy_threshold: float = 0.95
    latency_target_ms: float = 1.0
    memory_limit_mb: float = 50.0
    
    # Quantization параметры
    enable_quantization: bool = True
    quantization_precision: PrecisionLevel = PrecisionLevel.INT8
    quantization_mode: DynamicQuantizationMode = DynamicQuantizationMode.BALANCED
    
    # Pruning параметры
    enable_structured_pruning: bool = True
    enable_unstructured_pruning: bool = False
    structured_pruning_strategy: StructuredPruningStrategy = StructuredPruningStrategy.MAGNITUDE
    unstructured_pruning_strategy: UnstructuredPruningStrategy = UnstructuredPruningStrategy.MAGNITUDE
    
    # Distillation параметры
    enable_distillation: bool = False
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    
    # Crypto trading специфичные параметры
    crypto_features: bool = True
    hft_optimization: bool = True
    real_time_constraints: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        result = asdict(self)
        # Конвертируем enums в строки
        for key, value in result.items():
            if hasattr(value, 'value'):
                result[key] = value.value
        return result

@dataclass
class OptimizationResult:
    """Результат оптимизации модели"""
    # Модель и метрики
    optimized_model: nn.Module
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    
    # Performance метрики
    original_latency_ms: float
    optimized_latency_ms: float
    latency_improvement: float
    accuracy_retention: float
    
    # Примененные техники
    applied_techniques: List[str]
    optimization_config: OptimizationConfig
    
    # Детальная статистика
    detailed_stats: Dict[str, Any]
    optimization_time_sec: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь (исключая модель)"""
        result = asdict(self)
        # Удаляем модель из serialization
        del result['optimized_model']
        return result

class ModelOptimizer:
    """
    Универсальный оптимизатор моделей для crypto trading
    с поддержкой различных техник сжатия и оптимизации
    """
    
    def __init__(self, 
                 config: Optional[OptimizationConfig] = None,
                 device: Optional[torch.device] = None):
        """
        Args:
            config: Конфигурация оптимизации
            device: Устройство для вычислений
        """
        self.config = config or OptimizationConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger = logging.getLogger(f"{__name__}.ModelOptimizer")
        self.optimization_history = []
        self.performance_cache = {}
        
        # Инициализация оптимизаторов техник
        self._init_technique_optimizers()
    
    def _init_technique_optimizers(self):
        """Инициализация оптимизаторов для различных техник"""
        # Quantization optimizers
        if self.config.enable_quantization:
            self.quantizer = DynamicQuantizer(
                precision=self.config.quantization_precision,
                mode=self.config.quantization_mode,
                target_latency_us=self.config.latency_target_ms * 1000
            )
        
        # Pruning optimizers
        if self.config.enable_structured_pruning:
            self.structured_pruner = CryptoTradingStructuredPruner(
                target_compression_ratio=self.config.target_compression_ratio,
                accuracy_threshold=self.config.accuracy_threshold,
                latency_target_ms=self.config.latency_target_ms
            )
        
        if self.config.enable_unstructured_pruning:
            self.unstructured_pruner = CryptoTradingUnstructuredPruner(
                target_compression_ratio=self.config.target_compression_ratio,
                accuracy_threshold=self.config.accuracy_threshold,
                latency_target_ms=self.config.latency_target_ms
            )
    
    def optimize_model(self,
                      model: nn.Module,
                      train_data: torch.utils.data.DataLoader,
                      val_data: torch.utils.data.DataLoader,
                      teacher_model: Optional[nn.Module] = None) -> OptimizationResult:
        """
        Главный метод оптимизации модели
        
        Args:
            model: Исходная модель для оптимизации
            train_data: Обучающие данные
            val_data: Валидационные данные
            teacher_model: Teacher модель для distillation
            
        Returns:
            Результат оптимизации с оптимизированной моделью
        """
        start_time = time.time()
        
        self.logger.info(f"Начинаем оптимизацию модели с целью: {self.config.objective.value}")
        self.logger.info(f"Стратегия: {self.config.strategy.value}")
        
        # Получаем базовые метрики
        original_metrics = self._measure_model_performance(model, val_data)
        
        # Создаем копию модели для оптимизации
        optimized_model = copy.deepcopy(model)
        applied_techniques = []
        
        # Выбираем стратегию оптимизации
        if self.config.strategy == OptimizationStrategy.SEQUENTIAL:
            optimized_model, techniques = self._sequential_optimization(
                optimized_model, train_data, val_data, teacher_model
            )
        elif self.config.strategy == OptimizationStrategy.ADAPTIVE:
            optimized_model, techniques = self._adaptive_optimization(
                optimized_model, train_data, val_data, teacher_model
            )
        elif self.config.strategy == OptimizationStrategy.MULTI_OBJECTIVE:
            optimized_model, techniques = self._multi_objective_optimization(
                optimized_model, train_data, val_data, teacher_model
            )
        elif self.config.strategy == OptimizationStrategy.EVOLUTIONARY:
            optimized_model, techniques = self._evolutionary_optimization(
                optimized_model, train_data, val_data, teacher_model
            )
        else:  # PARALLEL
            optimized_model, techniques = self._parallel_optimization(
                optimized_model, train_data, val_data, teacher_model
            )
        
        applied_techniques.extend(techniques)
        
        # Финальные оптимизации
        optimized_model = self._apply_final_optimizations(optimized_model)
        
        # Измеряем финальные метрики
        final_metrics = self._measure_model_performance(optimized_model, val_data)
        
        # Создаем результат
        optimization_time = time.time() - start_time
        result = self._create_optimization_result(
            original_model=model,
            optimized_model=optimized_model,
            original_metrics=original_metrics,
            final_metrics=final_metrics,
            applied_techniques=applied_techniques,
            optimization_time=optimization_time
        )
        
        # Сохраняем в историю
        self.optimization_history.append({
            'timestamp': time.time(),
            'config': self.config.to_dict(),
            'result_summary': result.to_dict()
        })
        
        self.logger.info(f"Оптимизация завершена за {optimization_time:.2f}с. "
                        f"Compression ratio: {result.compression_ratio:.2f}x, "
                        f"Latency improvement: {result.latency_improvement:.1f}%")
        
        return result
    
    def _sequential_optimization(self,
                               model: nn.Module,
                               train_data: torch.utils.data.DataLoader,
                               val_data: torch.utils.data.DataLoader,
                               teacher_model: Optional[nn.Module]) -> Tuple[nn.Module, List[str]]:
        """Последовательная оптимизация"""
        applied_techniques = []
        current_model = model
        
        # 1. Knowledge Distillation (если есть teacher)
        if self.config.enable_distillation and teacher_model is not None:
            self.logger.info("Применяем knowledge distillation...")
            current_model = self._apply_knowledge_distillation(
                teacher_model, current_model, train_data, val_data
            )
            applied_techniques.append("knowledge_distillation")
        
        # 2. Structured Pruning
        if self.config.enable_structured_pruning:
            self.logger.info("Применяем structured pruning...")
            current_model = self.structured_pruner.prune_for_crypto_trading(
                current_model, train_data, val_data, self.config.structured_pruning_strategy
            )
            applied_techniques.append("structured_pruning")
        
        # 3. Unstructured Pruning
        if self.config.enable_unstructured_pruning:
            self.logger.info("Применяем unstructured pruning...")
            current_model = self.unstructured_pruner.adaptive_pruning(
                current_model, train_data, val_data
            )
            applied_techniques.append("unstructured_pruning")
        
        # 4. Quantization (последний шаг)
        if self.config.enable_quantization:
            self.logger.info("Применяем quantization...")
            input_shape = self._get_model_input_shape(current_model)
            current_model = self.quantizer.quantize_for_hft(
                current_model, input_shape
            )
            applied_techniques.append("quantization")
        
        return current_model, applied_techniques
    
    def _adaptive_optimization(self,
                              model: nn.Module,
                              train_data: torch.utils.data.DataLoader,
                              val_data: torch.utils.data.DataLoader,
                              teacher_model: Optional[nn.Module]) -> Tuple[nn.Module, List[str]]:
        """Адаптивная оптимизация с выбором лучших техник"""
        
        # Анализ модели для выбора оптимальной стратегии
        model_analysis = self._analyze_model_characteristics(model)
        
        applied_techniques = []
        current_model = model
        
        # Принятие решений на основе анализа
        if model_analysis['is_large_model'] and self.config.enable_distillation and teacher_model:
            current_model = self._apply_knowledge_distillation(
                teacher_model, current_model, train_data, val_data
            )
            applied_techniques.append("adaptive_distillation")
        
        # Выбираем pruning технику на основе архитектуры
        if model_analysis['has_conv_layers'] and self.config.enable_structured_pruning:
            current_model = self.structured_pruner.prune_for_crypto_trading(
                current_model, train_data, val_data
            )
            applied_techniques.append("adaptive_structured_pruning")
        elif model_analysis['has_dense_layers'] and self.config.enable_unstructured_pruning:
            current_model = self.unstructured_pruner.adaptive_pruning(
                current_model, train_data, val_data
            )
            applied_techniques.append("adaptive_unstructured_pruning")
        
        # Quantization если подходящая архитектура
        if model_analysis['supports_quantization'] and self.config.enable_quantization:
            input_shape = self._get_model_input_shape(current_model)
            current_model = self.quantizer.quantize_for_hft(current_model, input_shape)
            applied_techniques.append("adaptive_quantization")
        
        return current_model, applied_techniques
    
    def _multi_objective_optimization(self,
                                    model: nn.Module,
                                    train_data: torch.utils.data.DataLoader,
                                    val_data: torch.utils.data.DataLoader,
                                    teacher_model: Optional[nn.Module]) -> Tuple[nn.Module, List[str]]:
        """Многокритериальная оптимизация с Pareto-оптимальными решениями"""
        
        # Генерируем различные конфигурации оптимизации
        configurations = self._generate_pareto_configurations()
        
        best_models = []
        best_scores = []
        
        for config in configurations:
            try:
                # Применяем конфигурацию
                temp_model = copy.deepcopy(model)
                temp_model, techniques = self._apply_configuration(
                    temp_model, config, train_data, val_data, teacher_model
                )
                
                # Оценка по multiple criteria
                score = self._multi_objective_score(temp_model, val_data, config)
                
                best_models.append((temp_model, techniques, score))
                best_scores.append(score)
                
            except Exception as e:
                self.logger.warning(f"Ошибка в конфигурации: {e}")
                continue
        
        if not best_models:
            return model, []
        
        # Выбираем лучшую модель
        best_idx = np.argmax(best_scores)
        best_model, best_techniques, _ = best_models[best_idx]
        
        return best_model, ['multi_objective'] + best_techniques
    
    def _evolutionary_optimization(self,
                                 model: nn.Module,
                                 train_data: torch.utils.data.DataLoader,
                                 val_data: torch.utils.data.DataLoader,
                                 teacher_model: Optional[nn.Module]) -> Tuple[nn.Module, List[str]]:
        """Эволюционная оптимизация с генетическим алгоритмом"""
        
        population_size = 8
        generations = 5
        mutation_rate = 0.3
        
        # Инициализация популяции
        population = []
        for _ in range(population_size):
            config = self._mutate_config(copy.deepcopy(self.config), mutation_rate)
            temp_model = copy.deepcopy(model)
            try:
                temp_model, techniques = self._apply_configuration(
                    temp_model, config, train_data, val_data, teacher_model
                )
                fitness = self._calculate_fitness(temp_model, val_data)
                population.append((temp_model, techniques, config, fitness))
            except:
                continue
        
        # Эволюционный процесс
        for generation in range(generations):
            # Отбор лучших
            population.sort(key=lambda x: x[3], reverse=True)
            survivors = population[:population_size // 2]
            
            # Создание нового поколения
            new_population = list(survivors)
            
            while len(new_population) < population_size:
                # Скрещивание
                parent1, parent2 = np.random.choice(survivors, 2, replace=False)
                child_config = self._crossover_configs(parent1[2], parent2[2])
                child_config = self._mutate_config(child_config, mutation_rate)
                
                # Создание потомка
                try:
                    child_model = copy.deepcopy(model)
                    child_model, child_techniques = self._apply_configuration(
                        child_model, child_config, train_data, val_data, teacher_model
                    )
                    child_fitness = self._calculate_fitness(child_model, val_data)
                    new_population.append((child_model, child_techniques, child_config, child_fitness))
                except:
                    continue
            
            population = new_population[:population_size]
            
            best_fitness = max(p[3] for p in population)
            self.logger.info(f"Generation {generation + 1}: best fitness = {best_fitness:.4f}")
        
        # Возвращаем лучшую модель
        best_individual = max(population, key=lambda x: x[3])
        return best_individual[0], ['evolutionary'] + best_individual[1]
    
    def _parallel_optimization(self,
                             model: nn.Module,
                             train_data: torch.utils.data.DataLoader,
                             val_data: torch.utils.data.DataLoader,
                             teacher_model: Optional[nn.Module]) -> Tuple[nn.Module, List[str]]:
        """Параллельная оптимизация (простая версия)"""
        # В данной реализации применяем все техники одновременно
        return self._sequential_optimization(model, train_data, val_data, teacher_model)
    
    def _apply_knowledge_distillation(self,
                                    teacher_model: nn.Module,
                                    student_model: nn.Module,
                                    train_data: torch.utils.data.DataLoader,
                                    val_data: torch.utils.data.DataLoader) -> nn.Module:
        """Применение knowledge distillation"""
        
        distiller = ResponseDistiller(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=self.config.distillation_temperature,
            alpha=self.config.distillation_alpha
        )
        
        return distiller.distill(
            train_loader=train_data,
            val_loader=val_data,
            num_epochs=50
        )
    
    def _apply_final_optimizations(self, model: nn.Module) -> nn.Module:
        """Применение финальных оптимизаций"""
        try:
            # JIT optimization
            if hasattr(torch, 'jit'):
                dummy_input = torch.randn(1, *self._get_model_input_shape(model))
                traced_model = torch.jit.trace(model, dummy_input)
                optimized_model = torch.jit.optimize_for_inference(traced_model)
                return optimized_model
        except Exception as e:
            self.logger.warning(f"JIT оптимизация не удалась: {e}")
        
        return model
    
    def _measure_model_performance(self,
                                 model: nn.Module,
                                 val_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Измерение performance метрик модели"""
        model.eval()
        
        # Размер модели
        model_size = self._calculate_model_size_mb(model)
        
        # Latency measurement
        latency = self._measure_latency(model, val_data)
        
        # Accuracy measurement
        accuracy = self._measure_accuracy(model, val_data)
        
        # Memory usage
        memory_mb = self._measure_memory_usage(model)
        
        return {
            'size_mb': model_size,
            'latency_ms': latency,
            'accuracy': accuracy,
            'memory_mb': memory_mb
        }
    
    def _calculate_model_size_mb(self, model: nn.Module) -> float:
        """Расчет размера модели в MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / 1024 / 1024
    
    def _measure_latency(self,
                        model: nn.Module,
                        val_data: torch.utils.data.DataLoader,
                        num_iterations: int = 100) -> float:
        """Измерение латентности модели"""
        model.eval()
        
        # Получаем пример входа
        sample_batch = next(iter(val_data))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0][:1]  # Берем один образец
        else:
            sample_input = sample_batch[:1]
        
        # Прогрев
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Измерение
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(sample_input)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        return float(np.mean(latencies))
    
    def _measure_accuracy(self,
                         model: nn.Module,
                         val_data: torch.utils.data.DataLoader,
                         max_batches: int = 50) -> float:
        """Измерение точности модели"""
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_data):
                if i >= max_batches:
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets = batch, batch
                
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    outputs = outputs.get('trading_signal', outputs)
                
                # MSE loss для regression tasks
                loss = nn.MSELoss()(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        # Конвертируем в accuracy-like metric
        accuracy = max(0.0, 1.0 - avg_loss)
        return accuracy
    
    def _measure_memory_usage(self, model: nn.Module) -> float:
        """Измерение использования памяти"""
        # Простая аппроксимация на основе параметров
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        
        total_memory_mb = (param_memory + buffer_memory) / 1024 / 1024
        return total_memory_mb
    
    def _analyze_model_characteristics(self, model: nn.Module) -> Dict[str, bool]:
        """Анализ характеристик модели для выбора оптимизаций"""
        has_conv = False
        has_linear = False
        has_rnn = False
        total_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                has_conv = True
            elif isinstance(module, nn.Linear):
                has_linear = True
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                has_rnn = True
            
            if hasattr(module, 'weight'):
                total_params += module.weight.numel()
        
        is_large = total_params > 1_000_000  # 1M параметров
        
        return {
            'has_conv_layers': has_conv,
            'has_dense_layers': has_linear,
            'has_rnn_layers': has_rnn,
            'is_large_model': is_large,
            'supports_quantization': has_conv or has_linear,
            'total_parameters': total_params
        }
    
    def _get_model_input_shape(self, model: nn.Module) -> Tuple[int, ...]:
        """Определение размера входа модели"""
        first_layer = next(iter(model.modules()))
        
        if isinstance(first_layer, nn.Linear):
            return (first_layer.in_features,)
        elif isinstance(first_layer, nn.Conv1d):
            return (first_layer.in_channels, 100)
        elif isinstance(first_layer, nn.Conv2d):
            return (first_layer.in_channels, 32, 32)
        else:
            return (100,)  # Дефолтный размер
    
    def _create_optimization_result(self,
                                  original_model: nn.Module,
                                  optimized_model: nn.Module,
                                  original_metrics: Dict[str, float],
                                  final_metrics: Dict[str, float],
                                  applied_techniques: List[str],
                                  optimization_time: float) -> OptimizationResult:
        """Создание результата оптимизации"""
        
        compression_ratio = original_metrics['size_mb'] / final_metrics['size_mb']
        latency_improvement = (original_metrics['latency_ms'] - final_metrics['latency_ms']) / original_metrics['latency_ms'] * 100
        accuracy_retention = final_metrics['accuracy'] / original_metrics['accuracy']
        
        detailed_stats = {
            'original_metrics': original_metrics,
            'final_metrics': final_metrics,
            'techniques_applied': applied_techniques,
            'optimization_config': self.config.to_dict()
        }
        
        return OptimizationResult(
            optimized_model=optimized_model,
            original_size_mb=original_metrics['size_mb'],
            optimized_size_mb=final_metrics['size_mb'],
            compression_ratio=compression_ratio,
            original_latency_ms=original_metrics['latency_ms'],
            optimized_latency_ms=final_metrics['latency_ms'],
            latency_improvement=latency_improvement,
            accuracy_retention=accuracy_retention,
            applied_techniques=applied_techniques,
            optimization_config=self.config,
            detailed_stats=detailed_stats,
            optimization_time_sec=optimization_time
        )
    
    def _generate_pareto_configurations(self) -> List[OptimizationConfig]:
        """Генерация конфигураций для Pareto оптимизации"""
        configurations = []
        
        # Различные комбинации техник
        techniques_combinations = [
            {'quantization': True, 'structured_pruning': False, 'unstructured_pruning': False},
            {'quantization': False, 'structured_pruning': True, 'unstructured_pruning': False},
            {'quantization': True, 'structured_pruning': True, 'unstructured_pruning': False},
            {'quantization': True, 'structured_pruning': False, 'unstructured_pruning': True},
        ]
        
        for combo in techniques_combinations:
            config = copy.deepcopy(self.config)
            config.enable_quantization = combo['quantization']
            config.enable_structured_pruning = combo['structured_pruning']
            config.enable_unstructured_pruning = combo['unstructured_pruning']
            configurations.append(config)
        
        return configurations
    
    def _apply_configuration(self,
                           model: nn.Module,
                           config: OptimizationConfig,
                           train_data: torch.utils.data.DataLoader,
                           val_data: torch.utils.data.DataLoader,
                           teacher_model: Optional[nn.Module]) -> Tuple[nn.Module, List[str]]:
        """Применение конкретной конфигурации оптимизации"""
        
        # Временно заменяем config
        original_config = self.config
        self.config = config
        self._init_technique_optimizers()
        
        try:
            result = self._sequential_optimization(model, train_data, val_data, teacher_model)
        finally:
            # Восстанавливаем исходный config
            self.config = original_config
            self._init_technique_optimizers()
        
        return result
    
    def _multi_objective_score(self,
                              model: nn.Module,
                              val_data: torch.utils.data.DataLoader,
                              config: OptimizationConfig) -> float:
        """Многокритериальная оценка модели"""
        metrics = self._measure_model_performance(model, val_data)
        
        # Нормализованные score компоненты
        size_score = min(1.0, 100.0 / metrics['size_mb'])  # Меньше размер = лучше
        latency_score = min(1.0, config.latency_target_ms / metrics['latency_ms'])  # Меньше latency = лучше
        accuracy_score = metrics['accuracy']  # Больше accuracy = лучше
        
        # Взвешенная комбинация
        if config.objective == OptimizationObjective.LATENCY:
            return 0.5 * latency_score + 0.3 * accuracy_score + 0.2 * size_score
        elif config.objective == OptimizationObjective.MEMORY:
            return 0.5 * size_score + 0.3 * accuracy_score + 0.2 * latency_score
        elif config.objective == OptimizationObjective.ACCURACY:
            return 0.5 * accuracy_score + 0.25 * latency_score + 0.25 * size_score
        else:  # BALANCED
            return 0.33 * accuracy_score + 0.33 * latency_score + 0.34 * size_score
    
    def _calculate_fitness(self, model: nn.Module, val_data: torch.utils.data.DataLoader) -> float:
        """Вычисление fitness для эволюционного алгоритма"""
        return self._multi_objective_score(model, val_data, self.config)
    
    def _mutate_config(self, config: OptimizationConfig, mutation_rate: float) -> OptimizationConfig:
        """Мутация конфигурации для генетического алгоритма"""
        if np.random.random() < mutation_rate:
            config.enable_quantization = not config.enable_quantization
        
        if np.random.random() < mutation_rate:
            config.enable_structured_pruning = not config.enable_structured_pruning
        
        if np.random.random() < mutation_rate:
            config.target_compression_ratio *= np.random.uniform(0.8, 1.2)
        
        return config
    
    def _crossover_configs(self, config1: OptimizationConfig, config2: OptimizationConfig) -> OptimizationConfig:
        """Скрещивание конфигураций"""
        child_config = copy.deepcopy(config1)
        
        # Случайно выбираем признаки от родителей
        if np.random.random() < 0.5:
            child_config.enable_quantization = config2.enable_quantization
        
        if np.random.random() < 0.5:
            child_config.enable_structured_pruning = config2.enable_structured_pruning
        
        if np.random.random() < 0.5:
            child_config.target_compression_ratio = config2.target_compression_ratio
        
        return child_config
    
    def save_optimization_history(self, filepath: Union[str, Path]) -> None:
        """Сохранение истории оптимизации"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.optimization_history, f, indent=2, default=str)
        
        self.logger.info(f"История оптимизации сохранена в {filepath}")
    
    def load_optimization_history(self, filepath: Union[str, Path]) -> None:
        """Загрузка истории оптимизации"""
        filepath = Path(filepath)
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.optimization_history = json.load(f)
            
            self.logger.info(f"История оптимизации загружена из {filepath}")
        else:
            self.logger.warning(f"Файл истории не найден: {filepath}")
    
    def get_optimization_recommendations(self, model: nn.Module) -> Dict[str, Any]:
        """Получение рекомендаций по оптимизации модели"""
        analysis = self._analyze_model_characteristics(model)
        
        recommendations = {
            'model_analysis': analysis,
            'recommended_techniques': [],
            'expected_benefits': {},
            'risk_factors': []
        }
        
        # Рекомендации на основе анализа
        if analysis['is_large_model']:
            recommendations['recommended_techniques'].append('knowledge_distillation')
            recommendations['expected_benefits']['compression'] = 'high'
        
        if analysis['has_conv_layers']:
            recommendations['recommended_techniques'].append('structured_pruning')
            recommendations['expected_benefits']['hardware_efficiency'] = 'high'
        
        if analysis['has_dense_layers']:
            recommendations['recommended_techniques'].append('quantization')
            recommendations['expected_benefits']['memory_reduction'] = 'medium'
        
        # Факторы риска
        if analysis['has_rnn_layers']:
            recommendations['risk_factors'].append('RNN layers may be sensitive to aggressive compression')
        
        if analysis['total_parameters'] < 100_000:
            recommendations['risk_factors'].append('Small model - over-compression may hurt accuracy significantly')
        
        return recommendations