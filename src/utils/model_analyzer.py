"""
Модуль анализа ML-моделей для определения оптимальных стратегий сжатия.
Автоматический выбор техник compression на основе архитектуры модели.

Intelligent model analysis patterns для automated optimization
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import math

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Типы моделей для анализа"""
    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"
    CUSTOM = "custom"

class CompressionTechnique(Enum):
    """Техники сжатия"""
    QUANTIZATION = "quantization"
    STRUCTURED_PRUNING = "structured_pruning"
    UNSTRUCTURED_PRUNING = "unstructured_pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    LOW_RANK_APPROXIMATION = "low_rank_approximation"

class AnalysisLevel(Enum):
    """Уровни анализа модели"""
    BASIC = "basic"              # Базовые характеристики
    DETAILED = "detailed"       # Детальный анализ слоев
    COMPREHENSIVE = "comprehensive"  # Полный анализ с рекомендациями

@dataclass
class LayerAnalysis:
    """Анализ отдельного слоя"""
    name: str
    layer_type: str
    parameters: int
    memory_mb: float
    flops: int
    compression_potential: float  # 0-1 score
    recommended_techniques: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ModelAnalysisResult:
    """Результат анализа модели"""
    model_type: ModelType
    total_parameters: int
    total_memory_mb: float
    total_flops: int
    
    # Структурные характеристики
    num_linear_layers: int
    num_conv_layers: int
    num_rnn_layers: int
    num_attention_layers: int
    
    # Анализ по слоям
    layer_analyses: List[LayerAnalysis]
    bottleneck_layers: List[str]
    
    # Потенциал сжатия
    compression_potential: Dict[str, float]  # technique -> potential score
    
    # Рекомендации
    recommended_techniques: List[str]
    compression_strategy: str
    expected_compression_ratio: float
    
    # Риски и ограничения
    compression_risks: List[str]
    accuracy_impact_estimate: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['model_type'] = self.model_type.value
        return result

class ModelAnalyzer:
    """
    Интеллектуальный анализатор моделей для определения оптимальных
    стратегий сжатия с учетом специфики crypto trading
    """
    
    def __init__(self, crypto_domain_focus: bool = True):
        """
        Args:
            crypto_domain_focus: Фокус на crypto trading специфике
        """
        self.crypto_domain_focus = crypto_domain_focus
        self.logger = logging.getLogger(f"{__name__}.ModelAnalyzer")
        
        # Веса для различных техник сжатия
        self.technique_weights = self._initialize_technique_weights()
        
        # Patterns для распознавания архитектур
        self.architecture_patterns = self._initialize_architecture_patterns()
    
    def analyze_model(self, 
                     model: nn.Module,
                     sample_input: Optional[torch.Tensor] = None,
                     analysis_level: AnalysisLevel = AnalysisLevel.COMPREHENSIVE) -> ModelAnalysisResult:
        """
        Основной метод анализа модели
        
        Args:
            model: Модель для анализа
            sample_input: Пример входных данных
            analysis_level: Уровень детализации анализа
            
        Returns:
            Результат анализа с рекомендациями
        """
        self.logger.info(f"Начинаем анализ модели на уровне {analysis_level.value}")
        
        # Базовый анализ структуры
        basic_stats = self._analyze_basic_structure(model)
        
        # Определение типа модели
        model_type = self._classify_model_type(model, basic_stats)
        
        # Анализ слоев
        layer_analyses = []
        if analysis_level in [AnalysisLevel.DETAILED, AnalysisLevel.COMPREHENSIVE]:
            layer_analyses = self._analyze_layers(model, sample_input)
        
        # Идентификация bottleneck слоев
        bottleneck_layers = self._identify_bottlenecks(model, layer_analyses)
        
        # Анализ потенциала сжатия
        compression_potential = self._analyze_compression_potential(
            model, model_type, layer_analyses
        )
        
        # Генерация рекомендаций
        recommendations = self._generate_recommendations(
            model, model_type, compression_potential, layer_analyses
        )
        
        # Оценка рисков
        risks, accuracy_impact = self._assess_compression_risks(
            model, model_type, recommendations['recommended_techniques']
        )
        
        result = ModelAnalysisResult(
            model_type=model_type,
            total_parameters=basic_stats['total_parameters'],
            total_memory_mb=basic_stats['total_memory_mb'],
            total_flops=basic_stats['estimated_flops'],
            num_linear_layers=basic_stats['num_linear'],
            num_conv_layers=basic_stats['num_conv'],
            num_rnn_layers=basic_stats['num_rnn'],
            num_attention_layers=basic_stats['num_attention'],
            layer_analyses=layer_analyses,
            bottleneck_layers=bottleneck_layers,
            compression_potential=compression_potential,
            recommended_techniques=recommendations['recommended_techniques'],
            compression_strategy=recommendations['compression_strategy'],
            expected_compression_ratio=recommendations['expected_compression_ratio'],
            compression_risks=risks,
            accuracy_impact_estimate=accuracy_impact
        )
        
        self.logger.info(f"Анализ завершен. Тип модели: {model_type.value}")
        self.logger.info(f"Рекомендуемые техники: {recommendations['recommended_techniques']}")
        
        return result
    
    def _analyze_basic_structure(self, model: nn.Module) -> Dict[str, Any]:
        """Базовый анализ структуры модели"""
        
        stats = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'total_memory_mb': 0,
            'estimated_flops': 0,
            'num_linear': 0,
            'num_conv': 0,
            'num_rnn': 0,
            'num_attention': 0,
            'num_layers_total': 0,
            'max_layer_size': 0,
            'layer_size_distribution': []
        }
        
        layer_sizes = []
        
        for name, module in model.named_modules():
            # Считаем слои
            if isinstance(module, nn.Linear):
                stats['num_linear'] += 1
                layer_size = module.in_features * module.out_features
                layer_sizes.append(layer_size)
                stats['estimated_flops'] += layer_size * 2  # multiply + add
                
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                stats['num_conv'] += 1
                if isinstance(module, nn.Conv1d):
                    kernel_size = module.kernel_size[0]
                elif isinstance(module, nn.Conv2d):
                    kernel_size = module.kernel_size[0] * module.kernel_size[1]
                else:  # Conv3d
                    kernel_size = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
                
                layer_flops = module.in_channels * module.out_channels * kernel_size
                stats['estimated_flops'] += layer_flops
                layer_sizes.append(layer_flops)
                
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                stats['num_rnn'] += 1
                # Приблизительная оценка FLOPs для RNN
                hidden_size = module.hidden_size
                input_size = module.input_size
                rnn_flops = (input_size * hidden_size + hidden_size * hidden_size) * 4  # 4 gates for LSTM
                stats['estimated_flops'] += rnn_flops
                layer_sizes.append(rnn_flops)
                
            elif isinstance(module, nn.MultiheadAttention):
                stats['num_attention'] += 1
                # Приблизительная оценка для attention
                embed_dim = module.embed_dim
                attention_flops = embed_dim * embed_dim * 3  # Q, K, V projections
                stats['estimated_flops'] += attention_flops
                layer_sizes.append(attention_flops)
        
        # Подсчет параметров и памяти
        for param in model.parameters():
            param_count = param.numel()
            stats['total_parameters'] += param_count
            
            if param.requires_grad:
                stats['trainable_parameters'] += param_count
            
            stats['total_memory_mb'] += param_count * param.element_size()
        
        stats['total_memory_mb'] /= (1024 * 1024)  # Convert to MB
        stats['num_layers_total'] = len(list(model.modules())) - 1  # Exclude root module
        
        if layer_sizes:
            stats['max_layer_size'] = max(layer_sizes)
            stats['layer_size_distribution'] = self._calculate_size_distribution(layer_sizes)
        
        return stats
    
    def _classify_model_type(self, model: nn.Module, basic_stats: Dict[str, Any]) -> ModelType:
        """Классификация типа модели на основе структуры"""
        
        total_layers = max(1, basic_stats['num_layers_total'])
        
        # Соотношения различных типов слоев
        linear_ratio = basic_stats['num_linear'] / total_layers
        conv_ratio = basic_stats['num_conv'] / total_layers
        rnn_ratio = basic_stats['num_rnn'] / total_layers
        attention_ratio = basic_stats['num_attention'] / total_layers
        
        # Классификация на основе доминирующих компонентов
        if attention_ratio > 0.1:  # Больше 10% attention слоев
            return ModelType.TRANSFORMER
        
        elif rnn_ratio > 0.1:  # Больше 10% RNN слоев
            return ModelType.RECURRENT
        
        elif conv_ratio > 0.3:  # Больше 30% conv слоев
            if linear_ratio > 0.2:
                return ModelType.HYBRID  # Conv + Linear
            else:
                return ModelType.CONVOLUTIONAL
        
        elif linear_ratio > 0.5:  # Больше 50% linear слоев
            return ModelType.FEEDFORWARD
        
        elif linear_ratio > 0.2 and conv_ratio > 0.1:
            return ModelType.HYBRID
        
        else:
            return ModelType.CUSTOM
    
    def _analyze_layers(self, 
                       model: nn.Module,
                       sample_input: Optional[torch.Tensor]) -> List[LayerAnalysis]:
        """Детальный анализ каждого слоя"""
        
        layer_analyses = []
        
        # Если есть sample_input, можем измерить FLOPs более точно
        if sample_input is not None:
            try:
                flop_counts = self._profile_model_flops(model, sample_input)
            except:
                flop_counts = {}
        else:
            flop_counts = {}
        
        for name, module in model.named_modules():
            # Анализируем только leaf модули с параметрами
            if list(module.children()) or not any(p.numel() > 0 for p in module.parameters()):
                continue
            
            # Подсчет параметров слоя
            layer_params = sum(p.numel() for p in module.parameters())
            layer_memory = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 * 1024)
            
            # FLOPs для слоя
            layer_flops = flop_counts.get(name, self._estimate_layer_flops(module))
            
            # Оценка потенциала сжатия для слоя
            compression_potential = self._evaluate_layer_compression_potential(module, layer_params)
            
            # Рекомендуемые техники для слоя
            layer_techniques = self._recommend_layer_techniques(module, compression_potential)
            
            analysis = LayerAnalysis(
                name=name,
                layer_type=type(module).__name__,
                parameters=layer_params,
                memory_mb=layer_memory,
                flops=layer_flops,
                compression_potential=compression_potential,
                recommended_techniques=layer_techniques
            )
            
            layer_analyses.append(analysis)
        
        return layer_analyses
    
    def _identify_bottlenecks(self, 
                            model: nn.Module,
                            layer_analyses: List[LayerAnalysis]) -> List[str]:
        """Идентификация bottleneck слоев"""
        
        if not layer_analyses:
            return []
        
        bottlenecks = []
        
        # Сортируем слои по количеству параметров
        sorted_layers = sorted(layer_analyses, key=lambda x: x.parameters, reverse=True)
        
        total_params = sum(layer.parameters for layer in layer_analyses)
        
        # Слои, составляющие больше 20% от общих параметров
        param_threshold = total_params * 0.2
        
        for layer in sorted_layers:
            if layer.parameters > param_threshold:
                bottlenecks.append(layer.name)
        
        # Также добавляем слои с высоким FLOPs
        sorted_by_flops = sorted(layer_analyses, key=lambda x: x.flops, reverse=True)
        total_flops = sum(layer.flops for layer in layer_analyses)
        flops_threshold = total_flops * 0.15
        
        for layer in sorted_by_flops:
            if layer.flops > flops_threshold and layer.name not in bottlenecks:
                bottlenecks.append(layer.name)
        
        return bottlenecks[:10]  # Ограничиваем количество
    
    def _analyze_compression_potential(self,
                                     model: nn.Module,
                                     model_type: ModelType,
                                     layer_analyses: List[LayerAnalysis]) -> Dict[str, float]:
        """Анализ потенциала различных техник сжатия"""
        
        potential = {}
        
        # Quantization potential
        potential['quantization'] = self._evaluate_quantization_potential(model, model_type)
        
        # Structured pruning potential
        potential['structured_pruning'] = self._evaluate_structured_pruning_potential(
            model, model_type, layer_analyses
        )
        
        # Unstructured pruning potential
        potential['unstructured_pruning'] = self._evaluate_unstructured_pruning_potential(
            model, model_type, layer_analyses
        )
        
        # Knowledge distillation potential
        potential['knowledge_distillation'] = self._evaluate_distillation_potential(
            model, model_type
        )
        
        # Low-rank approximation potential
        potential['low_rank_approximation'] = self._evaluate_lowrank_potential(
            model, model_type, layer_analyses
        )
        
        return potential
    
    def _generate_recommendations(self,
                                model: nn.Module,
                                model_type: ModelType,
                                compression_potential: Dict[str, float],
                                layer_analyses: List[LayerAnalysis]) -> Dict[str, Any]:
        """Генерация рекомендаций по сжатию"""
        
        # Сортируем техники по потенциалу
        sorted_techniques = sorted(
            compression_potential.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Выбираем топ техники
        recommended_techniques = []
        for technique, potential in sorted_techniques:
            if potential > 0.3:  # Порог потенциала
                recommended_techniques.append(technique)
        
        # Определяем стратегию сжатия
        compression_strategy = self._determine_compression_strategy(
            model_type, recommended_techniques, compression_potential
        )
        
        # Оценка ожидаемого compression ratio
        expected_ratio = self._estimate_compression_ratio(
            recommended_techniques, compression_potential
        )
        
        return {
            'recommended_techniques': recommended_techniques,
            'compression_strategy': compression_strategy,
            'expected_compression_ratio': expected_ratio,
            'technique_priorities': sorted_techniques
        }
    
    def _assess_compression_risks(self,
                                model: nn.Module,
                                model_type: ModelType,
                                recommended_techniques: List[str]) -> Tuple[List[str], float]:
        """Оценка рисков сжатия модели"""
        
        risks = []
        accuracy_impact = 0.0
        
        # Риски в зависимости от типа модели
        if model_type == ModelType.TRANSFORMER:
            risks.append("Transformer модели чувствительны к агрессивному сжатию")
            accuracy_impact += 0.1
        
        if model_type == ModelType.RECURRENT:
            risks.append("RNN слои могут быть нестабильны после квантизации")
            accuracy_impact += 0.05
        
        # Риски в зависимости от техник
        if 'quantization' in recommended_techniques:
            risks.append("Квантизация может повлиять на точность для малых моделей")
            accuracy_impact += 0.03
        
        if 'structured_pruning' in recommended_techniques:
            risks.append("Структурированная pruning может удалить важные features")
            accuracy_impact += 0.05
        
        if len(recommended_techniques) > 2:
            risks.append("Комбинация множественных техник увеличивает риск потери точности")
            accuracy_impact += 0.1
        
        # Crypto trading специфические риски
        if self.crypto_domain_focus:
            risks.append("Для crypto trading критична стабильность предсказаний")
            risks.append("Directional accuracy может пострадать от aggressive compression")
            accuracy_impact += 0.05
        
        return risks, min(accuracy_impact, 0.5)  # Максимум 50% impact
    
    # Helper methods для анализа
    
    def _calculate_size_distribution(self, layer_sizes: List[int]) -> Dict[str, float]:
        """Расчет распределения размеров слоев"""
        
        if not layer_sizes:
            return {}
        
        sizes = np.array(layer_sizes)
        
        return {
            'mean': float(np.mean(sizes)),
            'std': float(np.std(sizes)),
            'min': float(np.min(sizes)),
            'max': float(np.max(sizes)),
            'q25': float(np.percentile(sizes, 25)),
            'q50': float(np.percentile(sizes, 50)),
            'q75': float(np.percentile(sizes, 75))
        }
    
    def _profile_model_flops(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, int]:
        """Профилирование FLOPs модели"""
        
        # Упрощенное профилирование - в production используем специальные tools
        flop_counts = {}
        
        def flop_count_hook(name):
            def hook(module, input, output):
                # Простая оценка FLOPs
                if isinstance(module, nn.Linear):
                    flops = module.in_features * module.out_features
                elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    # Упрощенная оценка для conv
                    if hasattr(output, 'numel'):
                        flops = output.numel() * module.weight.numel() // module.out_channels
                    else:
                        flops = 0
                else:
                    flops = 0
                
                flop_counts[name] = flops
            
            return hook
        
        # Регистрируем hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                handle = module.register_forward_hook(flop_count_hook(name))
                handles.append(handle)
        
        # Forward pass
        try:
            with torch.no_grad():
                model(sample_input)
        finally:
            # Удаляем hooks
            for handle in handles:
                handle.remove()
        
        return flop_counts
    
    def _estimate_layer_flops(self, module: nn.Module) -> int:
        """Оценка FLOPs для отдельного слоя"""
        
        if isinstance(module, nn.Linear):
            return module.in_features * module.out_features * 2  # multiply + add
        
        elif isinstance(module, nn.Conv1d):
            kernel_size = module.kernel_size[0]
            return module.in_channels * module.out_channels * kernel_size * 2
        
        elif isinstance(module, nn.Conv2d):
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            return module.in_channels * module.out_channels * kernel_size * 2
        
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            # Приблизительная оценка для RNN
            input_size = module.input_size
            hidden_size = module.hidden_size
            return (input_size * hidden_size + hidden_size * hidden_size) * 4 * 2
        
        elif isinstance(module, nn.MultiheadAttention):
            embed_dim = module.embed_dim
            return embed_dim * embed_dim * 6  # Q, K, V projections + output
        
        else:
            return 0
    
    def _evaluate_layer_compression_potential(self, module: nn.Module, layer_params: int) -> float:
        """Оценка потенциала сжатия для отдельного слоя"""
        
        potential = 0.0
        
        # Базовый потенциал на основе размера слоя
        if layer_params > 100000:  # Большие слои
            potential += 0.4
        elif layer_params > 10000:  # Средние слои
            potential += 0.3
        else:  # Маленькие слои
            potential += 0.1
        
        # Дополнительный потенциал в зависимости от типа слоя
        if isinstance(module, nn.Linear):
            potential += 0.3  # Linear слои хорошо сжимаются
        elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
            potential += 0.2  # Conv слои умеренно сжимаются
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            potential += 0.1  # RNN слои сложнее сжимать
        
        return min(potential, 1.0)
    
    def _recommend_layer_techniques(self, module: nn.Module, compression_potential: float) -> List[str]:
        """Рекомендация техник сжатия для слоя"""
        
        techniques = []
        
        if compression_potential > 0.5:
            # Высокий потенциал - можно применить агрессивные техники
            if isinstance(module, nn.Linear):
                techniques.extend(['quantization', 'structured_pruning', 'low_rank_approximation'])
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                techniques.extend(['quantization', 'structured_pruning'])
            else:
                techniques.append('quantization')
        
        elif compression_potential > 0.3:
            # Умеренный потенциал
            techniques.append('quantization')
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                techniques.append('unstructured_pruning')
        
        else:
            # Низкий потенциал - только консервативные техники
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                techniques.append('quantization')
        
        return techniques
    
    def _evaluate_quantization_potential(self, model: nn.Module, model_type: ModelType) -> float:
        """Оценка потенциала квантизации"""
        
        base_potential = 0.7  # Quantization обычно работает хорошо
        
        # Корректировка на основе типа модели
        if model_type == ModelType.FEEDFORWARD:
            base_potential += 0.2
        elif model_type == ModelType.CONVOLUTIONAL:
            base_potential += 0.1
        elif model_type == ModelType.TRANSFORMER:
            base_potential -= 0.1  # Attention может быть чувствительным
        elif model_type == ModelType.RECURRENT:
            base_potential -= 0.2  # RNN могут быть нестабильными
        
        # Проверка на наличие BatchNorm (улучшает quantization)
        has_batchnorm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())
        if has_batchnorm:
            base_potential += 0.1
        
        return min(base_potential, 1.0)
    
    def _evaluate_structured_pruning_potential(self,
                                             model: nn.Module,
                                             model_type: ModelType,
                                             layer_analyses: List[LayerAnalysis]) -> float:
        """Оценка потенциала структурированной pruning"""
        
        base_potential = 0.5
        
        # Высокий потенциал для моделей с большим количеством conv/linear слоев
        if model_type in [ModelType.CONVOLUTIONAL, ModelType.FEEDFORWARD]:
            base_potential += 0.3
        elif model_type == ModelType.HYBRID:
            base_potential += 0.2
        elif model_type == ModelType.TRANSFORMER:
            base_potential += 0.1
        
        # Корректировка на основе redundancy в слоях
        if layer_analyses:
            large_layers = [l for l in layer_analyses if l.parameters > 50000]
            redundancy_score = len(large_layers) / len(layer_analyses) if layer_analyses else 0
            base_potential += redundancy_score * 0.2
        
        return min(base_potential, 1.0)
    
    def _evaluate_unstructured_pruning_potential(self,
                                               model: nn.Module,
                                               model_type: ModelType,
                                               layer_analyses: List[LayerAnalysis]) -> float:
        """Оценка потенциала неструктурированной pruning"""
        
        base_potential = 0.6  # Unstructured обычно работает хорошо
        
        # Особенно эффективно для over-parameterized моделей
        if layer_analyses:
            total_params = sum(l.parameters for l in layer_analyses)
            if total_params > 1000000:  # Более 1M параметров
                base_potential += 0.2
            elif total_params > 100000:  # Более 100K параметров
                base_potential += 0.1
        
        # Корректировка по типу модели
        if model_type == ModelType.FEEDFORWARD:
            base_potential += 0.1
        elif model_type == ModelType.RECURRENT:
            base_potential -= 0.2  # RNN чувствительны к pruning
        
        return min(base_potential, 1.0)
    
    def _evaluate_distillation_potential(self, model: nn.Module, model_type: ModelType) -> float:
        """Оценка потенциала knowledge distillation"""
        
        # Distillation эффективно для больших моделей
        total_params = sum(p.numel() for p in model.parameters())
        
        if total_params > 10000000:  # Более 10M параметров
            base_potential = 0.8
        elif total_params > 1000000:  # Более 1M параметров
            base_potential = 0.6
        elif total_params > 100000:  # Более 100K параметров
            base_potential = 0.4
        else:
            base_potential = 0.2  # Малые модели плохо distill
        
        # Transformer и CNN модели особенно хорошо distill
        if model_type in [ModelType.TRANSFORMER, ModelType.CONVOLUTIONAL]:
            base_potential += 0.1
        
        return min(base_potential, 1.0)
    
    def _evaluate_lowrank_potential(self,
                                  model: nn.Module,
                                  model_type: ModelType,
                                  layer_analyses: List[LayerAnalysis]) -> float:
        """Оценка потенциала low-rank approximation"""
        
        base_potential = 0.3  # Консервативная оценка
        
        # Особенно эффективно для Linear слоев
        linear_layers = [l for l in layer_analyses if 'Linear' in l.layer_type]
        if linear_layers:
            large_linear = [l for l in linear_layers if l.parameters > 10000]
            if large_linear:
                base_potential += 0.4
        
        # Transformer модели хорошо подходят для low-rank
        if model_type == ModelType.TRANSFORMER:
            base_potential += 0.3
        
        return min(base_potential, 1.0)
    
    def _determine_compression_strategy(self,
                                      model_type: ModelType,
                                      recommended_techniques: List[str],
                                      compression_potential: Dict[str, float]) -> str:
        """Определение общей стратегии сжатия"""
        
        if not recommended_techniques:
            return "conservative"  # Консервативный подход
        
        # Агрессивная стратегия если высокий потенциал у нескольких техник
        high_potential_techniques = [
            t for t, p in compression_potential.items() if p > 0.7
        ]
        
        if len(high_potential_techniques) >= 2:
            return "aggressive"
        
        # Комбинированная стратегия
        if len(recommended_techniques) > 1:
            # Приоритизируем комбинации
            if 'quantization' in recommended_techniques and 'structured_pruning' in recommended_techniques:
                return "quantization_first_then_pruning"
            elif 'knowledge_distillation' in recommended_techniques:
                return "distillation_based"
            else:
                return "combined"
        
        # Одиночная техника
        primary_technique = recommended_techniques[0]
        
        if primary_technique == 'quantization':
            return "quantization_focused"
        elif primary_technique == 'structured_pruning':
            return "pruning_focused"
        elif primary_technique == 'knowledge_distillation':
            return "distillation_focused"
        else:
            return "technique_specific"
    
    def _estimate_compression_ratio(self,
                                  recommended_techniques: List[str],
                                  compression_potential: Dict[str, float]) -> float:
        """Оценка ожидаемого коэффициента сжатия"""
        
        if not recommended_techniques:
            return 1.0  # Нет сжатия
        
        # Базовые коэффициенты для техник
        technique_ratios = {
            'quantization': 2.0,  # INT8 обычно дает 2x
            'structured_pruning': 2.5,
            'unstructured_pruning': 3.0,
            'knowledge_distillation': 4.0,
            'low_rank_approximation': 2.0
        }
        
        total_ratio = 1.0
        
        for technique in recommended_techniques:
            base_ratio = technique_ratios.get(technique, 1.0)
            potential = compression_potential.get(technique, 0.5)
            
            # Корректируем ratio на основе потенциала
            effective_ratio = 1.0 + (base_ratio - 1.0) * potential
            
            # Комбинируем ratios (не просто умножаем, так как есть diminishing returns)
            total_ratio *= math.pow(effective_ratio, 0.8)
        
        return min(total_ratio, 10.0)  # Максимум 10x сжатие
    
    def _initialize_technique_weights(self) -> Dict[str, float]:
        """Инициализация весов техник для crypto trading"""
        
        if self.crypto_domain_focus:
            return {
                'quantization': 0.9,  # Высокий приоритет - меньше влияет на accuracy
                'structured_pruning': 0.7,
                'unstructured_pruning': 0.6,  # Может влиять на stability
                'knowledge_distillation': 0.8,  # Хорошо для сложных моделей
                'low_rank_approximation': 0.5  # Осторожно для trading
            }
        else:
            return {
                'quantization': 0.8,
                'structured_pruning': 0.8,
                'unstructured_pruning': 0.7,
                'knowledge_distillation': 0.9,
                'low_rank_approximation': 0.6
            }
    
    def _initialize_architecture_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Инициализация паттернов архитектур"""
        
        return {
            'feedforward': {
                'min_linear_ratio': 0.6,
                'max_conv_ratio': 0.1,
                'preferred_techniques': ['quantization', 'unstructured_pruning']
            },
            'convolutional': {
                'min_conv_ratio': 0.3,
                'preferred_techniques': ['quantization', 'structured_pruning']
            },
            'transformer': {
                'min_attention_ratio': 0.1,
                'preferred_techniques': ['knowledge_distillation', 'quantization']
            },
            'recurrent': {
                'min_rnn_ratio': 0.1,
                'preferred_techniques': ['quantization'],
                'avoid_techniques': ['unstructured_pruning']
            }
        }
    
    def get_compression_recommendations_summary(self, analysis_result: ModelAnalysisResult) -> Dict[str, Any]:
        """Получение краткого summary рекомендаций"""
        
        return {
            'model_type': analysis_result.model_type.value,
            'model_size_mb': analysis_result.total_memory_mb,
            'parameters_count': analysis_result.total_parameters,
            'primary_recommendation': analysis_result.recommended_techniques[0] if analysis_result.recommended_techniques else 'none',
            'expected_compression': f"{analysis_result.expected_compression_ratio:.1f}x",
            'compression_strategy': analysis_result.compression_strategy,
            'accuracy_risk': 'high' if analysis_result.accuracy_impact_estimate > 0.15 else 'medium' if analysis_result.accuracy_impact_estimate > 0.05 else 'low',
            'bottlenecks_count': len(analysis_result.bottleneck_layers),
            'top_bottleneck': analysis_result.bottleneck_layers[0] if analysis_result.bottleneck_layers else 'none'
        }