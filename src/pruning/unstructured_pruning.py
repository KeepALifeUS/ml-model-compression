"""
Модуль неструктурированной pruning для криптотрейдинговых моделей.
Удаляет отдельные веса с высокой гранулярностью для максимального сжатия.

Fine-grained optimization patterns для edge deployment
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from enum import Enum
import copy
import math

logger = logging.getLogger(__name__)

class UnstructuredPruningStrategy(Enum):
    """Стратегии неструктурированной pruning"""
    MAGNITUDE = "magnitude"              # По абсолютной величине весов
    RANDOM = "random"                   # Случайная pruning
    SNIP = "snip"                      # SNIP (Single-shot Network Pruning)
    GRASP = "grasp"                    # GraSP (Gradient Signal Preservation)
    LOTTERY_TICKET = "lottery_ticket"   # Lottery Ticket Hypothesis
    MAGNITUDE_STRUCTURED = "magnitude_structured"  # Гибридный подход

class PruningSchedule(Enum):
    """Расписания для gradual pruning"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    COSINE = "cosine"

class SparsityPattern(Enum):
    """Паттерны разреженности"""
    RANDOM = "random"
    BLOCK = "block"          # Block sparsity
    STRUCTURED = "structured"  # Структурированная разреженность
    N_M = "n_m"             # N:M sparsity (например 2:4)

class UnstructuredPruner:
    """
    Базовый класс неструктурированной pruning с поддержкой различных стратегий
    и расписаний для crypto trading моделей
    """
    
    def __init__(self, 
                 target_sparsity: float = 0.9,
                 strategy: UnstructuredPruningStrategy = UnstructuredPruningStrategy.MAGNITUDE,
                 schedule: PruningSchedule = PruningSchedule.EXPONENTIAL,
                 sparsity_pattern: SparsityPattern = SparsityPattern.RANDOM):
        """
        Args:
            target_sparsity: Целевая разреженность (0.9 = 90% весов обнулены)
            strategy: Стратегия pruning
            schedule: Расписание gradual pruning
            sparsity_pattern: Паттерн разреженности
        """
        self.target_sparsity = target_sparsity
        self.strategy = strategy
        self.schedule = schedule
        self.sparsity_pattern = sparsity_pattern
        
        self.logger = logging.getLogger(f"{__name__}.UnstructuredPruner")
        self.pruning_stats = {}
        self.sparsity_history = []
        
    def prune_model(self, 
                   model: nn.Module,
                   data_loader: Optional[torch.utils.data.DataLoader] = None,
                   num_steps: int = 10,
                   fine_tune_fn: Optional[Callable] = None) -> nn.Module:
        """
        Основной метод неструктурированной pruning
        
        Args:
            model: Модель для pruning
            data_loader: Данные для градиентов (для SNIP/GraSP)
            num_steps: Количество шагов gradual pruning
            fine_tune_fn: Функция fine-tuning между шагами
            
        Returns:
            Pruned модель
        """
        self.logger.info(f"Начинаем неструктурированную pruning "
                        f"до {self.target_sparsity*100:.1f}% sparsity")
        
        pruned_model = copy.deepcopy(model)
        
        if self.strategy in [UnstructuredPruningStrategy.SNIP, UnstructuredPruningStrategy.GRASP]:
            # Single-shot методы
            pruned_model = self._single_shot_pruning(pruned_model, data_loader)
        else:
            # Gradual pruning
            pruned_model = self._gradual_pruning(
                pruned_model, data_loader, num_steps, fine_tune_fn
            )
        
        # Финализация и анализ результатов
        final_sparsity = self._calculate_global_sparsity(pruned_model)
        
        self.pruning_stats = {
            "final_sparsity": final_sparsity,
            "target_sparsity": self.target_sparsity,
            "strategy": self.strategy.value,
            "schedule": self.schedule.value,
            "sparsity_pattern": self.sparsity_pattern.value,
            "num_parameters_pruned": self._count_pruned_parameters(pruned_model),
            "total_parameters": self._count_total_parameters(pruned_model)
        }
        
        self.logger.info(f"Неструктурированная pruning завершена. "
                        f"Достигнута sparsity: {final_sparsity*100:.1f}%")
        
        return pruned_model
    
    def _gradual_pruning(self, 
                        model: nn.Module,
                        data_loader: Optional[torch.utils.data.DataLoader],
                        num_steps: int,
                        fine_tune_fn: Optional[Callable]) -> nn.Module:
        """Постепенная pruning с использованием расписания"""
        current_sparsity = 0.0
        
        for step in range(num_steps):
            # Вычисляем целевую sparsity для текущего шага
            step_sparsity = self._calculate_step_sparsity(step, num_steps)
            
            if step_sparsity <= current_sparsity:
                continue
            
            self.logger.info(f"Шаг {step + 1}/{num_steps}: "
                           f"sparsity {current_sparsity*100:.1f}% -> {step_sparsity*100:.1f}%")
            
            # Применяем pruning для текущего шага
            model = self._apply_pruning_step(model, step_sparsity, data_loader)
            
            # Fine-tuning если предоставлена функция
            if fine_tune_fn and step < num_steps - 1:  # Не fine-tune на последнем шаге
                model = fine_tune_fn(model)
            
            current_sparsity = step_sparsity
            
            # Сохраняем историю
            actual_sparsity = self._calculate_global_sparsity(model)
            self.sparsity_history.append({
                "step": step,
                "target_sparsity": step_sparsity,
                "actual_sparsity": actual_sparsity,
                "model_size_mb": self._calculate_model_size(model)
            })
        
        return model
    
    def _single_shot_pruning(self, 
                            model: nn.Module,
                            data_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Single-shot pruning методы (SNIP, GraSP)"""
        if data_loader is None:
            raise ValueError(f"DataLoader необходим для {self.strategy.value}")
        
        if self.strategy == UnstructuredPruningStrategy.SNIP:
            return self._snip_pruning(model, data_loader)
        elif self.strategy == UnstructuredPruningStrategy.GRASP:
            return self._grasp_pruning(model, data_loader)
        else:
            raise ValueError(f"Неподдерживаемая single-shot стратегия: {self.strategy.value}")
    
    def _calculate_step_sparsity(self, step: int, total_steps: int) -> float:
        """Вычисление sparsity для текущего шага на основе расписания"""
        progress = (step + 1) / total_steps
        
        if self.schedule == PruningSchedule.LINEAR:
            return progress * self.target_sparsity
        
        elif self.schedule == PruningSchedule.EXPONENTIAL:
            # Экспоненциальное расписание: быстрый рост в начале
            return self.target_sparsity * (1 - (1 - progress) ** 3)
        
        elif self.schedule == PruningSchedule.POLYNOMIAL:
            # Полиномиальное расписание
            return self.target_sparsity * (progress ** 2)
        
        elif self.schedule == PruningSchedule.COSINE:
            # Cosine annealing schedule
            return self.target_sparsity * (1 - math.cos(progress * math.pi / 2))
        
        else:
            return progress * self.target_sparsity
    
    def _apply_pruning_step(self, 
                           model: nn.Module,
                           target_sparsity: float,
                           data_loader: Optional[torch.utils.data.DataLoader]) -> nn.Module:
        """Применение одного шага pruning"""
        if self.strategy == UnstructuredPruningStrategy.MAGNITUDE:
            return self._magnitude_pruning(model, target_sparsity)
        
        elif self.strategy == UnstructuredPruningStrategy.RANDOM:
            return self._random_pruning(model, target_sparsity)
        
        elif self.strategy == UnstructuredPruningStrategy.LOTTERY_TICKET:
            return self._lottery_ticket_pruning(model, target_sparsity)
        
        else:
            raise ValueError(f"Неподдерживаемая стратегия: {self.strategy.value}")
    
    def _magnitude_pruning(self, model: nn.Module, target_sparsity: float) -> nn.Module:
        """Pruning по абсолютной величине весов"""
        # Собираем все веса для глобального ranking
        all_weights = []
        layer_info = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weights = module.weight.data.abs().flatten()
                all_weights.append(weights)
                layer_info.append((name, module, len(weights)))
        
        if not all_weights:
            return model
        
        # Глобальный порог
        all_weights_tensor = torch.cat(all_weights)
        threshold_idx = int(target_sparsity * len(all_weights_tensor))
        threshold_value = torch.kthvalue(all_weights_tensor, threshold_idx + 1).values
        
        # Применяем pruning к каждому слою
        for name, module, _ in layer_info:
            mask = (module.weight.data.abs() >= threshold_value).float()
            
            if self.sparsity_pattern == SparsityPattern.BLOCK:
                mask = self._apply_block_sparsity(mask)
            elif self.sparsity_pattern == SparsityPattern.N_M:
                mask = self._apply_n_m_sparsity(mask, n=2, m=4)
            
            # Применяем маску
            prune.custom_from_mask(module, name='weight', mask=mask)
        
        return model
    
    def _random_pruning(self, model: nn.Module, target_sparsity: float) -> nn.Module:
        """Случайная pruning"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Создаем случайную маску
                mask = torch.rand_like(module.weight.data) > target_sparsity
                mask = mask.float()
                
                if self.sparsity_pattern == SparsityPattern.BLOCK:
                    mask = self._apply_block_sparsity(mask)
                elif self.sparsity_pattern == SparsityPattern.N_M:
                    mask = self._apply_n_m_sparsity(mask, n=2, m=4)
                
                prune.custom_from_mask(module, name='weight', mask=mask)
        
        return model
    
    def _lottery_ticket_pruning(self, model: nn.Module, target_sparsity: float) -> nn.Module:
        """
        Lottery Ticket Hypothesis pruning
        Требует сохраненных начальных весов
        """
        if not hasattr(self, 'initial_weights'):
            self.logger.warning("Начальные веса не сохранены. Используем magnitude pruning.")
            return self._magnitude_pruning(model, target_sparsity)
        
        # Реализация lottery ticket: находим "выигрышный билет"
        # Это упрощенная версия - в production нужна полная реализация
        return self._magnitude_pruning(model, target_sparsity)
    
    def _snip_pruning(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> nn.Module:
        """
        SNIP (Single-shot Network Pruning)
        Основан на градиентах сразу после инициализации
        """
        model.train()
        
        # Вычисляем градиенты на одном batch
        batch = next(iter(data_loader))
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
        else:
            outputs = model(batch)
            loss = nn.MSELoss()(outputs, batch)
        
        loss.backward()
        
        # Вычисляем SNIP scores для каждого параметра
        snip_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.weight.grad is not None:
                    # SNIP score = |weight * gradient|
                    score = (module.weight.data * module.weight.grad).abs()
                    snip_scores[name] = score
        
        # Применяем pruning на основе SNIP scores
        model = self._apply_score_based_pruning(model, snip_scores, self.target_sparsity)
        
        # Очищаем градиенты
        model.zero_grad()
        
        return model
    
    def _grasp_pruning(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> nn.Module:
        """
        GraSP (Gradient Signal Preservation)
        Сохраняет gradient flow через сеть
        """
        model.train()
        
        # Вычисляем Hessian-gradient product для GraSP
        batch = next(iter(data_loader))
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch, batch
        
        # Первый forward-backward pass
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        first_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Второй pass для GraSP score
        grad_norm = sum(g.pow(2).sum() for g in first_grads)
        second_grads = torch.autograd.grad(grad_norm, model.parameters())
        
        # Вычисляем GraSP scores
        grasp_scores = {}
        param_idx = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # GraSP score = |weight * hessian_gradient|
                hessian_grad = second_grads[param_idx]
                score = (module.weight.data * hessian_grad).abs()
                grasp_scores[name] = score
                param_idx += 1
        
        # Применяем pruning
        model = self._apply_score_based_pruning(model, grasp_scores, self.target_sparsity)
        model.zero_grad()
        
        return model
    
    def _apply_score_based_pruning(self, 
                                  model: nn.Module,
                                  scores: Dict[str, torch.Tensor],
                                  target_sparsity: float) -> nn.Module:
        """Применение pruning на основе importance scores"""
        # Собираем все scores для глобального ranking
        all_scores = []
        layer_info = []
        
        for name, score in scores.items():
            flat_scores = score.flatten()
            all_scores.append(flat_scores)
            module = dict(model.named_modules())[name]
            layer_info.append((name, module, score.shape))
        
        # Глобальный threshold
        all_scores_tensor = torch.cat(all_scores)
        threshold_idx = int(target_sparsity * len(all_scores_tensor))
        threshold_value = torch.kthvalue(all_scores_tensor, threshold_idx + 1).values
        
        # Применяем pruning
        for name, module, original_shape in layer_info:
            score = scores[name]
            mask = (score >= threshold_value).float()
            
            if self.sparsity_pattern == SparsityPattern.BLOCK:
                mask = self._apply_block_sparsity(mask)
            elif self.sparsity_pattern == SparsityPattern.N_M:
                mask = self._apply_n_m_sparsity(mask, n=2, m=4)
            
            prune.custom_from_mask(module, name='weight', mask=mask)
        
        return model
    
    def _apply_block_sparsity(self, mask: torch.Tensor, block_size: int = 4) -> torch.Tensor:
        """Применение block sparsity pattern"""
        # Простая реализация block sparsity
        original_shape = mask.shape
        
        if len(original_shape) >= 2:
            # Группируем в blocks и принимаем решение на уровне block
            h, w = original_shape[-2], original_shape[-1]
            
            # Padding до кратности block_size
            pad_h = (block_size - h % block_size) % block_size
            pad_w = (block_size - w % block_size) % block_size
            
            if pad_h > 0 or pad_w > 0:
                mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h))
            
            new_h, new_w = mask.shape[-2], mask.shape[-1]
            
            # Разбиваем на blocks
            blocks = mask.unfold(-2, block_size, block_size).unfold(-1, block_size, block_size)
            
            # Для каждого block: если сумма > половины элементов, сохраняем весь block
            block_mask = blocks.sum(dim=(-2, -1)) > (block_size * block_size / 2)
            
            # Восстанавливаем полную маску
            expanded_mask = block_mask.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)
            
            # Обрезаем до исходного размера
            mask = expanded_mask[..., :h, :w]
            
            # Восстанавливаем исходную форму
            if len(original_shape) > 2:
                mask = mask.view(original_shape)
        
        return mask
    
    def _apply_n_m_sparsity(self, mask: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
        """
        Применение N:M sparsity pattern
        Из каждых M весов оставляем N наибольших
        """
        original_shape = mask.shape
        flat_mask = mask.flatten()
        
        # Группируем по M элементов
        num_groups = len(flat_mask) // m
        remainder = len(flat_mask) % m
        
        # Обрабатываем полные группы
        grouped = flat_mask[:num_groups * m].view(num_groups, m)
        
        # Для каждой группы оставляем только n наибольших значений
        topk_values, topk_indices = torch.topk(grouped, n, dim=1)
        
        # Создаем новую маску для групп
        group_mask = torch.zeros_like(grouped)
        group_mask.scatter_(1, topk_indices, 1.0)
        
        # Восстанавливаем плоскую маску
        new_flat_mask = torch.cat([
            group_mask.flatten(),
            flat_mask[num_groups * m:]  # Остаток остается без изменений
        ])
        
        return new_flat_mask.view(original_shape)
    
    def _calculate_global_sparsity(self, model: nn.Module) -> float:
        """Вычисление глобального уровня sparsity"""
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    # Используем маску если есть
                    total_params += module.weight_mask.numel()
                    zero_params += (module.weight_mask == 0).sum().item()
                else:
                    # Считаем напрямую по весам
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def _count_pruned_parameters(self, model: nn.Module) -> int:
        """Подсчет количества обнуленных параметров"""
        zero_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    zero_params += (module.weight_mask == 0).sum().item()
                else:
                    zero_params += (module.weight == 0).sum().item()
        
        return zero_params
    
    def _count_total_parameters(self, model: nn.Module) -> int:
        """Подсчет общего количества параметров"""
        total_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                total_params += module.weight.numel()
        
        return total_params
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Расчет размера модели в MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        return param_size / 1024 / 1024
    
    def save_initial_weights(self, model: nn.Module) -> None:
        """Сохранение начальных весов для Lottery Ticket"""
        self.initial_weights = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self.initial_weights[name] = module.weight.data.clone()
    
    def restore_initial_weights(self, model: nn.Module) -> nn.Module:
        """Восстановление начальных весов"""
        if not hasattr(self, 'initial_weights'):
            self.logger.warning("Начальные веса не сохранены")
            return model
        
        for name, module in model.named_modules():
            if name in self.initial_weights:
                module.weight.data = self.initial_weights[name].clone()
        
        return model
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """Получение детальной статистики pruning"""
        stats = {
            "global_stats": self.pruning_stats,
            "sparsity_history": self.sparsity_history,
            "layer_wise_sparsity": self._get_layerwise_sparsity_stats()
        }
        
        return stats
    
    def _get_layerwise_sparsity_stats(self) -> Dict[str, Dict[str, float]]:
        """Получение статистики sparsity по слоям"""
        # Этот метод должен вызываться после pruning
        # В данной реализации возвращаем заглушку
        return {
            "note": "Статистика по слоям доступна после применения pruning к модели"
        }

class CryptoTradingUnstructuredPruner:
    """
    Специализированный unstructured pruner для crypto trading моделей
    с адаптивными стратегиями и crypto-specific оптимизациями
    """
    
    def __init__(self, 
                 target_compression_ratio: float = 10.0,
                 accuracy_threshold: float = 0.95,
                 latency_target_ms: float = 0.5):
        """
        Args:
            target_compression_ratio: Целевой коэффициент сжатия
            accuracy_threshold: Минимальная точность
            latency_target_ms: Целевая латентность в мс
        """
        self.target_compression_ratio = target_compression_ratio
        self.accuracy_threshold = accuracy_threshold
        self.latency_target_ms = latency_target_ms
        
        # Вычисляем target sparsity из compression ratio
        self.target_sparsity = 1.0 - (1.0 / target_compression_ratio)
        
        self.logger = logging.getLogger(f"{__name__}.CryptoTradingUnstructuredPruner")
        self.optimization_results = {}
        
    def adaptive_pruning(self, 
                        model: nn.Module,
                        training_data: torch.utils.data.DataLoader,
                        validation_data: torch.utils.data.DataLoader,
                        fine_tune_fn: Optional[Callable] = None) -> nn.Module:
        """
        Адаптивная pruning с автоматическим выбором оптимальной стратегии
        
        Args:
            model: Модель для pruning
            training_data: Обучающие данные
            validation_data: Валидационные данные  
            fine_tune_fn: Функция fine-tuning
            
        Returns:
            Оптимально pruned модель
        """
        self.logger.info(f"Начинаем адаптивную crypto pruning до {self.target_sparsity*100:.1f}% sparsity")
        
        # Сохраняем начальные веса для lottery ticket
        initial_model = copy.deepcopy(model)
        
        # Тестируем различные стратегии
        strategies_to_test = [
            (UnstructuredPruningStrategy.MAGNITUDE, PruningSchedule.EXPONENTIAL),
            (UnstructuredPruningStrategy.SNIP, PruningSchedule.LINEAR),
            (UnstructuredPruningStrategy.GRASP, PruningSchedule.POLYNOMIAL),
            (UnstructuredPruningStrategy.RANDOM, PruningSchedule.COSINE)
        ]
        
        best_model = None
        best_score = -float('inf')
        best_config = None
        
        for strategy, schedule in strategies_to_test:
            try:
                self.logger.info(f"Тестируем стратегию: {strategy.value} с расписанием {schedule.value}")
                
                # Создаем копию модели для тестирования
                test_model = copy.deepcopy(initial_model)
                
                # Создаем pruner с текущей конфигурацией
                pruner = UnstructuredPruner(
                    target_sparsity=self.target_sparsity,
                    strategy=strategy,
                    schedule=schedule,
                    sparsity_pattern=SparsityPattern.RANDOM
                )
                
                # Применяем pruning
                pruned_model = pruner.prune_model(
                    test_model,
                    data_loader=training_data,
                    num_steps=5,  # Меньше шагов для быстрого тестирования
                    fine_tune_fn=fine_tune_fn
                )
                
                # Оцениваем результат
                score = self._evaluate_crypto_model(pruned_model, validation_data)
                
                self.logger.info(f"Стратегия {strategy.value}: score = {score:.4f}")
                
                if score > best_score:
                    best_model = pruned_model
                    best_score = score
                    best_config = (strategy, schedule)
                    
            except Exception as e:
                self.logger.warning(f"Ошибка с стратегией {strategy.value}: {e}")
                continue
        
        if best_model is None:
            self.logger.error("Не удалось найти подходящую стратегию pruning")
            return model
        
        # Применяем лучшую стратегию с полными настройками
        self.logger.info(f"Лучшая стратегия: {best_config[0].value} с {best_config[1].value}")
        
        final_pruner = UnstructuredPruner(
            target_sparsity=self.target_sparsity,
            strategy=best_config[0],
            schedule=best_config[1],
            sparsity_pattern=SparsityPattern.N_M  # Используем N:M для лучшей hardware efficiency
        )
        
        final_model = final_pruner.prune_model(
            copy.deepcopy(initial_model),
            data_loader=training_data,
            num_steps=10,
            fine_tune_fn=fine_tune_fn
        )
        
        # Дополнительные оптимизации для crypto trading
        optimized_model = self._apply_crypto_optimizations(final_model)
        
        # Финальная оценка
        final_score = self._evaluate_crypto_model(optimized_model, validation_data)
        final_sparsity = self._calculate_global_sparsity(optimized_model)
        
        # Сохраняем результаты
        self.optimization_results = {
            "best_strategy": best_config[0].value,
            "best_schedule": best_config[1].value,
            "final_sparsity": final_sparsity,
            "final_score": final_score,
            "compression_ratio": 1.0 / (1.0 - final_sparsity),
            "meets_accuracy_threshold": final_score >= self.accuracy_threshold,
            "pruning_stats": final_pruner.pruning_stats
        }
        
        self.logger.info(f"Адаптивная crypto pruning завершена. "
                        f"Sparsity: {final_sparsity*100:.1f}%, Score: {final_score:.4f}")
        
        return optimized_model
    
    def _evaluate_crypto_model(self, 
                              model: nn.Module, 
                              validation_data: torch.utils.data.DataLoader) -> float:
        """Оценка модели для crypto trading задач"""
        model.eval()
        
        total_mse = 0.0
        total_directional_acc = 0.0
        total_sharpe = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in validation_data:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    outputs = model(inputs)
                    
                    # MSE Loss
                    mse = nn.MSELoss()(outputs, targets)
                    total_mse += mse.item()
                    
                    # Directional Accuracy для trading
                    dir_acc = self._calculate_directional_accuracy(outputs, targets)
                    total_directional_acc += dir_acc.item()
                    
                    # Simulated Sharpe ratio
                    sharpe = self._calculate_simulated_sharpe(outputs, targets)
                    total_sharpe += sharpe
                    
                    num_batches += 1
                    
                    if num_batches >= 50:  # Ограничиваем для скорости
                        break
        
        if num_batches == 0:
            return 0.0
        
        # Комбинированный score для crypto trading
        avg_mse = total_mse / num_batches
        avg_dir_acc = total_directional_acc / num_batches
        avg_sharpe = total_sharpe / num_batches
        
        # Weighted score: приоритет directional accuracy и sharpe ratio
        score = (
            0.3 * (1.0 / (1.0 + avg_mse)) +  # Обратно пропорционально MSE
            0.4 * avg_dir_acc +               # Directional accuracy
            0.3 * max(0.0, avg_sharpe)        # Sharpe ratio (только положительный)
        )
        
        return score
    
    def _calculate_directional_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Directional accuracy для trading signals"""
        if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
            pred_direction = torch.sign(predictions[1:, 0] - predictions[:-1, 0])
            target_direction = torch.sign(targets[1:, 0] - targets[:-1, 0])
        else:
            pred_direction = torch.sign(predictions[1:] - predictions[:-1])
            target_direction = torch.sign(targets[1:] - targets[:-1])
        
        correct = (pred_direction == target_direction).float()
        return correct.mean()
    
    def _calculate_simulated_sharpe(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Приближенное вычисление Sharpe ratio"""
        if len(predictions) < 2:
            return 0.0
        
        # Симулируем returns на основе предсказаний
        pred_returns = predictions[1:] - predictions[:-1]
        actual_returns = targets[1:] - targets[:-1]
        
        # Correlation между предсказанными и реальными returns
        correlation = torch.corrcoef(torch.stack([pred_returns.flatten(), actual_returns.flatten()]))[0, 1]
        
        if torch.isnan(correlation):
            return 0.0
        
        # Упрощенный Sharpe: корреляция * volatility adjustment
        volatility = torch.std(pred_returns)
        sharpe_approx = correlation.item() / (volatility.item() + 1e-8)
        
        return max(-2.0, min(2.0, sharpe_approx))  # Ограничиваем диапазон
    
    def _apply_crypto_optimizations(self, model: nn.Module) -> nn.Module:
        """Crypto-specific оптимизации после pruning"""
        optimized_model = model
        
        try:
            # 1. Удаление masks для compact representation
            optimized_model = self._finalize_sparse_model(optimized_model)
            
            # 2. Memory layout optimization
            optimized_model = self._optimize_sparse_memory_layout(optimized_model)
            
            # 3. JIT optimization если возможно
            optimized_model = self._apply_jit_optimization(optimized_model)
            
        except Exception as e:
            self.logger.warning(f"Некоторые crypto оптимизации не удались: {e}")
        
        return optimized_model
    
    def _finalize_sparse_model(self, model: nn.Module) -> nn.Module:
        """Финализация sparse модели - удаление masks"""
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                # Применяем маску окончательно
                module.weight.data = module.weight.data * module.weight_mask
                # Удаляем маску
                prune.remove(module, 'weight')
        
        return model
    
    def _optimize_sparse_memory_layout(self, model: nn.Module) -> nn.Module:
        """Оптимизация memory layout для sparse matrices"""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Обеспечиваем contiguous layout
                if not module.weight.is_contiguous():
                    module.weight.data = module.weight.data.contiguous()
                
                # Для очень sparse весов можно рассмотреть sparse tensor format
                sparsity = (module.weight == 0).float().mean()
                if sparsity > 0.9:  # Если >90% нулей
                    # В production здесь можно конвертировать в sparse format
                    pass
        
        return model
    
    def _apply_jit_optimization(self, model: nn.Module) -> nn.Module:
        """JIT оптимизация для sparse модели"""
        try:
            if hasattr(torch, 'jit'):
                # Создаем dummy input для трассировки
                dummy_input = torch.randn(1, self._estimate_input_size(model))
                traced_model = torch.jit.trace(model, dummy_input)
                return torch.jit.optimize_for_inference(traced_model)
            
        except Exception as e:
            self.logger.warning(f"JIT оптимизация не удалась: {e}")
        
        return model
    
    def _estimate_input_size(self, model: nn.Module) -> int:
        """Оценка размера входных данных"""
        first_layer = next(iter(model.modules()))
        if isinstance(first_layer, nn.Linear):
            return first_layer.in_features
        elif isinstance(first_layer, (nn.Conv1d, nn.Conv2d)):
            # Для crypto данных обычно временные ряды
            return 100  # Примерный размер
        else:
            return 100  # Дефолтное значение
    
    def _calculate_global_sparsity(self, model: nn.Module) -> float:
        """Вычисление глобального уровня sparsity"""
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Получение детального отчета об оптимизации"""
        return {
            "optimization_results": self.optimization_results,
            "configuration": {
                "target_compression_ratio": self.target_compression_ratio,
                "target_sparsity": self.target_sparsity,
                "accuracy_threshold": self.accuracy_threshold,
                "latency_target_ms": self.latency_target_ms
            },
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> Dict[str, str]:
        """Рекомендации по дальнейшей оптимизации"""
        recommendations = {}
        
        if not self.optimization_results:
            return {"general": "Запустите adaptive pruning для получения рекомендаций"}
        
        final_score = self.optimization_results.get("final_score", 0.0)
        meets_threshold = self.optimization_results.get("meets_accuracy_threshold", False)
        
        if not meets_threshold:
            recommendations["accuracy"] = "Рассмотрите менее агрессивную pruning или дообучение"
        
        if final_score > 0.8:
            recommendations["next_step"] = "Попробуйте комбинацию с quantization для дальнейшего сжатия"
        
        if self.optimization_results.get("compression_ratio", 1.0) < self.target_compression_ratio * 0.8:
            recommendations["compression"] = "Попробуйте более агрессивные настройки или structured pruning"
        
        return recommendations