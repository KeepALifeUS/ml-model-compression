"""
Базовый модуль Knowledge Distillation для криптотрейдинговых моделей.
Обеспечивает transfer знаний от больших teacher моделей к компактным student моделям.

Model efficiency patterns для production deployment
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
import copy
import math

logger = logging.getLogger(__name__)

class DistillationMode(Enum):
    """Режимы knowledge distillation"""
    RESPONSE = "response"           # Distillation выходов
    FEATURE = "feature"            # Feature matching
    ATTENTION = "attention"        # Attention transfer
    DARK_KNOWLEDGE = "dark"        # Dark knowledge (soft targets)
    MULTI_TEACHER = "multi_teacher"  # Ensemble teachers
    ONLINE = "online"              # Online distillation

class TemperatureSchedule(Enum):
    """Расписания изменения температуры"""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"

class DistillationLoss(nn.Module):
    """Базовый класс для loss функций distillation"""
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3):
        """
        Args:
            temperature: Температура для softmax
            alpha: Вес distillation loss
            beta: Вес task loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Вычисление комбинированного loss
        
        Args:
            student_logits: Выходы student модели
            teacher_logits: Выходы teacher модели  
            targets: Ground truth targets
            
        Returns:
            Dictionary с различными компонентами loss
        """
        # Soft targets от teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss (KL divergence)
        distillation_loss = F.kl_div(
            soft_predictions, soft_targets, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Task loss (обычный supervised loss)
        if targets.dtype == torch.long:  # Classification
            task_loss = F.cross_entropy(student_logits, targets)
        else:  # Regression
            task_loss = F.mse_loss(student_logits, targets)
        
        # Комбинированный loss
        total_loss = self.alpha * distillation_loss + self.beta * task_loss
        
        return {
            'total_loss': total_loss,
            'distillation_loss': distillation_loss,
            'task_loss': task_loss,
            'alpha': self.alpha,
            'beta': self.beta
        }

class FeatureDistillationLoss(nn.Module):
    """Loss для feature-based distillation"""
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 feature_weight: float = 0.1):
        super().__init__()
        self.base_loss = DistillationLoss(temperature, alpha, beta)
        self.feature_weight = feature_weight
        
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor,
                student_features: List[torch.Tensor],
                teacher_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Комбинированный loss с feature matching
        
        Args:
            student_logits: Выходы student модели
            teacher_logits: Выходы teacher модели
            targets: Ground truth
            student_features: Промежуточные features student
            teacher_features: Промежуточные features teacher
            
        Returns:
            Dictionary с loss компонентами
        """
        # Базовый distillation loss
        base_losses = self.base_loss(student_logits, teacher_logits, targets)
        
        # Feature matching loss
        feature_loss = 0.0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            if s_feat.shape != t_feat.shape:
                # Приводим к одинаковому размеру
                s_feat = self._align_features(s_feat, t_feat)
            
            # L2 loss между features
            feature_loss += F.mse_loss(s_feat, t_feat)
        
        # Нормализуем на количество feature layers
        if len(student_features) > 0:
            feature_loss = feature_loss / len(student_features)
        
        # Итоговый loss
        total_loss = base_losses['total_loss'] + self.feature_weight * feature_loss
        
        result = base_losses.copy()
        result.update({
            'total_loss': total_loss,
            'feature_loss': feature_loss,
            'feature_weight': self.feature_weight
        })
        
        return result
    
    def _align_features(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Приведение размеров features к совместимым"""
        if student_feat.shape == teacher_feat.shape:
            return student_feat
        
        # Для 2D features (batch_size, features)
        if len(student_feat.shape) == 2 and len(teacher_feat.shape) == 2:
            if student_feat.shape[1] != teacher_feat.shape[1]:
                # Добавляем linear projection
                projection = nn.Linear(student_feat.shape[1], teacher_feat.shape[1])
                return projection(student_feat)
        
        # Для других случаев используем adaptive pooling или interpolation
        if len(student_feat.shape) > 2:
            # Пример для 3D/4D tensors
            target_shape = teacher_feat.shape[2:]  # Пропускаем batch и channel dims
            
            if len(target_shape) == 1:  # 1D features
                return F.adaptive_avg_pool1d(student_feat, target_shape[0])
            elif len(target_shape) == 2:  # 2D features
                return F.adaptive_avg_pool2d(student_feat, target_shape)
        
        return student_feat

class BaseKnowledgeDistiller(ABC):
    """Базовый класс для knowledge distillation"""
    
    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3):
        """
        Args:
            teacher_model: Большая обученная модель
            student_model: Маленькая модель для обучения
            temperature: Температура для soft targets
            alpha: Вес distillation loss
            beta: Вес task loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.training_stats = {}
        
        # Замораживаем teacher модель
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    @abstractmethod
    def distill(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                num_epochs: int = 100,
                optimizer: Optional[torch.optim.Optimizer] = None) -> nn.Module:
        """Абстрактный метод для knowledge distillation"""
        pass
    
    def _validate_models(self) -> bool:
        """Валидация teacher и student моделей"""
        try:
            # Проверяем что модели имеют совместимые выходы
            dummy_input = self._create_dummy_input()
            
            with torch.no_grad():
                teacher_out = self.teacher_model(dummy_input)
                student_out = self.student_model(dummy_input)
            
            if teacher_out.shape != student_out.shape:
                self.logger.warning(f"Размеры выходов не совпадают: "
                                  f"teacher {teacher_out.shape}, student {student_out.shape}")
                # Но это не критично - можем адаптировать
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации моделей: {e}")
            return False
    
    def _create_dummy_input(self) -> torch.Tensor:
        """Создание dummy input для валидации"""
        # Простая эвристика для определения размера входа
        first_layer = next(iter(self.student_model.modules()))
        
        if isinstance(first_layer, nn.Linear):
            return torch.randn(1, first_layer.in_features)
        elif isinstance(first_layer, nn.Conv1d):
            return torch.randn(1, first_layer.in_channels, 100)
        elif isinstance(first_layer, nn.Conv2d):
            return torch.randn(1, first_layer.in_channels, 32, 32)
        else:
            # Дефолтный размер для временных рядов crypto
            return torch.randn(1, 100)
    
    def _calculate_model_compression(self) -> Dict[str, float]:
        """Вычисление коэффициента сжатия"""
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        teacher_size = sum(p.numel() * p.element_size() for p in self.teacher_model.parameters()) / 1024 / 1024
        student_size = sum(p.numel() * p.element_size() for p in self.student_model.parameters()) / 1024 / 1024
        
        return {
            'teacher_params': teacher_params,
            'student_params': student_params,
            'param_compression_ratio': teacher_params / student_params,
            'teacher_size_mb': teacher_size,
            'student_size_mb': student_size,
            'size_compression_ratio': teacher_size / student_size
        }

class ResponseDistiller(BaseKnowledgeDistiller):
    """Response-based knowledge distillation (классический подход)"""
    
    def distill(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                num_epochs: int = 100,
                optimizer: Optional[torch.optim.Optimizer] = None) -> nn.Module:
        """
        Обучение student модели через distillation выходов teacher
        
        Args:
            train_loader: Обучающие данные
            val_loader: Валидационные данные
            num_epochs: Количество эпох
            optimizer: Оптимизатор (создается автоматически если None)
            
        Returns:
            Обученная student модель
        """
        if not self._validate_models():
            raise ValueError("Модели не прошли валидацию")
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        criterion = DistillationLoss(self.temperature, self.alpha, self.beta)
        
        self.logger.info(f"Начинаем response distillation на {num_epochs} эпох")
        
        # История обучения
        train_history = []
        val_history = []
        
        for epoch in range(num_epochs):
            # Обучение
            train_stats = self._train_epoch(train_loader, optimizer, criterion)
            
            # Валидация
            val_stats = self._validate_epoch(val_loader, criterion)
            
            # Learning rate schedule
            scheduler.step()
            
            # Логирование
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                               f"Train Loss: {train_stats['total_loss']:.4f}, "
                               f"Val Loss: {val_stats['total_loss']:.4f}")
            
            train_history.append(train_stats)
            val_history.append(val_stats)
            
            # Early stopping если val loss растет
            if len(val_history) > 10:
                recent_losses = [s['total_loss'] for s in val_history[-10:]]
                if all(recent_losses[i] <= recent_losses[i+1] for i in range(9)):
                    self.logger.info(f"Early stopping на эпохе {epoch+1}")
                    break
        
        # Сохраняем статистики
        compression_stats = self._calculate_model_compression()
        
        self.training_stats = {
            'num_epochs_trained': epoch + 1,
            'final_train_loss': train_history[-1]['total_loss'],
            'final_val_loss': val_history[-1]['total_loss'],
            'compression_stats': compression_stats,
            'train_history': train_history,
            'val_history': val_history,
            'distillation_config': {
                'temperature': self.temperature,
                'alpha': self.alpha,
                'beta': self.beta
            }
        }
        
        self.logger.info(f"Response distillation завершен. "
                        f"Compression ratio: {compression_stats['param_compression_ratio']:.2f}x")
        
        return self.student_model
    
    def _train_epoch(self,
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: DistillationLoss) -> Dict[str, float]:
        """Обучение на одной эпохе"""
        self.student_model.train()
        
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch, batch
            
            optimizer.zero_grad()
            
            # Forward passes
            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs)
            
            student_logits = self.student_model(inputs)
            
            # Compute loss
            losses = criterion(student_logits, teacher_logits, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += losses['total_loss'].item()
            total_distill_loss += losses['distillation_loss'].item()
            total_task_loss += losses['task_loss'].item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'distillation_loss': total_distill_loss / num_batches,
            'task_loss': total_task_loss / num_batches
        }
    
    def _validate_epoch(self,
                       val_loader: torch.utils.data.DataLoader,
                       criterion: DistillationLoss) -> Dict[str, float]:
        """Валидация на одной эпохе"""
        self.student_model.eval()
        
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets = batch, batch
                
                # Forward passes
                teacher_logits = self.teacher_model(inputs)
                student_logits = self.student_model(inputs)
                
                # Compute loss
                losses = criterion(student_logits, teacher_logits, targets)
                
                # Statistics
                total_loss += losses['total_loss'].item()
                total_distill_loss += losses['distillation_loss'].item()
                total_task_loss += losses['task_loss'].item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'distillation_loss': total_distill_loss / num_batches,
            'task_loss': total_task_loss / num_batches
        }

class FeatureDistiller(BaseKnowledgeDistiller):
    """Feature-based knowledge distillation"""
    
    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 feature_weight: float = 0.1,
                 feature_layers: Optional[List[str]] = None):
        """
        Args:
            teacher_model: Teacher модель
            student_model: Student модель
            temperature: Температура
            alpha: Вес distillation loss
            beta: Вес task loss
            feature_weight: Вес feature matching loss
            feature_layers: Имена слоев для feature extraction
        """
        super().__init__(teacher_model, student_model, temperature, alpha, beta)
        self.feature_weight = feature_weight
        self.feature_layers = feature_layers or []
        
        # Hooks для extraction features
        self.teacher_features = {}
        self.student_features = {}
        self._register_feature_hooks()
    
    def _register_feature_hooks(self):
        """Регистрация hooks для извлечения features"""
        def get_teacher_hook(name):
            def hook(module, input, output):
                self.teacher_features[name] = output
            return hook
        
        def get_student_hook(name):
            def hook(module, input, output):
                self.student_features[name] = output
            return hook
        
        # Если не указаны конкретные слои, используем все подходящие
        if not self.feature_layers:
            self.feature_layers = []
            for name, module in self.teacher_model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU)):
                    self.feature_layers.append(name)
        
        # Регистрируем hooks
        for layer_name in self.feature_layers:
            teacher_layer = dict(self.teacher_model.named_modules()).get(layer_name)
            student_layer = dict(self.student_model.named_modules()).get(layer_name)
            
            if teacher_layer is not None:
                teacher_layer.register_forward_hook(get_teacher_hook(layer_name))
            
            if student_layer is not None:
                student_layer.register_forward_hook(get_student_hook(layer_name))
    
    def distill(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                num_epochs: int = 100,
                optimizer: Optional[torch.optim.Optimizer] = None) -> nn.Module:
        """Feature-based knowledge distillation"""
        if not self._validate_models():
            raise ValueError("Модели не прошли валидацию")
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        criterion = FeatureDistillationLoss(
            self.temperature, self.alpha, self.beta, self.feature_weight
        )
        
        self.logger.info(f"Начинаем feature distillation с {len(self.feature_layers)} слоями")
        
        train_history = []
        val_history = []
        
        for epoch in range(num_epochs):
            # Обучение
            train_stats = self._train_feature_epoch(train_loader, optimizer, criterion)
            
            # Валидация
            val_stats = self._validate_feature_epoch(val_loader, criterion)
            
            scheduler.step()
            
            # Логирование
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                               f"Train Loss: {train_stats['total_loss']:.4f} "
                               f"(Feature: {train_stats['feature_loss']:.4f}), "
                               f"Val Loss: {val_stats['total_loss']:.4f}")
            
            train_history.append(train_stats)
            val_history.append(val_stats)
            
            # Early stopping
            if len(val_history) > 15:
                recent_losses = [s['total_loss'] for s in val_history[-15:]]
                if all(recent_losses[i] <= recent_losses[i+1] for i in range(14)):
                    self.logger.info(f"Early stopping на эпохе {epoch+1}")
                    break
        
        # Статистики
        compression_stats = self._calculate_model_compression()
        
        self.training_stats = {
            'num_epochs_trained': epoch + 1,
            'final_train_loss': train_history[-1]['total_loss'],
            'final_val_loss': val_history[-1]['total_loss'],
            'final_feature_loss': train_history[-1]['feature_loss'],
            'compression_stats': compression_stats,
            'train_history': train_history,
            'val_history': val_history,
            'feature_layers_used': self.feature_layers,
            'distillation_config': {
                'temperature': self.temperature,
                'alpha': self.alpha,
                'beta': self.beta,
                'feature_weight': self.feature_weight
            }
        }
        
        self.logger.info(f"Feature distillation завершен. "
                        f"Compression ratio: {compression_stats['param_compression_ratio']:.2f}x")
        
        return self.student_model
    
    def _train_feature_epoch(self,
                            train_loader: torch.utils.data.DataLoader,
                            optimizer: torch.optim.Optimizer,
                            criterion: FeatureDistillationLoss) -> Dict[str, float]:
        """Обучение с feature matching"""
        self.student_model.train()
        
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0
        total_feature_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch, batch
            
            optimizer.zero_grad()
            
            # Очищаем feature caches
            self.teacher_features.clear()
            self.student_features.clear()
            
            # Forward passes с извлечением features
            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs)
            
            student_logits = self.student_model(inputs)
            
            # Собираем features в списки
            teacher_feat_list = [self.teacher_features.get(layer, torch.tensor(0.0)) 
                               for layer in self.feature_layers]
            student_feat_list = [self.student_features.get(layer, torch.tensor(0.0)) 
                               for layer in self.feature_layers]
            
            # Compute loss
            losses = criterion(
                student_logits, teacher_logits, targets,
                student_feat_list, teacher_feat_list
            )
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            total_loss += losses['total_loss'].item()
            total_distill_loss += losses['distillation_loss'].item()
            total_task_loss += losses['task_loss'].item()
            total_feature_loss += losses['feature_loss'].item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'distillation_loss': total_distill_loss / num_batches,
            'task_loss': total_task_loss / num_batches,
            'feature_loss': total_feature_loss / num_batches
        }
    
    def _validate_feature_epoch(self,
                               val_loader: torch.utils.data.DataLoader,
                               criterion: FeatureDistillationLoss) -> Dict[str, float]:
        """Валидация с feature matching"""
        self.student_model.eval()
        
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0
        total_feature_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets = batch, batch
                
                # Очищаем feature caches
                self.teacher_features.clear()
                self.student_features.clear()
                
                # Forward passes
                teacher_logits = self.teacher_model(inputs)
                student_logits = self.student_model(inputs)
                
                # Собираем features
                teacher_feat_list = [self.teacher_features.get(layer, torch.tensor(0.0)) 
                                   for layer in self.feature_layers]
                student_feat_list = [self.student_features.get(layer, torch.tensor(0.0)) 
                                   for layer in self.feature_layers]
                
                # Compute loss
                losses = criterion(
                    student_logits, teacher_logits, targets,
                    student_feat_list, teacher_feat_list
                )
                
                # Statistics
                total_loss += losses['total_loss'].item()
                total_distill_loss += losses['distillation_loss'].item()
                total_task_loss += losses['task_loss'].item()
                total_feature_loss += losses['feature_loss'].item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'distillation_loss': total_distill_loss / num_batches,
            'task_loss': total_task_loss / num_batches,
            'feature_loss': total_feature_loss / num_batches
        }