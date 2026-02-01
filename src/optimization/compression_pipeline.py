"""
Производственный pipeline для сжатия ML-моделей в криптотрейдинге.
Автоматизированный workflow с валидацией, rollback и deployment готовностью.

Context7: Production ML pipeline patterns для continuous deployment
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import json
import time
import traceback
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import shutil
import pickle
from contextlib import contextmanager

from .model_optimizer import ModelOptimizer, OptimizationConfig, OptimizationResult, OptimizationObjective

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Стадии compression pipeline"""
    VALIDATION = "validation"
    BACKUP = "backup"
    OPTIMIZATION = "optimization"
    TESTING = "testing"
    BENCHMARKING = "benchmarking"
    DEPLOYMENT_PREP = "deployment_prep"
    FINALIZATION = "finalization"

class PipelineStatus(Enum):
    """Статусы pipeline"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class PipelineConfig:
    """Конфигурация compression pipeline"""
    # Общие настройки
    name: str
    version: str = "1.0"
    description: str = ""
    
    # Optimization настройки
    optimization_config: OptimizationConfig = None
    
    # Validation настройки
    accuracy_tolerance: float = 0.05  # Максимальное снижение accuracy
    latency_improvement_threshold: float = 0.1  # Минимальное улучшение latency (10%)
    compression_ratio_threshold: float = 1.5  # Минимальный compression ratio
    
    # Testing настройки
    test_data_fraction: float = 0.2  # Доля данных для тестирования
    benchmark_iterations: int = 100
    stress_test_enabled: bool = True
    
    # Safety настройки
    enable_rollback: bool = True
    backup_models: bool = True
    max_pipeline_duration_hours: float = 24.0
    
    # Deployment настройки
    export_formats: List[str] = None  # ["onnx", "torchscript", "tflite"]
    target_platforms: List[str] = None  # ["cpu", "cuda", "edge"]
    
    def __post_init__(self):
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig()
        if self.export_formats is None:
            self.export_formats = ["onnx", "torchscript"]
        if self.target_platforms is None:
            self.target_platforms = ["cpu", "cuda"]

@dataclass
class PipelineResult:
    """Результат выполнения pipeline"""
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    
    # Результаты оптимизации
    optimization_result: Optional[OptimizationResult]
    
    # Метрики pipeline
    stages_completed: List[str]
    stages_failed: List[str]
    validation_passed: bool
    
    # Пути к артефактам
    model_paths: Dict[str, str]  # {"original": path, "optimized": path}
    export_paths: Dict[str, str]  # {"format": path}
    
    # Дополнительная информация
    logs: List[str]
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        result = asdict(self)
        # Обрабатываем специальные типы
        result['status'] = self.status.value
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        if self.optimization_result:
            result['optimization_result'] = self.optimization_result.to_dict()
        return result

class CompressionPipeline:
    """
    Production-ready compression pipeline для crypto trading моделей
    с полной автоматизацией, мониторингом и rollback capabilities
    """
    
    def __init__(self, 
                 workspace_dir: Union[str, Path],
                 config: Optional[PipelineConfig] = None):
        """
        Args:
            workspace_dir: Рабочая директория для pipeline
            config: Конфигурация pipeline
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or PipelineConfig(name="default_compression_pipeline")
        
        # Инициализация директорий
        self._setup_workspace()
        
        self.logger = logging.getLogger(f"{__name__}.CompressionPipeline")
        self._setup_logging()
        
        # Pipeline состояние
        self.current_pipeline_id = None
        self.current_stage = None
        self.pipeline_history = []
        
        # Backup и rollback
        self.backup_manager = BackupManager(self.workspace_dir / "backups")
        
        # Optimizer
        self.optimizer = ModelOptimizer(self.config.optimization_config)
        
    def _setup_workspace(self):
        """Настройка рабочего пространства"""
        (self.workspace_dir / "models").mkdir(exist_ok=True)
        (self.workspace_dir / "exports").mkdir(exist_ok=True)
        (self.workspace_dir / "backups").mkdir(exist_ok=True)
        (self.workspace_dir / "logs").mkdir(exist_ok=True)
        (self.workspace_dir / "configs").mkdir(exist_ok=True)
        (self.workspace_dir / "results").mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Настройка логирования для pipeline"""
        log_file = self.workspace_dir / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def run_compression_pipeline(self,
                                model: nn.Module,
                                train_data: torch.utils.data.DataLoader,
                                val_data: torch.utils.data.DataLoader,
                                test_data: Optional[torch.utils.data.DataLoader] = None,
                                teacher_model: Optional[nn.Module] = None,
                                pipeline_id: Optional[str] = None) -> PipelineResult:
        """
        Запуск полного compression pipeline
        
        Args:
            model: Исходная модель
            train_data: Обучающие данные
            val_data: Валидационные данные
            test_data: Тестовые данные (опционально)
            teacher_model: Teacher модель для distillation
            pipeline_id: ID pipeline (генерируется автоматически)
            
        Returns:
            Результат выполнения pipeline
        """
        pipeline_id = pipeline_id or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_pipeline_id = pipeline_id
        
        start_time = datetime.now()
        
        # Инициализация результата
        result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            start_time=start_time,
            end_time=None,
            duration_seconds=0,
            optimization_result=None,
            stages_completed=[],
            stages_failed=[],
            validation_passed=False,
            model_paths={},
            export_paths={},
            logs=[],
            error_message=None
        )
        
        self.logger.info(f"Начинаем compression pipeline {pipeline_id}")
        self.logger.info(f"Конфигурация: {self.config}")
        
        try:
            with self._pipeline_timeout_context():
                # Стадия 1: Validation
                self._execute_stage(PipelineStage.VALIDATION, result, 
                                  self._validate_inputs, model, train_data, val_data)
                
                # Стадия 2: Backup
                self._execute_stage(PipelineStage.BACKUP, result,
                                  self._backup_original_model, model)
                
                # Стадия 3: Optimization
                optimization_result = self._execute_stage(PipelineStage.OPTIMIZATION, result,
                                                        self._run_optimization, model, train_data, val_data, teacher_model)
                result.optimization_result = optimization_result
                
                # Стадия 4: Testing
                validation_passed = self._execute_stage(PipelineStage.TESTING, result,
                                                      self._run_comprehensive_testing, 
                                                      model, optimization_result.optimized_model, val_data, test_data)
                result.validation_passed = validation_passed
                
                if not validation_passed and self.config.enable_rollback:
                    self.logger.warning("Validation не пройдена, выполняем rollback")
                    self._rollback_pipeline(result)
                    result.status = PipelineStatus.ROLLED_BACK
                    return result
                
                # Стадия 5: Benchmarking
                benchmark_results = self._execute_stage(PipelineStage.BENCHMARKING, result,
                                                      self._run_benchmarking, optimization_result.optimized_model)
                
                # Стадия 6: Deployment Preparation
                export_paths = self._execute_stage(PipelineStage.DEPLOYMENT_PREP, result,
                                                 self._prepare_for_deployment, optimization_result.optimized_model)
                result.export_paths = export_paths
                
                # Стадия 7: Finalization
                self._execute_stage(PipelineStage.FINALIZATION, result,
                                  self._finalize_pipeline, result)
                
                result.status = PipelineStatus.SUCCESS
                
        except PipelineTimeoutError as e:
            self.logger.error(f"Pipeline timeout: {e}")
            result.error_message = str(e)
            result.status = PipelineStatus.FAILED
            if self.config.enable_rollback:
                self._rollback_pipeline(result)
                result.status = PipelineStatus.ROLLED_BACK
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            result.error_message = str(e)
            result.status = PipelineStatus.FAILED
            if self.config.enable_rollback:
                self._rollback_pipeline(result)
                result.status = PipelineStatus.ROLLED_BACK
        
        finally:
            end_time = datetime.now()
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            # Сохраняем результат
            self._save_pipeline_result(result)
            self.pipeline_history.append(result)
            
            self.logger.info(f"Pipeline {pipeline_id} завершен со статусом {result.status.value}")
            self.logger.info(f"Длительность: {result.duration_seconds:.2f} секунд")
        
        return result
    
    def _execute_stage(self, 
                      stage: PipelineStage,
                      result: PipelineResult,
                      stage_func: Callable,
                      *args, **kwargs) -> Any:
        """Выполнение стадии pipeline с обработкой ошибок"""
        self.current_stage = stage
        self.logger.info(f"Выполняем стадию: {stage.value}")
        
        try:
            stage_result = stage_func(*args, **kwargs)
            result.stages_completed.append(stage.value)
            self.logger.info(f"Стадия {stage.value} завершена успешно")
            return stage_result
            
        except Exception as e:
            self.logger.error(f"Стадия {stage.value} не удалась: {e}")
            result.stages_failed.append(stage.value)
            raise
    
    def _validate_inputs(self,
                        model: nn.Module,
                        train_data: torch.utils.data.DataLoader,
                        val_data: torch.utils.data.DataLoader) -> bool:
        """Валидация входных данных"""
        
        # Проверка модели
        if not isinstance(model, nn.Module):
            raise ValueError("model должен быть nn.Module")
        
        # Проверка данных
        if len(train_data) == 0:
            raise ValueError("train_data не может быть пустым")
        
        if len(val_data) == 0:
            raise ValueError("val_data не может быть пустым")
        
        # Тест forward pass
        try:
            sample_batch = next(iter(val_data))
            if isinstance(sample_batch, (list, tuple)):
                sample_input = sample_batch[0][:1]
            else:
                sample_input = sample_batch[:1]
            
            with torch.no_grad():
                output = model(sample_input)
            
            self.logger.info(f"Валидация входов прошла успешно. Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
            
        except Exception as e:
            raise ValueError(f"Модель не может обработать входные данные: {e}")
        
        return True
    
    def _backup_original_model(self, model: nn.Module) -> str:
        """Создание backup исходной модели"""
        backup_path = self.backup_manager.create_backup(
            model, 
            f"original_model_{self.current_pipeline_id}"
        )
        
        self.logger.info(f"Backup модели создан: {backup_path}")
        return backup_path
    
    def _run_optimization(self,
                         model: nn.Module,
                         train_data: torch.utils.data.DataLoader,
                         val_data: torch.utils.data.DataLoader,
                         teacher_model: Optional[nn.Module]) -> OptimizationResult:
        """Выполнение оптимизации модели"""
        
        self.logger.info("Запускаем оптимизацию модели...")
        
        # Измеряем базовые метрики
        original_metrics = self.optimizer._measure_model_performance(model, val_data)
        self.logger.info(f"Исходные метрики: {original_metrics}")
        
        # Оптимизация
        optimization_result = self.optimizer.optimize_model(
            model=model,
            train_data=train_data,
            val_data=val_data,
            teacher_model=teacher_model
        )
        
        self.logger.info(f"Оптимизация завершена. Compression ratio: {optimization_result.compression_ratio:.2f}x")
        
        # Сохраняем оптимизированную модель
        optimized_model_path = self.workspace_dir / "models" / f"optimized_model_{self.current_pipeline_id}.pt"
        torch.save(optimization_result.optimized_model.state_dict(), optimized_model_path)
        
        return optimization_result
    
    def _run_comprehensive_testing(self,
                                 original_model: nn.Module,
                                 optimized_model: nn.Module,
                                 val_data: torch.utils.data.DataLoader,
                                 test_data: Optional[torch.utils.data.DataLoader]) -> bool:
        """Комплексное тестирование оптимизированной модели"""
        
        self.logger.info("Запускаем комплексное тестирование...")
        
        # Измеряем метрики
        original_metrics = self.optimizer._measure_model_performance(original_model, val_data)
        optimized_metrics = self.optimizer._measure_model_performance(optimized_model, val_data)
        
        # Проверки
        validations = []
        
        # 1. Accuracy validation
        accuracy_drop = (original_metrics['accuracy'] - optimized_metrics['accuracy']) / original_metrics['accuracy']
        accuracy_ok = accuracy_drop <= self.config.accuracy_tolerance
        validations.append(("accuracy", accuracy_ok, f"drop: {accuracy_drop:.3f}"))
        
        # 2. Latency improvement
        latency_improvement = (original_metrics['latency_ms'] - optimized_metrics['latency_ms']) / original_metrics['latency_ms']
        latency_ok = latency_improvement >= self.config.latency_improvement_threshold
        validations.append(("latency", latency_ok, f"improvement: {latency_improvement:.3f}"))
        
        # 3. Compression ratio
        compression_ratio = original_metrics['size_mb'] / optimized_metrics['size_mb']
        compression_ok = compression_ratio >= self.config.compression_ratio_threshold
        validations.append(("compression", compression_ok, f"ratio: {compression_ratio:.2f}x"))
        
        # 4. Stress testing если включен
        if self.config.stress_test_enabled:
            stress_ok = self._run_stress_test(optimized_model, val_data)
            validations.append(("stress_test", stress_ok, "stress test"))
        
        # 5. Дополнительное тестирование на test_data
        if test_data is not None:
            test_metrics = self.optimizer._measure_model_performance(optimized_model, test_data)
            test_accuracy_drop = (original_metrics['accuracy'] - test_metrics['accuracy']) / original_metrics['accuracy']
            test_ok = test_accuracy_drop <= self.config.accuracy_tolerance
            validations.append(("test_data", test_ok, f"test accuracy drop: {test_accuracy_drop:.3f}"))
        
        # Результат
        all_passed = all(result for _, result, _ in validations)
        
        # Логирование
        for test_name, result, details in validations:
            status = "PASS" if result else "FAIL"
            self.logger.info(f"Тест {test_name}: {status} ({details})")
        
        self.logger.info(f"Комплексное тестирование: {'PASS' if all_passed else 'FAIL'}")
        
        return all_passed
    
    def _run_stress_test(self, 
                        model: nn.Module, 
                        val_data: torch.utils.data.DataLoader,
                        duration_seconds: int = 60) -> bool:
        """Stress testing модели"""
        
        self.logger.info(f"Запускаем stress test на {duration_seconds} секунд...")
        
        model.eval()
        start_time = time.time()
        iterations = 0
        errors = 0
        
        sample_batch = next(iter(val_data))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0][:1]
        else:
            sample_input = sample_batch[:1]
        
        while time.time() - start_time < duration_seconds:
            try:
                with torch.no_grad():
                    _ = model(sample_input)
                iterations += 1
            except Exception as e:
                errors += 1
                self.logger.warning(f"Ошибка в stress test: {e}")
            
            if iterations % 1000 == 0:
                self.logger.info(f"Stress test: {iterations} итераций, {errors} ошибок")
        
        error_rate = errors / iterations if iterations > 0 else 1.0
        stress_passed = error_rate < 0.01  # Менее 1% ошибок
        
        self.logger.info(f"Stress test завершен: {iterations} итераций, ошибок: {errors} ({error_rate:.3%})")
        
        return stress_passed
    
    def _run_benchmarking(self, model: nn.Module) -> Dict[str, Any]:
        """Бенчмаркинг производительности"""
        
        self.logger.info("Запускаем benchmarking...")
        
        # Здесь может быть более сложный benchmarking
        # Для примера используем простые метрики
        benchmark_results = {
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
            'forward_pass_time_ms': 0.0  # Будет заполнено ниже
        }
        
        # Измерение времени forward pass
        dummy_input = torch.randn(1, 100)  # Примерный размер
        model.eval()
        
        # Прогрев
        with torch.no_grad():
            for _ in range(10):
                try:
                    _ = model(dummy_input)
                except:
                    break
        
        # Измерение
        times = []
        with torch.no_grad():
            for _ in range(self.config.benchmark_iterations):
                try:
                    start_time = time.perf_counter()
                    _ = model(dummy_input)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)
                except:
                    break
        
        if times:
            benchmark_results['forward_pass_time_ms'] = float(np.mean(times))
        
        self.logger.info(f"Benchmarking завершен: {benchmark_results}")
        
        return benchmark_results
    
    def _prepare_for_deployment(self, model: nn.Module) -> Dict[str, str]:
        """Подготовка к deployment"""
        
        self.logger.info("Подготовка к deployment...")
        
        export_paths = {}
        
        for export_format in self.config.export_formats:
            try:
                if export_format == "torchscript":
                    export_path = self._export_torchscript(model)
                    export_paths["torchscript"] = export_path
                
                elif export_format == "onnx":
                    export_path = self._export_onnx(model)
                    export_paths["onnx"] = export_path
                
                elif export_format == "state_dict":
                    export_path = self._export_state_dict(model)
                    export_paths["state_dict"] = export_path
                
                else:
                    self.logger.warning(f"Неподдерживаемый формат экспорта: {export_format}")
                    
            except Exception as e:
                self.logger.error(f"Ошибка экспорта в {export_format}: {e}")
        
        self.logger.info(f"Deployment готов. Экспортированные форматы: {list(export_paths.keys())}")
        
        return export_paths
    
    def _export_torchscript(self, model: nn.Module) -> str:
        """Экспорт в TorchScript"""
        export_path = self.workspace_dir / "exports" / f"model_{self.current_pipeline_id}.pt"
        
        try:
            dummy_input = torch.randn(1, 100)
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(str(export_path))
            
            self.logger.info(f"TorchScript модель сохранена: {export_path}")
            
        except Exception as e:
            # Fallback to script mode
            self.logger.warning(f"Trace mode не удался, используем script mode: {e}")
            scripted_model = torch.jit.script(model)
            scripted_model.save(str(export_path))
        
        return str(export_path)
    
    def _export_onnx(self, model: nn.Module) -> str:
        """Экспорт в ONNX"""
        export_path = self.workspace_dir / "exports" / f"model_{self.current_pipeline_id}.onnx"
        
        try:
            import torch.onnx
            dummy_input = torch.randn(1, 100)
            
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            self.logger.info(f"ONNX модель сохранена: {export_path}")
            
        except ImportError:
            raise ImportError("ONNX не установлен. Установите: pip install onnx")
        
        return str(export_path)
    
    def _export_state_dict(self, model: nn.Module) -> str:
        """Экспорт state dict"""
        export_path = self.workspace_dir / "exports" / f"model_state_{self.current_pipeline_id}.pt"
        
        torch.save(model.state_dict(), export_path)
        
        self.logger.info(f"State dict сохранен: {export_path}")
        
        return str(export_path)
    
    def _finalize_pipeline(self, result: PipelineResult) -> None:
        """Финализация pipeline"""
        
        self.logger.info("Финализация pipeline...")
        
        # Сохраняем конфигурацию
        config_path = self.workspace_dir / "configs" / f"config_{self.current_pipeline_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
        
        # Создаем summary отчет
        summary_path = self.workspace_dir / "results" / f"summary_{self.current_pipeline_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Очистка temporary файлов
        self._cleanup_temporary_files()
        
        self.logger.info("Pipeline финализирован")
    
    def _rollback_pipeline(self, result: PipelineResult) -> None:
        """Rollback pipeline к исходному состоянию"""
        
        self.logger.info("Выполняем rollback pipeline...")
        
        try:
            # Восстановление из backup
            backup_path = f"original_model_{self.current_pipeline_id}"
            restored_model = self.backup_manager.restore_backup(backup_path)
            
            if restored_model:
                self.logger.info("Модель восстановлена из backup")
            
            # Очистка созданных файлов
            self._cleanup_pipeline_artifacts()
            
            result.logs.append("Pipeline rollback выполнен")
            
        except Exception as e:
            self.logger.error(f"Ошибка rollback: {e}")
            result.logs.append(f"Rollback failed: {e}")
    
    def _cleanup_temporary_files(self):
        """Очистка временных файлов"""
        # Здесь можно добавить логику очистки временных файлов
        pass
    
    def _cleanup_pipeline_artifacts(self):
        """Очистка артефактов pipeline"""
        try:
            # Удаляем созданные в рамках pipeline файлы
            artifacts_to_remove = [
                self.workspace_dir / "models" / f"optimized_model_{self.current_pipeline_id}.pt",
                self.workspace_dir / "exports" / f"model_{self.current_pipeline_id}.pt",
                self.workspace_dir / "exports" / f"model_{self.current_pipeline_id}.onnx",
            ]
            
            for artifact in artifacts_to_remove:
                if artifact.exists():
                    artifact.unlink()
                    
        except Exception as e:
            self.logger.warning(f"Ошибка при очистке артефактов: {e}")
    
    def _save_pipeline_result(self, result: PipelineResult):
        """Сохранение результата pipeline"""
        result_path = self.workspace_dir / "results" / f"result_{self.current_pipeline_id}.json"
        
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    
    @contextmanager
    def _pipeline_timeout_context(self):
        """Context manager для timeout pipeline"""
        timeout_seconds = self.config.max_pipeline_duration_hours * 3600
        
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise PipelineTimeoutError(f"Pipeline превысил лимит времени: {elapsed:.1f}s > {timeout_seconds:.1f}s")
    
    def get_pipeline_status(self, pipeline_id: Optional[str] = None) -> Optional[PipelineResult]:
        """Получение статуса pipeline"""
        target_id = pipeline_id or self.current_pipeline_id
        
        for result in self.pipeline_history:
            if result.pipeline_id == target_id:
                return result
        
        return None
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """Список всех pipeline"""
        return [
            {
                'pipeline_id': result.pipeline_id,
                'status': result.status.value,
                'start_time': result.start_time.isoformat(),
                'duration_seconds': result.duration_seconds,
                'compression_ratio': result.optimization_result.compression_ratio if result.optimization_result else None
            }
            for result in self.pipeline_history
        ]
    
    def cleanup_old_pipelines(self, keep_last: int = 10):
        """Очистка старых pipeline"""
        if len(self.pipeline_history) > keep_last:
            old_pipelines = self.pipeline_history[:-keep_last]
            
            for old_result in old_pipelines:
                try:
                    # Удаляем артефакты старых pipeline
                    old_artifacts = [
                        self.workspace_dir / "results" / f"result_{old_result.pipeline_id}.json",
                        self.workspace_dir / "configs" / f"config_{old_result.pipeline_id}.yaml",
                        self.workspace_dir / "models" / f"optimized_model_{old_result.pipeline_id}.pt"
                    ]
                    
                    for artifact in old_artifacts:
                        if artifact.exists():
                            artifact.unlink()
                    
                    # Удаляем backup
                    self.backup_manager.delete_backup(f"original_model_{old_result.pipeline_id}")
                    
                except Exception as e:
                    self.logger.warning(f"Ошибка при очистке pipeline {old_result.pipeline_id}: {e}")
            
            # Обновляем историю
            self.pipeline_history = self.pipeline_history[-keep_last:]
            
            self.logger.info(f"Очищены старые pipeline, оставлены последние {keep_last}")

class BackupManager:
    """Менеджер backup и restore моделей"""
    
    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.BackupManager")
    
    def create_backup(self, model: nn.Module, backup_name: str) -> str:
        """Создание backup модели"""
        backup_path = self.backup_dir / f"{backup_name}.pkl"
        
        backup_data = {
            'state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'architecture': str(model)
        }
        
        with open(backup_path, 'wb') as f:
            pickle.dump(backup_data, f)
        
        self.logger.info(f"Backup создан: {backup_path}")
        
        return str(backup_path)
    
    def restore_backup(self, backup_name: str) -> Optional[Dict[str, Any]]:
        """Восстановление backup"""
        backup_path = self.backup_dir / f"{backup_name}.pkl"
        
        if not backup_path.exists():
            self.logger.error(f"Backup не найден: {backup_path}")
            return None
        
        try:
            with open(backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            
            self.logger.info(f"Backup восстановлен: {backup_path}")
            
            return backup_data
            
        except Exception as e:
            self.logger.error(f"Ошибка восстановления backup: {e}")
            return None
    
    def delete_backup(self, backup_name: str) -> bool:
        """Удаление backup"""
        backup_path = self.backup_dir / f"{backup_name}.pkl"
        
        try:
            if backup_path.exists():
                backup_path.unlink()
                self.logger.info(f"Backup удален: {backup_path}")
                return True
            else:
                self.logger.warning(f"Backup не найден для удаления: {backup_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка удаления backup: {e}")
            return False
    
    def list_backups(self) -> List[str]:
        """Список всех backup"""
        backups = []
        for backup_file in self.backup_dir.glob("*.pkl"):
            backups.append(backup_file.stem)
        
        return backups

class PipelineTimeoutError(Exception):
    """Ошибка timeout pipeline"""
    pass