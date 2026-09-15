import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from typing import Dict, List, Any, Optional
import pandas as pd


class MetricsCalculator:
    """Calculate various metrics for emotion classification"""
    
    def __init__(self, multilabel: bool = False, label_names: Optional[List[str]] = None):
        self.multilabel = multilabel
        self.label_names = label_names or []
    
    def calculate_metrics(self, y_true, y_pred, y_scores=None) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        if self.multilabel:
            return self._calculate_multilabel_metrics(y_true, y_pred, y_scores)
        else:
            return self._calculate_multiclass_metrics(y_true, y_pred, y_scores)
    
    def _calculate_multilabel_metrics(self, y_true, y_pred, y_scores=None) -> Dict[str, Any]:
        """Calculate metrics for multilabel classification"""
        metrics = {}
        
        # Convert to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if y_scores is not None and torch.is_tensor(y_scores):
            y_scores = y_scores.cpu().numpy()
        
        # Overall metrics
        metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_f1'] = per_class_f1.tolist()
        metrics['per_class_precision'] = per_class_precision.tolist()
        metrics['per_class_recall'] = per_class_recall.tolist()
        
        # Label-specific metrics if label names are provided
        if self.label_names:
            for i, label in enumerate(self.label_names):
                if i < len(per_class_f1):
                    metrics[f'{label}_f1'] = per_class_f1[i]
                    metrics[f'{label}_precision'] = per_class_precision[i]
                    metrics[f'{label}_recall'] = per_class_recall[i]
        
        # AUC if scores are provided
        if y_scores is not None:
            try:
                metrics['macro_auc'] = roc_auc_score(y_true, y_scores, average='macro')
                metrics['micro_auc'] = roc_auc_score(y_true, y_scores, average='micro')
                
                # Per-class AUC
                per_class_auc = roc_auc_score(y_true, y_scores, average=None)
                metrics['per_class_auc'] = per_class_auc.tolist()
                
                if self.label_names:
                    for i, label in enumerate(self.label_names):
                        if i < len(per_class_auc):
                            metrics[f'{label}_auc'] = per_class_auc[i]
            except ValueError as e:
                print(f"Warning: Could not calculate AUC scores: {e}")
        
        return metrics
    
    def _calculate_multiclass_metrics(self, y_true, y_pred, y_scores=None) -> Dict[str, Any]:
        """Calculate metrics for multiclass classification"""
        metrics = {}
        
        # Convert to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if y_scores is not None and torch.is_tensor(y_scores):
            y_scores = y_scores.cpu().numpy()
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Per-class metrics
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
        
        metrics['per_class_f1'] = per_class_f1.tolist()
        metrics['per_class_precision'] = per_class_precision.tolist()
        metrics['per_class_recall'] = per_class_recall.tolist()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        if self.label_names and len(self.label_names) == len(unique_labels):
            report = classification_report(y_true, y_pred, target_names=self.label_names, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in a formatted way"""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        if self.multilabel:
            print(f"Subset Accuracy: {metrics.get('subset_accuracy', 0):.4f}")
        else:
            print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        print(f"Macro F1: {metrics.get('macro_f1', 0):.4f}")
        print(f"Micro F1: {metrics.get('micro_f1', 0):.4f}")
        print(f"Weighted F1: {metrics.get('weighted_f1', 0):.4f}")
        
        print(f"\nMacro Precision: {metrics.get('macro_precision', 0):.4f}")
        print(f"Macro Recall: {metrics.get('macro_recall', 0):.4f}")
        
        # Per-class metrics
        if self.label_names and 'per_class_f1' in metrics:
            print(f"\nPer-class F1 scores:")
            for i, label in enumerate(self.label_names):
                if i < len(metrics['per_class_f1']):
                    print(f"  {label}: {metrics['per_class_f1'][i]:.4f}")
        
        # AUC scores if available
        if 'macro_auc' in metrics:
            print(f"\nMacro AUC: {metrics['macro_auc']:.4f}")
            print(f"Micro AUC: {metrics['micro_auc']:.4f}")


def calculate_activation_sparsity(model, dataloader, device):
    """Calculate activation sparsity for ReLU layers"""
    model.eval()
    total_zeros = 0
    total_elements = 0
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _ = model(input_ids, attention_mask)

            for activation in activations:
                zeros = (activation == 0).sum().item()
                total_zeros += zeros
                total_elements += activation.numel()

            activations = []  # Reset per batch

    for hook in hooks:
        hook.remove()

    return total_zeros / total_elements if total_elements > 0 else 0


def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for a metric"""
    bootstrap_scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric
        score = metric_fn(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    
    return ci_lower, ci_upper, np.array(bootstrap_scores)


def compare_models_statistical_test(results1, results2, metric='macro_f1'):
    """Compare two models using paired t-test on cross-validation results"""
    try:
        from scipy import stats
        
        if len(results1) != len(results2):
            raise ValueError("Results arrays must have the same length")
        
        scores1 = [r[metric] for r in results1]
        scores2 = [r[metric] for r in results2]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            'mean_diff': np.mean(scores1) - np.mean(scores2),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    except ImportError:
        print("Warning: scipy not available for statistical tests")
        return None
