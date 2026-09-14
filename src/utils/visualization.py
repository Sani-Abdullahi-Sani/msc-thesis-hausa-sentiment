"""
Visualization utilities for training analysis and results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import pandas as pd
from typing import List, Dict, Any


def plot_training_history(history: List[Dict[str, Any]], output_path: str = None):
    """Plot training history including loss and metrics"""
    if not history:
        print("No training history to plot")
        return
    
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(df['epoch'], df['train_acc'], label='Training Accuracy', marker='o')
    axes[0, 1].plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score plot
    axes[1, 0].plot(df['epoch'], df['train_f1'], label='Training F1', marker='o')
    axes[1, 0].plot(df['epoch'], df['val_f1'], label='Validation F1', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate plot (if available)
    if 'learning_rate' in df.columns:
        axes[1, 1].plot(df['epoch'], df['learning_rate'], label='Learning Rate', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names: List[str] = None, 
                         output_path: str = None, normalize: bool = True):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_embedding_tsne(model, tokenizer, vocab_subset: int = 1000, 
                       output_path: str = None, random_state: int = 42):
    """Plot t-SNE visualization of token embeddings"""
    try:
        # Get embeddings
        model.eval()
        with torch.no_grad():
            # Take a subset of embeddings to avoid memory issues
            embeddings = model.embedding.weight[:vocab_subset].cpu().numpy()
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=random_state, 
                   perplexity=min(30, vocab_subset-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            alpha=0.6, s=20, c=range(len(embeddings_2d)), 
                            cmap='viridis')
        
        # Add labels for some points (to avoid clutter)
        if hasattr(tokenizer, 'id_to_token'):
            for i in range(0, min(50, vocab_subset), 5):  # Every 5th token
                if i in tokenizer.id_to_token:
                    token = tokenizer.id_to_token[i]
                    if len(token) < 10:  # Only short tokens for readability
                        plt.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                                   fontsize=8, alpha=0.7)
        
        plt.title('t-SNE Visualization of Token Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"Error creating t-SNE plot: {e}")


def plot_results_comparison(results_df: pd.DataFrame, output_path: str = None):
    """Plot comparison of results across different configurations"""
    if results_df is None or results_df.empty:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance by tokenizer
    if 'test_macro_f1' in results_df.columns:
        sns.boxplot(data=results_df, x='tokenizer', y='test_macro_f1', ax=axes[0, 0])
        axes[0, 0].set_title('F1-Score Distribution by Tokenization Method')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Performance by dataset and tokenizer
    if 'test_macro_f1' in results_df.columns:
        pivot_data = results_df.groupby(['dataset', 'tokenizer'])['test_macro_f1'].max().reset_index()
        sns.barplot(data=pivot_data, x='dataset', y='test_macro_f1', hue='tokenizer', ax=axes[0, 1])
        axes[0, 1].set_title('Best F1-Score by Dataset and Tokenizer')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Performance by initialization method
    if 'test_macro_f1' in results_df.columns:
        sns.boxplot(data=results_df, x='init_method', y='test_macro_f1', ax=axes[1, 0])
        axes[1, 0].set_title('F1-Score Distribution by Initialization Method')
    
    # Model size vs performance
    if 'model_params' in results_df.columns and 'test_macro_f1' in results_df.columns:
        sns.scatterplot(data=results_df, x='model_params', y='test_macro_f1', 
                       hue='tokenizer', ax=axes[1, 1])
        axes[1, 1].set_title('Model Size vs Performance')
        axes[1, 1].set_xlabel('Number of Parameters')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Results comparison plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_language_comparison(results_df: pd.DataFrame, output_path: str = None):
    """Plot performance comparison across languages"""
    if results_df is None or results_df.empty or 'test_macro_f1' not in results_df.columns:
        print("No suitable results to plot")
        return
    
    # Create language comparison plot
    plt.figure(figsize=(12, 8))
    
    # Get best performance for each language-tokenizer combination
    best_results = results_df.groupby(['dataset', 'tokenizer'])['test_macro_f1'].max().reset_index()
    
    # Create grouped bar plot
    tokenizers = best_results['tokenizer'].unique()
    datasets = best_results['dataset'].unique()
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, tokenizer in enumerate(tokenizers):
        tokenizer_data = best_results[best_results['tokenizer'] == tokenizer]
        scores = [tokenizer_data[tokenizer_data['dataset'] == dataset]['test_macro_f1'].iloc[0] 
                 if len(tokenizer_data[tokenizer_data['dataset'] == dataset]) > 0 else 0 
                 for dataset in datasets]
        
        plt.bar(x + i*width, scores, width, label=tokenizer)
    
    plt.xlabel('Language/Dataset')
    plt.ylabel('Best F1-Score')
    plt.title('Best Performance Comparison Across Languages')
    plt.xticks(x + width*1.5, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Language comparison plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_curves(train_losses: List[float], val_losses: List[float], 
                     output_path: str = None):
    """Plot simple loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(metrics_dict: Dict[str, List[float]], 
                           output_path: str = None):
    """Plot comparison of different metrics"""
    plt.figure(figsize=(12, 8))
    
    x_pos = np.arange(len(metrics_dict))
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        plt.bar(i, np.mean(values), yerr=np.std(values), 
                capsize=5, label=metric_name, alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x_pos, list(metrics_dict.keys()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()