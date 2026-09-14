import os
import json
import yaml
import datetime
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .data.dataset_loader import DatasetLoader
from .tokenizers.tokenizer_factory import TokenizerFactory
from .training.trainer import Trainer
from .training.metrics import MetricsCalculator
from .utils.visualization import plot_results_comparison, plot_language_comparison
from .utils.helpers import setup_logging, set_random_seeds


class ExperimentManager:
    """
    Manages multiple experiments, tracking results, and generating comprehensive reports
    """
    
    def __init__(self, config: Dict[str, Any], base_output_dir: str = "results"):
        self.config = config
        self.base_output_dir = base_output_dir
        self.experiment_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.output_dir = os.path.join(base_output_dir, self.experiment_id)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.output_dir, 'experiment_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        
        # Initialize results storage
        self.results = []
        self.dataset_cache = {}
        
        # Setup logging
        setup_logging(config.get('logging', {}))
        
        print(f"Experiment Manager initialized")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Output directory: {self.output_dir}")
    
    def get_dataset(self, dataset_name: str):
        """Get dataset, using cache if available"""
        if dataset_name not in self.dataset_cache:
            dataset_loader = DatasetLoader(self.config)
            dataset_dict = dataset_loader.load_dataset(dataset_name)
            self.dataset_cache[dataset_name] = dataset_dict
        
        return self.dataset_cache[dataset_name]
    
    def run_single_experiment(self, dataset_name: str, tokenizer_type: str, 
                            init_method: str, beta: float = 2.0) -> Dict[str, Any]:
        """Run a single experiment with given parameters"""
        
        print(f"\n{'='*60}")
        print(f"Running Experiment: {dataset_name} + {tokenizer_type} + {init_method}")
        if init_method == 'generalized':
            print(f"Beta value: {beta}")
        print(f"{'='*60}")
        
        # Set random seeds for reproducibility
        set_random_seeds(42)
        
        # Create experiment-specific output directory
        exp_name = f"{dataset_name}_{tokenizer_type}_{init_method}"
        if init_method == 'generalized':
            exp_name += f"_beta{beta}"
        
        exp_output_dir = os.path.join(self.output_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        try:
            # Load dataset
            dataset_dict = self.get_dataset(dataset_name)
            if dataset_dict is None:
                raise ValueError(f"Failed to load dataset: {dataset_name}")
            
            # Get dataset configuration
            dataset_config = self.config["datasets"][dataset_name]
            
            # Create corpus file for tokenizer training
            dataset_loader = DatasetLoader(self.config)
            corpus_file = os.path.join(exp_output_dir, f"{dataset_name}_corpus.txt")
            dataset_loader.create_corpus_file(dataset_dict, corpus_file, dataset_config["text_column"])
            
            # Train tokenizer
            tokenizer_path = self._train_tokenizer(
                tokenizer_type, corpus_file, dataset_name, dataset_config, exp_output_dir
            )
            
            # Get label information
            label_info = dataset_loader.get_label_info(dataset_dict, dataset_config)
            
            # Run training
            results = self._run_training(
                dataset_dict, tokenizer_path, tokenizer_type, dataset_name,
                init_method, beta, label_info, exp_output_dir
            )
            
            # Store results
            experiment_result = {
                'experiment_name': exp_name,
                'dataset': dataset_name,
                'tokenizer': tokenizer_type,
                'init_method': init_method,
                'beta': beta if init_method == 'generalized' else None,
                'timestamp': datetime.datetime.now().isoformat(),
                'output_dir': exp_output_dir,
                **results
            }
            
            self.results.append(experiment_result)
            
            # Save individual experiment result
            result_file = os.path.join(exp_output_dir, 'experiment_result.yaml')
            with open(result_file, 'w') as f:
                yaml.dump(experiment_result, f, indent=2)
            
            print(f"✓ Experiment completed successfully!")
            print(f"  Test F1: {results.get('test_macro_f1', 0):.4f}")
            print(f"  Results saved to: {exp_output_dir}")
            
            return experiment_result
            
        except Exception as e:
            print(f"✗ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Store failed experiment info
            failed_result = {
                'experiment_name': exp_name,
                'dataset': dataset_name,
                'tokenizer': tokenizer_type,
                'init_method': init_method,
                'beta': beta if init_method == 'generalized' else None,
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }
            
            self.results.append(failed_result)
            return failed_result
    
    def _train_tokenizer(self, tokenizer_type: str, corpus_file: str, 
                        dataset_name: str, dataset_config: Dict, output_dir: str) -> str:
        """Train tokenizer and return path"""
        tokenizer_config = self.config["tokenizers"][tokenizer_type]
        
        if tokenizer_type == "sentencepiece":
            model_prefix = os.path.join(output_dir, f"{dataset_name}_spm")
            return TokenizerFactory.train_sentencepiece_tokenizer(
                corpus_file, model_prefix, **tokenizer_config
            )
        elif tokenizer_type == "bpe":
            tokenizer_path = os.path.join(output_dir, f"{dataset_name}_bpe_tokenizer.json")
            return TokenizerFactory.train_bpe_tokenizer(
                corpus_file, tokenizer_path, **tokenizer_config
            )
        elif tokenizer_type in ["morphological", "phoneme"]:
            tokenizer = TokenizerFactory.create_tokenizer(
                tokenizer_type, 
                language=dataset_config["language_code"], 
                **tokenizer_config
            )
            tokenizer.train(corpus_file)
            tokenizer_path = os.path.join(output_dir, f"{dataset_name}_{tokenizer_type}_tokenizer.json")
            tokenizer.save(tokenizer_path)
            return tokenizer_path
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def _run_training(self, dataset_dict, tokenizer_path: str, tokenizer_type: str,
                     dataset_name: str, init_method: str, beta: float,
                     label_info: Dict, output_dir: str) -> Dict[str, Any]:
        """Run model training and evaluation"""
        from .data.data_utils import TokenizedDataset, collate_fn
        from .models.transformer_classifier import TransformerClassifier
        from torch.utils.data import DataLoader
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create datasets
        train_dataset = TokenizedDataset(dataset_dict['train'], tokenizer_path, tokenizer_type)
        val_dataset = TokenizedDataset(dataset_dict['val'], tokenizer_path, tokenizer_type)
        test_dataset = TokenizedDataset(dataset_dict['test'], tokenizer_path, tokenizer_type)
        
        # Create data loaders
        batch_size = self.config["training"]["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Create model
        model = TransformerClassifier(
            vocab_size=train_dataset.vocab_size,
            num_labels=label_info['num_labels'],
            embed_dim=self.config["model"]["embed_dim"],
            num_heads=self.config["model"]["num_heads"],
            num_layers=self.config["model"]["num_layers"],
            ff_dim=self.config["model"]["ff_dim"],
            dropout=self.config["model"]["dropout"],
            init_method=init_method,
            beta=beta,
            multilabel=label_info['multilabel']
        ).to(device)
        
        # Setup training
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        criterion = nn.BCEWithLogitsLoss() if label_info['multilabel'] else nn.CrossEntropyLoss()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=self.config["training"],
            output_dir=output_dir
        )
        
        # Train model
        trainer.train()
        
        # Final evaluation
        test_results = trainer.evaluate(test_loader)
        
        # Calculate additional metrics
        from .training.metrics import calculate_activation_sparsity
        sparsity = calculate_activation_sparsity(model, test_loader, device)
        
        return {
            'test_results': test_results,
            'sparsity': sparsity,
            'vocab_size': train_dataset.vocab_size,
            'model_params': sum(p.numel() for p in model.parameters()),
            'best_val_f1': trainer.best_val_f1,
            'best_epoch': trainer.best_epoch,
            'test_accuracy': test_results.get('accuracy', test_results.get('subset_accuracy', 0)),
            'test_macro_f1': test_results.get('macro_f1', 0),
            'test_micro_f1': test_results.get('micro_f1', 0)
        }
    
    def run_experiment_grid(self, datasets: List[str], tokenizers: List[str], 
                          init_methods: List[str], beta_values: List[float] = [1.0, 2.0, 3.0]):
        """Run a grid of experiments"""
        from itertools import product
        
        total_experiments = 0
        for dataset, tokenizer, init_method in product(datasets, tokenizers, init_methods):
            if init_method == 'generalized':
                total_experiments += len(beta_values)
            else:
                total_experiments += 1
        
        print(f"Starting experiment grid: {total_experiments} total experiments")
        
        completed = 0
        failed = 0
        
        for dataset, tokenizer, init_method in product(datasets, tokenizers, init_methods):
            if init_method == 'generalized':
                for beta in beta_values:
                    result = self.run_single_experiment(dataset, tokenizer, init_method, beta)
                    if 'error' in result:
                        failed += 1
                    else:
                        completed += 1
                    print(f"Progress: {completed + failed}/{total_experiments} experiments completed")
            else:
                result = self.run_single_experiment(dataset, tokenizer, init_method)
                if 'error' in result:
                    failed += 1
                else:
                    completed += 1
                print(f"Progress: {completed + failed}/{total_experiments} experiments completed")
        
        print(f"\n{'='*60}")
        print("EXPERIMENT GRID COMPLETED")
        print(f"{'='*60}")
        print(f"Total experiments: {total_experiments}")
        print(f"Completed successfully: {completed}")
        print(f"Failed: {failed}")
        
        # Generate final report
        self.generate_comprehensive_report()
        
        return self.results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        if not self.results:
            print("No results to analyze")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Filter out failed experiments
        successful_df = df[~df.get('status', '').eq('failed')].copy()
        
        if successful_df.empty:
            print("No successful experiments to analyze")
            return
        
        # Save comprehensive results
        results_file = os.path.join(self.output_dir, 'comprehensive_results.csv')
        successful_df.to_csv(results_file, index=False)
        print(f"Comprehensive results saved to: {results_file}")
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(successful_df)
        summary_file = os.path.join(self.output_dir, 'summary_statistics.yaml')
        with open(summary_file, 'w') as f:
            yaml.dump(summary_stats, f, indent=2)
        print(f"Summary statistics saved to: {summary_file}")
        
        # Generate visualizations
        self._generate_visualizations(successful_df)
        
        # Print summary to console
        self._print_summary(summary_stats)
    
    def _generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {}
        
        if 'test_macro_f1' in df.columns:
            # Best performance by tokenizer
            best_by_tokenizer = df.groupby('tokenizer')['test_macro_f1'].max().to_dict()
            summary['best_f1_by_tokenizer'] = best_by_tokenizer
            
            # Best performance by dataset
            best_by_dataset = df.groupby('dataset')['test_macro_f1'].max().to_dict()
            summary['best_f1_by_dataset'] = best_by_dataset
            
            # Best performance by initialization
            best_by_init = df.groupby('init_method')['test_macro_f1'].max().to_dict()
            summary['best_f1_by_initialization'] = best_by_init
            
            # Overall best result
            best_idx = df['test_macro_f1'].idxmax()
            best_result = df.loc[best_idx].to_dict()
            summary['overall_best'] = best_result
            
            # Average performance
            avg_performance = df.groupby(['dataset', 'tokenizer'])['test_macro_f1'].mean().to_dict()
            summary['average_f1_by_dataset_tokenizer'] = {
                f"{k[0]}_{k[1]}": v for k, v in avg_performance.items()
            }
        
        # Experiment statistics
        summary['experiment_statistics'] = {
            'total_experiments': len(df),
            'datasets': df['dataset'].unique().tolist(),
            'tokenizers': df['tokenizer'].unique().tolist(),
            'init_methods': df['init_method'].unique().tolist()
        }
        
        return summary
    
    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate visualization plots"""
        # Results comparison plot
        comparison_plot = os.path.join(self.output_dir, 'results_comparison.png')
        plot_results_comparison(df, comparison_plot)
        
        # Language comparison plot
        language_plot = os.path.join(self.output_dir, 'language_comparison.png')
        plot_language_comparison(df, language_plot)
        
        print(f"Visualizations saved to: {self.output_dir}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary to console"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        if 'overall_best' in summary:
            best = summary['overall_best']
            print(f"🏆 Best Overall Result:")
            print(f"   Dataset: {best.get('dataset', 'N/A')}")
            print(f"   Tokenizer: {best.get('tokenizer', 'N/A')}")
            print(f"   Initialization: {best.get('init_method', 'N/A')}")
            if best.get('beta'):
                print(f"   Beta: {best.get('beta', 'N/A')}")
            print(f"   Test F1: {best.get('test_macro_f1', 0):.4f}")
        
        if 'best_f1_by_tokenizer' in summary:
            print(f"\n📊 Best F1 by Tokenizer:")
            for tokenizer, f1 in summary['best_f1_by_tokenizer'].items():
                print(f"   {tokenizer:15}: {f1:.4f}")
        
        if 'best_f1_by_dataset' in summary:
            print(f"\n🌍 Best F1 by Dataset:")
            for dataset, f1 in summary['best_f1_by_dataset'].items():
                print(f"   {dataset:15}: {f1:.4f}")
        
        print(f"\n📁 All results saved in: {self.output_dir}")
    
    def load_results(self, results_file: str):
        """Load results from a previous experiment"""
        if results_file.endswith('.csv'):
            df = pd.read_csv(results_file)
            self.results = df.to_dict('records')
        elif results_file.endswith('.yaml') or results_file.endswith('.yml'):
            with open(results_file, 'r') as f:
                self.results = yaml.safe_load(f)
        else:
            raise ValueError("Results file must be CSV or YAML format")
        
        print(f"Loaded {len(self.results)} results from {results_file}")