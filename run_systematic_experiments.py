#!/usr/bin/env python3
"""
SYSTEMATIC EXPERIMENT RUNNER
Runs all experiments for thesis with multiple seeds
- 2 datasets × 2 tokenizers × 3 initializations × 3 seeds = 36 experiments
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import random
import numpy as np

# Import your existing modules
from models.transformer_classifier import TransformerClassifier
from custom_tokenizers.bpe_tokenizer import BPETokenizer
from custom_tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Encode
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Create attention mask
        attention_mask = [1] * len(token_ids)

        # Pad
        padding_length = self.max_length - len(token_ids)
        token_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data_tsv(data_path, split='train', label2id=None):
    """Load from TSV files"""
    file_path = Path(data_path) / f'{split}.tsv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path, sep='\t')
    
    # Detect column names
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]
    
    texts = df[text_col].tolist()
    labels_raw = df[label_col].tolist()
    
    # Handle string labels
    if isinstance(labels_raw[0], str):
        labels_raw = [str(l).lower() for l in labels_raw]
        
        # Create label2id if not provided
        if label2id is None:
            unique = sorted(set(labels_raw))
            label2id = {l: i for i, l in enumerate(unique)}
        
        labels = [label2id[l] for l in labels_raw]
        return texts, labels, label2id
    
    # Numeric labels
    return texts, [int(l) for l in labels_raw], label2id

def train_or_load_tokenizer(tokenizer_type, dataset_path, output_dir, vocab_size=6452):
    """Train or load tokenizer"""
    tokenizer_dir = Path(output_dir) / 'tokenizers'
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(dataset_path).name

    if tokenizer_type == 'bpe':
        tokenizer_file = tokenizer_dir / f'{dataset_name}_bpe.json'

        if tokenizer_file.exists():
            print(f"   Loading existing BPE tokenizer: {tokenizer_file.name}")
            tokenizer = BPETokenizer(vocab_size=vocab_size)
            tokenizer.load(str(tokenizer_file))
            return tokenizer

        # Train new
        print(f"   Training BPE tokenizer on {dataset_name}...")
        train_texts, _, _ = load_data_tsv(dataset_path, 'train')

        # Create corpus file
        corpus_file = tokenizer_dir / f'{dataset_name}_corpus.txt'
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in train_texts:
                f.write(text + '\n')

        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(str(corpus_file))
        tokenizer.save(str(tokenizer_file))
        return tokenizer

    elif tokenizer_type == 'sentencepiece':
        model_prefix = str(tokenizer_dir / f'{dataset_name}_spm')
        tokenizer_config = tokenizer_dir / f'{dataset_name}_spm_config.json'

        if tokenizer_config.exists():
            print(f"   Loading existing SentencePiece tokenizer: {tokenizer_config.name}")
            tokenizer = SentencePieceTokenizer(vocab_size=vocab_size)
            tokenizer.load(str(tokenizer_config))
            return tokenizer

        # Train new
        print(f"   Training SentencePiece tokenizer on {dataset_name}...")
        train_texts, _, _ = load_data_tsv(dataset_path, 'train')

        corpus_file = tokenizer_dir / f'{dataset_name}_corpus.txt'
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in train_texts:
                f.write(text + '\n')

        tokenizer = SentencePieceTokenizer(vocab_size=vocab_size)
        tokenizer.train(str(corpus_file), model_prefix)
        tokenizer.save(str(tokenizer_config))
        return tokenizer

    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(dataloader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='macro', zero_division=0))

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(dataloader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='macro', zero_division=0))

def run_experiment(dataset_path, tokenizer_type, init_method, beta_input, beta_output,
                   output_dir, seed, epochs=20, batch_size=32, lr=2e-4):

    # Set seed first
    set_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {Path(dataset_path).name} | {tokenizer_type} | {init_method} | seed={seed}")
    print(f"β_in={beta_input}, β_out={beta_output}")
    print(f"{'='*70}\n")

    # Load data
    print("📊 Loading data...")
    train_texts, train_labels, label2id = load_data_tsv(dataset_path, 'train')
    val_texts, val_labels, _ = load_data_tsv(dataset_path, 'dev', label2id)
    test_texts, test_labels, _ = load_data_tsv(dataset_path, 'test', label2id)

    print(f"   Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    # Tokenizer
    print("🔤 Loading/training tokenizer...")
    tokenizer = train_or_load_tokenizer(tokenizer_type, dataset_path, output_dir)
    vocab_size = tokenizer.get_vocab_size()
    print(f"   Vocab size: {vocab_size:,}")

    # Datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    # Model
    print("🏗️  Building model...")
    model = TransformerClassifier(
        vocab_size=vocab_size,
        num_labels=len(label2id),
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        ff_dim=512,
        dropout=0.1,
        init_method=init_method,
        beta=beta_input,  # For balanced
        beta_input=beta_input,
        beta_output=beta_output
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training
    print("\n🚀 Starting training...")
    history = {
        'train_losses': [], 'train_accs': [], 'train_f1s': [],
        'val_losses': [], 'val_accs': [], 'val_f1s': []
    }

    best_val_acc = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['train_f1s'].append(train_f1)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        history['val_f1s'].append(val_f1)

        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, Path(output_dir) / 'best_model.pt')
            print(f"  ⭐ New best!")

    # Final test evaluation
    print("\n📊 Final test evaluation...")
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"  Test: Acc={test_acc:.4f}, F1={test_f1:.4f}")

    # Save results
    results = {
        'dataset': Path(dataset_path).name,
        'tokenizer': tokenizer_type,
        'init_method': init_method,
        'seed': seed,
        'best_val_accuracy': best_val_acc,
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'f1_macro': test_f1
        },
        'label2id': label2id,
        'initialization': {
            'method': init_method,
            'beta_input': beta_input,
            'beta_output': beta_output
        }
    }

    with open(Path(output_dir) / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(Path(output_dir) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ COMPLETE: Val={best_val_acc:.4f}, Test={test_acc:.4f}")
    print(f"📁 Results saved to: {output_dir}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Systematic Sentiment Analysis Experiments')
    parser.add_argument('--dataset', required=True, choices=['native', 'translated'],
                       help='Dataset type')
    parser.add_argument('--tokenizer', required=True, choices=['bpe', 'sentencepiece'],
                       help='Tokenizer type')
    parser.add_argument('--init', required=True, choices=['he', 'balanced', 'imbalanced'],
                       help='Initialization strategy')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_base', default='results_systematic',
                       help='Base output directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')

    args = parser.parse_args()

    # Determine data path
    if args.dataset == 'native':
        data_path = 'data/raw/hausa'
    else:
        data_path = 'data/raw/english_translated_hausa'

    # Determine initialization parameters
    if args.init == 'he':
        init_method = 'he'
        beta_input, beta_output = 1.0, 1.0
    elif args.init == 'balanced':
        init_method = 'generalized'
        beta_input, beta_output = 1.0, 1.0
    else:  # imbalanced
        init_method = 'generalized_imbalanced'
        beta_input, beta_output = 1.0, 3.0

    # Create output directory
    output_dir = Path(args.output_base) / f"{args.dataset}_{args.tokenizer}_{args.init}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    run_experiment(
        dataset_path=data_path,
        tokenizer_type=args.tokenizer,
        init_method=init_method,
        beta_input=beta_input,
        beta_output=beta_output,
        output_dir=str(output_dir),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == '__main__':
    main()
