#!/usr/bin/env python3
"""
Create matched subsamples for controlled cross-lingual comparison
Ensures all datasets have identical size, splits, and class distributions
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

def create_matched_subsample(input_dir, output_dir, dataset_name, target_sizes, seed=42):
    """
    Create stratified subsample matching target sizes
    
    Parameters:
    -----------
    input_dir : str
        Directory containing train.tsv, dev.tsv, test.tsv
    output_dir : str
        Directory to save matched subsamples
    dataset_name : str
        Name for logging
    target_sizes : dict
        {'train': N, 'val': N, 'test': N}
    seed : int
        Random seed for reproducibility
    """
    
    print(f"\n{'='*80}")
    print(f"Creating matched subsample: {dataset_name}")
    print(f"{'='*80}")
    
    # Load all splits
    train_df = pd.read_csv(f'{input_dir}/train.tsv', sep='\t')
    dev_df = pd.read_csv(f'{input_dir}/dev.tsv', sep='\t')
    test_df = pd.read_csv(f'{input_dir}/test.tsv', sep='\t')
    
    # Combine all data
    combined = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    print(f"Original data: {len(combined)} samples")
    print(f"Target total: {sum(target_sizes.values())} samples")
    
    # First split: separate test set
    train_val, test = train_test_split(
        combined,
        test_size=target_sizes['test'],
        stratify=combined['label'],
        random_state=seed
    )
    
    # Second split: separate train and validation
    train, val = train_test_split(
        train_val,
        test_size=target_sizes['val'],
        train_size=target_sizes['train'],
        stratify=train_val['label'],
        random_state=seed
    )
    
    # Verify distributions
    print(f"\nClass distributions:")
    print(f"Train ({len(train)} samples):")
    train_dist = train['label'].value_counts(normalize=True) * 100
    for label, pct in train_dist.items():
        print(f"  {label}: {pct:.1f}%")
    
    print(f"\nValidation ({len(val)} samples):")
    val_dist = val['label'].value_counts(normalize=True) * 100
    for label, pct in val_dist.items():
        print(f"  {label}: {pct:.1f}%")
    
    print(f"\nTest ({len(test)} samples):")
    test_dist = test['label'].value_counts(normalize=True) * 100
    for label, pct in test_dist.items():
        print(f"  {label}: {pct:.1f}%")
    
    # Save subsamples
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(f'{output_dir}/train.tsv', sep='\t', index=False)
    val.to_csv(f'{output_dir}/dev.tsv', sep='\t', index=False)
    test.to_csv(f'{output_dir}/test.tsv', sep='\t', index=False)
    
    print(f"\n✅ Saved to: {output_dir}")
    
    return {
        'train': train,
        'val': val,
        'test': test
    }

def main():
    print("="*80)
    print("CREATING MATCHED DATASETS FOR CONTROLLED COMPARISON")
    print("="*80)
    
    # Define target sizes (matching smallest dataset: Native Hausa)
    # Using standard 70/15/15 split
    total_samples = 22152  # Native Hausa total
    
    target_sizes = {
        'train': 15506,  # 70%
        'val': 3323,     # 15%
        'test': 3323     # 15%
    }
    
    print(f"\nTarget distribution (70/15/15):")
    print(f"  Train: {target_sizes['train']:,} samples (70%)")
    print(f"  Val:   {target_sizes['val']:,} samples (15%)")
    print(f"  Test:  {target_sizes['test']:,} samples (15%)")
    print(f"  Total: {sum(target_sizes.values()):,} samples")
    
    # Create matched subsamples for all datasets
    datasets = {
        'Native Hausa': {
            'input': 'data/raw/hausa',
            'output': 'data/matched/hausa'
        },
        'Native English': {
            'input': 'data/raw/english',
            'output': 'data/matched/english'
        },
        'Translated Hausa': {
            'input': 'data/raw/english_translated_hausa',
            'output': 'data/matched/english_translated_hausa'
        }
    }
    
    all_samples = {}
    
    for dataset_name, paths in datasets.items():
        samples = create_matched_subsample(
            input_dir=paths['input'],
            output_dir=paths['output'],
            dataset_name=dataset_name,
            target_sizes=target_sizes,
            seed=42
        )
        all_samples[dataset_name] = samples
    
    # Verification: Check class distributions are similar
    print("\n" + "="*80)
    print("VERIFICATION: Class Distribution Comparison")
    print("="*80)
    
    print("\nTraining Set Distributions:")
    print(f"{'Dataset':<25} {'Positive':>12} {'Negative':>12} {'Neutral':>12}")
    print("-"*65)
    
    for dataset_name, samples in all_samples.items():
        train = samples['train']
        dist = train['label'].value_counts()
        total = len(train)
        
        pos = dist.get('positive', dist.get('Positive', 0))
        neg = dist.get('negative', dist.get('Negative', 0))
        neu = dist.get('neutral', dist.get('Neutral', 0))
        
        print(f"{dataset_name:<25} {pos:>5} ({pos/total*100:>4.1f}%) "
              f"{neg:>5} ({neg/total*100:>4.1f}%) "
              f"{neu:>5} ({neu/total*100:>4.1f}%)")
    
    # Create README
    readme_path = 'data/matched/README.md'
    with open(readme_path, 'w') as f:
        f.write("# Matched Datasets for Controlled Cross-Lingual Comparison\n\n")
        f.write("## Overview\n\n")
        f.write("These datasets have been created through stratified sampling to ensure:\n")
        f.write("- Identical sample sizes across all languages\n")
        f.write("- Standard 70/15/15 train/val/test split\n")
        f.write("- Matched class distributions\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- Total samples per dataset: {sum(target_sizes.values()):,}\n")
        f.write(f"- Train: {target_sizes['train']:,} (70%)\n")
        f.write(f"- Validation: {target_sizes['val']:,} (15%)\n")
        f.write(f"- Test: {target_sizes['test']:,} (15%)\n\n")
        f.write("## Datasets\n\n")
        f.write("1. Native Hausa: `data/matched/hausa/`\n")
        f.write("2. Native English: `data/matched/english/`\n")
        f.write("3. Translated Hausa: `data/matched/english_translated_hausa/`\n\n")
        f.write("## Methodology\n\n")
        f.write("Stratified random sampling with seed=42 for reproducibility.\n")
    
    print(f"\n✅ Created README: {readme_path}")
    
    print("\n" + "="*80)
    print("✅ MATCHED DATASETS CREATED SUCCESSFULLY")
    print("="*80)
    print("\nNext steps:")
    print("1. Use data/matched/ for main experiments")
    print("2. Report full dataset results in appendix")
    print("3. Update thesis to explain matched sampling methodology")

if __name__ == "__main__":
    main()
