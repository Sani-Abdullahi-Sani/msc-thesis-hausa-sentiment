"""
Data utilities for tokenization and dataset preparation - Sentiment Analysis
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any
import sentencepiece as smp
from tokenizers import Tokenizer

from ..tokenizers.morphological_tokenizer import MorphologicalTokenizer
from ..tokenizers.phoneme_tokenizer import PhonemeTokenizer


class TokenizedDataset(Dataset):
    """Dataset class supporting all four tokenization methods for sentiment analysis"""

    def __init__(self, data, tokenizer_path, tokenizer_type="sentencepiece", max_length=128):
        self.tokenizer_type = tokenizer_type
        self.max_length = max_length

        # Initialize tokenizer based on type
        if tokenizer_type == "sentencepiece":
            self.tokenizer = smp.SentencePieceProcessor()
            self.tokenizer.load(tokenizer_path)
            self.tokenize_fn = self._tokenize_sentencepiece
            self.vocab_size = self.tokenizer.get_piece_size()
        elif tokenizer_type == "bpe":
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenize_fn = self._tokenize_bpe
            self.vocab_size = self.tokenizer.get_vocab_size()
        elif tokenizer_type == "morphological":
            self.tokenizer = MorphologicalTokenizer()
            self.vocab_size = self.tokenizer.load(tokenizer_path)
            self.tokenize_fn = self._tokenize_morphological
        elif tokenizer_type == "phoneme":
            self.tokenizer = PhonemeTokenizer()
            self.vocab_size = self.tokenizer.load(tokenizer_path)
            self.tokenize_fn = self._tokenize_phoneme
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

        # Identify text column - for sentiment analysis, it's typically 'tweet'
        potential_text_cols = ['tweet', 'text', 'sentence', 'content']
        self.text_column = next((col for col in potential_text_cols if col in data.column_names), data.column_names[0])

        # For sentiment analysis - single label classification
        potential_label_cols = ['label', 'sentiment', 'emotion', 'class']
        self.label_column = next((col for col in potential_label_cols if col in data.column_names), None)

        if not self.label_column:
            raise ValueError(f"Could not find label column. Available columns: {data.column_names}")

        # Get unique labels and create a mapping for sentiment analysis
        self.unique_labels = sorted(data.unique(self.label_column))
        self.label_to_id = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.num_labels = len(self.unique_labels)
        
        # Convert labels to IDs
        self.labels = [self.label_to_id[label] for label in data[self.label_column]]

        # Store the text data
        self.texts = data[self.text_column]

        # Store reference to the original data
        self._data = data
        
        # This is single-label classification for sentiment
        self.multilabel = False
        
        print(f"🔤 Tokenized Dataset initialized:")
        print(f"   📊 Samples: {len(self.texts)}")
        print(f"   📝 Text column: {self.text_column}")
        print(f"   🎯 Label column: {self.label_column}")
        print(f"   🏷️  Labels: {self.unique_labels}")
        print(f"   📏 Vocab size: {self.vocab_size}")
        print(f"   🎭 Task type: Single-label sentiment classification")

    def _tokenize_sentencepiece(self, text):
        """Tokenize text using SentencePiece"""
        token_ids = self.tokenizer.encode(text, out_type=int)
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        return token_ids

    def _tokenize_bpe(self, text):
        """Tokenize text using BPE tokenizer"""
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        return token_ids

    def _tokenize_morphological(self, text):
        """Tokenize text using morphological tokenizer"""
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        return token_ids

    def _tokenize_phoneme(self, text):
        """Tokenize text using phoneme tokenizer"""
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        return token_ids

    @property
    def data(self):
        return self._data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):  # Handle case where idx is a single index
            text = self.texts[idx]  # Fetch text using the index
            token_ids = self.tokenize_fn(text)
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            label = self.labels[idx]  # Fetch label using the index
            # For sentiment classification, labels are integers (class indices)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label_tensor
            }
        elif isinstance(idx, (list, slice, np.ndarray)):
            # Create a batch of data using list comprehension and indexing data directly
            batch = [{
                "input_ids": torch.tensor(self.tokenize_fn(self.texts[i]), dtype=torch.long),
                "attention_mask": torch.ones_like(torch.tensor(self.tokenize_fn(self.texts[i]), dtype=torch.long)),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)
            } for i in idx]

            # Convert list of dictionaries to a single dictionary with batched tensors
            return {key: [d[key] for d in batch] for key in batch[0]}
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")


def collate_fn(batch):
    """
    Collate function for DataLoader that handles variable length sequences.
    Optimized for sentiment analysis (single-label classification).
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad input_ids and attention_masks
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # For sentiment analysis, labels are single integers
    labels = torch.stack(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }


def create_sentiment_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Create a DataLoader specifically configured for sentiment analysis.
    
    Args:
        dataset: TokenizedDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader configured for sentiment analysis
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=torch.cuda.is_available()
    )


def analyze_sentiment_distribution(dataset):
    """
    Analyze and print the sentiment distribution in the dataset.
    
    Args:
        dataset: TokenizedDataset instance
    """
    from collections import Counter
    
    label_counts = Counter(dataset.labels)
    total_samples = len(dataset.labels)
    
    print(f"\n📊 Sentiment Distribution Analysis:")
    print(f"   Total samples: {total_samples}")
    print(f"   Label distribution:")
    
    for label_id, count in sorted(label_counts.items()):
        label_name = dataset.id_to_label[label_id]
        percentage = (count / total_samples) * 100
        print(f"     {label_name}: {count} ({percentage:.1f}%)")
    
    # Check for class imbalance
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 2.0:
        print(f"   ⚠️  Class imbalance detected: {imbalance_ratio:.1f}:1 ratio")
        print(f"   💡 Consider using weighted loss or data balancing techniques")
    else:
        print(f"   ✅ Classes are relatively balanced ({imbalance_ratio:.1f}:1 ratio)")


def prepare_sentiment_datasets(dataset_dict, tokenizer_path, tokenizer_type, max_length=128):
    """
    Prepare train, validation, and test datasets for sentiment analysis.
    
    Args:
        dataset_dict: DatasetDict from DatasetLoader
        tokenizer_path: Path to trained tokenizer
        tokenizer_type: Type of tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print(f"🔧 Preparing sentiment datasets with {tokenizer_type} tokenizer...")
    
    train_dataset = TokenizedDataset(
        dataset_dict['train'], 
        tokenizer_path, 
        tokenizer_type, 
        max_length
    )
    
    val_dataset = TokenizedDataset(
        dataset_dict['val'], 
        tokenizer_path, 
        tokenizer_type, 
        max_length
    )
    
    test_dataset = TokenizedDataset(
        dataset_dict['test'], 
        tokenizer_path, 
        tokenizer_type, 
        max_length
    )
    
    # Analyze distributions
    print(f"\n📈 Training set analysis:")
    analyze_sentiment_distribution(train_dataset)
    
    return train_dataset, val_dataset, test_dataset