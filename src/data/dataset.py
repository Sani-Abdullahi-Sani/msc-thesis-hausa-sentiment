import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, Dataset as HFDataset, concatenate_datasets

class DatasetManager:
    """Manager for loading and processing datasets"""
    
    def __init__(self, config):
        """
        Initialize dataset manager with configuration.
        
        Args:
            config: Dictionary containing dataset configuration parameters
        """
        self.config = config
        
    def load_dataset(self, dataset_name, output_dir=None):
        """
        Load and prepare dataset for experiments.
        
        Args:
            dataset_name: Name of the dataset
            output_dir: Directory to save dataset visualizations
            
        Returns:
            DatasetDict containing train, val, test splits
        """
        if dataset_name not in self.config["datasets"]:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
        
        dataset_config = self.config["datasets"][dataset_name]
        
        def read_csv_from_drive(path):
            """Reads a CSV file"""
            return pd.read_csv(path)
        
        # Read CSVs as pandas DataFrames
        try:
            train_df = read_csv_from_drive(dataset_config["train_path"])
            val_df = read_csv_from_drive(dataset_config["val_path"])
            test_df = read_csv_from_drive(dataset_config["test_path"])
        except Exception as e:
            print(f"Error loading {dataset_name} dataset: {e}")
            return None
        
        # Convert emotion columns to int64 before creating Hugging Face Datasets
        emotion_columns = dataset_config.get("label_columns", [])
        for df in [train_df, val_df, test_df]:
            for col in emotion_columns:
                if col in df.columns:
                    # Fill NaN or Inf values with 0 before converting to int64
                    df[col] = df[col].fillna(0).astype(int)
        
        # Convert to Hugging Face Datasets
        dataset_dict = DatasetDict({
            "train": HFDataset.from_pandas(train_df),
            "val": HFDataset.from_pandas(val_df),
            "test": HFDataset.from_pandas(test_df)
        })
        
        # Display stats
        print(f"Dataset {dataset_name} loaded successfully!")
        print(f"Train samples: {len(dataset_dict['train'])}")
        print(f"Validation samples: {len(dataset_dict['val'])}")
        print(f"Test samples: {len(dataset_dict['test'])}")
        
        # Analyze dataset
        if output_dir:
            self.analyze_dataset(dataset_dict, dataset_name, output_dir)
        
        return dataset_dict
    
    def analyze_dataset(self, dataset_dict, dataset_name, output_dir):
        """
        Analyze dataset and create visualizations.
        
        Args:
            dataset_dict: Dataset dictionary containing train, val, test splits
            dataset_name: Name of the dataset
            output_dir: Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine datasets for overall statistics
        combined_dataset = concatenate_datasets([dataset_dict["train"], dataset_dict["val"], dataset_dict["test"]])
        
        # Get label columns from config
        label_columns = self.config["datasets"][dataset_name].get("label_columns", [])
        existing_labels = [col for col in label_columns if col in dataset_dict['train'].column_names]
        
        if existing_labels:
            # Calculate emotion counts
            emotion_counts = {emotion: combined_dataset[emotion].count(1) for emotion in existing_labels}
            
            print(f"\nEmotion Counts for {dataset_name}:", emotion_counts)
            
            # Visualize emotion distribution
            plt.figure(figsize=(8, 8))
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange'][:len(emotions)]
            
            plt.pie(counts, labels=emotions, colors=colors,
                   autopct='%1.1f%%', startangle=140,
                   explode=[0.1 if i == 0 else 0 for i in range(len(emotions))])
            plt.title(f'Distribution of Emotions in {dataset_name} Dataset', fontsize=16)
            plt.savefig(f"{output_dir}/{dataset_name}_emotion_distribution.png")
            plt.close()
        else:
            print(f"\nNo emotion columns found in {dataset_name} dataset. Available columns:",
                 dataset_dict['train'].column_names)


class TokenizedDataset(Dataset):
    """Dataset class for tokenized text with support for different tokenizers"""
    
    def __init__(self, data, tokenizer_path, tokenizer_type="sentencepiece", max_length=128):
        """
        Initialize tokenized dataset.
        
        Args:
            data: HuggingFace dataset
            tokenizer_path: Path to the tokenizer model
            tokenizer_type: Type of tokenizer ('sentencepiece' or 'bpe')
            max_length: Maximum sequence length
        """
        self.tokenizer_type = tokenizer_type
        self.max_length = max_length
        
        # Import here to avoid circular imports
        import sentencepiece as spm
        from tokenizers import Tokenizer
        
        if tokenizer_type == "sentencepiece":
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(tokenizer_path)
            self.tokenize_fn = self._tokenize_sentencepiece
            self.vocab_size = self.tokenizer.get_piece_size()
        elif tokenizer_type == "bpe":
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenize_fn = self._tokenize_bpe
            self.vocab_size = self.tokenizer.get_vocab_size()
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        
        # Identify text column
        potential_text_cols = ['text', 'sentence', 'content']
        self.text_column = next((col for col in potential_text_cols if col in data.column_names), data.column_names[0])
        
        # Determine label type
        emotion_columns = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        self.multilabel = any(col in data.column_names for col in emotion_columns)
        
        if self.multilabel:
            self.label_columns = [col for col in emotion_columns if col in data.column_names]
            self.num_labels = len(self.label_columns)
            self.labels = data.to_pandas()[self.label_columns].values.tolist()
        else:
            potential_label_cols = ['label', 'emotion', 'sentiment', 'class']
            self.label_column = next((col for col in potential_label_cols if col in data.column_names), None)
            
            if not self.label_column:
                raise ValueError(f"Could not find label column. Available columns: {data.column_names}")
            
            # Get unique labels and create a mapping
            self.unique_labels = sorted(data.unique(self.label_column))
            self.label_to_id = {label: idx for idx, label in enumerate(self.unique_labels)}
            self.num_labels = len(self.unique_labels)
            self.labels = [self.label_to_id[label] for label in data[self.label_column]]
        
        # Store the text data
        self.texts = data[self.text_column]
        
        # Store reference to the original data
        self._data = data
    
    def _tokenize_sentencepiece(self, text):
        """
        Tokenize text using SentencePiece.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        token_ids = self.tokenizer.encode(text, out_type=int)
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        return token_ids
    
    def _tokenize_bpe(self, text):
        """
        Tokenize text using BPE.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        return token_ids
    
    @property
    def data(self):
        """Get the original data"""
        return self._data
    
    def __len__(self):
        """Get the length of the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index, list of indices, or slice
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        if isinstance(idx, int):  # Handle case where idx is a single index
            text = self.texts[idx]  # Fetch text using the index
            token_ids = self.tokenize_fn(text)
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            
            label = self.labels[idx]  # Fetch label using the index
            label_tensor = torch.tensor(label, dtype=torch.float if self.multilabel else torch.long)
            
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
                "labels": torch.tensor(self.labels[i], dtype=torch.float if self.multilabel else torch.long)
            } for i in idx]
            
            # Convert list of dictionaries to a single dictionary with batched tensors
            return {key: [d[key] for d in batch] for key in batch[0]}
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
    
    def create_dataloader(self, batch_size=32, shuffle=True, collate_fn=None):
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            collate_fn: Collate function for DataLoader
            
        Returns:
            DataLoader instance
        """
        from src.models.transformer import collate_fn as default_collate_fn
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_fn or default_collate_fn
        )
