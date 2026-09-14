"""
Dataset loading and preprocessing utilities for sentiment analysis
Updated with data cleaning functionality
"""
import os
import re
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict, List, Optional


class DatasetLoader:
    """Load and preprocess datasets for sentiment analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def load_dataset(self, dataset_name: str) -> Optional[DatasetDict]:
        """Load and prepare dataset for sentiment analysis experiments"""
        if dataset_name not in self.config["datasets"]:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")

        dataset_config = self.config["datasets"][dataset_name]
        file_format = dataset_config.get("file_format", "csv")

        try:
            # Read files based on format
            if file_format == "tsv":
                # Read TSV files
                train_df = pd.read_csv(dataset_config["train_path"], sep='\t', encoding='utf-8')
                val_df = pd.read_csv(dataset_config["val_path"], sep='\t', encoding='utf-8')
                test_df = pd.read_csv(dataset_config["test_path"], sep='\t', encoding='utf-8')
            else:
                # Read CSV files (fallback)
                train_df = pd.read_csv(dataset_config["train_path"])
                val_df = pd.read_csv(dataset_config["val_path"])
                test_df = pd.read_csv(dataset_config["test_path"])

            print(f"📊 Raw data loaded for {dataset_name}:")
            print(f"   Train: {len(train_df)} samples")
            print(f"   Val: {len(val_df)} samples") 
            print(f"   Test: {len(test_df)} samples")

        except Exception as e:
            print(f"Error loading {dataset_name} dataset: {e}")
            return None

        # Clean and preprocess the data
        train_df = self._preprocess_sentiment_data(train_df, dataset_config)
        val_df = self._preprocess_sentiment_data(val_df, dataset_config)
        test_df = self._preprocess_sentiment_data(test_df, dataset_config)

        # Convert to Hugging Face Datasets
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "val": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df)
        })

        # Display stats after preprocessing
        print(f"📊 Processed sentiment dataset {dataset_name}:")
        print(f"   Train samples: {len(dataset_dict['train'])}")
        print(f"   Validation samples: {len(dataset_dict['val'])}")
        print(f"   Test samples: {len(dataset_dict['test'])}")
        
        # Show label distribution
        self._show_label_distribution(dataset_dict, dataset_config)

        return dataset_dict
    
    def _preprocess_sentiment_data(self, df: pd.DataFrame, dataset_config: Dict) -> pd.DataFrame:
        """Preprocess sentiment data with cleaning"""
        # Get column names
        text_column = dataset_config["text_column"]
        label_column = dataset_config["label_column"]
        
        # Clean the dataframe
        df = df.dropna(subset=[text_column, label_column])
        
        # === NEW: Data Cleaning for Social Media Text ===
        print(f"   🧹 Cleaning {text_column} data...")
        
        # Count original lengths for statistics
        original_lengths = df[text_column].str.len()
        original_count = len(df)
        
        # 1. Remove @user mentions (anonymized usernames)
        df[text_column] = df[text_column].str.replace(r'@user\b', '', regex=True, flags=re.IGNORECASE)
        
        # 2. Remove any remaining @username mentions
        df[text_column] = df[text_column].str.replace(r'@\w+', '', regex=True)
        
        # 3. Remove URLs (if any)
        df[text_column] = df[text_column].str.replace(r'http[s]?://\S+', '', regex=True)
        df[text_column] = df[text_column].str.replace(r'www\.\S+', '', regex=True)
        
        # 4. Clean up RT (retweet indicators)
        df[text_column] = df[text_column].str.replace(r'\bRT\b', '', regex=True, flags=re.IGNORECASE)
        
        # 5. Clean up excessive whitespace
        df[text_column] = df[text_column].str.replace(r'\s+', ' ', regex=True)
        
        # 6. Strip leading/trailing whitespace
        df[text_column] = df[text_column].str.strip()
        
        # Remove empty texts after cleaning
        df = df[df[text_column].str.len() > 0]
        
        # Calculate cleaning statistics
        cleaned_lengths = df[text_column].str.len()
        removed_samples = original_count - len(df)
        avg_reduction = original_lengths.mean() - cleaned_lengths.mean()
        
        print(f"   ✨ Cleaning results:")
        print(f"      Average length reduction: {avg_reduction:.1f} characters")
        print(f"      Removed empty texts: {removed_samples}")
        print(f"      Final samples: {len(df)}")
        # === END: Data Cleaning ===
        
        # Clean and standardize labels
        df[label_column] = df[label_column].astype(str).str.strip().str.lower()
        
        # Map labels to standard format if needed
        label_mapping = {
            'pos': 'positive',
            'neg': 'negative', 
            'neu': 'neutral',
            'neutral': 'neutral',
            'positive': 'positive',
            'negative': 'negative'
        }
        
        df[label_column] = df[label_column].map(label_mapping).fillna(df[label_column])
        
        # Filter out any invalid labels
        valid_labels = set(dataset_config.get("label_names", ["positive", "negative", "neutral"]))
        df = df[df[label_column].isin(valid_labels)]
        
        # Remove very short texts (likely not meaningful after cleaning)
        df = df[df[text_column].str.len() > 5]
        
        print(f"   ✅ After preprocessing: {len(df)} samples")
        return df.reset_index(drop=True)
    
    def _show_label_distribution(self, dataset_dict: DatasetDict, dataset_config: Dict):
        """Show label distribution across splits"""
        label_column = dataset_config["label_column"]
        
        print(f"📈 Label distribution:")
        for split_name, split_data in dataset_dict.items():
            labels = split_data[label_column]
            label_counts = pd.Series(labels).value_counts()
            print(f"   {split_name.capitalize()}:")
            for label, count in label_counts.items():
                percentage = (count / len(labels)) * 100
                print(f"     {label}: {count} ({percentage:.1f}%)")
    
    def create_corpus_file(self, dataset_dict: DatasetDict, output_path: str, text_column: str = "tweet"):
        """Create a corpus file from dataset for tokenizer training"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for split in dataset_dict:
                for i in range(len(dataset_dict[split])):
                    text = dataset_dict[split][i][text_column]
                    f.write(text + '\n')
        
        print(f"📝 Corpus file created at: {output_path}")
        return output_path
    
    def get_label_info(self, dataset_dict: DatasetDict, dataset_config: Dict) -> Dict:
        """Get information about labels in the sentiment dataset"""
        label_column = dataset_config.get("label_column", "label")
        
        if label_column not in dataset_dict['train'].column_names:
            raise ValueError(f"Label column '{label_column}' not found. Available columns: {dataset_dict['train'].column_names}")
        
        # Single-label sentiment classification
        unique_labels = sorted(dataset_dict['train'].unique(label_column))
        num_labels = len(unique_labels)
        
        # Create label to ID mapping for consistent encoding
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        
        return {
            'multilabel': False,  # Sentiment is single-label classification
            'num_labels': num_labels,
            'label_names': unique_labels,
            'label_to_id': label_to_id,
            'label_column': label_column
        }
    
    def convert_tsv_to_csv(self, dataset_name: str):
        """Utility function to convert TSV files to CSV format if needed"""
        if dataset_name not in self.config["datasets"]:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
            
        dataset_config = self.config["datasets"][dataset_name]
        
        for split in ["train", "val", "test"]:
            tsv_path = dataset_config[f"{split}_path"]
            csv_path = tsv_path.replace('.tsv', '.csv')
            
            if os.path.exists(tsv_path):
                df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"✅ Converted {tsv_path} to {csv_path}")
            else:
                print(f"⚠️  File not found: {tsv_path}")
    
    def show_cleaning_examples(self, dataset_name: str, num_examples: int = 3):
        """Show before/after examples of data cleaning"""
        if dataset_name not in self.config["datasets"]:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
            
        dataset_config = self.config["datasets"][dataset_name]
        text_column = dataset_config["text_column"]
        
        # Load a small sample to show cleaning examples
        train_path = dataset_config["train_path"]
        if os.path.exists(train_path):
            if train_path.endswith('.tsv'):
                df_sample = pd.read_csv(train_path, sep='\t', encoding='utf-8', nrows=num_examples)
            else:
                df_sample = pd.read_csv(train_path, nrows=num_examples)
            
            print(f"🧹 DATA CLEANING EXAMPLES for {dataset_name}")
            print(f"=" * 60)
            
            for i, original_text in enumerate(df_sample[text_column]):
                # Apply the same cleaning as in preprocessing
                cleaned_text = original_text
                
                # Remove @user mentions
                cleaned_text = re.sub(r'@user\b', '', cleaned_text, flags=re.IGNORECASE)
                cleaned_text = re.sub(r'@\w+', '', cleaned_text)
                cleaned_text = re.sub(r'http[s]?://\S+', '', cleaned_text)
                cleaned_text = re.sub(r'www\.\S+', '', cleaned_text)
                cleaned_text = re.sub(r'\bRT\b', '', cleaned_text, flags=re.IGNORECASE)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                cleaned_text = cleaned_text.strip()
                
                print(f"Example {i+1}:")
                print(f"  Original: {original_text}")
                print(f"  Cleaned:  {cleaned_text}")
                print(f"  Length:   {len(original_text)} → {len(cleaned_text)} characters")
                print()
        else:
            print(f"⚠️  File not found: {train_path}")
