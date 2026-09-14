import os
import math
import json
import datetime
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, concatenate_datasets


class ExperimentManager:
    """Manages experiments across multiple datasets and tokenizers"""

    def __init__(self, config, output_dir=None):
        """Initialize the experiment manager with configuration"""
        self.config = config
        self.results = {}
        
        # Generate unique experiment ID if output directory not specified
        if output_dir is None:
            self.experiment_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            self.output_dir = os.path.join("experiment_results", self.experiment_id)
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)

        # Save configuration
        with open(os.path.join(self.output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"Experiment results will be saved to: {self.output_dir}")

    def load_dataset(self, dataset_name):
        """Load and prepare dataset for experiments"""
        if dataset_name not in self.config["datasets"]:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")

        dataset_config = self.config["datasets"][dataset_name]

        def read_csv(path):
            """Reads a CSV file"""
            return pd.read_csv(path)

        # Read CSVs as pandas DataFrames
        try:
            train_df = read_csv(dataset_config["train_path"])
            val_df = read_csv(dataset_config["val_path"])
            test_df = read_csv(dataset_config["test_path"])
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
            "train": Dataset.from_pandas(train_df),
            "val": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df)
        })

        # Display stats
        print(f"Dataset {dataset_name} loaded successfully!")
        print(f"Train samples: {len(dataset_dict['train'])}")
        print(f"Validation samples: {len(dataset_dict['val'])}")
        print(f"Test samples: {len(dataset_dict['test'])}")

        # Analyze dataset and save visualization
        self.analyze_dataset(dataset_dict, dataset_name)

        return dataset_dict

    def analyze_dataset(self, dataset_dict, dataset_name):
        """Analyze dataset and create visualizations"""
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
            plt.savefig(os.path.join(self.output_dir, f"{dataset_name}_emotion_distribution.png"))
            plt.close()
        else:
            print(f"\nNo emotion columns found in {dataset_name} dataset. Available columns:",
                  dataset_dict['train'].column_names)