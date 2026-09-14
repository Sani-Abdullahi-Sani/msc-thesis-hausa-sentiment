import os
import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


def train_sentencepiece_tokenizer(dataset_dict, dataset_name, text_column, output_dir, config):
    """Train a SentencePiece tokenizer on the text data"""
    tokenizer_config = config["tokenizers"]["sentencepiece"]
    model_prefix = os.path.join(output_dir, f"{dataset_name}_spm")

    # Create a temporary file with all text data
    corpus_file = os.path.join(output_dir, f"{dataset_name}_corpus.txt")

    with open(corpus_file, 'w', encoding='utf-8') as f:
        for split in dataset_dict:
            for i in range(len(dataset_dict[split])):
                text = dataset_dict[split][i][text_column]
                f.write(text + '\n')

    # Train SentencePiece model
    # Reduced vocab_size to 6000 or lower if needed based on error message
    vocab_size = min(tokenizer_config["vocab_size"], 6452)  # Ensure vocab_size is within allowed limit
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=tokenizer_config.get("character_coverage", 0.9995),
        model_type=tokenizer_config.get("model_type", "unigram"),
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf'
    )

    print(f"SentencePiece model trained and saved as {model_prefix}.model and {model_prefix}.vocab")
    return f"{model_prefix}.model"


def train_bpe_tokenizer(dataset_dict, dataset_name, text_column, output_dir, config):
    """Train a BPE tokenizer on the text data"""
    tokenizer_config = config["tokenizers"]["bpe"]
    model_path = os.path.join(output_dir, f"{dataset_name}_bpe_tokenizer.json")

    # Create a temporary file with all text data
    corpus_file = os.path.join(output_dir, f"{dataset_name}_corpus.txt")

    if not os.path.exists(corpus_file):
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for split in dataset_dict:
                for i in range(len(dataset_dict[split])):
                    text = dataset_dict[split][i][text_column]
                    f.write(text + '\n')

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Configure pre-tokenization and post-processing
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Prepare trainer
    trainer = trainers.BpeTrainer(
        vocab_size=tokenizer_config["vocab_size"],
        min_frequency=tokenizer_config.get("min_frequency", 2),
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
    )

    # Train the tokenizer
    tokenizer.train([corpus_file], trainer)

    # Save the tokenizer
    tokenizer.save(model_path)

    print(f"BPE tokenizer trained and saved at {model_path}")
    return model_path


class TokenizerManager:
    """Manager for tokenizer training and handling"""
    
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        
    def train_tokenizer(self, dataset_dict, dataset_name, tokenizer_type):
        """Train a tokenizer of the specified type"""
        text_column = self.config["datasets"][dataset_name].get("text_column", "text")
        
        if tokenizer_type == "sentencepiece":
            return train_sentencepiece_tokenizer(dataset_dict, dataset_name, text_column, self.output_dir, self.config)
        elif tokenizer_type == "bpe":
            return train_bpe_tokenizer(dataset_dict, dataset_name, text_column, self.output_dir, self.config)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")