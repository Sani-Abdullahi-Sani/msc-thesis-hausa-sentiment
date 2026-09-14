import os
from typing import List, Dict, Any
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from .base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    BPE tokenizer wrapper that provides a consistent interface
    with other tokenizers in the project.
    """
    
    def __init__(self, language: str = "en", vocab_size: int = 8000):
        super().__init__(language, vocab_size)
        self.bpe_tokenizer = None
        self.min_frequency = 2
        self.add_prefix_space = False
        
    def train(self, corpus_file: str, **kwargs) -> int:
        """Train BPE tokenizer on corpus"""
        print(f"Training BPE tokenizer for {self.language}...")
        
        # Update parameters from kwargs
        self.min_frequency = kwargs.get('min_frequency', self.min_frequency)
        self.add_prefix_space = kwargs.get('add_prefix_space', self.add_prefix_space)
        
        # Initialize BPE tokenizer
        self.bpe_tokenizer = Tokenizer(models.BPE())
        
        # Configure pre-tokenization and post-processing
        self.bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=self.add_prefix_space
        )
        self.bpe_tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        # Prepare trainer
        special_tokens = self.get_special_tokens()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=special_tokens
        )
        
        # Train the tokenizer
        self.bpe_tokenizer.train([corpus_file], trainer)
        
        # Build vocabulary mappings
        self._build_vocab_mappings()
        
        self.is_trained = True
        print(f"BPE tokenizer trained with {self.get_vocab_size()} tokens")
        return self.get_vocab_size()
    
    def _build_vocab_mappings(self):
        """Build token-to-id and id-to-token mappings"""
        if self.bpe_tokenizer is None:
            raise ValueError("Tokenizer not trained")
        
        self.token_to_id = self.bpe_tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.bpe_tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        encoding = self.bpe_tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        if self.bpe_tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        return self.bpe_tokenizer.decode(token_ids)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts"""
        if self.bpe_tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        encodings = self.bpe_tokenizer.encode_batch(texts)
        return [encoding.ids for encoding in encodings]
    
    def decode_batch(self, token_ids_batch: List[List[int]]) -> List[str]:
        """Decode a batch of token ID sequences"""
        if self.bpe_tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        return self.bpe_tokenizer.decode_batch(token_ids_batch)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens (subwords)"""
        if self.bpe_tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        encoding = self.bpe_tokenizer.encode(text)
        return encoding.tokens
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.bpe_tokenizer is None:
            return 0
        return self.bpe_tokenizer.get_vocab_size()
    
    def save(self, filepath: str) -> None:
        """Save BPE tokenizer"""
        if self.bpe_tokenizer is None:
            raise ValueError("Tokenizer not trained")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the tokenizer
        self.bpe_tokenizer.save(filepath)
        
        # Save additional configuration
        config_data = {
            'language': self.language,
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'add_prefix_space': self.add_prefix_space,
            'actual_vocab_size': self.get_vocab_size(),
            'is_trained': self.is_trained,
            'tokenizer_type': 'bpe'
        }
        
        config_filepath = filepath.replace('.json', '_config.json')
        import json
        with open(config_filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        print(f"BPE tokenizer saved to: {filepath}")
        print(f"BPE tokenizer config saved to: {config_filepath}")
    
    def load(self, filepath: str) -> int:
        """Load BPE tokenizer"""
        # Load the tokenizer
        self.bpe_tokenizer = Tokenizer.from_file(filepath)
        
        # Try to load configuration
        config_filepath = filepath.replace('.json', '_config.json')
        if os.path.exists(config_filepath):
            import json
            with open(config_filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.language = config_data.get('language', self.language)
            self.vocab_size = config_data.get('vocab_size', self.vocab_size)
            self.min_frequency = config_data.get('min_frequency', self.min_frequency)
            self.add_prefix_space = config_data.get('add_prefix_space', self.add_prefix_space)
            self.is_trained = config_data.get('is_trained', True)
        
        # Build vocabulary mappings
        self._build_vocab_mappings()
        
        print(f"BPE tokenizer loaded with {self.get_vocab_size()} tokens")
        return self.get_vocab_size()
    
    def get_special_tokens(self) -> List[str]:
        """Get special tokens for BPE"""
        return ["<unk>", "<pad>", "<s>", "</s>"]
    
    def tokenize_and_convert_to_ids(self, text: str, max_length: int = None) -> List[int]:
        """Tokenize text and convert to IDs with optional truncation"""
        token_ids = self.encode(text)
        
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids
    
    def batch_encode_plus(self, texts: List[str], max_length: int = None, 
                         padding: bool = False, truncation: bool = True) -> Dict[str, List[List[int]]]:
        """Enhanced batch encoding with padding and truncation options"""
        all_token_ids = []
        all_attention_masks = []
        
        for text in texts:
            token_ids = self.encode(text)
            
            # Truncation
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(token_ids)
            
            all_token_ids.append(token_ids)
            all_attention_masks.append(attention_mask)
        
        # Padding
        if padding and max_length:
            pad_token_id = self.token_to_id.get("<pad>", 0)
            
            for i in range(len(all_token_ids)):
                current_length = len(all_token_ids[i])
                if current_length < max_length:
                    padding_length = max_length - current_length
                    all_token_ids[i].extend([pad_token_id] * padding_length)
                    all_attention_masks[i].extend([0] * padding_length)
        
        return {
            'input_ids': all_token_ids,
            'attention_mask': all_attention_masks
        }
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Additional data to save (overrides base class method)"""
        return {
            'min_frequency': self.min_frequency,
            'add_prefix_space': self.add_prefix_space,
            'actual_vocab_size': self.get_vocab_size()
        }
    
    def _load_additional_data(self, data: Dict[str, Any]) -> None:
        """Load additional data (overrides base class method)"""
        self.min_frequency = data.get('min_frequency', 2)
        self.add_prefix_space = data.get('add_prefix_space', False)
    
    def train_from_iterator(self, text_iterator, vocab_size: int = None, **kwargs):
        """Train tokenizer from an iterator of texts"""
        if vocab_size:
            self.vocab_size = vocab_size
        
        # Update parameters from kwargs
        self.min_frequency = kwargs.get('min_frequency', self.min_frequency)
        
        # Initialize BPE tokenizer
        self.bpe_tokenizer = Tokenizer(models.BPE())
        
        # Configure pre-tokenization and post-processing
        self.bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=self.add_prefix_space
        )
        self.bpe_tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        # Prepare trainer
        special_tokens = self.get_special_tokens()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=special_tokens
        )
        
        # Train from iterator
        self.bpe_tokenizer.train_from_iterator(text_iterator, trainer)
        
        # Build vocabulary mappings
        self._build_vocab_mappings()
        
        self.is_trained = True
        print(f"BPE tokenizer trained from iterator with {self.get_vocab_size()} tokens")
        return self.get_vocab_size()