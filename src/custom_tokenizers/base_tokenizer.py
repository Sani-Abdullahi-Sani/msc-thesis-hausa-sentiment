"""
Base tokenizer interface and common functionality
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
import os


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers"""
    
    def __init__(self, language: str = "en", vocab_size: int = 8000):
        self.language = language
        self.vocab_size = vocab_size
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.is_trained = False
    
    @abstractmethod
    def train(self, corpus_file: str) -> int:
        """Train the tokenizer on a corpus file"""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        pass
    
    def save(self, filepath: str) -> None:
        """Save the trained tokenizer"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        tokenizer_data = {
            'language': self.language,
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'is_trained': self.is_trained,
            'tokenizer_type': self.__class__.__name__
        }
        
        # Add any additional data from subclasses
        additional_data = self._get_save_data()
        tokenizer_data.update(additional_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str) -> int:
        """Load a trained tokenizer"""
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        self.language = tokenizer_data['language']
        self.vocab_size = tokenizer_data['vocab_size']
        self.token_to_id = tokenizer_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in tokenizer_data['id_to_token'].items()}
        self.is_trained = tokenizer_data.get('is_trained', True)
        
        # Load any additional data for subclasses
        self._load_additional_data(tokenizer_data)
        
        return len(self.token_to_id)
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Override in subclasses to add additional data to save"""
        return {}
    
    def _load_additional_data(self, data: Dict[str, Any]) -> None:
        """Override in subclasses to load additional data"""
        pass
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.token_to_id)
    
    def get_special_tokens(self) -> List[str]:
        """Get special tokens"""
        return ["<unk>", "<pad>", "<s>", "</s>"]