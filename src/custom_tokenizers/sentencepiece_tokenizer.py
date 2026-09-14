import os
import sentencepiece as spm
from typing import List, Dict, Any
from .base_tokenizer import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """
    SentencePiece tokenizer wrapper that provides a consistent interface
    with other tokenizers in the project.
    """
    
    def __init__(self, language: str = "en", vocab_size: int = 8000):
        super().__init__(language, vocab_size)
        self.sp_model = None
        self.model_path = None
        
        # SentencePiece specific parameters
        self.model_type = "unigram"
        self.character_coverage = 0.9995
    
    def train(self, corpus_file: str, model_prefix: str = None, **kwargs) -> int:
        """Train SentencePiece model on corpus"""
        if model_prefix is None:
            model_prefix = corpus_file.replace('.txt', '_spm')
        
        print(f"Training SentencePiece tokenizer for {self.language}...")
        
        # Update parameters from kwargs
        self.model_type = kwargs.get('model_type', self.model_type)
        self.character_coverage = kwargs.get('character_coverage', self.character_coverage)
        
        # Ensure vocab_size is within allowed limit
        effective_vocab_size = min(self.vocab_size, 6452)
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=effective_vocab_size,
            character_coverage=self.character_coverage,
            model_type=self.model_type,
            input_sentence_size=1000000,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc_cf'
        )
        
        # Load the trained model
        self.model_path = f"{model_prefix}.model"
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(self.model_path)
        
        # Build vocabulary mappings
        self._build_vocab_mappings()
        
        self.is_trained = True
        print(f"SentencePiece model trained with {self.get_vocab_size()} tokens")
        return self.get_vocab_size()
    
    def _build_vocab_mappings(self):
        """Build token-to-id and id-to-token mappings"""
        if self.sp_model is None:
            raise ValueError("Model not trained or loaded")
        
        vocab_size = self.sp_model.get_piece_size()
        
        self.token_to_id = {}
        self.id_to_token = {}
        
        for i in range(vocab_size):
            piece = self.sp_model.id_to_piece(i)
            self.token_to_id[piece] = i
            self.id_to_token[i] = piece
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.sp_model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.sp_model.encode(text, out_type=int)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        if self.sp_model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.sp_model.decode(token_ids)
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode text to token pieces (subwords)"""
        if self.sp_model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.sp_model.encode(text, out_type=str)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.sp_model is None:
            return 0
        return self.sp_model.get_piece_size()
    
    def save(self, filepath: str) -> None:
        """Save tokenizer configuration and model path"""
        if self.sp_model is None:
            raise ValueError("Model not trained")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save configuration
        tokenizer_data = {
            'language': self.language,
            'vocab_size': self.vocab_size,
            'model_type': self.model_type,
            'character_coverage': self.character_coverage,
            'model_path': self.model_path,
            'actual_vocab_size': self.get_vocab_size(),
            'is_trained': self.is_trained,
            'tokenizer_type': 'sentencepiece'
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
        print(f"SentencePiece tokenizer config saved to: {filepath}")
    
    def load(self, filepath: str) -> int:
        """Load tokenizer from saved configuration"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Load configuration
        self.language = tokenizer_data['language']
        self.vocab_size = tokenizer_data['vocab_size']
        self.model_type = tokenizer_data['model_type']
        self.character_coverage = tokenizer_data['character_coverage']
        self.model_path = tokenizer_data['model_path']
        self.is_trained = tokenizer_data.get('is_trained', True)
        
        # Load SentencePiece model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SentencePiece model file not found: {self.model_path}")
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(self.model_path)
        
        # Build vocabulary mappings
        self._build_vocab_mappings()
        
        print(f"SentencePiece tokenizer loaded with {self.get_vocab_size()} tokens")
        return self.get_vocab_size()
    
    def get_special_tokens(self) -> List[str]:
        """Get special tokens for SentencePiece"""
        return ["<unk>", "<s>", "</s>"]  # SentencePiece uses different special tokens
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Additional data to save (overrides base class method)"""
        return {
            'model_type': self.model_type,
            'character_coverage': self.character_coverage,
            'model_path': self.model_path,
            'actual_vocab_size': self.get_vocab_size() if self.sp_model else 0
        }
    
    def _load_additional_data(self, data: Dict[str, Any]) -> None:
        """Load additional data (overrides base class method)"""
        self.model_type = data.get('model_type', 'unigram')
        self.character_coverage = data.get('character_coverage', 0.9995)
        self.model_path = data.get('model_path')
        
        if self.model_path and os.path.exists(self.model_path):
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(self.model_path)
            self._build_vocab_mappings()
    
    def tokenize_and_convert_to_ids(self, text: str, max_length: int = None) -> List[int]:
        """Tokenize text and convert to IDs with optional truncation"""
        token_ids = self.encode(text)
        
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids
    
    def batch_encode(self, texts: List[str], max_length: int = None) -> List[List[int]]:
        """Encode a batch of texts"""
        return [self.tokenize_and_convert_to_ids(text, max_length) for text in texts]