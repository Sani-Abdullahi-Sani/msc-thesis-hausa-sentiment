import os
import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

from .sentencepiece_tokenizer import SentencePieceTokenizer
from .bpe_tokenizer import BPETokenizer
from .morphological_tokenizer import MorphologicalTokenizer
from .phoneme_tokenizer import PhonemeTokenizer


class TokenizerFactory:
    """Factory class for creating and managing different tokenizers"""
    
    @staticmethod
    def create_tokenizer(tokenizer_type: str, language: str = "en", vocab_size: int = 8000, **kwargs):
        """Create a tokenizer of the specified type"""
        if tokenizer_type == "morphological":
            return MorphologicalTokenizer(language=language, vocab_size=vocab_size)
        elif tokenizer_type == "phoneme":
            return PhonemeTokenizer(language=language, vocab_size=vocab_size)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    
    @staticmethod
    def train_sentencepiece_tokenizer(corpus_file: str, model_prefix: str, vocab_size: int = 8000, **kwargs):
        """Train a SentencePiece tokenizer"""
        # Ensure vocab_size is within allowed limit
        vocab_size = min(vocab_size, 6452)
        
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=kwargs.get("character_coverage", 0.9995),
            model_type=kwargs.get("model_type", "unigram"),
            input_sentence_size=1000000,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc_cf'
        )
        
        print(f"SentencePiece model trained and saved as {model_prefix}.model and {model_prefix}.vocab")
        return f"{model_prefix}.model"
    
    @staticmethod
    def train_bpe_tokenizer(corpus_file: str, model_path: str, vocab_size: int = 8000, **kwargs):
        """Train a BPE tokenizer"""
        # Initialize a BPE tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Configure pre-tokenization and post-processing
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # Prepare trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=kwargs.get("min_frequency", 2),
            special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
        )

        # Train the tokenizer
        tokenizer.train([corpus_file], trainer)

        # Save the tokenizer
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        tokenizer.save(model_path)

        print(f"BPE tokenizer trained and saved at {model_path}")
        return model_path
    
    @staticmethod
    def load_tokenizer(tokenizer_type: str, model_path: str, language: str = "en"):
        """Load a trained tokenizer"""
        if tokenizer_type == "sentencepiece":
            tokenizer = smp.SentencePieceProcessor()
            tokenizer.load(model_path)
            return tokenizer
        elif tokenizer_type == "bpe":
            return Tokenizer.from_file(model_path)
        elif tokenizer_type == "morphological":
            tokenizer = MorphologicalTokenizer(language=language)
            tokenizer.load(model_path)
            return tokenizer
        elif tokenizer_type == "phoneme":
            tokenizer = PhonemeTokenizer(language=language)
            tokenizer.load(model_path)
            return tokenizer
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")