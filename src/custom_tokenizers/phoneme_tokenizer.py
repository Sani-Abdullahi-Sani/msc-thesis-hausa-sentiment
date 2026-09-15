import re
from collections import Counter
from typing import List, Dict, Any
from .base_tokenizer import BaseTokenizer

# Optional imports with fallbacks
try:
    import epitran
    EPITRAN_AVAILABLE = True
except ImportError:
    print("Warning: epitran not available. Phoneme tokenizer will use rule-based fallback.")
    EPITRAN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available. Using basic tokenization.")
    NLTK_AVAILABLE = False


class PhonemeTokenizer(BaseTokenizer):
    """
    Phoneme-based tokenizer that converts text to phonetic representations
    using epitran for transliteration and custom phoneme segmentation.
    """

    def __init__(self, language="en", vocab_size=8000):
        super().__init__(language, vocab_size)
        self.phoneme_vocab = {}
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        self.phoneme_freq = Counter()

        # Initialize epitran transliterator
        self.epi = self._initialize_epitran()

        # Phoneme patterns for segmentation
        self.phoneme_patterns = self._get_phoneme_patterns()

    def _initialize_epitran(self):
        """Initialize epitran transliterator based on language"""
        if not EPITRAN_AVAILABLE:
            return None

        try:
            if self.language == "en":
                return epitran.Epitran('eng-Latn')
            elif self.language == "sw":  # Swahili
                return epitran.Epitran('swa-Latn')
            elif self.language == "ha":  # Hausa
                return epitran.Epitran('hau-Latn')
            else:
                # Default to English if language not supported
                return epitran.Epitran('eng-Latn')
        except Exception as e:
            print(f"Warning: Epitran not available for {self.language}, using rule-based approach: {e}")
            return None

    def _get_phoneme_patterns(self):
        """Define phoneme segmentation patterns"""
        return {
            # Common IPA phoneme boundaries
            'vowels': r'[aeiouæɑɒɔəɛɪʊʌʏyøœ]',
            'consonants': r'[bcdfghjklmnpqrstvwxzʃʒθðŋtʃdʒ]',
            'diphthongs': r'(aɪ|aʊ|eɪ|oʊ|ɔɪ)',
            'consonant_clusters': r'([bcdfghjklmnpqrstvwxz]{2,3})'
        }

    def _text_to_phonemes_rule_based(self, text):
        """
        Rule-based phoneme conversion for when epitran is not available
        This is a simplified approach - in practice, you'd want more sophisticated rules
        """
        # Simple English phoneme approximation rules
        phoneme_rules = {
            'ph': 'f', 'th': 'θ', 'sh': 'ʃ', 'ch': 'tʃ', 'ng': 'ŋ',
            'oo': 'u', 'ee': 'i', 'ea': 'i', 'ou': 'aʊ', 'ow': 'aʊ',
            'ai': 'eɪ', 'ay': 'eɪ', 'oy': 'ɔɪ', 'oi': 'ɔɪ',
            'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɒ', 'u': 'ʌ'
        }

        text = text.lower()
        for rule, phoneme in phoneme_rules.items():
            text = text.replace(rule, phoneme)

        return text

    def _segment_phonemes(self, phonetic_text):
        """Segment phonetic text into individual phonemes or phoneme clusters"""
        if not phonetic_text:
            return []

        phonemes = []
        i = 0

        while i < len(phonetic_text):
            # Try to match diphthongs first
            if i < len(phonetic_text) - 1:
                diphthong = phonetic_text[i:i+2]
                if re.match(self.phoneme_patterns['diphthongs'], diphthong):
                    phonemes.append(diphthong)
                    i += 2
                    continue

            # Match single phonemes
            if re.match(self.phoneme_patterns['vowels'] + '|' + self.phoneme_patterns['consonants'],
                       phonetic_text[i]):
                phonemes.append(phonetic_text[i])
                i += 1
            else:
                # Handle unknown characters
                if phonetic_text[i] not in [' ', '.', ',', '!', '?']:
                    phonemes.append(phonetic_text[i])
                i += 1

        return [p for p in phonemes if p.strip()]

    def train(self, corpus_file):
        """Train the phoneme tokenizer on a corpus"""
        print(f"Training phoneme tokenizer for {self.language}...")

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if NLTK_AVAILABLE:
                    words = word_tokenize(line.strip())
                else:
                    words = line.strip().split()

                for word in words:
                    # Skip punctuation
                    if re.match(r'^[a-zA-ZÀ-ÿ]+$', word):
                        # Convert to phonemes
                        if self.epi:
                            try:
                                phonetic = self.epi.transliterate(word.lower())
                            except:
                                phonetic = self._text_to_phonemes_rule_based(word)
                        else:
                            phonetic = self._text_to_phonemes_rule_based(word)

                        # Segment into phonemes
                        phonemes = self._segment_phonemes(phonetic)
                        for phoneme in phonemes:
                            if phoneme:
                                self.phoneme_freq[phoneme] += 1

        # Build vocabulary from most frequent phonemes
        most_common_phonemes = self.phoneme_freq.most_common(self.vocab_size - 4)

        # Add special tokens
        special_tokens = self.get_special_tokens()
        self.phoneme_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        self.id_to_phoneme = {idx: token for idx, token in enumerate(special_tokens)}

        # Add phonemes to vocabulary
        for phoneme, freq in most_common_phonemes:
            if phoneme not in self.phoneme_to_id:
                idx = len(self.phoneme_to_id)
                self.phoneme_to_id[phoneme] = idx
                self.id_to_phoneme[idx] = phoneme

        # Update base class attributes
        self.token_to_id = self.phoneme_to_id
        self.id_to_token = self.id_to_phoneme
        self.is_trained = True

        print(f"Built phoneme vocabulary with {len(self.phoneme_to_id)} phonemes")
        return len(self.phoneme_to_id)

    def encode(self, text):
        """Encode text into phoneme token IDs"""
        if NLTK_AVAILABLE:
            words = word_tokenize(text)
        else:
            words = text.split()

        token_ids = []

        for word in words:
            if re.match(r'^[a-zA-ZÀ-ÿ]+$', word):
                # Convert to phonemes
                if self.epi:
                    try:
                        phonetic = self.epi.transliterate(word.lower())
                    except:
                        phonetic = self._text_to_phonemes_rule_based(word)
                else:
                    phonetic = self._text_to_phonemes_rule_based(word)

                # Segment and encode
                phonemes = self._segment_phonemes(phonetic)
                for phoneme in phonemes:
                    if phoneme:
                        token_id = self.phoneme_to_id.get(phoneme, self.phoneme_to_id["<unk>"])
                        token_ids.append(token_id)
            elif word in self.phoneme_to_id:
                token_ids.append(self.phoneme_to_id[word])
            else:
                token_ids.append(self.phoneme_to_id["<unk>"])

        return token_ids

    def decode(self, token_ids):
        """Decode token IDs back to phonetic text"""
        phonemes = [self.id_to_phoneme.get(tid, "<unk>") for tid in token_ids]
        return " ".join(phonemes)

    def _get_save_data(self) -> Dict[str, Any]:
        """Add phoneme-specific data to save"""
        return {
            'phoneme_freq': dict(self.phoneme_freq),
            'phoneme_patterns': self.phoneme_patterns
        }

    def _load_additional_data(self, data: Dict[str, Any]) -> None:
        """Load phoneme-specific data"""
        self.phoneme_freq = Counter(data.get('phoneme_freq', {}))
        self.phoneme_patterns = data.get('phoneme_patterns', self._get_phoneme_patterns())
        self.phoneme_to_id = self.token_to_id
        self.id_to_phoneme = self.id_to_token
        
        # Reinitialize epitran
        self.epi = self._initialize_epitran()

    def get_special_tokens(self) -> List[str]:
        """Get special tokens for phoneme tokenizer"""
        return ["<unk>", "<pad>", "<s>", "</s>"]

    def tokenize_and_convert_to_ids(self, text: str, max_length: int = None) -> List[int]:
        """Tokenize text and convert to IDs with optional truncation"""
        token_ids = self.encode(text)
        
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids

    def batch_encode(self, texts: List[str], max_length: int = None) -> List[List[int]]:
        """Encode a batch of texts"""
        return [self.tokenize_and_convert_to_ids(text, max_length) for text in texts]
