import re
from collections import Counter
from typing import List, Dict, Any
from .base_tokenizer import BaseTokenizer

# Optional imports with fallbacks
try:
    import nltk
    from nltk.stem import SnowballStemmer
    from nltk.tokenize import word_tokenize
    # Download required NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available. Using basic tokenization.")
    NLTK_AVAILABLE = False


class MorphologicalTokenizer(BaseTokenizer):
    """
    Morphological tokenizer that segments words into morphemes using rule-based
    and statistical approaches, with language-specific handling for Hausa, Swahili, and English.
    """

    def __init__(self, language="en", vocab_size=8000):
        super().__init__(language, vocab_size)
        self.morpheme_vocab = {}
        self.morpheme_to_id = {}
        self.id_to_morpheme = {}
        self.morpheme_freq = Counter()

        # Language-specific morphological patterns
        self.morphological_patterns = self._get_morphological_patterns()

        # Initialize stemmer for the language
        if NLTK_AVAILABLE:
            if language == "en":
                self.stemmer = SnowballStemmer("english")
            elif language == "sw":  # Swahili
                # Use English stemmer as approximation for Swahili
                self.stemmer = SnowballStemmer("english")
            else:  # Hausa and others
                self.stemmer = SnowballStemmer("english")
        else:
            self.stemmer = None

    def _get_morphological_patterns(self):
        """Define language-specific morphological patterns"""
        patterns = {
            "en": {
                "prefixes": ["un", "re", "pre", "dis", "mis", "over", "under", "out", "up"],
                "suffixes": ["ing", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment", "able", "ible", "ful", "less"],
                "inflections": ["s", "es", "d", "ed", "ing", "er", "est"]
            },
            "sw": {  # Swahili patterns
                "prefixes": ["a", "wa", "m", "mi", "ki", "vi", "i", "zi", "u", "ku", "pa", "mu"],
                "suffixes": ["a", "e", "i", "o", "u", "wa", "ye", "za", "na", "ta", "ka"],
                "inflections": ["ni", "we", "ye", "tu", "mu", "wa"]
            },
            "ha": {  # Hausa patterns
                "prefixes": ["ma", "mu", "ba", "ya", "ta", "za", "ka", "su"],
                "suffixes": ["a", "i", "u", "e", "o", "wa", "ya", "ta", "na", "ka", "su"],
                "inflections": ["ni", "ka", "ki", "ya", "ta", "mu", "ku", "su", "na"]
            }
        }

        lang_code = self.language[:2] if len(self.language) > 2 else self.language
        return patterns.get(lang_code, patterns["en"])

    def _segment_morphemes(self, word):
        """
        Segment a word into morphemes using rule-based approach
        """
        if len(word) <= 2:
            return [word]

        morphemes = []
        remaining = word.lower()

        # Check for prefixes
        for prefix in sorted(self.morphological_patterns["prefixes"], key=len, reverse=True):
            if remaining.startswith(prefix) and len(remaining) > len(prefix):
                morphemes.append(prefix)
                remaining = remaining[len(prefix):]
                break

        # Check for suffixes
        for suffix in sorted(self.morphological_patterns["suffixes"], key=len, reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix):
                root = remaining[:-len(suffix)]
                if len(root) >= 2:  # Ensure root is meaningful
                    morphemes.append(root)
                    morphemes.append(suffix)
                    return morphemes

        # If no suffix found, check for inflections
        for inflection in sorted(self.morphological_patterns["inflections"], key=len, reverse=True):
            if remaining.endswith(inflection) and len(remaining) > len(inflection):
                root = remaining[:-len(inflection)]
                if len(root) >= 2:
                    morphemes.append(root)
                    morphemes.append(inflection)
                    return morphemes

        # If no morphological segmentation found, use stemming
        if not morphemes:
            if self.stemmer:
                stem = self.stemmer.stem(remaining)
                if stem != remaining and len(stem) >= 2:
                    morphemes.append(stem)
                    suffix = remaining[len(stem):]
                    if suffix:
                        morphemes.append(suffix)
                else:
                    morphemes.append(remaining)
            else:
                morphemes.append(remaining)
        else:
            morphemes.append(remaining)

        return [m for m in morphemes if m]  # Filter empty strings

    def train(self, corpus_file):
        """Train the morphological tokenizer on a corpus"""
        print(f"Training morphological tokenizer for {self.language}...")

        # Read corpus and segment into morphemes
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if NLTK_AVAILABLE:
                    words = word_tokenize(line.strip().lower())
                else:
                    words = line.strip().lower().split()

                for word in words:
                    # Skip punctuation and very short words
                    if re.match(r'^[a-zA-ZÀ-ÿ]+$', word) and len(word) > 1:
                        morphemes = self._segment_morphemes(word)
                        for morpheme in morphemes:
                            self.morpheme_freq[morpheme] += 1

        # Build vocabulary from most frequent morphemes
        most_common_morphemes = self.morpheme_freq.most_common(self.vocab_size - 4)  # Reserve space for special tokens

        # Add special tokens
        special_tokens = self.get_special_tokens()
        self.morpheme_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        self.id_to_morpheme = {idx: token for idx, token in enumerate(special_tokens)}

        # Add morphemes to vocabulary
        for morpheme, freq in most_common_morphemes:
            if morpheme not in self.morpheme_to_id:
                idx = len(self.morpheme_to_id)
                self.morpheme_to_id[morpheme] = idx
                self.id_to_morpheme[idx] = morpheme

        # Update base class attributes
        self.token_to_id = self.morpheme_to_id
        self.id_to_token = self.id_to_morpheme
        self.is_trained = True

        print(f"Built morphological vocabulary with {len(self.morpheme_to_id)} morphemes")
        return len(self.morpheme_to_id)

    def encode(self, text):
        """Encode text into morpheme token IDs"""
        if NLTK_AVAILABLE:
            words = word_tokenize(text.lower())
        else:
            words = text.lower().split()

        token_ids = []

        for word in words:
            if re.match(r'^[a-zA-ZÀ-ÿ]+$', word) and len(word) > 1:
                morphemes = self._segment_morphemes(word)
                for morpheme in morphemes:
                    token_id = self.morpheme_to_id.get(morpheme, self.morpheme_to_id["<unk>"])
                    token_ids.append(token_id)
            elif word in self.morpheme_to_id:
                token_ids.append(self.morpheme_to_id[word])
            else:
                token_ids.append(self.morpheme_to_id["<unk>"])

        return token_ids

    def decode(self, token_ids):
        """Decode token IDs back to text"""
        morphemes = [self.id_to_morpheme.get(tid, "<unk>") for tid in token_ids]
        return " ".join(morphemes)

    def _get_save_data(self) -> Dict[str, Any]:
        """Add morphological-specific data to save"""
        return {
            'morpheme_freq': dict(self.morpheme_freq),
            'morphological_patterns': self.morphological_patterns
        }

    def _load_additional_data(self, data: Dict[str, Any]) -> None:
        """Load morphological-specific data"""
        self.morpheme_freq = Counter(data.get('morpheme_freq', {}))
        self.morphological_patterns = data.get('morphological_patterns', self._get_morphological_patterns())
        self.morpheme_to_id = self.token_to_id
        self.id_to_morpheme = self.id_to_token