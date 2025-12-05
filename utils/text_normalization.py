import re
import unicodedata
from typing import List

class TextNormalizer:
    @staticmethod
    def strip_accents(text: str) -> str:
        """
        Strip accents and diacritics from text.
        Example: 'Émile' -> 'Emile'
        """
        text = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in text if not unicodedata.combining(c))

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for robust matching:
        - Unicode normalization (NFKD)
        - Accent stripping
        - Lowercasing
        - Removing punctuation and special characters
        - Collapsing spaces
        """
        if not isinstance(text, str):
            text = str(text)
        text = TextNormalizer.strip_accents(text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def normalize_list(items: List[str]) -> List[str]:
        """
        Apply normalize_text to each element in a list.
        """
        return [TextNormalizer.normalize_text(i) for i in items]

    @staticmethod
    def normalize_dict_keys(d: dict) -> dict:
        """
        Return a new dict with normalized keys but original values.
        Useful for label_to_id mappings in entity linking.
        """
        return {TextNormalizer.normalize_text(k): v for k, v in d.items()}
