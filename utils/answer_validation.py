import re
from typing import Optional, Tuple, List
from datetime import datetime
import unicodedata

try:
    from word2number import w2n
    HAS_WORD2NUMBER = True
except ImportError:
    HAS_WORD2NUMBER = False
    print("Warning: word2number not installed. Install with: pip install word2number")

try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    print("Warning: python-dateutil not installed. Install with: pip install python-dateutil")


class AnswerValidator:
    """
    Validates model answers against gold answers with proper normalization.
    Handles:
    - Numeric values (5 vs five, 1945 vs 1946)
    - Dates (Jan 1 vs January 1st)
    - Text normalization
    - Semantic similarity fallback
    """
    
    # Common date patterns
    DATE_PATTERNS = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY, DD-MM-YYYY
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY-MM-DD
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
    ]
    
    # Number word patterns (0-100)
    NUMBER_WORDS = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100
    }
    
    @staticmethod
    def normalize_text_basic(text: str) -> str:
        """Basic text normalization (lowercase, unicode, punctuation removal)."""
        if not isinstance(text, str):
            text = str(text)
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = text.lower()
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def extract_numbers(text: str) -> List[Optional[float]]:
        """
        Extract all numeric values from text.
        Returns list of numbers (as floats) found in the text.
        Handles: digits, word numbers (five -> 5), ordinals (first -> 1).
        """
        numbers = []
        
        # 1. Extract digit numbers (including decimals, negatives, with commas)
        digit_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
        for match in re.finditer(digit_pattern, text):
            num_str = match.group().replace(',', '')
            try:
                numbers.append(float(num_str))
            except ValueError:
                pass
        
        # 2. Extract word numbers (if word2number is available)
        if HAS_WORD2NUMBER:
            # Try to convert word numbers using word2number
            words = text.lower().split()
            i = 0
            while i < len(words):
                # Try to parse word sequences as numbers
                for j in range(min(5, len(words) - i), 0, -1):
                    phrase = ' '.join(words[i:i+j])
                    try:
                        num = w2n.word_to_num(phrase)
                        numbers.append(float(num))
                        i += j
                        break
                    except (ValueError, AttributeError):
                        pass
                else:
                    i += 1
        else:
            # Fallback: simple word-to-number for common words
            text_lower = text.lower()
            for word, value in AnswerValidator.NUMBER_WORDS.items():
                if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    numbers.append(float(value))
        
        return numbers
    
    @staticmethod
    def extract_dates(text: str) -> List[Optional[datetime]]:
        """
        Extract and normalize dates from text.
        Returns list of datetime objects (or None if parsing fails).
        """
        dates = []
        
        if not HAS_DATEUTIL:
            # Basic fallback: try to parse common patterns
            for pattern in AnswerValidator.DATE_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    date_str = match.group()
                    try:
                        # Try common formats
                        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y']:
                            try:
                                dt = datetime.strptime(date_str, fmt)
                                dates.append(dt)
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass
        else:
            # Use dateutil for robust parsing
            # Find potential date strings
            for pattern in AnswerValidator.DATE_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    date_str = match.group()
                    try:
                        dt = date_parser.parse(date_str, fuzzy=False)
                        dates.append(dt)
                    except (ValueError, TypeError):
                        pass
            
            # Also try fuzzy parsing on the whole text (for dates not matching patterns)
            try:
                dt = date_parser.parse(text, fuzzy=True, default=datetime(1900, 1, 1))
                # Only add if it's a reasonable date (not the default)
                if dt.year > 1900 and dt.year < 2100:
                    dates.append(dt)
            except (ValueError, TypeError):
                pass
        
        return dates
    
    @staticmethod
    def normalize_numeric_answer(text: str) -> Optional[float]:
        """
        Extract the primary numeric value from an answer.
        For answers that are primarily numeric (e.g., "1945", "five", "3.14").
        Returns None if no clear numeric value is found.
        """
        numbers = AnswerValidator.extract_numbers(text)
        if not numbers:
            return None
        
        # If multiple numbers, prefer the one that appears to be the main answer
        # (e.g., if text is mostly just a number, use that)
        text_norm = AnswerValidator.normalize_text_basic(text)
        
        # If text is mostly just a number, return the first/largest
        if len(text_norm.split()) <= 3:  # Short answer likely to be the number
            return numbers[0]
        
        # For longer text, prefer integers over decimals (for years, counts)
        integers = [n for n in numbers if n == int(n)]
        if integers:
            return integers[0]
        
        return numbers[0]
    
    @staticmethod
    def normalize_date_answer(text: str) -> Optional[datetime]:
        """
        Extract the primary date from an answer.
        Returns None if no clear date is found.
        """
        dates = AnswerValidator.extract_dates(text)
        if not dates:
            return None
        
        # Prefer dates with full year-month-day
        full_dates = [d for d in dates if d.year > 1900 and d.month and d.day]
        if full_dates:
            return full_dates[0]
        
        return dates[0]
    
    @staticmethod
    def is_numeric_match(pred: str, gold: str, tolerance: float = 0.0) -> bool:
        """
        Check if two answers match numerically.
        For numeric answers, requires exact match (tolerance=0) or within tolerance.
        """
        pred_num = AnswerValidator.normalize_numeric_answer(pred)
        gold_num = AnswerValidator.normalize_numeric_answer(gold)
        
        if pred_num is None or gold_num is None:
            return False
        
        # For exact numeric matching (e.g., years, counts), use strict equality
        # tolerance=0 means exact match only
        return abs(pred_num - gold_num) <= tolerance
    
    @staticmethod
    def is_date_match(pred: str, gold: str) -> bool:
        """
        Check if two answers match as dates.
        Normalizes different date formats to compare.
        """
        pred_date = AnswerValidator.normalize_date_answer(pred)
        gold_date = AnswerValidator.normalize_date_answer(gold)
        
        if pred_date is None or gold_date is None:
            return False
        
        # Compare dates (year, month, day must match)
        # datetime objects always have year, month, day
        return (pred_date.year == gold_date.year and
                pred_date.month == gold_date.month and
                pred_date.day == gold_date.day)
    
    @staticmethod
    def is_exact_match_after_normalization(pred: str, gold: str) -> bool:
        """Check exact match after basic text normalization."""
        pred_norm = AnswerValidator.normalize_text_basic(pred)
        gold_norm = AnswerValidator.normalize_text_basic(gold)
        return pred_norm == gold_norm
    
    @staticmethod
    def validate_answer(
        pred: str,
        gold: str,
        bertscore_threshold: float = 0.90,
        use_bertscore: bool = True,
        bertscore_func=None
    ) -> Tuple[bool, str]:
        """
        Main validation function that checks if prediction matches gold answer.
        
        Returns:
            (is_correct: bool, reason: str)
        
        Strategy:
        1. Check exact match after normalization
        2. Check numeric match (strict, no tolerance for years/counts)
        3. Check date match
        4. Fall back to BERTScore if available
        5. Otherwise, return False
        
        Args:
            pred: Predicted answer
            gold: Gold standard answer
            bertscore_threshold: BERTScore F1 threshold
            use_bertscore: Whether to use BERTScore as fallback
            bertscore_func: Function to compute BERTScore (pred, gold) -> float
        """
        if not pred or not gold:
            return False, "empty_answer"
        
        # 1. Exact match after normalization
        if AnswerValidator.is_exact_match_after_normalization(pred, gold):
            return True, "exact_match_normalized"
        
        # 2. Check if both are primarily date answers (BEFORE numeric check)
        # Dates often contain numbers, so we need to check dates first
        pred_date = AnswerValidator.normalize_date_answer(pred)
        gold_date = AnswerValidator.normalize_date_answer(gold)
        
        if pred_date is not None and gold_date is not None:
            if AnswerValidator.is_date_match(pred, gold):
                return True, "date_match"
            else:
                return False, f"date_mismatch"
        
        # 3. Check if both are primarily numeric answers (AFTER date check)
        pred_num = AnswerValidator.normalize_numeric_answer(pred)
        gold_num = AnswerValidator.normalize_numeric_answer(gold)
        
        if pred_num is not None and gold_num is not None:
            # STRICT numeric matching: no tolerance for years, counts, etc.
            if pred_num == gold_num:
                return True, "numeric_exact_match"
            else:
                # Different numbers = wrong answer (e.g., 1945 vs 1946)
                return False, f"numeric_mismatch_{pred_num}_vs_{gold_num}"
        
        # 4. Check if prediction is a subset/entity within gold answer
        # (e.g., "Ian Watkins" should match "Lostprophets disbanded...Watkins was charged...")
        pred_norm = AnswerValidator.normalize_text_basic(pred)
        gold_norm = AnswerValidator.normalize_text_basic(gold)
        
        pred_words = pred.split()
        
        # If prediction is short (<=3 words) and appears in gold answer, consider it correct
        if len(pred_words) <= 3 and pred_norm in gold_norm:
            return True, "subset_match_in_gold"
        
        # Also check if all significant words from prediction appear in gold
        # (handles cases like "Ian Watkins" in "Watkins was charged...")
        if len(pred_words) <= 5:
            # Extract significant words (not common stopwords, allow short names like "Ian")
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'was', 'is', 'are', 'were', 'been', 'be', 'after', 'before', 'during'}
            pred_significant = {w.lower() for w in pred_words if w.lower() not in stopwords}
            gold_words_lower = {w.lower() for w in gold.split()}
            
            if pred_significant and pred_significant.issubset(gold_words_lower):
                return True, "significant_words_subset_match"
            
            # Special case: Person names - if prediction is 2 words (first+last name)
            # and the last name appears in gold, consider it correct
            # (e.g., "Ian Watkins" where "Watkins" appears in gold)
            # not best practice, but it works for now
            if len(pred_words) == 2 and pred_words[0][0].isupper() and pred_words[1][0].isupper():
                last_name = pred_words[-1].lower()
                if last_name in gold_words_lower:
                    return True, "person_name_lastname_match"
        
        # Also check if key capitalized phrases (entities) from prediction appear in gold
        pred_caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', pred)
        if pred_caps and len(pred_words) <= 5:
            # Check if all capitalized phrases from prediction appear in gold
            gold_caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', gold)
            gold_caps_norm = {AnswerValidator.normalize_text_basic(cap) for cap in gold_caps}
            pred_caps_norm = {AnswerValidator.normalize_text_basic(cap) for cap in pred_caps}
            if pred_caps_norm and pred_caps_norm.issubset(gold_caps_norm):
                return True, "entity_subset_match"
        
        # 5. For short answers (<=5 words), require exact match
        gold_words = gold.split()
        if len(pred_words) <= 5 and len(gold_words) <= 5:
            # Already checked exact match above, so this is a mismatch
            return False, "short_answer_mismatch"
        
        # 6. Fall back to BERTScore for semantic similarity
        if use_bertscore and bertscore_func is not None:
            try:
                f1_score = bertscore_func(pred, gold)
                if f1_score >= bertscore_threshold:
                    return True, f"bertscore_match_{f1_score:.3f}"
                else:
                    return False, f"bertscore_below_threshold_{f1_score:.3f}"
            except Exception as e:
                return False, f"bertscore_error_{str(e)}"
        
        # Default: no match
        return False, "no_match"


def validate_answer_batch(
    preds: List[str],
    golds: List[str],
    bertscore_threshold: float = 0.90,
    use_bertscore: bool = True,
    bertscore_batch_func=None
) -> List[Tuple[bool, str]]:

    results = []
    
    # First pass: try exact/numeric/date matching
    for pred, gold in zip(preds, golds):
        is_correct, reason = AnswerValidator.validate_answer(
            pred, gold,
            bertscore_threshold=bertscore_threshold,
            use_bertscore=False,  # Skip BERTScore in first pass
            bertscore_func=None
        )
        results.append((is_correct, reason))
    
    # Second pass: BERTScore for remaining mismatches
    if use_bertscore and bertscore_batch_func is not None:
        # Collect indices that need BERTScore check
        needs_bertscore = []
        indices = []
        for i, (is_correct, reason) in enumerate(results):
            if not is_correct and "short_answer" not in reason and "numeric" not in reason and "date" not in reason:
                needs_bertscore.append((preds[i], golds[i]))
                indices.append(i)
        
        if needs_bertscore:
            try:
                batch_preds = [p for p, g in needs_bertscore]
                batch_golds = [g for p, g in needs_bertscore]
                f1_scores = bertscore_batch_func(batch_preds, batch_golds)
                
                # Update results
                for idx, f1_score in zip(indices, f1_scores):
                    if f1_score >= bertscore_threshold:
                        results[idx] = (True, f"bertscore_match_{f1_score:.3f}")
                    else:
                        _, old_reason = results[idx]
                        results[idx] = (False, f"{old_reason}_bertscore_{f1_score:.3f}")
            except Exception as e:
                # If BERTScore fails, keep original results
                pass
    
    return results

