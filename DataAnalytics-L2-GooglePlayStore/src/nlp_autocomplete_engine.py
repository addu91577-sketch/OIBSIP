"""Natural Language Processing Corpus Ingestion, N-Gram Modeling & Autocomplete Engine.

Day 4 Analytical Core for Google Play Store Analysis & NLP Analytics (Task 5).
Provides analytical computations, statistical language modeling, and visualization routines:
1. Mobile app reviews text corpus ingestion and preprocessing pipeline.
2. Vocabulary frequency distribution, lexical diversity (TTR), and Zipf's Law modeling.
3. Unigram, Bigram, and Trigram statistical language models with maximum likelihood estimation.
4. Add-k Laplace smoothing, Jelinek-Mercer linear interpolation, and Katz / Stupid Backoff.
5. Interactive Top-3 next-word prediction engine across >= 10 mobile prompt prefixes.
6. Empirical precision, recall (Hit Rate@3), Mean Reciprocal Rank (MRR), and perplexity evaluation.
7. Publication-quality vocabulary frequency and autocomplete performance figures.
"""

from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================================
# 0. Lexical Constants & Fallback Stopwords
# ============================================================================

STANDARD_ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Try loading NLTK stopwords if available, blending with standard list
try:
    import nltk
    try:
        from nltk.corpus import stopwords as _nltk_sw
        ENGLISH_STOPWORDS = set(_nltk_sw.words("english")).union(STANDARD_ENGLISH_STOPWORDS)
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as _nltk_sw
        ENGLISH_STOPWORDS = set(_nltk_sw.words("english")).union(STANDARD_ENGLISH_STOPWORDS)
except Exception:
    ENGLISH_STOPWORDS = STANDARD_ENGLISH_STOPWORDS


DEFAULT_PROMPT_PREFIXES: List[str] = [
    "this app",
    "easy to",
    "very good",
    "thank you",
    "i love",
    "please add",
    "customer service",
    "waste of",
    "great",
    "best app",
    "too many",
    "after the update"
]


# ============================================================================
# 1. Corpus Preprocessing & Vocabulary Extraction
# ============================================================================

class CorpusPreprocessor:
    """Preprocesses raw user review text into structured token sequences."""

    TOKEN_PATTERN = re.compile(r"\b[a-z]+(?:'[a-z]+)?\b")
    SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?\n\r;:]+")

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Normalize raw text characters, apostrophes, and whitespace."""
        if not isinstance(text, str):
            return ""
        # Standardize curly apostrophes and single quotes
        text = text.replace("’", "'").replace("`", "'").replace("‘", "'")
        # Strip URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        return text.lower()

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Extract lowercase alphabetic tokens preserving standard contractions."""
        cleaned = cls.clean_text(text)
        return cls.TOKEN_PATTERN.findall(cleaned)

    @classmethod
    def split_sentences(cls, text: str) -> List[List[str]]:
        """Segment a review into sentences and tokenize each individual sentence.

        Args:
            text: Raw input review string.

        Returns:
            List of token lists, one per valid sentence.
        """
        cleaned = cls.clean_text(text)
        raw_segments = cls.SENTENCE_SPLIT_PATTERN.split(cleaned)
        sentence_tokens = []
        for segment in raw_segments:
            tokens = cls.TOKEN_PATTERN.findall(segment)
            if tokens:
                sentence_tokens.append(tokens)
        return sentence_tokens

    @classmethod
    def preprocess_corpus(
        cls,
        texts: List[str],
        min_sentence_len: int = 1
    ) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Preprocess an entire corpus of text reviews.

        Args:
            texts: List of review strings.
            min_sentence_len: Minimum token length for a sentence to be retained.

        Returns:
            Tuple of:
                - List of tokenized sentences.
                - Dictionary of corpus statistics (token count, unique vocab, TTR, etc.).
        """
        sentences: List[List[str]] = []
        for text in texts:
            for s in cls.split_sentences(text):
                if len(s) >= min_sentence_len:
                    sentences.append(s)

        total_tokens = sum(len(s) for s in sentences)
        token_counter = Counter(token for s in sentences for token in s)
        vocab_size = len(token_counter)
        ttr = (vocab_size / total_tokens) if total_tokens > 0 else 0.0

        stats = {
            "total_documents": len(texts),
            "total_sentences": len(sentences),
            "total_tokens": total_tokens,
            "unique_vocabulary": vocab_size,
            "type_token_ratio": ttr,
            "token_counter": token_counter
        }

        return sentences, stats

    @classmethod
    def build_vocabulary_summary(
        cls,
        token_counter: Counter,
        top_n: int = 50,
        stopwords_set: Optional[Set[str]] = None
    ) -> pd.DataFrame:
        """Construct detailed frequency and ranking table for the vocabulary.

        Args:
            token_counter: Frequency counter of tokens.
            top_n: Number of top tokens to include.
            stopwords_set: Set of stopwords to flag. Defaults to ENGLISH_STOPWORDS.

        Returns:
            pd.DataFrame: Table with rank, token, count, frequency_pct, cumulative_pct, is_stopword.
        """
        if stopwords_set is None:
            stopwords_set = ENGLISH_STOPWORDS

        total_tokens = sum(token_counter.values())
        most_common = token_counter.most_common(top_n)

        records = []
        cumulative_count = 0
        for rank, (token, count) in enumerate(most_common, start=1):
            cumulative_count += count
            records.append({
                "Rank": rank,
                "Token": token,
                "Count": count,
                "Frequency_Pct": (count / total_tokens * 100.0) if total_tokens > 0 else 0.0,
                "Cumulative_Pct": (cumulative_count / total_tokens * 100.0) if total_tokens > 0 else 0.0,
                "Is_Stopword": token in stopwords_set
            })

        return pd.DataFrame(records)


# ============================================================================
# 2. N-Gram Frequency Language Model
# ============================================================================

class NGramLanguageModel:
    """N-Gram statistical language model supporting Unigram, Bigram, and Trigram orders."""

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    def __init__(self, n: int = 3):
        """Initialize language model up to order n (1, 2, or 3)."""
        if n not in (1, 2, 3):
            raise ValueError(f"Model order n must be 1, 2, or 3; got {n}")
        self.n = n
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()

        # Indexed successor mappings for O(1) autocomplete candidate lookups
        self.bigram_successors: Dict[str, Counter] = defaultdict(Counter)
        self.trigram_successors: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

        self.total_tokens: int = 0
        self.vocabulary: Set[str] = set()
        self.is_fitted: bool = False

    def fit(self, sentences: List[List[str]]) -> "NGramLanguageModel":
        """Train language model by counting n-grams across tokenized sentences.

        Args:
            sentences: List of tokenized sentences.

        Returns:
            Self (fitted model instance).
        """
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.bigram_successors.clear()
        self.trigram_successors.clear()
        self.total_tokens = 0
        self.vocabulary.clear()

        for s in sentences:
            # Pad sentence with (n - 1) START tokens and 1 END token
            padded = [self.START_TOKEN] * (self.n - 1) + s + [self.END_TOKEN]
            length = len(padded)

            for i in range(self.n - 1, length):
                word = padded[i]
                self.unigram_counts[word] += 1
                self.total_tokens += 1
                if word not in (self.START_TOKEN, self.END_TOKEN):
                    self.vocabulary.add(word)

                if self.n >= 2:
                    c1 = padded[i - 1]
                    self.bigram_counts[(c1, word)] += 1
                    self.bigram_successors[c1][word] += 1

                if self.n >= 3:
                    c2 = padded[i - 2]
                    c1 = padded[i - 1]
                    self.trigram_counts[(c2, c1, word)] += 1
                    self.trigram_successors[(c2, c1)][word] += 1

        self.is_fitted = True
        return self

    def score_mle(self, word: str, context: Tuple[str, ...]) -> float:
        """Calculate Maximum Likelihood Estimate (MLE) probability P(word | context)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        ctx_len = len(context)
        if ctx_len == 0 or self.n == 1:
            return self.unigram_counts[word] / self.total_tokens if self.total_tokens > 0 else 0.0

        if ctx_len == 1 or self.n == 2:
            c1 = context[-1]
            c1_count = sum(self.bigram_successors[c1].values())
            if c1_count == 0:
                return 0.0
            return self.bigram_successors[c1][word] / c1_count

        # Trigram MLE
        c2, c1 = context[-2], context[-1]
        trig_ctx_count = sum(self.trigram_successors[(c2, c1)].values())
        if trig_ctx_count == 0:
            return 0.0
        return self.trigram_successors[(c2, c1)][word] / trig_ctx_count

    def score_laplace(self, word: str, context: Tuple[str, ...], k: float = 0.01) -> float:
        """Calculate Add-k Laplace-smoothed probability P_Laplace(word | context)."""
        vocab_size = max(len(self.vocabulary), 1)
        ctx_len = len(context)

        if ctx_len == 0 or self.n == 1:
            word_count = self.unigram_counts[word]
            return (word_count + k) / (self.total_tokens + k * vocab_size)

        if ctx_len == 1 or self.n == 2:
            c1 = context[-1]
            pair_count = self.bigram_successors[c1][word]
            c1_count = sum(self.bigram_successors[c1].values())
            return (pair_count + k) / (c1_count + k * vocab_size)

        c2, c1 = context[-2], context[-1]
        tri_count = self.trigram_successors[(c2, c1)][word]
        tri_ctx_count = sum(self.trigram_successors[(c2, c1)].values())
        return (tri_count + k) / (tri_ctx_count + k * vocab_size)

    def score_interpolated(
        self,
        word: str,
        context: Tuple[str, ...],
        lambdas: Tuple[float, float, float] = (0.6, 0.3, 0.1)
    ) -> float:
        """Jelinek-Mercer linear interpolation across Trigram, Bigram, and Unigram MLEs.

        P_Interp(w | c2, c1) = λ3 * P(w|c2, c1) + λ2 * P(w|c1) + λ1 * P(w)
        """
        l3, l2, l1 = lambdas
        p_uni = self.unigram_counts[word] / self.total_tokens if self.total_tokens > 0 else 1e-9

        c1 = context[-1] if len(context) >= 1 else ""
        c1_count = sum(self.bigram_successors[c1].values()) if c1 else 0
        p_bi = (self.bigram_successors[c1][word] / c1_count) if c1_count > 0 else 0.0

        if len(context) >= 2:
            c2 = context[-2]
            tri_ctx_count = sum(self.trigram_successors[(c2, c1)].values())
            p_tri = (self.trigram_successors[(c2, c1)][word] / tri_ctx_count) if tri_ctx_count > 0 else 0.0
        else:
            p_tri = 0.0

        return l3 * p_tri + l2 * p_bi + l1 * p_uni

    def score_stupid_backoff(
        self,
        word: str,
        context: Tuple[str, ...],
        alpha: float = 0.4
    ) -> float:
        """Katz-inspired Stupid Backoff scoring (Brants et al., 2007)."""
        if len(context) >= 2:
            c2, c1 = context[-2], context[-1]
            tri_ctx_count = sum(self.trigram_successors[(c2, c1)].values())
            if tri_ctx_count > 0 and self.trigram_successors[(c2, c1)][word] > 0:
                return self.trigram_successors[(c2, c1)][word] / tri_ctx_count

        if len(context) >= 1:
            c1 = context[-1]
            c1_count = sum(self.bigram_successors[c1].values())
            if c1_count > 0 and self.bigram_successors[c1][word] > 0:
                return alpha * (self.bigram_successors[c1][word] / c1_count)

        uni_count = self.unigram_counts[word]
        if self.total_tokens > 0 and uni_count > 0:
            return (alpha ** 2) * (uni_count / self.total_tokens)

        return 1e-8

    def compute_perplexity(
        self,
        test_sentences: List[List[str]],
        strategy: str = "interpolated"
    ) -> float:
        """Compute corpus perplexity PP(W) on held-out test sentences.

        PP(W) = exp( - 1/M * sum(log P(w_i | context)) )
        """
        log_prob_sum = 0.0
        token_count = 0

        for s in test_sentences:
            padded = [self.START_TOKEN] * (self.n - 1) + s + [self.END_TOKEN]
            for i in range(self.n - 1, len(padded)):
                word = padded[i]
                if self.n == 1:
                    ctx: Tuple[str, ...] = ()
                elif self.n == 2:
                    ctx = (padded[i - 1],)
                else:
                    ctx = (padded[i - 2], padded[i - 1])

                if strategy == "laplace":
                    prob = self.score_laplace(word, ctx)
                elif strategy == "stupid_backoff":
                    prob = self.score_stupid_backoff(word, ctx)
                else:
                    prob = self.score_interpolated(word, ctx)

                prob = max(prob, 1e-12)
                log_prob_sum += math.log(prob)
                token_count += 1

        if token_count == 0:
            return float("inf")

        cross_entropy = - (log_prob_sum / token_count)
        return float(math.exp(cross_entropy))


# ============================================================================
# 3. Autocomplete Prediction Engine
# ============================================================================

class AutocompleteEngine:
    """Production Autocomplete Engine generating Top-k candidate word completions."""

    def __init__(self, model: NGramLanguageModel):
        """Initialize engine with a trained NGramLanguageModel."""
        self.model = model

    def predict_next_words(
        self,
        prefix: str,
        top_k: int = 3,
        strategy: str = "stupid_backoff",
        exclude_special_tokens: bool = True
    ) -> List[Tuple[str, float]]:
        """Generate top-k next word predictions for a given input prefix string.

        Args:
            prefix: User input prompt prefix (e.g., 'this app', 'easy to').
            top_k: Number of candidate completions to return (default: 3).
            strategy: Scoring strategy ('stupid_backoff', 'interpolated', 'mle', 'laplace').
            exclude_special_tokens: Exclude <s> and </s> tokens from suggestions.

        Returns:
            List of (candidate_word, conditional_probability) tuples sorted descending.
        """
        tokens = CorpusPreprocessor.tokenize(prefix)

        # Determine context based on token length
        if len(tokens) >= 2 and self.model.n >= 3:
            ctx = (tokens[-2], tokens[-1])
        elif len(tokens) >= 1 and self.model.n >= 2:
            ctx = (tokens[-1],)
        else:
            ctx = ()

        candidate_pool: Set[str] = set()

        # 1. Trigram successors
        if len(ctx) >= 2 and ctx in self.model.trigram_successors:
            candidate_pool.update(self.model.trigram_successors[ctx].keys())

        # 2. Bigram successors
        if len(ctx) >= 1:
            last_word = ctx[-1]
            if last_word in self.model.bigram_successors:
                candidate_pool.update(self.model.bigram_successors[last_word].keys())

        # 3. Fallback to top unigrams if candidate pool is too small
        if len(candidate_pool) < top_k:
            top_unis = [w for w, _ in self.model.unigram_counts.most_common(50)]
            candidate_pool.update(top_unis)

        if exclude_special_tokens:
            candidate_pool.discard(self.model.START_TOKEN)
            candidate_pool.discard(self.model.END_TOKEN)

        scored_candidates: List[Tuple[str, float]] = []
        for word in candidate_pool:
            if strategy == "stupid_backoff":
                score = self.model.score_stupid_backoff(word, ctx)
            elif strategy == "interpolated":
                score = self.model.score_interpolated(word, ctx)
            elif strategy == "laplace":
                score = self.model.score_laplace(word, ctx)
            else:
                score = self.model.score_mle(word, ctx)
            scored_candidates.append((word, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:top_k]

        # Normalize relative probabilities among returned top_k candidates for display
        total_score = sum(score for _, score in top_candidates)
        if total_score > 0:
            top_candidates = [(w, score / total_score) for w, score in top_candidates]

        return top_candidates

    def predict_for_prefixes(
        self,
        prefixes: Optional[List[str]] = None,
        top_k: int = 3,
        strategy: str = "stupid_backoff"
    ) -> pd.DataFrame:
        """Generate formatted top-k predictions for multiple prompt prefixes.

        Args:
            prefixes: List of input prompt strings. Defaults to DEFAULT_PROMPT_PREFIXES.
            top_k: Number of predictions per prefix (default 3).
            strategy: Scoring algorithm.

        Returns:
            pd.DataFrame: Table of predictions with columns:
                ['Prefix', 'Clean_Tokens', 'Top_1', 'Prob_1', 'Top_2', 'Prob_2', 'Top_3', 'Prob_3']
        """
        if prefixes is None:
            prefixes = DEFAULT_PROMPT_PREFIXES

        records = []
        for p in prefixes:
            preds = self.predict_next_words(p, top_k=top_k, strategy=strategy)
            row: Dict[str, Any] = {
                "Prefix": p,
                "Clean_Tokens": " ".join(CorpusPreprocessor.tokenize(p)),
                "Strategy": strategy
            }
            for i in range(top_k):
                if i < len(preds):
                    row[f"Top_{i+1}"] = preds[i][0]
                    row[f"Prob_{i+1}"] = round(float(preds[i][1]), 4)
                else:
                    row[f"Top_{i+1}"] = "N/A"
                    row[f"Prob_{i+1}"] = 0.0
            records.append(row)

        return pd.DataFrame(records)


# ============================================================================
# 4. Evaluation Suite (Precision, Recall@k, Hit Rate, MRR, Comparative Analysis)
# ============================================================================

class AutocompleteEvaluator:
    """Evaluates autocomplete accuracy, precision, recall, and reciprocal rank."""

    @classmethod
    def evaluate_heldout_corpus(
        cls,
        engine: AutocompleteEngine,
        test_sentences: List[List[str]],
        top_k: int = 3,
        sample_size: int = 3000,
        seed: int = 42,
        strategy: str = "stupid_backoff"
    ) -> Dict[str, Any]:
        """Compute evaluation metrics on sampled test context-continuation pairs.

        Metrics computed:
            - Top-1 Accuracy: Target matches rank 1 prediction.
            - Top-k Recall (Hit Rate@k): Target is in top-k suggestions.
            - Precision@k: Average (Hit / k) across predictions.
            - Mean Reciprocal Rank (MRR): Average (1 / rank) for hits, 0 otherwise.

        Args:
            engine: AutocompleteEngine instance.
            test_sentences: Tokenized test sentence sequences.
            top_k: Evaluated suggestion capacity (default 3).
            sample_size: Maximum test context instances to evaluate.
            seed: Random seed for deterministic reproducibility.
            strategy: Scoring strategy to test.

        Returns:
            Dict containing accuracy, recall, precision, MRR, and sample stats.
        """
        rng = random.Random(seed)

        # Extract context-target pairs from test sentences
        pairs: List[Tuple[List[str], str]] = []
        for s in test_sentences:
            for i in range(1, len(s)):
                context = s[:i]
                target = s[i]
                pairs.append((context, target))

        if len(pairs) > sample_size:
            eval_pairs = rng.sample(pairs, sample_size)
        else:
            eval_pairs = pairs

        top1_hits = 0
        top_k_hits = 0
        reciprocal_rank_sum = 0.0
        oov_count = 0

        for context_tokens, target_word in eval_pairs:
            prefix_str = " ".join(context_tokens)
            preds = engine.predict_next_words(
                prefix_str, top_k=top_k, strategy=strategy, exclude_special_tokens=True
            )
            candidate_words = [w for w, _ in preds]

            if target_word not in engine.model.vocabulary:
                oov_count += 1

            # Top-1 Accuracy
            if candidate_words and candidate_words[0] == target_word:
                top1_hits += 1

            # Top-k Recall / Hit Rate
            if target_word in candidate_words:
                top_k_hits += 1
                rank = candidate_words.index(target_word) + 1
                reciprocal_rank_sum += 1.0 / rank

        total = len(eval_pairs) if len(eval_pairs) > 0 else 1
        top1_acc = (top1_hits / total) * 100.0
        recall_at_k = (top_k_hits / total) * 100.0
        precision_at_k = (recall_at_k / top_k)
        mrr = reciprocal_rank_sum / total
        oov_rate = (oov_count / total) * 100.0

        return {
            "sample_size": total,
            "top_k": top_k,
            "top_1_hits": top1_hits,
            "top_1_accuracy": top1_acc,
            "top_k_hits": top_k_hits,
            "top_k_recall": recall_at_k,
            "precision_at_k": precision_at_k,
            "mean_reciprocal_rank": mrr,
            "oov_rate": oov_rate
        }

    @classmethod
    def evaluate_prefix_benchmarks(
        cls,
        engine: AutocompleteEngine,
        test_sentences: List[List[str]],
        prefixes: Optional[List[str]] = None,
        top_k: int = 3
    ) -> pd.DataFrame:
        """Measure prefix-specific hit rate across real test set sentence occurrences."""
        if prefixes is None:
            prefixes = DEFAULT_PROMPT_PREFIXES

        records = []
        for prefix in prefixes:
            p_tokens = CorpusPreprocessor.tokenize(prefix)
            p_len = len(p_tokens)
            preds = engine.predict_next_words(prefix, top_k=top_k)
            pred_words = [w for w, _ in preds]

            # Find actual occurrences of prefix in test sentences
            actual_continuations = []
            for s in test_sentences:
                for idx in range(len(s) - p_len):
                    if s[idx:idx + p_len] == p_tokens:
                        actual_continuations.append(s[idx + p_len])

            occ_count = len(actual_continuations)
            if occ_count > 0:
                hits = sum(1 for target in actual_continuations if target in pred_words)
                hit_rate = (hits / occ_count) * 100.0
                top1_hits = sum(1 for target in actual_continuations if pred_words and target == pred_words[0])
                top1_acc = (top1_hits / occ_count) * 100.0
            else:
                hits = 0
                hit_rate = 0.0
                top1_acc = 0.0

            records.append({
                "Prefix": prefix,
                "Test_Occurrences": occ_count,
                "Top_1_Prediction": pred_words[0] if pred_words else "N/A",
                "Top_3_Suggestions": ", ".join(pred_words),
                "Empirical_Top1_Accuracy": round(top1_acc, 2),
                "Empirical_Recall@3": round(hit_rate, 2),
                "Empirical_Precision@3": round(hit_rate / top_k, 2)
            })

        return pd.DataFrame(records)

    @classmethod
    def compare_model_orders(
        cls,
        train_sentences: List[List[str]],
        test_sentences: List[List[str]],
        sample_size: int = 2500,
        seed: int = 42
    ) -> pd.DataFrame:
        """Train and evaluate Unigram, Bigram, Trigram, Laplace, and Backoff models."""
        configs = [
            ("Unigram (Order 1)", 1, "mle"),
            ("Bigram (Order 2, MLE)", 2, "mle"),
            ("Trigram (Order 3, MLE)", 3, "mle"),
            ("Trigram (Add-k Laplace)", 3, "laplace"),
            ("Trigram (Jelinek-Mercer Interp)", 3, "interpolated"),
            ("Trigram (Stupid Backoff)", 3, "stupid_backoff")
        ]

        results = []
        for name, n_order, strat in configs:
            lm = NGramLanguageModel(n=n_order).fit(train_sentences)
            eng = AutocompleteEngine(lm)
            metrics = cls.evaluate_heldout_corpus(
                eng, test_sentences, top_k=3, sample_size=sample_size, seed=seed, strategy=strat
            )
            # Evaluate test perplexity on a sample of test sentences
            pp = lm.compute_perplexity(test_sentences[:200], strategy=strat if strat != "mle" else "laplace")

            results.append({
                "Model_Configuration": name,
                "Order_n": n_order,
                "Strategy": strat,
                "Top_1_Accuracy_Pct": round(metrics["top_1_accuracy"], 2),
                "Top_3_Recall_Pct": round(metrics["top_k_recall"], 2),
                "Precision@3_Pct": round(metrics["precision_at_k"], 2),
                "MRR": round(metrics["mean_reciprocal_rank"], 4),
                "Test_Perplexity": round(pp, 2)
            })

        return pd.DataFrame(results)


# ============================================================================
# 5. Visualizations
# ============================================================================

def visualize_top_vocabulary(
    token_counter: Counter,
    output_path: Path,
    top_n: int = 20
) -> None:
    """Generate publication-quality Top 20 vocabulary frequency figure (Figure 09).

    Dual-panel layout:
    - Panel A: Top 20 overall corpus tokens (including function/stopwords).
    - Panel B: Top 20 content words (filtering stopwords to isolate domain semantics).
    - Inset / Subplot: Zipf's Law rank-frequency log-log plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Overall top 20
    overall_top = token_counter.most_common(top_n)
    words_overall = [w for w, _ in overall_top]
    counts_overall = [c for _, c in overall_top]

    # 2. Content top 20 (excluding stopwords)
    content_top = [
        (w, c) for w, c in token_counter.most_common(1000)
        if w not in ENGLISH_STOPWORDS and w not in ("<s>", "</s>") and len(w) > 1
    ][:top_n]
    words_content = [w for w, _ in content_top]
    counts_content = [c for _, c in content_top]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 0.7], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # Subplot 1: Overall Vocabulary
    y_pos1 = np.arange(len(words_overall))
    colors1 = sns.color_palette("mako", len(words_overall))[::-1]
    bars1 = ax1.barh(y_pos1, counts_overall, color=colors1, edgecolor="black", alpha=0.85)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels([f"'{w}'" for w in words_overall], fontsize=10, fontweight="medium")
    ax1.invert_yaxis()
    ax1.set_xlabel("Token Frequency (Occurrences)", fontsize=11, fontweight="bold")
    ax1.set_title("Top 20 Overall Corpus Tokens (Raw Vocabulary)", fontsize=13, fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    # Annotate bar counts
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + max(counts_overall) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{int(w):,}", va="center", ha="left", fontsize=9, color="#222222")

    # Subplot 2: Content Vocabulary
    y_pos2 = np.arange(len(words_content))
    colors2 = sns.color_palette("viridis", len(words_content))[::-1]
    bars2 = ax2.barh(y_pos2, counts_content, color=colors2, edgecolor="black", alpha=0.85)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([f"'{w}'" for w in words_content], fontsize=10, fontweight="medium")
    ax2.invert_yaxis()
    ax2.set_xlabel("Token Frequency (Occurrences)", fontsize=11, fontweight="bold")
    ax2.set_title("Top 20 Domain Content Words (Stopwords Filtered)", fontsize=13, fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + max(counts_content) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{int(w):,}", va="center", ha="left", fontsize=9, color="#222222")

    # Subplot 3: Zipf's Law Log-Log Distribution
    top_500 = token_counter.most_common(500)
    ranks = np.arange(1, len(top_500) + 1)
    freqs = np.array([c for _, c in top_500])

    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    fit_line = 10 ** (slope * log_ranks + intercept)

    ax3.loglog(ranks, freqs, marker="o", markersize=3, color="#2b5c8f", alpha=0.7, label="Observed Corpus Frequencies")
    ax3.loglog(ranks, fit_line, color="#c0392b", linestyle="--", linewidth=2,
               label=f"Zipf's Law Theoretical Fit (Slope s = {abs(slope):.2f})")
    ax3.set_xlabel("Log Rank (Frequency Rank r)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Log Frequency (f)", fontsize=11, fontweight="bold")
    ax3.set_title("Zipf's Law Power-Law Scaling Across Top 500 Vocabulary Tokens", fontsize=12, fontweight="bold")
    ax3.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax3.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.suptitle("Google Play Store Reviews Corpus: Vocabulary Frequency & Linguistic Scaling",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()


def visualize_autocomplete_performance(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generate publication-quality Autocomplete Performance figure (Figure 10).

    Multi-panel layout:
    - Panel A: Top-3 prediction probability distribution across representative Play Store prefixes.
    - Panel B: Comparative Top-1 Accuracy, Recall@3, and MRR across language model orders.
    - Panel C: Test Perplexity progression across models.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Panel A: Top-3 Candidates Probability Stack across 6 iconic prefixes
    sample_prefixes = [
        "this app", "easy to", "waste of", "please add", "customer service", "very good"
    ]
    sub_pred = predictions_df[predictions_df["Prefix"].isin(sample_prefixes)].copy()
    if len(sub_pred) > 0:
        x_indices = np.arange(len(sub_pred))
        p1 = sub_pred["Prob_1"].values
        p2 = sub_pred["Prob_2"].values
        p3 = sub_pred["Prob_3"].values
        bar_width = 0.55

        ax1.bar(x_indices, p1, width=bar_width, label="Top 1 Candidate", color="#1f77b4", edgecolor="black")
        ax1.bar(x_indices, p2, width=bar_width, bottom=p1, label="Top 2 Candidate", color="#ff7f0e", edgecolor="black")
        ax1.bar(x_indices, p3, width=bar_width, bottom=p1 + p2, label="Top 3 Candidate", color="#2ca02c", edgecolor="black")

        # Label each bar with predicted words
        for idx, row in enumerate(sub_pred.itertuples()):
            text_str = f"1: {row.Top_1}\n2: {row.Top_2}\n3: {row.Top_3}"
            ax1.text(idx, (row.Prob_1 + row.Prob_2 + row.Prob_3) + 0.03, text_str,
                     ha="center", va="bottom", fontsize=8.5, fontweight="medium",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8f9fa", edgecolor="#dddddd"))

        ax1.set_xticks(x_indices)
        ax1.set_xticklabels([f"\"{p}\"" for p in sub_pred["Prefix"]], fontsize=11, fontweight="bold")
        ax1.set_ylabel("Normalized Probability Share", fontsize=11, fontweight="bold")
        ax1.set_ylim(0, 1.35)
        ax1.set_title("Top-3 Autocomplete Word Predictions and Relative Likelihoods Across Play Store Prefixes",
                      fontsize=13, fontweight="bold", pad=12)
        ax1.legend(loc="upper right", frameon=True, facecolor="white")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel B: Comparative Top-1 Acc vs Recall@3 vs MRR
    models = metrics_df["Model_Configuration"].tolist()
    short_names = [m.replace("Trigram (", "").replace("Bigram (", "").replace("Unigram (", "").replace(")", "") for m in models]
    x = np.arange(len(short_names))
    width = 0.28

    acc = metrics_df["Top_1_Accuracy_Pct"].values
    rec = metrics_df["Top_3_Recall_Pct"].values
    mrr_pct = (metrics_df["MRR"].values * 100.0)

    b1 = ax2.bar(x - width, acc, width, label="Top-1 Accuracy (%)", color="#2980b9", edgecolor="black")
    b2 = ax2.bar(x, rec, width, label="Top-3 Recall@3 (%)", color="#27ae60", edgecolor="black")
    b3 = ax2.bar(x + width, mrr_pct, width, label="MRR (*100)", color="#8e44ad", edgecolor="black")

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9, fontweight="medium")
    ax2.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Autocomplete Accuracy, Recall & MRR Benchmarks", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(frameon=True, facecolor="white", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel C: Perplexity Progression
    pp_values = metrics_df["Test_Perplexity"].values
    # Filter infs if any
    clean_pp = [min(p, 5000.0) for p in pp_values]
    bars_pp = ax3.bar(x, clean_pp, width=0.5, color="#d35400", edgecolor="black", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9, fontweight="medium")
    ax3.set_ylabel("Perplexity (Lower is Better)", fontsize=11, fontweight="bold")
    ax3.set_yscale("log")
    ax3.set_title("Language Model Test Perplexity (Log Scale)", fontsize=12, fontweight="bold", pad=10)
    ax3.grid(axis="y", which="both", linestyle="--", alpha=0.5)

    for bar, val in zip(bars_pp, clean_pp):
        ax3.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                 f"{int(val):,}" if val < 10000 else f"{val:.1e}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle("N-Gram Statistical Language Model & Autocomplete Engine Performance Evaluation",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================================
# 6. Pipeline Orchestrator & CLI Runner
# ============================================================================

def generate_all_day4_reports(
    reviews_csv_path: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    train_split_ratio: float = 0.8,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Execute complete Day 4 NLP Corpus & Autocomplete Engine pipeline.

    Steps:
        1. Ingest cleaned user reviews dataset.
        2. Clean, segment, and tokenize corpus into sentences.
        3. Build vocabulary distribution and export top 50 summary CSV.
        4. Split sentences deterministically into Train (80%) and Test (20%) sets.
        5. Train Unigram, Bigram, Trigram, Laplace, Interpolated, and Stupid Backoff models.
        6. Compute Top-3 next-word predictions for >= 10 mobile prompt prefixes.
        7. Compute empirical evaluation metrics (Top-1 Accuracy, Recall@3, Precision@3, MRR, Perplexity).
        8. Export Figure 09 (Top 20 Vocabulary Frequency) and Figure 10 (Autocomplete Performance).
        9. Save all analytical CSV summaries.

    Returns:
        Dict[str, Any]: Dictionary containing generated artifacts and core metrics.
    """
    root_dir = Path(__file__).resolve().parent.parent

    if reviews_csv_path is None:
        reviews_csv_path = root_dir / "Data" / "cleaned_user_reviews.csv"
    if reports_dir is None:
        reports_dir = root_dir / "reports"
    if figures_dir is None:
        figures_dir = reports_dir / "figures"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("DAY 4: NLP CORPUS INGESTION & AUTOCOMPLETE ENGINE PIPELINE")
    print("=" * 76)

    # 1. Corpus Ingestion
    print(f"\n[Step 1/6] Ingesting reviews corpus from: {reviews_csv_path}")
    if not reviews_csv_path.exists():
        raise FileNotFoundError(f"Cleaned reviews dataset missing at {reviews_csv_path}")

    reviews_df = pd.read_csv(reviews_csv_path)
    raw_texts = reviews_df["Translated_Review"].dropna().tolist()
    print(f"  -> Ingested {len(raw_texts):,} valid review documents.")

    # 2. Preprocessing & Vocabulary Analysis
    print("\n[Step 2/6] Segmenting sentences and tokenizing lexical items...")
    sentences, corpus_stats = CorpusPreprocessor.preprocess_corpus(raw_texts, min_sentence_len=1)
    token_counter = corpus_stats["token_counter"]

    print(f"  -> Total Extracted Sentences: {corpus_stats['total_sentences']:,}")
    print(f"  -> Total Lexical Tokens:      {corpus_stats['total_tokens']:,}")
    print(f"  -> Unique Vocabulary Size:    {corpus_stats['unique_vocabulary']:,}")
    print(f"  -> Type-Token Ratio (TTR):    {corpus_stats['type_token_ratio']:.4f}")

    vocab_summary_df = CorpusPreprocessor.build_vocabulary_summary(token_counter, top_n=50)
    vocab_csv_path = reports_dir / "vocabulary_frequency_summary.csv"
    vocab_summary_df.to_csv(vocab_csv_path, index=False)
    print(f"  -> Exported vocabulary frequency table to: {vocab_csv_path}")

    # 3. Train/Test Split
    print(f"\n[Step 3/6] Partitioning corpus into Train ({train_split_ratio*100:.0f}%) / Test ({(1-train_split_ratio)*100:.0f}%)...")
    rng = random.Random(random_seed)
    shuffled_sentences = list(sentences)
    rng.shuffle(shuffled_sentences)

    split_idx = int(len(shuffled_sentences) * train_split_ratio)
    train_sentences = shuffled_sentences[:split_idx]
    test_sentences = shuffled_sentences[split_idx:]
    print(f"  -> Training Sentences: {len(train_sentences):,} | Test Sentences: {len(test_sentences):,}")

    # 4. Fit Language Models & Autocomplete Engine
    print("\n[Step 4/6] Fitting N-Gram Language Models (Orders 1, 2, 3)...")
    trigram_model = NGramLanguageModel(n=3).fit(train_sentences)
    print(f"  -> Trigram Model fitted. Bigrams: {len(trigram_model.bigram_counts):,}, Trigrams: {len(trigram_model.trigram_counts):,}")

    engine = AutocompleteEngine(trigram_model)

    # 5. Top-3 Predictions across Prompt Prefixes
    print(f"\n[Step 5/6] Generating Top-3 Predictions across {len(DEFAULT_PROMPT_PREFIXES)} prompt prefixes...")
    predictions_df = engine.predict_for_prefixes(DEFAULT_PROMPT_PREFIXES, top_k=3, strategy="stupid_backoff")
    preds_csv_path = reports_dir / "ngram_predictions_summary.csv"
    predictions_df.to_csv(preds_csv_path, index=False)
    print(f"  -> Exported predictions table to: {preds_csv_path}")

    for _, row in predictions_df.iterrows():
        p = row["Prefix"]
        print(f"     \"{p:18}\" -> [1] {row['Top_1']} ({row['Prob_1']*100:.1f}%), [2] {row['Top_2']} ({row['Prob_2']*100:.1f}%), [3] {row['Top_3']} ({row['Prob_3']*100:.1f}%)")

    # 6. Evaluation & Benchmarking
    print("\n[Step 6/6] Computing comparative model metrics & generating figures...")
    comparison_metrics_df = AutocompleteEvaluator.compare_model_orders(
        train_sentences, test_sentences, sample_size=3000, seed=random_seed
    )
    metrics_csv_path = reports_dir / "autocomplete_metrics_summary.csv"
    comparison_metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  -> Exported comparative metrics table to: {metrics_csv_path}")

    # Figure 09: Top 20 Vocabulary
    fig09_path = figures_dir / "09_top20_vocabulary_frequency.png"
    visualize_top_vocabulary(token_counter, fig09_path, top_n=20)
    print(f"  -> Saved Figure 09 to: {fig09_path}")

    # Figure 10: Autocomplete Performance
    fig10_path = figures_dir / "10_ngram_autocomplete_evaluation.png"
    visualize_autocomplete_performance(comparison_metrics_df, predictions_df, fig10_path)
    print(f"  -> Saved Figure 10 to: {fig10_path}")

    print("\n" + "=" * 76)
    print("DAY 4 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 76)

    return {
        "corpus_stats": corpus_stats,
        "vocabulary_summary_df": vocab_summary_df,
        "predictions_df": predictions_df,
        "metrics_df": comparison_metrics_df,
        "artifacts": {
            "vocab_csv": vocab_csv_path,
            "predictions_csv": preds_csv_path,
            "metrics_csv": metrics_csv_path,
            "fig09": fig09_path,
            "fig10": fig10_path
        }
    }


if __name__ == "__main__":
    generate_all_day4_reports()
