"""Unit and regression tests for Day 4 NLP Corpus Ingestion & Autocomplete Engine Module.
"""

from collections import Counter
from pathlib import Path
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp_autocomplete_engine import (
    CorpusPreprocessor,
    NGramLanguageModel,
    AutocompleteEngine,
    AutocompleteEvaluator,
    DEFAULT_PROMPT_PREFIXES,
    generate_all_day4_reports
)


@pytest.fixture(scope="module")
def toy_sentences():
    """Simple deterministic tokenized sentence set for mathematical testing."""
    return [
        ["this", "app", "is", "great"],
        ["this", "app", "is", "useful"],
        ["this", "game", "is", "fun"],
        ["easy", "to", "use"],
        ["easy", "to", "learn"],
        ["easy", "to", "play"]
    ]


@pytest.fixture(scope="module")
def toy_model(toy_sentences):
    """Trained Trigram language model on toy sentences."""
    lm = NGramLanguageModel(n=3)
    lm.fit(toy_sentences)
    return lm


def test_corpus_preprocessor_tokenization():
    """Verify text normalization, contraction retention, and special character stripping."""
    sample_text = "I love this app! It's super fast, doesn't crash: https://example.com"
    tokens = CorpusPreprocessor.tokenize(sample_text)

    assert "i" in tokens
    assert "love" in tokens
    assert "it's" in tokens
    assert "doesn't" in tokens
    assert "crash" in tokens
    # Ensure URL is stripped
    assert not any("http" in t for t in tokens)
    # Ensure punctuation is stripped
    assert "!" not in tokens and ":" not in tokens


def test_corpus_preprocessor_sentence_splitting():
    """Verify multi-sentence review splitting across diverse delimiters."""
    multi_sent = "Great graphics. Easy to play! Why did it update? Because reasons."
    sents = CorpusPreprocessor.split_sentences(multi_sent)

    assert len(sents) == 4
    assert sents[0] == ["great", "graphics"]
    assert sents[1] == ["easy", "to", "play"]
    assert sents[2] == ["why", "did", "it", "update"]
    assert sents[3] == ["because", "reasons"]


def test_corpus_preprocessor_vocabulary_summary():
    """Verify vocabulary summary DataFrame construction, ranking, and cumulative math."""
    counter = Counter({"app": 100, "good": 50, "the": 30, "fast": 20})
    df = CorpusPreprocessor.build_vocabulary_summary(counter, top_n=4)

    assert len(df) == 4
    assert list(df["Token"]) == ["app", "good", "the", "fast"]
    assert list(df["Rank"]) == [1, 2, 3, 4]
    assert np.isclose(df["Cumulative_Pct"].iloc[-1], 100.0)
    assert df.loc[df["Token"] == "the", "Is_Stopword"].values[0] == True
    assert df.loc[df["Token"] == "app", "Is_Stopword"].values[0] == False


def test_ngram_model_fitting_and_mle(toy_model):
    """Test n-gram counts and Maximum Likelihood Estimation conditional probabilities."""
    assert toy_model.is_fitted
    assert len(toy_model.vocabulary) > 0

    # Trigram: ("this", "app") -> "is" (count = 2, total = 2) => P("is" | "this", "app") == 1.0
    p_is = toy_model.score_mle("is", ("this", "app"))
    assert np.isclose(p_is, 1.0)

    # Bigram: ("easy",) -> "to" (count = 3, total = 3) => P("to" | "easy") == 1.0
    p_to = toy_model.score_mle("to", ("easy",))
    assert np.isclose(p_to, 1.0)

    # Unseen transition should return 0.0 under MLE
    p_unseen = toy_model.score_mle("banana", ("easy", "to"))
    assert p_unseen == 0.0


def test_ngram_smoothing_and_backoff(toy_model):
    """Verify Laplace smoothing and Stupid Backoff handle unseen transitions without zero probability."""
    # Add-k Laplace smoothing
    p_laplace_unseen = toy_model.score_laplace("banana", ("easy", "to"), k=0.1)
    assert p_laplace_unseen > 0.0

    # Stupid Backoff
    score_backoff = toy_model.score_stupid_backoff("use", ("easy", "to"))
    assert score_backoff > 0.0

    # Backoff to bigram or unigram for unseen trigram context
    score_unseen_ctx = toy_model.score_stupid_backoff("use", ("completely", "unseen"))
    assert score_unseen_ctx > 0.0


def test_autocomplete_engine_predictions(toy_model):
    """Test next-word prediction engine generates sorted top-k suggestions."""
    engine = AutocompleteEngine(toy_model)

    preds = engine.predict_next_words("easy to", top_k=3)
    assert len(preds) == 3
    pred_words = [w for w, _ in preds]
    assert set(pred_words) == {"use", "learn", "play"}

    # Relative probabilities sum to ~1.0
    total_prob = sum(prob for _, prob in preds)
    assert np.isclose(total_prob, 1.0)


def test_autocomplete_predictions_across_prompt_prefixes():
    """Verify autocomplete engine executes across >= 10 prompt prefixes."""
    # Load actual cleaned dataset to test real prefixes
    data_path = Path(__file__).parent.parent / "Data" / "cleaned_user_reviews.csv"
    assert data_path.exists()

    df = pd.read_csv(data_path)
    sample_texts = df["Translated_Review"].dropna().tolist()[:2000]
    sentences, _ = CorpusPreprocessor.preprocess_corpus(sample_texts)

    model = NGramLanguageModel(n=3).fit(sentences)
    engine = AutocompleteEngine(model)

    assert len(DEFAULT_PROMPT_PREFIXES) >= 10
    preds_df = engine.predict_for_prefixes(DEFAULT_PROMPT_PREFIXES, top_k=3)

    assert len(preds_df) == len(DEFAULT_PROMPT_PREFIXES)
    assert "Top_1" in preds_df.columns
    assert "Top_2" in preds_df.columns
    assert "Top_3" in preds_df.columns

    for _, row in preds_df.iterrows():
        assert isinstance(row["Top_1"], str) and len(row["Top_1"]) > 0
        assert 0.0 <= row["Prob_1"] <= 1.0


def test_autocomplete_evaluator_metrics(toy_sentences, toy_model):
    """Verify precision, recall@3, and MRR metric computation properties."""
    engine = AutocompleteEngine(toy_model)
    metrics = AutocompleteEvaluator.evaluate_heldout_corpus(
        engine, toy_sentences, top_k=3, sample_size=50
    )

    assert 0.0 <= metrics["top_1_accuracy"] <= 100.0
    assert 0.0 <= metrics["top_k_recall"] <= 100.0
    assert metrics["top_1_accuracy"] <= metrics["top_k_recall"]
    assert 0.0 <= metrics["mean_reciprocal_rank"] <= 1.0
    assert np.isclose(metrics["precision_at_k"], metrics["top_k_recall"] / 3.0)


def test_day4_generated_artifacts_exist():
    """Verify all Day 4 reports CSV and figures PNG files exist and are non-empty."""
    reports_dir = Path(__file__).parent.parent / "reports"
    figures_dir = reports_dir / "figures"

    required_artifacts = [
        reports_dir / "vocabulary_frequency_summary.csv",
        reports_dir / "ngram_predictions_summary.csv",
        reports_dir / "autocomplete_metrics_summary.csv",
        figures_dir / "09_top20_vocabulary_frequency.png",
        figures_dir / "10_ngram_autocomplete_evaluation.png"
    ]

    for artifact in required_artifacts:
        assert artifact.exists(), f"Artifact {artifact.name} does not exist at {artifact}"
        assert artifact.stat().st_size > 0, f"Artifact {artifact.name} is empty (0 bytes)"
