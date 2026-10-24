"""Unit and regression tests for Day 5 NLP Autocorrect & Benchmarking Module.
"""

from collections import Counter
from pathlib import Path
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp_autocorrect_engine import (
    DistanceMetrics,
    NorvigCandidateGenerator,
    LevenshteinAutocorrectEngine,
    DamerauLevenshteinAutocorrectEngine,
    BayesianFrequencyAutocorrectEngine,
    GboardReferenceEngine,
    AutocorrectEvaluator,
    DEFAULT_MISSPELLED_TEST_CORPUS,
    generate_all_day5_reports,
    PY_SPELLCHECKER_AVAILABLE
)
if PY_SPELLCHECKER_AVAILABLE:
    from src.nlp_autocorrect_engine import PySpellCheckerEngine


@pytest.fixture(scope="module")
def toy_vocab():
    """Toy vocabulary with frequency counts."""
    return {
        "the": 10000,
        "tea": 50,
        "ten": 30,
        "to": 5000,
        "update": 1200,
        "updated": 800,
        "game": 3000,
        "amazing": 1500,
        "graphics": 600,
        "download": 700,
        "review": 400,
        "screen": 350,
        "free": 900,
        "mobile": 450
    }


def test_levenshtein_distance_properties():
    """Verify mathematical properties of classic Levenshtein distance."""
    # Identity
    assert DistanceMetrics.levenshtein_distance("amazing", "amazing") == 0
    assert DistanceMetrics.levenshtein_distance("", "") == 0

    # Empty string edge cases
    assert DistanceMetrics.levenshtein_distance("", "test") == 4
    assert DistanceMetrics.levenshtein_distance("test", "") == 4

    # Single canonical operations
    assert DistanceMetrics.levenshtein_distance("cat", "cats") == 1  # Insertion
    assert DistanceMetrics.levenshtein_distance("cats", "cat") == 1  # Deletion
    assert DistanceMetrics.levenshtein_distance("cat", "bat") == 1   # Substitution

    # Symmetry
    assert DistanceMetrics.levenshtein_distance("kitten", "sitting") == DistanceMetrics.levenshtein_distance("sitting", "kitten")
    assert DistanceMetrics.levenshtein_distance("kitten", "sitting") == 3


def test_damerau_levenshtein_transposition():
    """Verify that adjacent transposition is cost 1 in Damerau-Levenshtein but cost 2 in Levenshtein."""
    # Transposition 'teh' -> 'the'
    assert DistanceMetrics.damerau_levenshtein_distance("teh", "the") == 1
    assert DistanceMetrics.levenshtein_distance("teh", "the") == 2

    # Transposition 'udpate' -> 'update'
    assert DistanceMetrics.damerau_levenshtein_distance("udpate", "update") == 1
    assert DistanceMetrics.levenshtein_distance("udpate", "update") == 2

    # Non-adjacent swap should NOT be distance 1
    assert DistanceMetrics.damerau_levenshtein_distance("abcde", "ebcda") > 1


def test_qwerty_spatial_cost():
    """Verify physical key adjacency scoring."""
    # Identical character
    assert DistanceMetrics.qwerty_spatial_cost("a", "a") == 0.0

    # Direct physical neighbors on QWERTY keyboard
    assert DistanceMetrics.qwerty_spatial_cost("a", "s") == 0.5
    assert DistanceMetrics.qwerty_spatial_cost("u", "y") == 0.5
    assert DistanceMetrics.qwerty_spatial_cost("k", "l") == 0.5

    # Distant keys
    assert DistanceMetrics.qwerty_spatial_cost("a", "p") == 1.0
    assert DistanceMetrics.qwerty_spatial_cost("z", "p") == 1.0


def test_norvig_candidate_generator():
    """Verify combinatorial edit generation and vocabulary filtering."""
    edits = NorvigCandidateGenerator.edits1("cat", include_transpositions=True)

    # Must contain deletes, transpositions, substitutions, insertions
    assert "at" in edits      # Deletion
    assert "act" in edits     # Transposition
    assert "bat" in edits     # Substitution
    assert "cats" in edits    # Insertion

    # Filtering against known vocabulary
    vocab = {"bat", "cats", "dog", "random"}
    known_cands = NorvigCandidateGenerator.known(edits, vocab)
    assert "bat" in known_cands
    assert "cats" in known_cands
    assert "dog" not in known_cands


def test_engines_on_toy_vocab(toy_vocab):
    """Verify correction accuracy on toy vocabulary."""
    dam_engine = DamerauLevenshteinAutocorrectEngine(toy_vocab)
    bayes_engine = BayesianFrequencyAutocorrectEngine(toy_vocab)

    # Both should correct 'teh' -> 'the' (due to transposition + high frequency prior)
    assert dam_engine.correct("teh") == "the"
    assert bayes_engine.correct("teh") == "the"

    # Both should correct 'amzing' -> 'amazing'
    assert dam_engine.correct("amzing") == "amazing"
    assert bayes_engine.correct("amzing") == "amazing"

    # Clean control word should not be modified (0 False Positives)
    assert dam_engine.correct("game") == "game"
    assert bayes_engine.correct("game") == "game"


def test_default_test_corpus_structure():
    """Verify test corpus integrity, size (>= 20 words), and error category coverage."""
    corpus = DEFAULT_MISSPELLED_TEST_CORPUS
    assert len(corpus) >= 20

    error_types = {item["Error_Type"] for item in corpus}
    assert "Insertion" in error_types
    assert "Deletion" in error_types
    assert "Substitution" in error_types
    assert "Transposition" in error_types
    assert "Multi-Edit" in error_types
    assert "Control (Clean)" in error_types

    for item in corpus:
        assert "Misspelled" in item
        assert "Target" in item
        assert "Error_Type" in item
        assert "Distance" in item
        assert isinstance(item["Distance"], int)


def test_autocorrect_evaluator_metrics_computation(toy_vocab):
    """Verify information retrieval formulas for accuracy, precision, recall, F1, and MRR."""
    engine = DamerauLevenshteinAutocorrectEngine(toy_vocab)
    toy_test_suite = [
        {"Misspelled": "amzing", "Target": "amazing", "Error_Type": "Deletion", "Distance": 1},
        {"Misspelled": "teh", "Target": "the", "Error_Type": "Transposition", "Distance": 1},
        {"Misspelled": "game", "Target": "game", "Error_Type": "Control (Clean)", "Distance": 0},
    ]

    res = AutocorrectEvaluator.evaluate_engine(engine, toy_test_suite)

    assert res["Total_Words"] == 3
    assert res["Total_Errors"] == 2
    assert res["Total_Controls"] == 1

    # Both typos should be corrected and control left intact -> 100% across the board
    assert np.isclose(res["Top1_Accuracy"], 100.0)
    assert np.isclose(res["Precision"], 100.0)
    assert np.isclose(res["Recall"], 100.0)
    assert np.isclose(res["F1_Score"], 100.0)
    assert np.isclose(res["MRR"], 1.0)
    assert res["Mean_Latency_ms"] >= 0.0


@pytest.mark.skipif(not PY_SPELLCHECKER_AVAILABLE, reason="pyspellchecker not installed")
def test_pyspellchecker_integration(toy_vocab):
    """Verify pyspellchecker baseline integration."""
    spell_engine = PySpellCheckerEngine(additional_vocab=set(toy_vocab.keys()))
    corr = spell_engine.correct("teh")
    assert corr == "the"
    cands = spell_engine.predict_candidates("amzing", top_k=3)
    assert len(cands) > 0


def test_all_day5_artifacts_exist():
    """Verify all Day 5 CSV reports and figures exist and are non-empty."""
    root_dir = Path(__file__).resolve().parent.parent
    reports_dir = root_dir / "reports"
    figures_dir = reports_dir / "figures"

    # CSV reports
    assert (reports_dir / "autocorrect_test_corpus.csv").exists()
    assert (reports_dir / "autocorrect_benchmark_summary.csv").exists()
    assert (reports_dir / "autocorrect_confusion_matrix.csv").exists()
    assert (reports_dir / "autocorrect_word_predictions.csv").exists()

    # Figures
    assert (figures_dir / "11_autocorrect_performance_metrics.png").exists()
    assert (figures_dir / "12_autocorrect_confusion_matrices.png").exists()

    # Verify non-empty files
    assert (reports_dir / "autocorrect_benchmark_summary.csv").stat().st_size > 100
    assert (figures_dir / "11_autocorrect_performance_metrics.png").stat().st_size > 1000
