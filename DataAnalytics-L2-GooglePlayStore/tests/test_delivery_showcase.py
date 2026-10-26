"""Unit tests for Day 8 Project Delivery Showcase & Presentation Deck.

Validates the integrity of the interactive HTML5 presentation deck, video narration guide,
Figure 17 lifecycle graphic, and the master project delivery scorecard.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.delivery_showcase import ProjectDeliveryShowcase


@pytest.fixture
def showcase():
    """Fixture providing initialized ProjectDeliveryShowcase instance."""
    return ProjectDeliveryShowcase()


def test_presentation_html_structure(showcase):
    """Verify that presentation/index.html is valid, complete, and contains all 14 slides."""
    html_file = showcase.presentation_dir / "index.html"
    assert html_file.exists(), f"Presentation deck not found at {html_file}"
    content = html_file.read_text(encoding="utf-8")

    # Document integrity
    assert "<!DOCTYPE html>" in content
    assert "<title>" in content
    assert "deck-container" in content
    assert "progressBar" in content
    assert "notesDrawer" in content

    # All 14 slides present
    for i in range(1, 15):
        assert f'data-slide="{i}"' in content, f"Slide {i} missing from presentation/index.html"

    # Interactive components and speaker notes
    assert "speakerNotes" in content
    assert "simulateTyping" in content
    assert "toggleNotes" in content
    assert "toggleFullscreen" in content


def test_presentation_guide_completeness(showcase):
    """Verify that PRESENTATION_GUIDE.md contains scripts for all slides and AI prompts."""
    guide_file = showcase.base_dir / "PRESENTATION_GUIDE.md"
    assert guide_file.exists(), f"Presentation guide not found at {guide_file}"
    content = guide_file.read_text(encoding="utf-8")

    # Structure checks
    assert "# 🎬 Video Presentation Guide & Narration Script" in content
    assert "3 Ways to Record Your Video" in content
    assert "Gamma.app" in content
    assert "ElevenLabs" in content
    assert "HeyGen" in content

    # Check that all 14 slides are covered in the storyboard
    for i in range(1, 15):
        assert f"Slide {i}:" in content, f"Slide {i} missing from PRESENTATION_GUIDE.md storyboard"


def test_project_delivery_scorecard_csv(showcase):
    """Verify the structure and contents of reports/project_delivery_scorecard.csv."""
    csv_file = showcase.reports_dir / "project_delivery_scorecard.csv"
    assert csv_file.exists(), f"Delivery scorecard CSV not found at {csv_file}"

    df = pd.read_csv(csv_file)
    expected_cols = ["Day", "Phase", "Core Deliverables", "Metric / KPI", "Target / SLA", "Status"]
    assert list(df.columns) == expected_cols

    # Exactly 8 milestone rows
    assert len(df) == 8
    assert list(df["Day"]) == [f"Day {i}" for i in range(1, 9)]

    # Check Day 8 row details
    day8_row = df[df["Day"] == "Day 8"].iloc[0]
    assert "Project Delivery Deck" in day8_row["Phase"]
    assert "Interactive 14-slide HTML deck" in day8_row["Core Deliverables"]
    assert "Completed & Verified" in day8_row["Status"]


def test_figure_17_existence_and_size(showcase):
    """Verify that Figure 17 exists and meets high-resolution production standards."""
    fig17_file = showcase.figures_dir / "17_project_delivery_lifecycle_and_scorecard.png"
    assert fig17_file.exists(), f"Figure 17 not found at {fig17_file}"
    size = fig17_file.stat().st_size
    # Must be a high-resolution 300 DPI render (> 200 KB)
    assert size > 200_000, f"Figure 17 size too small ({size} bytes)"


def test_delivery_showcase_verification_suite(showcase):
    """Verify that automated deliverable verification reports 100% PASS."""
    results = showcase.verify_delivery_artifacts()
    assert results["all_passed"] is True, f"Artifact verification failed: {results}"
    assert results["passed_count"] == results["total_artifacts_checked"]
    assert results["passed_count"] >= 17
