"""
Integration tests for GEO-INFER-BIO: sequence analysis pipeline feeding into visualization.

Tests the SequenceAnalyzer and BioVisualizer working together in a real pipeline,
using in-memory sequence data and spatial coordinates.
"""

import pytest
import pandas as pd

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import matplotlib

matplotlib.use("Agg")


pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def sample_sequences():
    """Create synthetic DNA sequences with spatial metadata."""
    sequences = [
        SeqRecord(
            Seq("ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"),
            id="seq_001",
            name="Sample_A",
        ),
        SeqRecord(
            Seq("GCGCGCGCATATATATATAGCGCGCGCGCATATATATAT"),
            id="seq_002",
            name="Sample_B",
        ),
        SeqRecord(
            Seq("ATGATGATGATGATGATGATGATGATGATGATGATGATGATG"),
            id="seq_003",
            name="Sample_C",
        ),
    ]
    return sequences


@pytest.fixture
def spatial_data():
    """Create spatial metadata for sequence samples."""
    return pd.DataFrame(
        {
            "latitude": [34.05, 36.77, 37.77],
            "longitude": [-118.24, -119.42, -122.42],
            "collection_site": ["Los Angeles", "Fresno", "San Francisco"],
        }
    )


class TestSequenceAnalysisPipeline:
    """Test sequence analysis methods producing real outputs."""

    def test_gc_content_calculation(self, sample_sequences):
        """Test GC content calculation produces valid percentages."""
        from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer

        analyzer = SequenceAnalyzer()

        gc_values = []
        for record in sample_sequences:
            gc = analyzer.calculate_gc_content(record.seq)
            gc_values.append(gc)
            assert 0.0 <= gc <= 100.0, f"GC content {gc} out of range for {record.id}"

        # Verify distinct GC values for different sequences
        assert len(set(round(v, 2) for v in gc_values)) > 1, (
            "Expected distinct GC values across sequences"
        )

    def test_motif_finding(self, sample_sequences):
        """Test motif detection on synthetic sequences with known repeats."""
        from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer

        analyzer = SequenceAnalyzer()

        # seq_001 has repeating CGATCG pattern
        motifs = analyzer.find_motifs(sample_sequences[0].seq, motif_length=6)
        assert isinstance(motifs, dict)
        # Should find at least one repeated motif in the repeating sequence
        assert len(motifs) > 0, "Expected repeated motifs in CGATCG-repeat sequence"

        # Verify motif positions are valid indices
        for motif, positions in motifs.items():
            assert len(motif) == 6
            for pos in positions:
                assert 0 <= pos <= len(sample_sequences[0].seq) - 6

    def test_coding_region_prediction(self, sample_sequences):
        """Test coding region prediction returns valid frame/position data."""
        from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer

        analyzer = SequenceAnalyzer()

        # seq_003 starts with ATG (start codon) so should find coding regions
        # if there is a downstream stop codon within the min_length
        regions = analyzer.predict_coding_regions(sample_sequences[2].seq, min_length=9)
        assert isinstance(regions, list)

        for region in regions:
            assert "frame" in region
            assert "start" in region
            assert "end" in region
            assert 0 <= region["frame"] <= 2
            assert region["start"] < region["end"]


class TestSpatialAnalysisPipeline:
    """Test the full pipeline from sequence analysis through spatial distribution."""

    def test_analyze_spatial_distribution(self, sample_sequences, spatial_data):
        """Test spatial distribution analysis end-to-end."""
        from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer

        analyzer = SequenceAnalyzer()
        results = analyzer.analyze_spatial_distribution(sample_sequences, spatial_data)

        # Verify all three analysis types are present
        assert "gc_content" in results
        assert "motif_density" in results
        assert "coding_potential" in results

        # Verify each is a DataFrame with expected columns
        gc_df = results["gc_content"]
        assert isinstance(gc_df, pd.DataFrame)
        assert "sequence_id" in gc_df.columns
        assert "gc_content" in gc_df.columns
        assert "latitude" in gc_df.columns
        assert "longitude" in gc_df.columns
        assert len(gc_df) == len(sample_sequences)

        motif_df = results["motif_density"]
        assert isinstance(motif_df, pd.DataFrame)
        assert "motif_count" in motif_df.columns
        assert len(motif_df) == len(sample_sequences)

    def test_gc_content_varies_by_sequence(self, sample_sequences, spatial_data):
        """Test that GC content values differ across sequences with different compositions."""
        from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer

        analyzer = SequenceAnalyzer()
        results = analyzer.analyze_spatial_distribution(sample_sequences, spatial_data)

        gc_values = results["gc_content"]["gc_content"].tolist()
        # Sequences have different GC compositions, so values should differ
        assert max(gc_values) - min(gc_values) > 1.0, (
            "GC content should vary across diverse sequences"
        )

    def test_visualization_from_analysis_output(
        self, sample_sequences, spatial_data, tmp_path
    ):
        """Test that visualization consumes analysis output without error."""
        from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer
        from geo_infer_bio.utils.visualization import BioVisualizer

        analyzer = SequenceAnalyzer()
        results = analyzer.analyze_spatial_distribution(sample_sequences, spatial_data)

        visualizer = BioVisualizer()

        # Test spatial distribution plotting
        output_file = str(tmp_path / "spatial_dist.png")
        visualizer.plot_spatial_distribution(results, output_path=output_file)

        import os

        assert os.path.exists(output_file), "Visualization file should be created"

    def test_gc_distribution_plot(self, sample_sequences, spatial_data, tmp_path):
        """Test GC content distribution plot from analysis pipeline."""
        from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer
        from geo_infer_bio.utils.visualization import BioVisualizer

        analyzer = SequenceAnalyzer()
        results = analyzer.analyze_spatial_distribution(sample_sequences, spatial_data)

        visualizer = BioVisualizer()
        output_file = str(tmp_path / "gc_dist.png")
        visualizer.plot_gc_distribution(results["gc_content"], output_path=output_file)

        import os

        assert os.path.exists(output_file), "GC distribution plot should be created"


class TestSequenceSimilarityPipeline:
    """Test sequence comparison capabilities."""

    def test_sequence_similarity_self_comparison(self, sample_sequences):
        """Test that a sequence is maximally similar to itself."""
        from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer

        analyzer = SequenceAnalyzer()
        similarity = analyzer.calculate_sequence_similarity(
            sample_sequences[0].seq, sample_sequences[0].seq
        )
        assert abs(similarity - 1.0) < 0.01, "Self-similarity should be ~1.0"

    def test_sequence_similarity_different_sequences(self, sample_sequences):
        """Test that different sequences have lower similarity than self."""
        from geo_infer_bio.core.sequence_analysis import SequenceAnalyzer

        analyzer = SequenceAnalyzer()
        self_sim = analyzer.calculate_sequence_similarity(
            sample_sequences[0].seq, sample_sequences[0].seq
        )
        cross_sim = analyzer.calculate_sequence_similarity(
            sample_sequences[0].seq, sample_sequences[1].seq
        )
        assert cross_sim < self_sim, (
            "Cross-sequence similarity should be less than self-similarity"
        )
