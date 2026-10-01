"""
GraphQL API for GEO-INFER-BIO.
"""

from typing import cast
import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI
import pandas as pd
from pathlib import Path
import tempfile
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from ..core.sequence_analysis import SequenceAnalyzer
from ..utils.validation import DataValidator
from ..utils.visualization import BioVisualizer


@strawberry.type
class SpatialData:
    """Spatial data type."""

    latitude: float
    longitude: float


@strawberry.type
class SequenceData:
    """Sequence data type."""

    id: str
    sequence: str
    spatial_data: SpatialData | None = None


@strawberry.type
class AnalysisResult:
    """Analysis result type."""

    sequence_id: str
    gc_content: float
    motif_count: int
    coding_regions: int
    spatial_data: SpatialData | None = None


@strawberry.type
class VisualizationData:
    """Visualization data type."""

    gc_content: str
    motif_density: str
    coding_potential: str


@strawberry.input
class SpatialDataInput:
    """Input variant of SpatialData for GraphQL arguments."""

    latitude: float
    longitude: float


@strawberry.input
class SequenceDataInput:
    """Input variant of SequenceData for GraphQL arguments."""

    id: str
    sequence: str
    spatial_data: SpatialDataInput | None = None


@strawberry.input
class AnalysisResultInput:
    """Input variant of AnalysisResult for GraphQL arguments."""

    sequence_id: str
    gc_content: float
    motif_count: int
    coding_regions: int
    spatial_data: SpatialDataInput | None = None


@strawberry.type
class Query:
    """Query type."""

    @strawberry.field
    def analyze_sequence(self, sequence_data: SequenceDataInput) -> AnalysisResult:
        """
        Analyze a single sequence.

        Args:
            sequence_data: Sequence data with optional spatial context

        Returns:
            Analysis results
        """
        analyzer = SequenceAnalyzer()
        validator = DataValidator()

        # Validate sequence
        if not validator.validate_sequence(sequence_data.sequence):
            raise ValueError("Invalid sequence")

        # Create SeqRecord
        record = SeqRecord(
            Seq(sequence_data.sequence),
            id=sequence_data.id,
        )

        # Add spatial data if provided
        if sequence_data.spatial_data:
            record.spatial_data = pd.DataFrame(
                [
                    {
                        "latitude": sequence_data.spatial_data.latitude,
                        "longitude": sequence_data.spatial_data.longitude,
                    }
                ]
            )

        # Perform analysis
        seq_value = cast("Seq", record.seq)
        gc_content = analyzer.calculate_gc_content(seq_value)
        motifs = analyzer.find_motifs(seq_value)
        coding_regions = analyzer.predict_coding_regions(seq_value)

        return AnalysisResult(
            sequence_id=sequence_data.id,
            gc_content=gc_content,
            motif_count=len(motifs),
            coding_regions=len(coding_regions),
            spatial_data=sequence_data.spatial_data,
        )

    @strawberry.field
    def analyze_file(
        self, file_path: str, spatial_data_path: str | None = None
    ) -> list[AnalysisResult]:
        """
        Analyze sequences from a file.

        Args:
            file_path: Path to FASTA file
            spatial_data_path: Optional path to CSV file with spatial data

        Returns:
            List of analysis results
        """
        analyzer = SequenceAnalyzer()
        validator = DataValidator()

        # Load sequences
        loaded = analyzer.load_sequence(file_path)
        sequences: list[SeqRecord] = loaded if isinstance(loaded, list) else [loaded]

        # Load spatial data if provided
        spatial_df = None
        if spatial_data_path:
            spatial_df = pd.read_csv(spatial_data_path)

        results: list[AnalysisResult] = []
        for i, record in enumerate(sequences):
            # Add spatial data if available
            if spatial_df is not None and i < len(spatial_df):
                record.spatial_data = spatial_df.iloc[[i]]

            # Validate sequence
            validation = validator.validate_sequence_record(record)
            if not all(validation.values()):
                continue

            # Perform analysis
            seq_value = cast("Seq", record.seq)
            gc_content = analyzer.calculate_gc_content(seq_value)
            motifs = analyzer.find_motifs(seq_value)
            coding_regions = analyzer.predict_coding_regions(seq_value)

            result = AnalysisResult(
                sequence_id=record.id or "",
                gc_content=gc_content,
                motif_count=len(motifs),
                coding_regions=len(coding_regions),
            )

            spatial_attr = getattr(record, "spatial_data", None)
            if spatial_attr is not None:
                result.spatial_data = SpatialData(
                    latitude=float(spatial_attr.iloc[0]["latitude"]),
                    longitude=float(spatial_attr.iloc[0]["longitude"]),
                )

            results.append(result)

        return results

    @strawberry.field
    def visualize_spatial(
        self, analysis_results: list[AnalysisResultInput]
    ) -> VisualizationData:
        """
        Generate spatial visualizations of analysis results.

        Args:
            analysis_results: List of analysis results

        Returns:
            Visualization data
        """
        visualizer = BioVisualizer()

        # Convert results to DataFrame
        df = pd.DataFrame(
            [
                {
                    "sequence_id": result.sequence_id,
                    "gc_content": result.gc_content,
                    "motif_count": result.motif_count,
                    "coding_regions": result.coding_regions,
                    "latitude": result.spatial_data.latitude
                    if result.spatial_data
                    else None,
                    "longitude": result.spatial_data.longitude
                    if result.spatial_data
                    else None,
                }
                for result in analysis_results
            ]
        )

        # Generate visualizations
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # GC content distribution
            gc_plot = output_dir / "gc_content.png"
            visualizer.plot_gc_distribution(df, output_path=str(gc_plot))

            # Motif density
            motif_plot = output_dir / "motif_density.png"
            visualizer.plot_motif_density(df, output_path=str(motif_plot))

            # Coding potential
            coding_plot = output_dir / "coding_potential.png"
            visualizer.plot_coding_potential(df, output_path=str(coding_plot))

            # Read visualization files
            with open(gc_plot, "rb") as f:
                gc_bytes = f.read()
            with open(motif_plot, "rb") as f:
                motif_bytes = f.read()
            with open(coding_plot, "rb") as f:
                coding_bytes = f.read()

            return VisualizationData(
                gc_content=gc_bytes.decode("latin1"),
                motif_density=motif_bytes.decode("latin1"),
                coding_potential=coding_bytes.decode("latin1"),
            )

    @strawberry.field
    def health_check(self) -> str:
        """Health check query."""
        return "healthy"


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
app = FastAPI()
app.include_router(graphql_app, prefix="/graphql")
