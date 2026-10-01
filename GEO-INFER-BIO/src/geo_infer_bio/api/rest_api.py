"""
REST API for GEO-INFER-BIO.
"""

from typing import Any, cast
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import tempfile
import base64
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from .. import __version__

from ..core.sequence_analysis import SequenceAnalyzer
from ..utils.validation import DataValidator
from ..utils.visualization import BioVisualizer


class SpatialData(BaseModel):
    """Spatial data model."""

    latitude: float
    longitude: float


class SequenceData(BaseModel):
    """Sequence data model."""

    id: str
    sequence: str
    spatial_data: SpatialData | None = None


class AnalysisResult(BaseModel):
    """Analysis result model."""

    sequence_id: str
    gc_content: float
    motif_count: int
    coding_regions: int
    spatial_data: SpatialData | None = None


app = FastAPI(
    title="GEO-INFER-BIO API",
    description="API for biological sequence analysis with spatial context",
    version=__version__,
)


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "name": "GEO-INFER-BIO API",
        "version": __version__,
        "description": "API for biological sequence analysis with spatial context",
    }


@app.post("/analyze/sequence", response_model=AnalysisResult)
async def analyze_sequence(sequence_data: SequenceData) -> AnalysisResult:
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
        raise HTTPException(status_code=400, detail="Invalid sequence")

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


@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    spatial_data: UploadFile | None = File(None),
) -> list[dict[str, Any]]:
    """
    Analyze sequences from a file.

    Args:
        file: FASTA file containing sequences
        spatial_data: Optional CSV file containing spatial data

    Returns:
        Analysis results
    """
    analyzer = SequenceAnalyzer()
    validator = DataValidator()

    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as fasta_temp:
        fasta_temp.write(await file.read())
        fasta_path = fasta_temp.name
    spatial_df = None
    spatial_path: str | None = None
    if spatial_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as spatial_temp:
            spatial_temp.write(await spatial_data.read())
            spatial_path = spatial_temp.name

    try:
        # Read after the temp-file handle is closed so written bytes are flushed
        spatial_df: pd.DataFrame | None = None
        if spatial_path is not None:
            spatial_df = pd.read_csv(spatial_path)

        # Load and validate sequences
        loaded = analyzer.load_sequence(fasta_path)
        sequences: list[SeqRecord] = loaded if isinstance(loaded, list) else [loaded]
        results: list[dict[str, Any]] = []

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

            result: dict[str, Any] = {
                "sequence_id": record.id,
                "gc_content": gc_content,
                "motif_count": len(motifs),
                "coding_regions": len(coding_regions),
            }

            spatial_attr = getattr(record, "spatial_data", None)
            if spatial_attr is not None:
                result["spatial_data"] = {
                    "latitude": spatial_attr.iloc[0]["latitude"],
                    "longitude": spatial_attr.iloc[0]["longitude"],
                }

            results.append(result)

        return results
    finally:
        # Remove any temporary files that were created, even on failure
        for temp_path in (fasta_path, spatial_path):
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)


@app.post("/visualize/spatial")
async def visualize_spatial(
    analysis_results: list[AnalysisResult],
) -> dict[str, str]:
    """
    Generate spatial visualizations of analysis results.

    Every result must carry spatial_data; the plots place each sequence on a
    latitude/longitude map.

    Args:
        analysis_results: List of analysis results with spatial context

    Returns:
        Mapping of plot name to base64-encoded PNG data
    """
    if not analysis_results:
        raise HTTPException(
            status_code=400, detail="at least one analysis result is required"
        )
    if any(result.spatial_data is None for result in analysis_results):
        raise HTTPException(
            status_code=400,
            detail="every analysis result must include spatial_data for spatial visualization",
        )

    visualizer = BioVisualizer()

    # Convert results to DataFrame with flattened spatial columns
    df = pd.DataFrame(
        [
            {
                "sequence_id": result.sequence_id,
                "gc_content": result.gc_content,
                "motif_count": result.motif_count,
                "coding_regions": result.coding_regions,
                "latitude": result.spatial_data.latitude,
                "longitude": result.spatial_data.longitude,
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

        # Read visualization files and encode as base64 for JSON transport
        visualizations: dict[str, str] = {}
        for plot_file in [gc_plot, motif_plot, coding_plot]:
            with open(plot_file, "rb") as f:
                visualizations[plot_file.stem] = base64.b64encode(f.read()).decode(
                    "ascii"
                )

        return visualizations


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy"}
