"""
Sequence analysis module for GEO-INFER-BIO.
"""

from typing import Any, cast
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment, PairwiseAligner, substitution_matrices
from Bio.Data import CodonTable

_BLOSUM62 = substitution_matrices.load("BLOSUM62")

from ..utils.validation import DataValidator
from ..utils.visualization import BioVisualizer


class SequenceAnalyzer:
    """A class for analyzing biological sequences with spatial context."""

    def __init__(self) -> None:
        """Initialize the SequenceAnalyzer."""
        self.validator = DataValidator()
        self.visualizer = BioVisualizer()
        self.codon_table = CodonTable.standard_dna_table

    def load_sequence(
        self, file_path: str, format: str = "fasta"
    ) -> SeqRecord | list[SeqRecord]:
        """
        Load sequence data from a file.

        Args:
            file_path: Path to the sequence file
            format: File format (default: "fasta")

        Returns:
            Sequence record(s) from the file
        """
        with open(file_path, encoding="utf-8") as handle:
            return list(SeqIO.parse(handle, format))

    def align_sequences(
        self,
        sequences: list[SeqRecord],
        algorithm: str = "global",
        gap_open: float = -10,
        gap_extend: float = -0.5,
    ) -> MultipleSeqAlignment:
        """
        Perform a pairwise alignment of the first two sequence records.

        Note: despite returning a MultipleSeqAlignment container, this performs a
        two-sequence pairwise alignment (not a progressive multiple-sequence
        alignment). Records beyond the first two are ignored.

        Args:
            sequences: List of sequence records (exactly the first two are used)
            algorithm: Alignment algorithm ("global" or "local")
            gap_open: Gap opening penalty
            gap_extend: Gap extension penalty

        Returns:
            Pairwise alignment of the first two records, wrapped in a
            MultipleSeqAlignment for compatibility with downstream BIO tools
        """
        if len(sequences) < 2:
            raise ValueError("at least two sequences are required for alignment")
        if algorithm not in {"global", "local"}:
            raise ValueError("algorithm must be 'global' or 'local'")

        # PairwiseAligner is the supported Biopython alignment implementation.
        # The result remains a MultipleSeqAlignment for downstream BIO tools.
        aligner = PairwiseAligner()
        aligner.mode = algorithm
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extend
        alignment = aligner.align(sequences[0].seq, sequences[1].seq)[0]
        aligned_records = [
            SeqRecord(Seq(str(alignment[row])), id=sequences[row].id)
            for row in range(2)
        ]
        return MultipleSeqAlignment(aligned_records)

    def calculate_gc_content(self, sequence: Seq) -> float:
        """
        Calculate GC content of a sequence.

        Args:
            sequence: DNA/RNA sequence

        Returns:
            GC content as a percentage
        """
        gc_count = int(sequence.count("G")) + int(sequence.count("C"))
        return (gc_count / len(sequence)) * 100

    def find_motifs(self, sequence: Seq, motif_length: int = 6) -> dict[str, list[int]]:
        """
        Find repeated motifs in a sequence.

        Args:
            sequence: DNA/RNA sequence
            motif_length: Length of motifs to search for

        Returns:
            Dictionary of motifs and their positions
        """
        motifs: dict[str, list[int]] = {}
        for i in range(len(sequence) - motif_length + 1):
            motif = str(sequence[i : i + motif_length])
            if motif in motifs:
                motifs[motif].append(i)
            else:
                motifs[motif] = [i]
        return {k: v for k, v in motifs.items() if len(v) > 1}

    def calculate_sequence_similarity(self, seq1: Seq, seq2: Seq) -> float:
        """
        Calculate sequence similarity using BLOSUM62 matrix.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity score
        """
        matrix = _BLOSUM62

        def _lookup(x: str, y: str) -> float:
            try:
                return float(matrix[x, y])
            except KeyError:
                try:
                    return float(matrix[y, x])
                except KeyError:
                    return 0.0

        score = sum(_lookup(a, b) for a, b in zip(seq1, seq2))

        # Compute maximum possible score (self-alignment) for normalization
        max_score1 = sum(_lookup(a, a) for a in seq1)
        max_score2 = sum(_lookup(b, b) for b in seq2)
        max_score = max(max_score1, max_score2)

        if max_score == 0:
            return 0.0
        return float(score / max_score)

    def predict_coding_regions(
        self, sequence: Seq, min_length: int = 100
    ) -> list[dict[str, int]]:
        """
        Predict potential coding regions in a sequence.

        Args:
            sequence: DNA sequence
            min_length: Minimum length of coding region

        Returns:
            List of predicted coding regions with start/end positions
        """
        coding_regions = []
        start_codons = ["ATG"]
        stop_codons = ["TAA", "TAG", "TGA"]

        for frame in range(3):
            for i in range(frame, len(sequence) - 2, 3):
                codon = str(sequence[i : i + 3])
                if codon in start_codons:
                    start_pos = i
                    for j in range(i + 3, len(sequence) - 2, 3):
                        codon = str(sequence[j : j + 3])
                        if codon in stop_codons:
                            if j - start_pos >= min_length:
                                coding_regions.append(
                                    {
                                        "frame": frame,
                                        "start": start_pos,
                                        "end": j + 2,
                                    }
                                )
                            break
        return coding_regions

    def analyze_spatial_distribution(
        self,
        sequences: list[SeqRecord],
        spatial_data: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Analyze spatial distribution of sequence features.

        Args:
            sequences: List of sequence records
            spatial_data: DataFrame containing spatial coordinates

        Returns:
            Dictionary of spatial analyses
        """
        results: dict[str, list[dict[str, Any]]] = {
            "gc_content": [],
            "motif_density": [],
            "coding_potential": [],
        }

        for seq, spatial in zip(sequences, spatial_data.itertuples()):
            if seq.seq is None:
                continue
            seq_value = cast("Seq", seq.seq)
            gc = self.calculate_gc_content(seq_value)
            motifs = self.find_motifs(seq_value)
            coding_regions = self.predict_coding_regions(seq_value)

            results["gc_content"].append(
                {
                    "sequence_id": seq.id,
                    "gc_content": gc,
                    "latitude": spatial.latitude,
                    "longitude": spatial.longitude,
                }
            )

            results["motif_density"].append(
                {
                    "sequence_id": seq.id,
                    "motif_count": len(motifs),
                    "latitude": spatial.latitude,
                    "longitude": spatial.longitude,
                }
            )

            results["coding_potential"].append(
                {
                    "sequence_id": seq.id,
                    "coding_regions": len(coding_regions),
                    "latitude": spatial.latitude,
                    "longitude": spatial.longitude,
                }
            )

        return {k: pd.DataFrame(v) for k, v in results.items()}

    def visualize_spatial_patterns(
        self,
        spatial_analysis: dict[str, pd.DataFrame],
        output_path: str | None = None,
    ) -> None:
        """
        Visualize spatial patterns in sequence features.

        Args:
            spatial_analysis: Dictionary of spatial analyses
            output_path: Optional path to save visualization
        """
        self.visualizer.plot_spatial_distribution(
            spatial_analysis,
            output_path=output_path,
        )
