"""
Data validation utilities for GEO-INFER-BIO.
"""

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from typing import cast


class DataValidator:
    """A class for validating biological data."""

    def __init__(self) -> None:
        """Initialize the DataValidator."""
        self.valid_nucleotides = set("ATCG")
        self.valid_rna_nucleotides = set("AUCG")
        self.valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        self.valid_coordinates = {
            "latitude": (-90, 90),
            "longitude": (-180, 180),
        }

    def validate_sequence(
        self, sequence: str | Seq, sequence_type: str = "DNA"
    ) -> bool:
        """
        Validate a biological sequence.

        Args:
            sequence: The sequence to validate
            sequence_type: Type of sequence ("DNA", "RNA", or "protein").
                DNA uses A/T/C/G, RNA uses A/U/C/G, protein uses the standard
                one-letter amino-acid alphabet.

        Returns:
            bool: True if sequence is valid, False otherwise
        """
        if isinstance(sequence, str):
            sequence = Seq(sequence)

        valid_chars = {
            "DNA": self.valid_nucleotides,
            "RNA": self.valid_rna_nucleotides,
            "protein": self.valid_amino_acids,
        }

        if sequence_type not in valid_chars:
            raise ValueError(f"Invalid sequence type: {sequence_type}")

        return all(char in valid_chars[sequence_type] for char in str(sequence))

    def validate_spatial_coordinates(self, latitude: float, longitude: float) -> bool:
        """
        Validate spatial coordinates.

        Args:
            latitude: Latitude value
            longitude: Longitude value

        Returns:
            bool: True if coordinates are valid, False otherwise
        """
        lat_valid = (
            self.valid_coordinates["latitude"][0]
            <= latitude
            <= self.valid_coordinates["latitude"][1]
        )
        lon_valid = (
            self.valid_coordinates["longitude"][0]
            <= longitude
            <= self.valid_coordinates["longitude"][1]
        )
        return lat_valid and lon_valid

    def validate_spatial_dataframe(
        self, df: pd.DataFrame, required_columns: list[str] | None = None
    ) -> bool:
        """
        Validate spatial data DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: List of required columns

        Returns:
            bool: True if DataFrame is valid, False otherwise
        """
        if required_columns is None:
            required_columns = ["latitude", "longitude"]

        # Check required columns
        if not all(col in df.columns for col in required_columns):
            return False

        # Check coordinate ranges
        for _, row in df.iterrows():
            if not self.validate_spatial_coordinates(row["latitude"], row["longitude"]):
                return False

        return True

    def validate_sequence_record(
        self, record: SeqRecord, check_spatial: bool = True
    ) -> dict[str, bool]:
        """
        Validate a sequence record.

        Args:
            record: SeqRecord to validate
            check_spatial: Whether to check spatial attributes

        Returns:
            Dict containing validation results
        """
        results = {
            "sequence_valid": self.validate_sequence(
                cast(str | Seq, record.seq) if record.seq is not None else ""
            ),
            "id_valid": bool(record.id),
            "description_valid": bool(record.description),
            "spatial_valid": True,
        }

        if check_spatial and hasattr(record, "spatial_data"):
            results["spatial_valid"] = self.validate_spatial_dataframe(
                record.spatial_data
            )

        return results

    def validate_alignment(self, alignment: MultipleSeqAlignment) -> dict[str, bool]:
        """
        Validate a multiple sequence alignment.

        Args:
            alignment: MultipleSeqAlignment to validate

        Returns:
            Dict containing validation results
        """
        results = {
            "sequences_valid": True,
            "lengths_equal": True,
            "spatial_valid": True,
        }

        # Check sequence validity
        for record in alignment:
            if not self.validate_sequence(record.seq):
                results["sequences_valid"] = False
                break

        # Check alignment lengths
        lengths = [len(record.seq) for record in alignment]
        results["lengths_equal"] = len(set(lengths)) == 1

        # Check spatial data if present
        if hasattr(alignment[0], "spatial_data"):
            for record in alignment:
                if not self.validate_spatial_dataframe(record.spatial_data):
                    results["spatial_valid"] = False
                    break

        return results

    def validate_gc_content(self, gc_content: float, sequence_length: int) -> bool:
        """
        Validate GC content calculation.

        Args:
            gc_content: Calculated GC content
            sequence_length: Length of the sequence

        Returns:
            bool: True if GC content is valid, False otherwise
        """
        if sequence_length == 0:
            return False
        return 0 <= gc_content <= 100

    def validate_motif(self, motif: str, sequence_type: str = "DNA") -> bool:
        """
        Validate a DNA/RNA motif.

        Args:
            motif: Motif to validate
            sequence_type: Type of sequence ("DNA" or "RNA")

        Returns:
            bool: True if motif is valid, False otherwise
        """
        if sequence_type not in ["DNA", "RNA"]:
            raise ValueError("sequence_type must be 'DNA' or 'RNA'")
        return self.validate_sequence(motif, sequence_type)

    def validate_coding_region(
        self,
        start: int,
        end: int,
        sequence_length: int,
        min_length: int = 100,
    ) -> bool:
        """
        Validate a coding region.

        Args:
            start: Start position
            end: End position
            sequence_length: Length of the sequence
            min_length: Minimum length of coding region

        Returns:
            bool: True if coding region is valid, False otherwise
        """
        if start < 0 or end >= sequence_length:
            return False
        if end - start + 1 < min_length:
            return False
        return True
