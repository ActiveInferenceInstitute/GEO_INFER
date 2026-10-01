#!/usr/bin/env python3
"""GEO-INFER-BIO module orchestrator.

Runs one documented end-to-end BIO operation on synthetic data: synthesize
three contigs (one carrying a designed 87-nt ORF, one a mutated variant of
it, one random background), write them as FASTA, and run the
``SequenceAnalyzer`` pipeline over them — global pairwise alignment, GC
content, repeated-motif detection, BLOSUM62 similarity, and coding-region
prediction. All work goes through the real ``geo_infer_bio`` public API.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    import numpy as np

    from geo_infer_bio import SequenceAnalyzer

    rng = np.random.default_rng(7)
    bases = np.array(list("ACGT"))
    sense_codons = [
        "TTC",
        "TTA",
        "CTG",
        "ATT",
        "ATC",
        "GTT",
        "GCT",
        "CCC",
        "ACC",
        "GTC",
        "GAG",
        "TGG",
        "TAC",
        "AAG",
        "GAA",
        "CAA",
        "GGC",
        "CGT",
    ]

    def _random_bases(count: int) -> list[str]:
        return [str(base) for base in rng.choice(bases, size=count)]

    # Designed 87-nt ORF: ATG, 27 sense codons, TAA (j - start = 84 >= 60).
    orf = ["ATG"] + [sense_codons[i % len(sense_codons)] for i in range(27)] + ["TAA"]
    orf_string = "".join(orf)

    background_a = _random_bases(240)
    orf_start = 30
    contig_a = (
        "".join(background_a[:orf_start])
        + orf_string
        + "".join(background_a[orf_start + len(orf_string) :])
    )

    # Mutated variant: three point substitutions inside the ORF.
    mutated = list(contig_a)
    for offset in (35, 60, 90):
        original = mutated[offset]
        mutated[offset] = next(base for base in "ACGT" if base != original)
    contig_b = "".join(mutated)
    contig_c = "".join(_random_bases(240))

    records_payload = [
        ("contig_alpha", contig_a),
        ("contig_beta", contig_b),
        ("contig_gamma", contig_c),
    ]

    analyzer = SequenceAnalyzer()
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_path = Path(tmp_dir) / "synthetic_contigs.fasta"
        fasta_path.write_text(
            "".join(f">{name}\n{sequence}\n" for name, sequence in records_payload),
            encoding="utf-8",
        )
        records = analyzer.load_sequence(str(fasta_path))

    if not isinstance(records, list):
        raise RuntimeError("expected a list of SeqRecords from the FASTA file")

    alignment = analyzer.align_sequences(records[:2], algorithm="global")
    aligned_pair = [str(record.seq) for record in alignment]
    matches = sum(1 for a, b in zip(aligned_pair[0], aligned_pair[1]) if a == b)
    alignment_identity = matches / len(aligned_pair[0])

    gc_contents = {
        record.id: round(analyzer.calculate_gc_content(record.seq), 3)
        for record in records
    }
    repeated_motifs = analyzer.find_motifs(records[0].seq, motif_length=6)
    similarity = analyzer.calculate_sequence_similarity(records[0].seq, records[1].seq)
    coding_regions = analyzer.predict_coding_regions(records[0].seq, min_length=60)

    return {
        "operation": "synthetic_contig_sequence_analysis_pipeline",
        "n_records": len(records),
        "contig_lengths": {name: len(seq) for name, seq in records_payload},
        "gc_content_percent": gc_contents,
        "alignment_identity": round(float(alignment_identity), 4),
        "alignment_length": int(len(aligned_pair[0])),
        "repeated_hexamer_motif_count": len(repeated_motifs),
        "blosum62_similarity_alpha_beta": round(float(similarity), 4),
        "predicted_coding_regions": coding_regions,
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("BIO", _operation))
