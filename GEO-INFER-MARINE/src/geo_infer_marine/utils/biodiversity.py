"""Shared biodiversity index computations.

Single implementation of the standard alpha-diversity metrics
(Shannon, Simpson, Pielou evenness) used by the ecosystem and
coral reef modules, so threshold and smoothing choices cannot
silently diverge between callers.
"""

import numpy as np

# Log smoothing constant: avoids log(0) for zero proportions while
# contributing negligibly (< 1e-8) to the final index values.
_LOG_SMOOTHING = 1e-10


def biodiversity_metrics(species_counts: dict[str, int]) -> dict[str, float]:
    """Compute standard alpha-diversity metrics from species counts.

    Args:
        species_counts: Mapping of species name to individual count.

    Returns:
        Dictionary with ``species_richness``, ``total_abundance``,
        ``shannon``, ``simpson`` (1 - D, Gini-Simpson) and
        ``evenness`` (Pielou's J). Empty input yields all zeros.
    """
    if not species_counts:
        return {
            "species_richness": 0,
            "total_abundance": 0,
            "shannon": 0.0,
            "simpson": 0.0,
            "evenness": 0.0,
        }

    counts = np.array(list(species_counts.values()), dtype=float)
    total = counts.sum()
    richness = len(counts)
    proportions = counts / total

    shannon = -float(np.sum(proportions * np.log(proportions + _LOG_SMOOTHING)))
    simpson = 1.0 - float(np.sum(proportions**2))
    max_shannon = np.log(richness) if richness > 1 else 1.0
    evenness = shannon / max_shannon if max_shannon > 0 else 0.0

    return {
        "species_richness": richness,
        "total_abundance": int(total),
        "shannon": shannon,
        "simpson": simpson,
        "evenness": float(evenness),
    }
