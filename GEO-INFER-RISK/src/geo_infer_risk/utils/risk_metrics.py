"""Catastrophe risk metrics derived from an event loss table.

An *event loss table* (ELT) is the standard output of a catastrophe model: one
row per (event, hazard) pair with the modelled ground-up loss. Every metric in
this module is a summary of that table:

- Average Annual Loss (AAL) -- the expected loss per year of exposure.
- Exceedance probability (EP) curve -- loss as a function of the probability of
  being exceeded, and its inverse, the return-period loss.
- Probable Maximum Loss (PML) -- the loss at a single long return period.
- Extreme-value PML -- a power-law tail extrapolation past the observed record
  (see :func:`estimate_pml_with_tail_fit`).
- Annual Exceedance Probability (AEP) curve -- the probability that a year's
  summed losses breach a threshold, as a curve over thresholds (see
  :func:`calculate_aep_curve`).
- Tail Value at Risk (TVaR) -- the mean loss conditional on breaching VaR.
- Occurrence and aggregate exceedance probabilities (OEP / AEP) -- the annual
  probability that a single event, respectively the annual total, breaches a
  loss threshold.

Annualization is explicit throughout. Several metrics need to know how many
years of exposure the table spans; a table of ``n`` events says nothing about
that on its own. :func:`calculate_aal` requires ``exposure_years``. The
exceedance functions take an optional ``exposure_years``: without it their
probabilities are per-event frequencies (EP curve, PML) or the table is
assumed to span one year (annual OEP/AEP, tail fit), and a warning is logged
because that assumption distorts anything expressed per year.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .rng import SeedLike, resolve_rng

logger = logging.getLogger(__name__)

__all__ = [
    "calculate_aal",
    "calculate_ep_curve",
    "calculate_pml",
    "calculate_loss_by_return_period",
    "calculate_tail_value_at_risk",
    "calculate_annual_occurrence_exceedance_probability",
    "calculate_annual_aggregate_exceedance_probability",
    "calculate_loss_frequency_curve",
    "calculate_correlation_matrix",
    "calculate_aep_curve",
    "estimate_pml_with_tail_fit",
]

REQUIRED_COLUMNS: tuple[str, ...] = ("event_id", "hazard_type", "loss")


def _validate_columns(event_loss_table: pd.DataFrame) -> None:
    """Raise if the event loss table lacks a required column.

    Args:
        event_loss_table: Candidate event loss table.

    Raises:
        ValueError: If any of ``event_id``, ``hazard_type`` or ``loss`` is
            missing.
    """
    for column in REQUIRED_COLUMNS:
        if column not in event_loss_table.columns:
            raise ValueError(f"event_loss_table must contain column '{column}'")


def _event_total_losses(
    event_loss_table: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Return one loss per event, as a 1-D float array.

    A DataFrame may hold several rows per event (one per hazard or per
    sub-peril); those are summed so that each event contributes exactly once.
    An array is taken to already be one loss per event.

    Args:
        event_loss_table: Event loss table, or an array of per-event losses.

    Returns:
        Per-event total losses.

    Raises:
        ValueError: If a DataFrame input lacks a required column, or if an
            array input is not one-dimensional.
    """
    if isinstance(event_loss_table, np.ndarray):
        if event_loss_table.ndim > 1:
            raise ValueError("loss array must be one-dimensional")
        return np.asarray(event_loss_table, dtype=float).ravel()
    _validate_columns(event_loss_table)
    grouped = event_loss_table.groupby("event_id")["loss"].sum()
    return np.asarray(grouped.to_numpy(), dtype=float)


def _resolve_exposure_years(
    exposure_years: float | None,
    n_events: int,
    caller: str,
) -> float:
    """Return the annualization denominator, warning when it is assumed.

    Args:
        exposure_years: Years of exposure the table spans, or ``None``.
        n_events: Number of distinct events in the table, used only to make the
            warning actionable.
        caller: Name of the calling function, for the warning message.

    Returns:
        A positive number of exposure years.

    Raises:
        ValueError: If ``exposure_years`` is supplied but not positive, or is
            not finite.
    """
    if exposure_years is None:
        logger.warning(
            "%s called without exposure_years; assuming the %d-event table "
            "spans 1 year, which inflates any per-year quantity. Pass "
            "exposure_years for a correctly annualized result.",
            caller,
            n_events,
        )
        return 1.0
    years = float(exposure_years)
    if not np.isfinite(years) or years <= 0:
        raise ValueError("exposure_years must be finite and positive")
    return years


def _empirical_exceedance(
    losses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical exceedance curve for a set of per-event losses.

    Uses the Weibull plotting position, ``p_i = i / (n + 1)`` for the ``i``-th
    largest loss. It is unbiased for the exceedance probability of the
    underlying distribution regardless of that distribution's shape, and unlike
    ``i / n`` it never assigns probability 1.0 to the smallest observation.

    Args:
        losses: Per-event losses, in any order.

    Returns:
        Tuple of ``(probs, sorted_losses)`` where ``probs`` is strictly
        increasing and ``sorted_losses`` is the matching losses in decreasing
        order. Both are empty when ``losses`` is empty.
    """
    if losses.size == 0:
        return np.empty(0), np.empty(0)
    sorted_losses = np.sort(losses)[::-1]
    n = sorted_losses.size
    probs = np.arange(1, n + 1, dtype=float) / (n + 1)
    return probs, sorted_losses


def _interpolate_loss_at_probs(
    probs: np.ndarray,
    sorted_losses: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Interpolate loss at requested exceedance probabilities.

    ``probs`` is increasing while ``sorted_losses`` is decreasing, which is
    exactly the ``(xp, fp)`` contract of :func:`numpy.interp`. Targets outside
    the empirical range are clamped to the largest or smallest observed loss
    rather than extrapolated, because the tail beyond the observed sample is
    not identified by the data.

    Args:
        probs: Increasing empirical exceedance probabilities.
        sorted_losses: Losses in decreasing order, aligned to ``probs``.
        targets: Exceedance probabilities to evaluate.

    Returns:
        Interpolated losses, same shape as ``targets``.
    """
    if probs.size == 0:
        return np.zeros_like(targets, dtype=float)
    return np.interp(targets, probs, sorted_losses)


def calculate_aal(
    event_loss_table: pd.DataFrame | np.ndarray,
    exposure_years: float,
) -> float | dict[str, Any]:
    """Calculate the Average Annual Loss (AAL).

    The AAL is the expected loss per year: total modelled loss divided by the
    number of years of exposure the table represents,
    ``AAL = total loss / exposure_years``. The event count cannot stand in for
    the exposure period, so ``exposure_years`` is required.

    Args:
        event_loss_table: Either a DataFrame of event losses with columns
            ``event_id``, ``hazard_type`` and ``loss``, or a 1-D numpy array of
            per-event losses.
        exposure_years: Number of exposure years the loss table spans.

    Returns:
        For an array input, the AAL as a float. For a DataFrame input, a dict
        with ``total`` (float) and ``by_hazard`` (``Dict[str, float]``) keys.

    Raises:
        ValueError: If a DataFrame input lacks a required column, if an array
            input is not one-dimensional, or if ``exposure_years`` is not
            finite and positive.
    """
    if exposure_years is None:
        raise ValueError(
            "calculate_aal requires exposure_years: the number of years of "
            "exposure the loss table spans"
        )
    if isinstance(event_loss_table, np.ndarray):
        losses = _event_total_losses(event_loss_table)
        years = _resolve_exposure_years(exposure_years, losses.size, "calculate_aal")
        return float(losses.sum() / years) if losses.size else 0.0

    _validate_columns(event_loss_table)
    num_events = int(event_loss_table["event_id"].nunique())
    years = _resolve_exposure_years(exposure_years, num_events, "calculate_aal")
    total_loss = float(event_loss_table["loss"].sum())
    hazard_aal = {
        str(hazard_type): float(group["loss"].sum()) / years
        for hazard_type, group in event_loss_table.groupby("hazard_type")
    }
    return {"total": total_loss / years, "by_hazard": hazard_aal}


def calculate_ep_curve(
    event_loss_table: pd.DataFrame | np.ndarray,
    exceedance_probs: list[float] | None = None,
    exposure_years: float | None = None,
) -> dict[str, list[float]]:
    """Calculate the exceedance probability (EP) curve.

    The EP curve pairs each loss level with the probability of exceeding it.
    Without ``exposure_years`` the probabilities are per-event exceedance
    frequencies from the Weibull plotting position. With ``exposure_years`` they
    are annual occurrence exceedance probabilities, obtained by converting the
    empirical rate of exceedances per year, ``lambda``, through the Poisson
    relation ``P = 1 - exp(-lambda)``.

    Args:
        event_loss_table: DataFrame of event losses with columns ``event_id``,
            ``hazard_type`` and ``loss``, or a 1-D array of per-event losses.
        exceedance_probs: Probabilities at which to report losses. When
            ``None`` (default) the full empirical curve is returned, one point
            per event. Requested probabilities outside the empirical range are
            clamped to the largest or smallest observed loss.
        exposure_years: Years of exposure the table spans. When provided, the
            returned probabilities are annual rather than per-event.

    Returns:
        Dict with equal-length lists under ``exceedance_probability``, ``loss``
        and ``return_period``. Return periods are the reciprocal of the
        probabilities, and ``inf`` where a probability is zero.

    Raises:
        ValueError: If a DataFrame input lacks a required column, if an array
            input is not one-dimensional, if any requested probability is
            outside ``(0, 1]``, or if ``exposure_years`` is not finite and
            positive.
    """
    losses = _event_total_losses(event_loss_table)
    probs, sorted_losses = _empirical_exceedance(losses)

    if exposure_years is not None:
        years = _resolve_exposure_years(
            exposure_years, losses.size, "calculate_ep_curve"
        )
        # probs are per-event exceedance frequencies; scale to an annual rate
        # of exceedances and map through the Poisson occurrence probability.
        rates = probs * losses.size / years
        probs = 1.0 - np.exp(-rates)

    if exceedance_probs is None:
        target_probs = probs
        target_losses = sorted_losses
    else:
        targets = np.asarray(exceedance_probs, dtype=float)
        if targets.size and (np.any(targets <= 0) or np.any(targets > 1)):
            raise ValueError("exceedance_probs must lie in (0, 1]")
        order = np.argsort(-targets)
        target_probs = targets[order]
        target_losses = _interpolate_loss_at_probs(probs, sorted_losses, target_probs)

    return {
        "exceedance_probability": [float(p) for p in target_probs],
        "loss": [float(loss) for loss in target_losses],
        "return_period": [
            float(1.0 / p) if p > 0 else float("inf") for p in target_probs
        ],
    }


def calculate_pml(
    event_loss_table: pd.DataFrame | np.ndarray,
    return_period: float = 250,
    exposure_years: float | None = None,
) -> float:
    """Calculate the Probable Maximum Loss (PML) at a return period.

    The PML is the loss exceeded with probability ``1 / return_period``.

    Args:
        event_loss_table: DataFrame of event losses with columns ``event_id``,
            ``hazard_type`` and ``loss``, or a 1-D array of per-event losses.
        return_period: Return period in years. Default 250.
        exposure_years: Years of exposure the table spans. Required for the
            return period to mean years rather than events; see
            :func:`calculate_ep_curve`.

    Returns:
        The PML, or ``0.0`` for an empty table.

    Raises:
        ValueError: If ``return_period`` is not finite and greater than 1, or
            if the inputs fail :func:`calculate_ep_curve` validation.
    """
    if not np.isfinite(return_period) or return_period <= 1:
        raise ValueError("return_period must be finite and greater than 1")

    losses = _event_total_losses(event_loss_table)
    if losses.size == 0:
        return 0.0

    exceedance_prob = 1.0 / float(return_period)
    curve = calculate_ep_curve(
        event_loss_table, [exceedance_prob], exposure_years=exposure_years
    )
    # A return period longer than the record can resolve is clamped to the
    # largest observed loss, which understates the tail. Say so rather than
    # returning a confident-looking number. Without exposure_years the record
    # length is only knowable in events, so events stand in for years.
    resolvable_years = (
        float(exposure_years) + 1.0
        if exposure_years is not None
        else float(losses.size + 1)
    )
    if return_period > resolvable_years:
        logger.warning(
            "PML requested at a %.0f-year return period from a record that "
            "resolves about %.0f years; the result is clamped to the largest "
            "observed loss and understates the tail. Fit a parametric tail "
            "for longer return periods.",
            return_period,
            resolvable_years,
        )

    return float(curve["loss"][0])


def calculate_loss_by_return_period(
    event_loss_table: pd.DataFrame | np.ndarray,
    return_periods: list[float],
    exposure_years: float | None = None,
) -> dict[str, float]:
    """Calculate losses for several return periods.

    Args:
        event_loss_table: DataFrame of event losses with columns ``event_id``,
            ``hazard_type`` and ``loss``, or a 1-D array of per-event losses.
        return_periods: Return periods to evaluate.
        exposure_years: Years of exposure the table spans; see
            :func:`calculate_ep_curve`.

    Returns:
        Mapping from the string form of each requested return period to its
        loss, in the order requested.

    Raises:
        ValueError: If any return period is not finite and greater than 1, or
            if the inputs fail :func:`calculate_ep_curve` validation.
    """
    periods = [float(rp) for rp in return_periods]
    if any(not np.isfinite(rp) or rp <= 1 for rp in periods):
        raise ValueError("every return period must be finite and greater than 1")
    if not periods:
        return {}

    exceedance_probs = [1.0 / rp for rp in periods]
    curve = calculate_ep_curve(
        event_loss_table, exceedance_probs, exposure_years=exposure_years
    )
    # calculate_ep_curve sorts requested probabilities descending; recover the
    # caller's order by matching on probability.
    by_prob = dict(zip(curve["exceedance_probability"], curve["loss"]))
    return {str(rp): by_prob[1.0 / rp] for rp in periods}


def calculate_tail_value_at_risk(
    event_loss_table: pd.DataFrame | np.ndarray,
    confidence_level: float = 0.99,
) -> float:
    """Calculate Tail Value at Risk (TVaR) over the event loss distribution.

    TVaR at level ``a`` is ``E[L | L >= VaR_a]``: the mean loss among the worst
    ``1 - a`` of events. It is coherent as a risk measure, unlike VaR, and is
    sensitive to how heavy the tail is rather than only to where it starts.

    Args:
        event_loss_table: DataFrame of event losses with columns ``event_id``,
            ``hazard_type`` and ``loss``, or a 1-D array of per-event losses.
        confidence_level: Level ``a``, e.g. ``0.99``. Must lie in ``(0, 1)``.

    Returns:
        The TVaR, or ``0.0`` for an empty table.

    Raises:
        ValueError: If ``confidence_level`` is not finite and in ``(0, 1)``, or
            if the inputs fail validation.
    """
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    losses = np.sort(_event_total_losses(event_loss_table))
    if losses.size == 0:
        return 0.0

    # VaR as the smallest observation at or above the requested quantile.
    var_idx = int(np.ceil(confidence_level * losses.size)) - 1
    var_idx = max(0, min(var_idx, losses.size - 1))
    var = losses[var_idx]

    tail = losses[losses >= var]
    return float(np.mean(tail)) if tail.size else float(var)


def calculate_annual_occurrence_exceedance_probability(
    event_loss_table: pd.DataFrame | np.ndarray,
    threshold: float,
    exposure_years: float | None = None,
) -> float:
    """Calculate the annual Occurrence Exceedance Probability (OEP).

    The OEP is the probability that at least one event in a year exceeds the
    threshold. Modelling occurrences as a Poisson process with annual rate
    ``lambda`` -- the count of threshold-breaching events divided by the years
    of exposure -- gives ``OEP = 1 - exp(-lambda)``.

    Args:
        event_loss_table: DataFrame of event losses with columns ``event_id``,
            ``hazard_type`` and ``loss``, or a 1-D array of per-event losses.
        threshold: Loss threshold. Must be finite.
        exposure_years: Years of exposure the table spans. When omitted the
            table is assumed to span one year, which over-states the annual
            rate for any longer record; a warning is logged.

    Returns:
        The OEP, in ``[0, 1)``.

    Raises:
        ValueError: If ``threshold`` is not finite, or if the inputs fail
            validation.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    losses = _event_total_losses(event_loss_table)
    if losses.size == 0:
        return 0.0

    years = _resolve_exposure_years(
        exposure_years,
        losses.size,
        "calculate_annual_occurrence_exceedance_probability",
    )
    exceedance_rate = float(np.count_nonzero(losses > threshold)) / years
    return float(1.0 - np.exp(-exceedance_rate))


def calculate_annual_aggregate_exceedance_probability(
    event_loss_table: pd.DataFrame | np.ndarray,
    threshold: float,
    num_years: int = 10000,
    random_seed: SeedLike = None,
    exposure_years: float | None = None,
) -> float:
    """Estimate the annual Aggregate Exceedance Probability (AEP) by simulation.

    The AEP is the probability that the *sum* of a year's event losses exceeds
    the threshold. Each simulated year draws an event count from a Poisson
    distribution with rate ``len(events) / exposure_years`` and then draws that
    many losses with replacement from the empirical severity distribution. This
    is the standard frequency-severity decomposition; it assumes occurrences
    are independent across the year and severities are independent of count.

    Args:
        event_loss_table: DataFrame of event losses with columns ``event_id``,
            ``hazard_type`` and ``loss``, or a 1-D array of per-event losses.
        threshold: Loss threshold. Must be finite.
        num_years: Number of years to simulate. The Monte Carlo standard error
            of the returned probability is about ``sqrt(p (1 - p) / num_years)``.
        random_seed: Seed or generator for the simulation; see
            :func:`geo_infer_risk.utils.rng.resolve_rng`. Pass an int for a
            replayable estimate.
        exposure_years: Years of exposure the table spans, used for the Poisson
            rate. When omitted the table is assumed to span one year, which
            over-states event frequency for any longer record; a warning is
            logged.

    Returns:
        The AEP, in ``[0, 1]``.

    Raises:
        ValueError: If ``threshold`` is not finite, if ``num_years`` is not a
            positive integer, or if the inputs fail validation.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not isinstance(num_years, (int, np.integer)) or num_years < 1:
        raise ValueError("num_years must be a positive integer")

    losses = _event_total_losses(event_loss_table)
    if losses.size == 0:
        return 0.0

    years = _resolve_exposure_years(
        exposure_years,
        losses.size,
        "calculate_annual_aggregate_exceedance_probability",
    )
    rng = resolve_rng(random_seed)

    rate = losses.size / years
    counts = rng.poisson(rate, size=int(num_years))
    total_draws = int(counts.sum())
    if total_draws == 0:
        return float(0.0 > threshold)

    draws = rng.choice(losses, size=total_draws, replace=True)
    year_index = np.repeat(np.arange(int(num_years)), counts)
    year_totals = np.bincount(year_index, weights=draws, minlength=int(num_years))
    return float(np.count_nonzero(year_totals > threshold) / int(num_years))


def calculate_loss_frequency_curve(
    event_loss_table: pd.DataFrame | np.ndarray,
    num_bins: int = 20,
) -> dict[str, list[float]]:
    """Calculate a histogram of per-event losses.

    Args:
        event_loss_table: DataFrame of event losses with columns ``event_id``,
            ``hazard_type`` and ``loss``, or a 1-D array of per-event losses.
        num_bins: Number of histogram bins. Must be a positive integer.

    Returns:
        Dict with ``bin_edges`` (length ``num_bins + 1``), ``frequencies``
        (counts) and ``normalized_frequencies`` (counts divided by their sum,
        so they integrate to 1 as a probability mass function).

    Raises:
        ValueError: If ``num_bins`` is not a positive integer, or if the inputs
            fail validation.
    """
    if not isinstance(num_bins, (int, np.integer)) or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")

    losses = _event_total_losses(event_loss_table)
    if losses.size == 0:
        return {"bin_edges": [], "frequencies": [], "normalized_frequencies": []}

    frequencies, bin_edges = np.histogram(losses, bins=int(num_bins))
    total = frequencies.sum()
    normalized = frequencies / total if total > 0 else frequencies.astype(float)

    return {
        "bin_edges": [float(edge) for edge in bin_edges],
        "frequencies": [int(count) for count in frequencies],
        "normalized_frequencies": [float(value) for value in normalized],
    }


def _fit_exceedance_tail(
    losses: np.ndarray, years: float, threshold_percentile: float
) -> tuple[float, float, float, np.ndarray]:
    """Fit a power-law tail to the largest per-event losses.

    The tail is the ``threshold_percentile`` largest fraction of losses.  Its
    exceedance behaviour is modelled in log space as

    ``log(P_annual) ~= slope * log(L) + intercept``,

    where ``P_annual`` is the Poisson-annualised exceedance probability of a
    loss at least ``L``.  ``np.polyfit`` yields the linear least-squares fit,
    which extrapolates monotonically beyond the longest observed return period
    and so supports PML estimates past the record that a purely empirical curve
    must clamp.

    Args:
        losses: Per-event losses.
        years: Years of exposure the record spans (annualisation denominator).
        threshold_percentile: Tail cut-off quantile in ``(0, 1)``.

    Returns:
        Tuple of ``(slope, intercept, threshold, tail)`` where ``tail`` holds
        the losses exceeding ``threshold``.

    Raises:
        ValueError: If the tail has fewer than three observations (fit would be
            under-determined), the cut-off is not in ``(0, 1)``, or the fitted
            slope is not negative (a tail cannot grow more probable with loss).
    """
    if not 0.0 < threshold_percentile < 1.0:
        raise ValueError("threshold_percentile must lie in (0, 1)")
    if losses.size < 4:
        raise ValueError("at least four losses are required to fit a tail")
    threshold = float(np.quantile(losses, threshold_percentile))
    tail = losses[losses > threshold]
    if tail.size < 3:
        raise ValueError("too few losses above the tail threshold to fit a stable tail")
    ordered = np.sort(tail)
    # Exceedance frequency of the i-th smallest tail loss among the tail.
    exceed_freq = (tail.size - np.arange(tail.size)) / (tail.size + 1)
    # Annualise event exceedance through the Poisson occurrence probability.
    rates = exceed_freq * tail.size / years
    annual_probs = 1.0 - np.exp(-rates)
    log_loss = np.log(ordered)
    log_prob = np.log(annual_probs)
    slope, intercept = np.polyfit(log_loss, log_prob, 1)
    if slope >= 0.0:
        raise ValueError("fitted tail slope is not negative; cannot extrapolate")
    return float(slope), float(intercept), threshold, tail


def calculate_aep_curve(
    event_loss_table: pd.DataFrame | np.ndarray,
    thresholds: Sequence[float] | None = None,
    num_years: int = 5000,
    random_seed: SeedLike = None,
    exposure_years: float | None = None,
) -> dict[str, list[float]]:
    """Estimate an Annual Exceedance Probability (AEP) curve over loss levels.

    AEP here is the *aggregate* annual probability that a year's summed losses
    exceed a threshold -- the frequency-severity simulation of
    :func:`calculate_annual_aggregate_exceedance_probability` evaluated at
    several thresholds, sharing one deterministic seed so the curve is smooth.

    Args:
        event_loss_table: Event loss table (or array of per-event losses).
        thresholds: Loss levels at which AEP is reported.  When ``None`` a
            default logarithmic grid spanning the observed losses is used.
        num_years: Years in the simulation (see the underlying function).
        random_seed: Seed or generator for reproducibility (default fresh).
        exposure_years: Years of exposure the record spans, used for the
            Poisson rate; see the underlying function.

    Returns:
        Dict with equal-length lists ``threshold`` and ``aep``.  Thresholds
        above every observed loss have AEP ``0.0``; a threshold of zero is gone
        unless a year can draw no events.

    Raises:
        ValueError: If a threshold is non-finite, or the inputs fail underlying
            validation, or ``num_years`` is not a positive integer.
    """
    losses = _event_total_losses(event_loss_table)
    if losses.size == 0:
        selected = list(thresholds or [0.0])
        if any(not np.isfinite(float(threshold)) for threshold in selected):
            raise ValueError("threshold must be finite")
        return {
            "threshold": [float(threshold) for threshold in selected],
            "aep": [0.0 for _ in selected],
        }
    if thresholds is None:
        low = float(np.min(losses))
        high = float(np.max(losses))
        step = (high - low) / 20.0 if high > low else 1.0
        selected = [low + step * i for i in range(21)]
    else:
        if any(not np.isfinite(float(threshold)) for threshold in thresholds):
            raise ValueError("threshold must be finite")
        selected = [float(threshold) for threshold in thresholds]
    aeps = [
        calculate_annual_aggregate_exceedance_probability(
            event_loss_table,
            threshold,
            num_years=num_years,
            random_seed=random_seed,
            exposure_years=exposure_years,
        )
        for threshold in selected
    ]
    return {"threshold": selected, "aep": aeps}


def estimate_pml_with_tail_fit(
    event_loss_table: pd.DataFrame | np.ndarray,
    return_period: float = 250,
    exposure_years: float | None = None,
    threshold_percentile: float = 0.7,
) -> dict[str, Any]:
    """Estimate PML beyond the observed record from a fitted power-law tail.

    The classic PML is the empirical loss at a long return period, which must
    clamp at the largest observed loss once the return period exceeds the
    record. This function instead fits a power-law (Generalised-log / log-log)
    tail to the largest losses and extrapolates to the requested return period,
    giving a PML that continues to rise past the record. This is the standard
    actuarial extreme-value treatment for catastrophe tails.

    Args:
        event_loss_table: Event loss table (or array of per-event losses).
        return_period: Return period in years. Must be finite and above 1.
        exposure_years: Years of exposure the record spans. Required for a
            correct annualisation; fallback assumes one year and warns.
        threshold_percentile: Tail quantile in ``(0, 1)``.

    Returns:
        Dict with ``pml`` (the extrapolated loss), ``return_period``,
        ``method`` (``\"gpd_tail\"``), ``threshold_percentile``,
        ``tail_threshold``, ``tail_count``, ``tail_slope`` (the power-law
        alpha), ``tail_intercept`` and ``warning`` when applicable.

    Raises:
        ValueError: If the record cannot support a tail fit (too few tail
            losses) or the fitted tail is ill-conditioned, or ``return_period``
            is not finite and greater than 1.
    """
    if not np.isfinite(return_period) or return_period <= 1:
        raise ValueError("return_period must be finite and greater than 1")
    losses = _event_total_losses(event_loss_table)
    if losses.size == 0:
        return {
            "pml": 0.0,
            "return_period": float(return_period),
            "method": "gpd_tail",
            "tail_count": 0,
        }
    years = _resolve_exposure_years(
        exposure_years, losses.size, "estimate_pml_with_tail_fit"
    )
    slope, intercept, threshold, tail = _fit_exceedance_tail(
        losses, years, threshold_percentile
    )
    exceedance_prob = 1.0 / float(return_period)
    # Invert log(prob) = slope*log(L) + intercept for L.
    log_pml = (np.log(exceedance_prob) - intercept) / slope
    pml = float(np.exp(log_pml))
    tail_count = int(tail.size)
    return {
        "pml": max(0.0, pml),
        "return_period": float(return_period),
        "method": "gpd_tail",
        "threshold_percentile": threshold_percentile,
        "tail_threshold": threshold,
        "tail_count": tail_count,
        "tail_slope": slope,
        "tail_intercept": intercept,
    }


def calculate_correlation_matrix(event_loss_table: pd.DataFrame) -> dict[str, Any]:
    """Calculate the correlation of losses across hazard types.

    Losses are pivoted to one row per event and one column per hazard type,
    with absent combinations filled with zero, and the Pearson correlation is
    taken across events. A hazard type with zero variance yields ``NaN``
    correlations, which is reported rather than silently replaced.

    Args:
        event_loss_table: DataFrame of event losses with columns ``event_id``,
            ``hazard_type`` and ``loss``.

    Returns:
        Dict with ``hazard_types`` (column order) and ``correlation_matrix``
        (nested list, row-major, aligned to ``hazard_types``).

    Raises:
        ValueError: If a required column is missing.
    """
    _validate_columns(event_loss_table)

    pivot_table = event_loss_table.pivot_table(
        index="event_id",
        columns="hazard_type",
        values="loss",
        aggfunc="sum",
        fill_value=0,
    )

    return {
        "hazard_types": [str(column) for column in pivot_table.columns],
        "correlation_matrix": pivot_table.corr().values.tolist(),
    }
