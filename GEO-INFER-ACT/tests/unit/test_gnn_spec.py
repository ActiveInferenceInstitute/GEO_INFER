"""Exact GNN v1 Markdown rendering for categorical models."""

from __future__ import annotations

from fractions import Fraction

import pytest

from geo_infer_act.core.gnn_spec import CategoricalGNNSpec, exact_decimal, is_dyadic

F = Fraction


def _spec(**changes) -> CategoricalGNNSpec:
    values = dict(
        model_name="Two cell test",
        annotation="Two H3-free categorical states.",
        A=((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))),
        B=(((F(1), F(1, 2)), (F(1, 4), F(0))), ((F(0), F(1, 2)), (F(3, 4), F(1)))),
        C=(0.0, 2.0),
        D=(F(1, 2), F(1, 2)),
        E=(F(1, 4), F(3, 4)),
        step_seconds=60,
        action_labels=("stay", "move"),
    )
    values.update(changes)
    return CategoricalGNNSpec(**values)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (F(3, 4), "0.75"),
        (F(9, 64), "0.140625"),
        (F(1), "1.0"),
        (F(0), "0.0"),
        (F(21, 32), "0.65625"),
    ],
)
def test_exact_decimal_literals_are_exact_binary_values(
    value: Fraction, text: str
) -> None:
    """Dyadic probabilities render as terminating decimals that round-trip exactly."""
    assert exact_decimal(value) == text
    assert Fraction(float(text)) == value
    assert is_dyadic(value)


def test_non_dyadic_values_have_no_exact_literal() -> None:
    """One third is rejected rather than rounded."""
    assert not is_dyadic(F(1, 3))
    with pytest.raises(ValueError, match="exact binary"):
        exact_decimal(F(1, 3))


def test_validation_rejects_unnormalized_transition_columns() -> None:
    """A B column that does not sum to one fails at construction."""
    bad = (((F(1), F(1, 2)), (F(1, 4), F(0))), ((F(0), F(1, 4)), (F(3, 4), F(1))))
    with pytest.raises(ValueError, match="B"):
        _spec(B=bad)


def test_rendered_source_declares_axes_time_and_space() -> None:
    """The rendered Markdown carries canonical axes, explicit time and literals."""
    spec = _spec()
    source = spec.render()
    assert "b_tensor_order: next_state_previous_state_action" in source
    assert "StepSeconds=60" in source and "SpaceKind=categorical" in source
    assert "A[2,2,type=float]" in source and "B[2,2,2,type=float]" in source
    assert "E={(0.25, 0.75)}" in source
    assert spec.float_matrices()["B"][0][0] == [1.0, 0.5]
    assert len(spec.sha256()) == 64
