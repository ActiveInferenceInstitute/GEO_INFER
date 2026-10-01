"""
Contrast analysis for Statistical Parametric Mapping

This module implements contrast definition and testing for SPM analysis.
Contrasts allow specification of linear combinations of regression coefficients
to test specific hypotheses about experimental conditions or covariates.

The implementation supports:
- Simple contrasts (e.g., A > B)
- Complex contrasts with multiple conditions
- F-contrasts for testing multiple coefficients simultaneously
- Automatic contrast generation for common experimental designs

Mathematical Foundation:
For a contrast vector c, the contrast statistic is:
t = c^T β / sqrt(c^T Var(β) c)

where β are the regression coefficients and Var(β) is their covariance matrix.
"""

import numpy as np
from scipy.stats import t, f
import re

from ..models.data_models import SPMResult, ContrastResult, DesignMatrix


class Contrast:
    """
    Contrast specification for SPM hypothesis testing.

    This class represents a linear combination of regression coefficients
    defining a specific hypothesis to be tested.

    Attributes:
        vector: Contrast vector (length = n_regressors)
        name: Descriptive name for the contrast
        type: Type of contrast ('t' or 'F')
        weights: Optional dictionary of condition weights for interpretation
    """

    def __init__(
        self,
        vector: np.ndarray,
        name: str = "",
        contrast_type: str = "t",
        weights: dict[str, float] | None = None,
    ):
        """
        Initialize contrast.

        Args:
            vector: Contrast vector defining the hypothesis
            name: Descriptive name
            contrast_type: 't' for t-contrast, 'F' for F-contrast
            weights: Dictionary mapping condition names to weights
        """
        self.vector = np.asarray(vector, dtype=float)
        self.name = name or f"Contrast_{len(vector)}"
        self.type = contrast_type.lower()
        self.weights = weights or {}

        if self.type not in ["t", "f"]:
            raise ValueError("Contrast type must be 't' or 'F'")

        if self.type == "f" and self.vector.ndim != 2:
            raise ValueError("F-contrasts require 2D contrast matrix")

    @classmethod
    def from_string(
        cls, contrast_str: str, design_names: list[str], contrast_type: str = "t"
    ) -> "Contrast":
        """
        Create contrast from string specification.

        Args:
            contrast_str: String like "A > B" or "A + B - C"
            design_names: Names of design matrix columns
            contrast_type: Type of contrast

        Returns:
            Contrast object

        Example:
            >>> names = ['intercept', 'condition_A', 'condition_B', 'covariate']
            >>> contrast = Contrast.from_string("condition_A > condition_B", names)
        """
        # Save original string for name (before stripping spaces)
        original_str = contrast_str

        # Parse contrast string
        contrast_str = contrast_str.replace(" ", "")

        # Handle simple comparisons
        if ">" in contrast_str:
            left, right = contrast_str.split(">")
            weights = cls._parse_condition_weights(left, design_names, +1)
            weights.update(cls._parse_condition_weights(right, design_names, -1))
        elif "<" in contrast_str:
            left, right = contrast_str.split("<")
            weights = cls._parse_condition_weights(left, design_names, -1)
            weights.update(cls._parse_condition_weights(right, design_names, +1))
        else:
            # General linear combination
            weights = cls._parse_linear_combination(contrast_str, design_names)

        # Convert to contrast vector
        vector = np.zeros(len(design_names))
        for i, name in enumerate(design_names):
            if name in weights:
                vector[i] = weights[name]

        # Validate that at least one design column was matched
        if not np.any(vector):
            raise ValueError(
                f"Contrast string '{original_str}' did not match any design matrix columns: {design_names}"
            )

        return cls(vector, original_str, contrast_type, weights)

    @staticmethod
    def _parse_condition_weights(
        condition: str, design_names: list[str], weight: float
    ) -> dict[str, float]:
        """Parse condition weights from contrast string component."""
        weights = {}

        # Handle intercept
        if condition.lower() in ["intercept", "const", "1"]:
            weights["intercept"] = weight
            return weights

        # Split on + and -
        parts = re.split(r"([+-])", condition)
        if not parts[0]:  # Starts with sign
            parts = parts[1:]

        current_weight = weight
        for part in parts:
            part = part.strip()
            if part in ["+", "-"]:
                current_weight = weight if part == "+" else -weight
            elif part:
                # Check if coefficient is specified (e.g., "2*A")
                coeff_match = re.match(r"(\d*\.?\d*)\*(\w+)", part)
                if coeff_match:
                    coeff, var_name = coeff_match.groups()
                    coeff_val = float(coeff) if coeff else 1.0
                else:
                    var_name = part
                    coeff_val = 1.0

                weights[var_name] = current_weight * coeff_val

        return weights

    @staticmethod
    def _parse_linear_combination(
        expr: str, design_names: list[str]
    ) -> dict[str, float]:
        """Parse general linear combination expression."""
        weights = {}

        # Split on + and - while keeping the operators
        terms = re.split(r"([+-])", expr)

        if not terms[0]:  # Expression starts with operator
            terms = terms[1:]

        # Process terms
        for i in range(0, len(terms), 2):
            operator = terms[i - 1] if i > 0 else "+"
            term = terms[i].strip()

            sign = 1 if operator == "+" else -1

            # Parse coefficient and variable
            if "*" in term:
                coeff_str, var = term.split("*", 1)
                try:
                    coeff = float(coeff_str)
                except ValueError:
                    coeff = 1.0
                    var = term
            else:
                coeff = 1.0
                var = term

            # Remove parentheses if present
            var = var.strip("()")

            if var in design_names:
                weights[var] = sign * coeff

        return weights

    def __str__(self) -> str:
        """String representation of contrast."""
        return f"Contrast('{self.name}': {self.vector})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Contrast(vector={self.vector}, name='{self.name}', type='{self.type}')"


def contrast(
    model_result: SPMResult,
    contrast_spec: str | np.ndarray | Contrast,
    contrast_type: str = "t",
) -> ContrastResult:
    """
    Define and compute a contrast for SPM analysis.

    Args:
        model_result: Fitted GLM results
        contrast_spec: Contrast specification (string, vector, or Contrast object)
        contrast_type: Type of contrast ('t' or 'F')

    Returns:
        ContrastResult with statistics and significance

    Example:
        >>> # String specification
        >>> result = contrast(model, "condition_A > condition_B")
        >>>
        >>> # Vector specification
        >>> result = contrast(model, [0, 1, -1, 0])
        >>>
        >>> # Contrast object
        >>> c = Contrast([0, 1, -1], "A_vs_B")
        >>> result = contrast(model, c)
    """
    # Convert contrast specification to Contrast object
    design_names = model_result.design_matrix.names or []
    if isinstance(contrast_spec, str):
        contrast_obj = Contrast.from_string(contrast_spec, design_names, contrast_type)
    elif isinstance(contrast_spec, np.ndarray):
        contrast_obj = Contrast(contrast_spec, contrast_type=contrast_type)
    elif isinstance(contrast_spec, (list, tuple)):
        if contrast_spec and all(isinstance(item, str) for item in contrast_spec):
            vector = np.zeros(model_result.design_matrix.n_regressors)
            for name in contrast_spec:
                if name not in design_names:
                    raise ValueError(
                        f"Contrast regressor '{name}' not found in design matrix"
                    )
                vector[design_names.index(name)] = 1.0
            contrast_obj = Contrast(
                vector, name="+".join(contrast_spec), contrast_type=contrast_type
            )
        else:
            contrast_obj = Contrast(
                np.array(contrast_spec, dtype=float), contrast_type=contrast_type
            )
    elif isinstance(contrast_spec, Contrast):
        contrast_obj = contrast_spec
    else:
        raise TypeError("contrast_spec must be string, array, list, or Contrast object")

    # Validate contrast dimensions
    n_regressors = model_result.design_matrix.n_regressors
    if contrast_obj.type == "t":
        if len(contrast_obj.vector) != n_regressors:
            raise ValueError(
                f"T-contrast vector length {len(contrast_obj.vector)} "
                f"does not match number of regressors {n_regressors}"
            )
    else:  # F-contrast
        if contrast_obj.vector.shape[1] != n_regressors:
            raise ValueError(
                f"F-contrast matrix columns {contrast_obj.vector.shape[1]} "
                f"does not match number of regressors {n_regressors}"
            )

    # Compute contrast statistics
    if contrast_obj.type == "t":
        result = _compute_t_contrast(model_result, contrast_obj)
    else:
        result = _compute_f_contrast(model_result, contrast_obj)

    return result


def _compute_t_contrast(
    model_result: SPMResult, contrast_obj: Contrast
) -> ContrastResult:
    """Compute t-contrast statistics."""
    c = contrast_obj.vector
    beta = model_result.beta_coefficients
    cov_beta = model_result.cov_beta

    # Contrast estimate: c^T β
    if beta.ndim == 1:
        effect_size = c @ beta
    else:
        effect_size = np.mean(beta.T @ c)

    # Standard error: sqrt(c^T Var(β) c)
    if cov_beta is not None and cov_beta.ndim == 2:
        var_contrast = c @ cov_beta @ c
    elif cov_beta is not None:
        var_contrast = np.sum(cov_beta * c[:, np.newaxis] * c[np.newaxis, :])
    else:
        # cov_beta not stored — compute from residuals and design matrix
        X = model_result.design_matrix.matrix
        n_points, n_regressors = X.shape
        if (
            model_result.residuals is not None
            and len(model_result.residuals) > n_regressors
        ):
            sigma2 = float(
                np.sum(model_result.residuals**2) / (n_points - n_regressors)
            )
            XtX_inv = np.linalg.pinv(X.T @ X)
            cov_beta_computed = sigma2 * XtX_inv
            var_contrast = c @ cov_beta_computed @ c
        else:
            var_contrast = float(np.sum(c**2))

    standard_error = np.sqrt(float(np.mean(var_contrast)))

    # T-statistic
    t_statistic = effect_size / standard_error

    # Degrees of freedom
    n_points = model_result.design_matrix.n_points
    n_regressors = model_result.design_matrix.n_regressors
    df = n_points - n_regressors

    # P-values (two-tailed)
    p_values = 2 * t.sf(np.abs(t_statistic), df)

    # Reshape to match data dimensions if needed
    if model_result.spm_data.data.ndim > 1 and np.ndim(t_statistic) > 0:
        target_shape = model_result.spm_data.data.shape
        t_statistic = t_statistic.reshape(target_shape)
        effect_size = effect_size.reshape(target_shape)
        standard_error = standard_error.reshape(target_shape)
        p_values = p_values.reshape(target_shape)

    return ContrastResult(
        contrast_vector=c,
        t_statistic=t_statistic,
        effect_size=effect_size,
        standard_error=standard_error,
        p_values=p_values,
    )


def _compute_f_contrast(
    model_result: SPMResult, contrast_obj: Contrast
) -> ContrastResult:
    """Compute F-contrast statistics."""
    C = contrast_obj.vector  # F-contrast matrix (n_hypotheses x n_regressors)
    beta = model_result.beta_coefficients
    cov_beta = model_result.cov_beta

    # F-statistic: (Cβ)^T (C Var(β) C^T)^(-1) (Cβ) / rank(C) — the matrix
    # form handles both single- and multi-hypothesis contrasts.
    C_beta = C @ beta

    if C.ndim == 1:
        C = C.reshape(1, -1)
        C_beta = C_beta.reshape(1, -1)

    # F-statistic
    rank_C = np.linalg.matrix_rank(C)
    if rank_C == 0:
        raise ValueError("F-contrast matrix has zero rank")
    if C.shape[0] == 1:
        c = C[0]
        var_contrast = c @ cov_beta @ c
        f_statistic = (c @ beta) ** 2 / var_contrast / rank_C
    else:
        # Multiple contrast F-test
        C_cov_C_inv = np.linalg.pinv(C @ cov_beta @ C.T)
        f_statistic = (C_beta.T @ C_cov_C_inv @ C_beta) / rank_C

    # Degrees of freedom
    n_points = model_result.design_matrix.n_points
    n_regressors = model_result.design_matrix.n_regressors
    df_num = rank_C
    df_den = n_points - n_regressors

    # P-values
    p_values = f.sf(f_statistic, df_num, df_den)

    # For F-contrasts, we use F-statistic instead of t-statistic
    return ContrastResult(
        contrast_vector=C.flatten(),
        t_statistic=f_statistic,  # Store F-statistic here for compatibility
        effect_size=C_beta.flatten(),
        standard_error=np.sqrt(np.diag(C @ cov_beta @ C.T)),
        p_values=p_values,
    )


def generate_common_contrasts(
    design_matrix: "DesignMatrix", design_type: str = "categorical"
) -> list[Contrast]:
    """
    Generate common contrasts for standard experimental designs.

    Args:
        design_matrix: Design matrix specification
        design_type: Type of experimental design

    Returns:
        List of common contrasts for the design

    Example:
        >>> contrasts = generate_common_contrasts(design, "categorical")
        >>> for c in contrasts:
        ...     result = contrast(model, c)
    """
    names = design_matrix.names or []
    contrasts = []

    if design_type == "categorical":
        # Find categorical variables (assuming they don't include 'intercept')
        cat_vars = [name for name in names if name != "intercept"]

        if len(cat_vars) >= 2:
            # Pairwise comparisons
            for i in range(len(cat_vars)):
                for j in range(i + 1, len(cat_vars)):
                    # A > B
                    vec = np.zeros(len(names))
                    vec[names.index(cat_vars[i])] = 1
                    vec[names.index(cat_vars[j])] = -1
                    contrasts.append(Contrast(vec, f"{cat_vars[i]} > {cat_vars[j]}"))

                    # A < B
                    vec = np.zeros(len(names))
                    vec[names.index(cat_vars[i])] = -1
                    vec[names.index(cat_vars[j])] = 1
                    contrasts.append(Contrast(vec, f"{cat_vars[i]} < {cat_vars[j]}"))

    elif design_type == "trend":
        # Linear trend contrasts
        if len(names) > 2:
            # Assume ordered conditions
            n_conditions = len(names) - 1  # Excluding intercept
            coeffs = np.linspace(-1, 1, n_conditions)

            vec = np.zeros(len(names))
            for i, name in enumerate(names):
                if name != "intercept":
                    condition_idx = int(name.split("_")[-1]) if "_" in name else 0
                    vec[i] = coeffs[min(condition_idx, len(coeffs) - 1)]

            contrasts.append(Contrast(vec, "linear_trend"))

    return contrasts
