"""
Cascadian Agricultural Mortgage & Debt Module

Provides analysis of agricultural mortgage and debt levels, including
loan-to-value ratios and total debt estimation per land parcel.
"""

try:
    from .geo_infer_mortgage_debt import GeoInferMortgageDebt
except ImportError:
    GeoInferMortgageDebt = None

__all__ = ["GeoInferMortgageDebt"]
