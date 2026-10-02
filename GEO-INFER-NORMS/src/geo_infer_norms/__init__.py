"""
GEO-INFER-NORMS: Social-technical compliance modeling with deterministic and probabilistic aspects.

This module provides tools and frameworks for understanding, modeling, and analyzing social norms,
regulatory frameworks, and compliance requirements in spatial contexts.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Team"
__email__ = "blanket@activeinference.institute"

from .core import (
    legal_frameworks,
    zoning_analysis,
    compliance_tracking,
    policy_impact,
    normative_inference,
)

from .models import legal_entity, regulation, compliance_status, zoning, policy

__all__ = [
    "legal_frameworks",
    "zoning_analysis",
    "compliance_tracking",
    "policy_impact",
    "normative_inference",
    "legal_entity",
    "regulation",
    "compliance_status",
    "zoning",
    "policy",
]
