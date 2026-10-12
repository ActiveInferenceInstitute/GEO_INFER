"""
Spatial Ecology Module

Provides landscape economics analysis, habitat connectivity assessment,
ecosystem network analysis, conservation prioritization using Zonation-like
scoring, and restoration cost-benefit analysis.
"""

from typing import Any
import numpy as np
import pandas as pd
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)


class LandscapeEconomics:
    """Landscape economics analysis using a patch-matrix model.

    Quantifies the economic value of landscape heterogeneity, edge effects,
    and patch size–value relationships following principles from landscape
    ecology and environmental economics.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize landscape economics.

        Args:
            config: Optional overrides. Supported keys:
                - ``per_ha_values`` (dict): Per-hectare values by land cover type.
                - ``edge_bonus_factor`` (float): Value multiplier for edge habitat (default 1.15).
        """
        self.config = config or {}
        self.per_ha_values: dict[str, float] = self.config.get(
            "per_ha_values",
            {
                "forest": 3800.0,
                "wetland": 6500.0,
                "grassland": 1200.0,
                "cropland": 2800.0,
                "urban_green": 1600.0,
            },
        )
        self.edge_bonus = float(self.config.get("edge_bonus_factor", 1.15))
        logger.info("LandscapeEconomics initialized")

    def analyze_landscape(self, landscape_data: gpd.GeoDataFrame) -> dict[str, Any]:
        """Analyze landscape economics from a patch GeoDataFrame.

        Expects columns: ``land_cover`` (str), and geometry must be polygon/multi-polygon.
        Optionally ``condition`` (float 0-1).

        Returns:
            Dict with patch_count, total_area_ha, total_value, per_cover breakdown,
            landscape_diversity (Shannon index), and mean_patch_size_ha.
        """
        logger.info("Analyzing landscape economics for %d patches", len(landscape_data))
        gdf = landscape_data.copy()

        # Compute areas in hectares
        if gdf.crs and not gdf.crs.is_projected:
            gdf_proj = gdf.to_crs(epsg=3857)
        else:
            gdf_proj = gdf
        gdf["area_ha"] = gdf_proj.geometry.area / 1e4  # m² → ha

        condition = (
            gdf["condition"].values if "condition" in gdf.columns else np.ones(len(gdf))
        )
        land_cover = (
            gdf["land_cover"].values
            if "land_cover" in gdf.columns
            else ["forest"] * len(gdf)
        )

        # Per-patch value
        values = []
        for i, lc in enumerate(land_cover):
            unit = self.per_ha_values.get(str(lc).lower(), 1000.0)
            values.append(gdf["area_ha"].iloc[i] * unit * float(condition[i]))
        gdf["value_usd"] = [round(v, 2) for v in values]

        # Aggregate by cover
        cover_summary = {}
        for lc in set(land_cover):
            mask = np.array([str(c).lower() == str(lc).lower() for c in land_cover])
            cover_summary[str(lc).lower()] = {
                "patch_count": int(mask.sum()),
                "total_area_ha": round(float(gdf.loc[mask, "area_ha"].sum()), 2),
                "total_value_usd": round(float(gdf.loc[mask, "value_usd"].sum()), 2),
            }

        # Shannon diversity index
        proportions = np.array([cs["total_area_ha"] for cs in cover_summary.values()])
        proportions = proportions / (proportions.sum() + 1e-10)
        shannon = -float(np.sum(proportions * np.log(proportions + 1e-10)))

        result = {
            "patch_count": len(gdf),
            "total_area_ha": round(float(gdf["area_ha"].sum()), 2),
            "total_value_usd": round(float(gdf["value_usd"].sum()), 2),
            "mean_patch_size_ha": round(float(gdf["area_ha"].mean()), 2),
            "landscape_diversity_shannon": round(shannon, 4),
            "cover_summary": cover_summary,
        }
        logger.info(
            "Landscape value: $%.2f across %.2f ha",
            result["total_value_usd"],
            result["total_area_ha"],
        )
        return result


class HabitatConnectivity:
    """Habitat connectivity analysis using graph-theoretic metrics.

    Computes connectivity indices between habitat patches based on
    inter-patch distances and patch quality, following the Integral
    Index of Connectivity (IIC) approach.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.dispersal_distance_m = float(
            self.config.get("dispersal_distance_m", 5000.0)
        )
        logger.info(
            "HabitatConnectivity initialized: dispersal=%.0f m",
            self.dispersal_distance_m,
        )

    def analyze_connectivity(self, habitat_data: gpd.GeoDataFrame) -> dict[str, Any]:
        """Analyze habitat connectivity from patch GeoDataFrame.

        Expects polygon geometries. Optionally ``habitat_quality`` (0-1) column.

        Returns:
            Dict with integral_index_of_connectivity, n_components, mean_distance_m,
            connectivity_matrix (nested list), and isolated_patches list.
        """
        logger.info("Analyzing connectivity for %d patches", len(habitat_data))
        gdf = habitat_data.copy()
        n = len(gdf)

        # Ensure projected CRS for distance calculations
        if gdf.crs and not gdf.crs.is_projected:
            gdf = gdf.to_crs(epsg=3857)

        centroids = gdf.geometry.centroid
        quality = (
            gdf["habitat_quality"].values
            if "habitat_quality" in gdf.columns
            else np.ones(n)
        )
        areas = gdf.geometry.area / 1e4  # ha

        # Build distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = centroids.iloc[i].distance(centroids.iloc[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Adjacency based on dispersal distance
        adj = (dist_matrix <= self.dispersal_distance_m) & (dist_matrix > 0)

        # Connected components via BFS
        visited = set()
        components: list[list[int]] = []
        for start in range(n):
            if start in visited:
                continue
            comp: list[int] = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                for nb in range(n):
                    if adj[node, nb] and nb not in visited:
                        queue.append(nb)
            components.append(comp)

        # Integral Index of Connectivity (IIC)
        total_area = float(areas.sum())
        iic_sum = 0.0
        for i in range(n):
            for j in range(n):
                # Shortest path length (simplified: 1 if connected, inf otherwise)
                same_comp = any(i in c and j in c for c in components)
                if same_comp:
                    nl = 1 if i != j else 0
                    iic_sum += (
                        areas.iloc[i] * quality[i] * areas.iloc[j] * quality[j]
                    ) / (1 + nl)
        iic = float(iic_sum / (total_area**2 + 1e-10))

        isolated = [int(c[0]) for c in components if len(c) == 1]
        mean_dist = (
            float(np.mean(dist_matrix[dist_matrix > 0]))
            if (dist_matrix > 0).any()
            else 0.0
        )

        result = {
            "integral_index_of_connectivity": round(iic, 6),
            "n_components": len(components),
            "n_isolated_patches": len(isolated),
            "mean_inter_patch_distance_m": round(mean_dist, 2),
            "isolated_patch_indices": isolated,
            "total_patches": n,
        }
        logger.info("IIC=%.6f, components=%d", iic, len(components))
        return result


class EcosystemNetworkAnalysis:
    """Ecosystem network analysis using trophic and functional flow networks.

    Models ecosystems as directed networks where nodes represent species
    groups or functional units and edges represent energy/material flows.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("EcosystemNetworkAnalysis initialized")

    def analyze_network(self, network_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze an ecosystem network.

        Args:
            network_data: Dict with:
                - ``nodes`` (list[dict]): Each with ``id``, ``name``, ``biomass`` (float).
                - ``edges`` (list[dict]): Each with ``source``, ``target``, ``flow`` (float).

        Returns:
            Dict with network metrics: n_nodes, n_edges, total_flow,
            connectance, mean_node_degree, network_efficiency, cycling_index.
        """
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])
        n_nodes = len(nodes)
        n_edges = len(edges)
        logger.info("Analyzing ecosystem network: %d nodes, %d edges", n_nodes, n_edges)

        if n_nodes == 0:
            return {"n_nodes": 0, "n_edges": 0, "total_flow": 0.0}

        # Build adjacency
        node_ids = [n["id"] for n in nodes]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        adj = np.zeros((n_nodes, n_nodes))
        total_flow = 0.0
        for e in edges:
            si = id_to_idx.get(e["source"])
            ti = id_to_idx.get(e["target"])
            if si is not None and ti is not None:
                flow = float(e.get("flow", 1.0))
                adj[si, ti] = flow
                total_flow += flow

        # Connectance = realized links / possible links
        max_links = n_nodes * (n_nodes - 1)
        connectance = n_edges / max(max_links, 1)

        # Degree distribution
        out_degree = (adj > 0).sum(axis=1)
        in_degree = (adj > 0).sum(axis=0)
        mean_degree = float((out_degree + in_degree).mean())

        # Cycling index (fraction of flow in cycles via diagonal of adj^n)
        cycling = 0.0
        if n_nodes <= 100:
            adj_norm = adj / (total_flow + 1e-10)
            power = adj_norm.copy()
            for _ in range(min(n_nodes, 10)):
                power = power @ adj_norm
                cycling += float(np.trace(power))
            cycling = min(1.0, cycling)

        # Network efficiency (harmonic mean of shortest path lengths on weighted graph)
        # Simplified: use inverse flow as distance
        flow_dist = np.where(adj > 0, 1.0 / adj, np.inf)
        np.fill_diagonal(flow_dist, 0)
        # Floyd-Warshall
        dist = flow_dist.copy()
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        inv_dist = np.where(dist > 0, 1.0 / dist, 0)
        np.fill_diagonal(inv_dist, 0)
        efficiency = float(inv_dist.sum() / max(n_nodes * (n_nodes - 1), 1))

        result = {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "total_flow": round(total_flow, 4),
            "connectance": round(connectance, 4),
            "mean_node_degree": round(mean_degree, 4),
            "network_efficiency": round(efficiency, 6),
            "cycling_index": round(cycling, 6),
        }
        logger.info(
            "Network: connectance=%.4f, efficiency=%.6f", connectance, efficiency
        )
        return result


class ConservationPrioritization:
    """Conservation prioritization using Zonation-inspired ranking.

    Ranks spatial units by conservation value considering species richness,
    threat level, cost, and complementarity with existing protected areas.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.weights = self.config.get(
            "weights",
            {
                "species_richness": 0.30,
                "threat_level": 0.25,
                "habitat_quality": 0.25,
                "cost_efficiency": 0.20,
            },
        )
        logger.info("ConservationPrioritization initialized")

    def prioritize_areas(self, conservation_data: dict[str, Any]) -> pd.DataFrame:
        """Prioritize conservation areas using multi-criteria scoring.

        Args:
            conservation_data: Dict with ``areas`` key — list of dicts each having:
                - ``name`` (str): Area identifier.
                - ``species_richness`` (float): Species count or relative richness.
                - ``threat_level`` (float, 0-1): Level of threat (higher = more threatened).
                - ``habitat_quality`` (float, 0-1): Habitat condition.
                - ``cost_per_ha`` (float): Protection cost per hectare.
                - ``area_ha`` (float): Area size.
                - ``existing_protection`` (float, 0-1): Fraction already protected.

        Returns:
            DataFrame ranked by priority_score (descending), with composite
            score breakdown.
        """
        areas = conservation_data.get("areas", [])
        logger.info("Prioritizing %d conservation areas", len(areas))

        if not areas:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for area in areas:
            sr = float(area.get("species_richness", 0))
            tl = float(area.get("threat_level", 0))
            hq = float(area.get("habitat_quality", 0))
            cost = float(area.get("cost_per_ha", 1.0))
            area_ha = float(area.get("area_ha", 1.0))
            existing = float(area.get("existing_protection", 0.0))

            # Normalize species richness to 0-1 (will be re-normalized below)
            # Cost efficiency: inverse cost, normalized
            cost_eff = 1.0 / (cost + 1e-6)

            rows.append(
                {
                    "name": area.get("name", "unnamed"),
                    "area_ha": area_ha,
                    "species_richness_raw": sr,
                    "threat_level": tl,
                    "habitat_quality": hq,
                    "cost_per_ha": cost,
                    "existing_protection": existing,
                    "cost_efficiency_raw": cost_eff,
                }
            )

        df = pd.DataFrame(rows)

        # Min-max normalize scores
        for col, norm_col in [
            ("species_richness_raw", "species_richness_norm"),
            ("cost_efficiency_raw", "cost_efficiency_norm"),
        ]:
            vals = df[col].values
            vmin, vmax = vals.min(), vals.max()
            df[norm_col] = (vals - vmin) / (vmax - vmin + 1e-10)

        # Composite score
        w = self.weights
        df["priority_score"] = (
            w.get("species_richness", 0.3) * df["species_richness_norm"]
            + w.get("threat_level", 0.25) * df["threat_level"]
            + w.get("habitat_quality", 0.25) * df["habitat_quality"]
            + w.get("cost_efficiency", 0.2) * df["cost_efficiency_norm"]
        )

        # Complementarity bonus: less-protected areas get a boost
        df["priority_score"] *= 1.0 + 0.2 * (1.0 - df["existing_protection"])
        df["priority_score"] = df["priority_score"].round(4)
        df["rank"] = (
            df["priority_score"].rank(ascending=False, method="min").astype(int)
        )
        df = df.sort_values("rank").reset_index(drop=True)

        # Clean up intermediate columns
        df = df.drop(
            columns=[
                "species_richness_raw",
                "cost_efficiency_raw",
                "species_richness_norm",
                "cost_efficiency_norm",
            ]
        )

        logger.info(
            "Top priority: %s (score=%.4f)",
            df.iloc[0]["name"],
            df.iloc[0]["priority_score"],
        )
        return df


class RestorationEconomics:
    """Restoration economics cost-benefit analysis.

    Estimates restoration costs, expected ecosystem service gains,
    and computes benefit-cost ratios and payback periods for
    ecological restoration projects.
    """

    # Typical restoration costs (USD/ha) by ecosystem type
    RESTORATION_COSTS: dict[str, float] = {
        "forest": 3500.0,
        "wetland": 8000.0,
        "grassland": 1500.0,
        "coastal": 12000.0,
        "freshwater": 6000.0,
        "degraded_farmland": 2500.0,
    }

    # Expected annual ecosystem service gains post-restoration (USD/ha/yr)
    SERVICE_GAINS: dict[str, float] = {
        "forest": 2800.0,
        "wetland": 5000.0,
        "grassland": 900.0,
        "coastal": 6500.0,
        "freshwater": 4000.0,
        "degraded_farmland": 1800.0,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.discount_rate = float(self.config.get("discount_rate", 0.04))
        self.time_horizon = int(self.config.get("time_horizon", 30))
        logger.info(
            "RestorationEconomics initialized: discount=%.2f, horizon=%d yr",
            self.discount_rate,
            self.time_horizon,
        )

    def analyze_restoration(self, restoration_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze restoration economics for proposed projects.

        Args:
            restoration_data: Dict with ``projects`` key — list of dicts each having:
                - ``name`` (str): Project identifier.
                - ``type`` (str): Ecosystem type to restore.
                - ``area_ha`` (float): Restoration area.
                - ``current_service_value`` (float, optional): Current ES value/ha/yr.
                - ``restoration_cost_per_ha`` (float, optional): Override default cost.
                - ``expected_recovery_years`` (int, optional): Years to full recovery.

        Returns:
            Dict with per-project economics and portfolio summary.
        """
        projects = restoration_data.get("projects", [])
        logger.info("Analyzing %d restoration projects", len(projects))

        project_results: list[dict[str, Any]] = []
        total_cost = 0.0
        total_npv_benefit = 0.0

        for proj in projects:
            p_type = proj.get("type", "forest").lower()
            area = float(proj.get("area_ha", 0.0))
            current_sv = float(proj.get("current_service_value", 0.0))
            cost_ha = float(
                proj.get(
                    "restoration_cost_per_ha",
                    self.RESTORATION_COSTS.get(p_type, 5000.0),
                )
            )
            recovery_years = int(proj.get("expected_recovery_years", 10))
            target_sv = self.SERVICE_GAINS.get(p_type, 2000.0)

            # Total upfront cost
            upfront = cost_ha * area

            # Annual benefit = gain in ES value, ramping up over recovery period
            annual_benefits = []
            for t in range(1, self.time_horizon + 1):
                ramp = min(1.0, t / max(recovery_years, 1))
                annual_gain = (target_sv * ramp - current_sv) * area
                annual_benefits.append(max(0.0, annual_gain))

            # NPV of benefits
            npv_benefit = sum(
                b / (1 + self.discount_rate) ** t
                for t, b in enumerate(annual_benefits, 1)
            )

            bcr = npv_benefit / (upfront + 1e-10)

            # Payback period
            cumulative = 0.0
            payback = self.time_horizon
            for t, b in enumerate(annual_benefits, 1):
                cumulative += b / (1 + self.discount_rate) ** t
                if cumulative >= upfront:
                    payback = t
                    break

            p_result = {
                "name": proj.get("name", "unnamed"),
                "type": p_type,
                "area_ha": area,
                "upfront_cost_usd": round(upfront, 2),
                "npv_benefits_usd": round(npv_benefit, 2),
                "benefit_cost_ratio": round(bcr, 3),
                "payback_period_years": payback,
                "annual_benefit_at_maturity_usd": round(
                    max(0, target_sv - current_sv) * area, 2
                ),
            }
            project_results.append(p_result)
            total_cost += upfront
            total_npv_benefit += npv_benefit

        result = {
            "n_projects": len(projects),
            "total_upfront_cost_usd": round(total_cost, 2),
            "total_npv_benefits_usd": round(total_npv_benefit, 2),
            "portfolio_bcr": round(total_npv_benefit / (total_cost + 1e-10), 3),
            "projects": project_results,
        }
        logger.info("Portfolio BCR: %.3f", result["portfolio_bcr"])
        return result
