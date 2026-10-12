#!/usr/bin/env python3
"""
GEO-INFER-MARINE Example: Marine Ecosystem Analysis

This example demonstrates comprehensive marine ecosystem analysis
including coral reef assessment, biodiversity analysis, and MPA planning.
"""

from geo_infer_marine import (
    MarineEcosystemModeler,
    MarineHabitatType,
    SpeciesData,
)
import xarray as xr


def main():
    print("=" * 60)
    print("GEO-INFER-MARINE: Marine Ecosystem Analysis")
    print("=" * 60)

    # 1. Initialize Ecosystem Modeler
    print("\n1. Initializing Marine Ecosystem Modeler...")

    modeler = MarineEcosystemModeler()
    print("   Ecosystem modeler ready")

    # 2. Register Species of Interest
    print("\n2. Registering Key Species...")

    species_list = [
        SpeciesData(
            species_id="clownfish",
            common_name="Clownfish",
            scientific_name="Amphiprion ocellaris",
            trophic_level=3.0,
            habitat_preference=[MarineHabitatType.CORAL_REEF],
            temperature_range=(24.0, 30.0),
            depth_range=(1.0, 15.0),
            conservation_status="LC",
        ),
        SpeciesData(
            species_id="bluefin_tuna",
            common_name="Pacific Bluefin Tuna",
            scientific_name="Thunnus orientalis",
            trophic_level=4.2,
            habitat_preference=[MarineHabitatType.OPEN_OCEAN],
            temperature_range=(14.0, 26.0),
            depth_range=(50.0, 500.0),
            conservation_status="VU",
        ),
        SpeciesData(
            species_id="sea_turtle",
            common_name="Green Sea Turtle",
            scientific_name="Chelonia mydas",
            trophic_level=2.0,
            habitat_preference=[
                MarineHabitatType.SEAGRASS,
                MarineHabitatType.CORAL_REEF,
            ],
            temperature_range=(20.0, 30.0),
            depth_range=(0.0, 40.0),
            conservation_status="EN",
        ),
    ]

    for species in species_list:
        modeler.register_species(species)
        print(f"   {species.common_name} ({species.conservation_status})")

    # 3. Coral Reef Health Assessment
    print("\n3. Assessing Coral Reef Health...")

    # Simulated temperature and pH data
    locations = ["site_1", "site_2", "site_3", "site_4"]
    temperature = xr.DataArray(
        [26.5, 28.0, 29.5, 31.0], dims=["location"], coords={"location": locations}
    )
    ph = xr.DataArray(
        [8.1, 8.0, 7.9, 7.7], dims=["location"], coords={"location": locations}
    )

    coral_health = modeler.assess_coral_reef_health(temperature, ph)

    print("   Site Assessment:")
    for i, loc in enumerate(locations):
        risk = float(coral_health["bleaching_risk"][i])
        stress = float(coral_health["acidification_stress"][i])
        status = (
            "✓ Healthy" if risk < 0.3 else "⚠ At Risk" if risk < 0.6 else "✗ Critical"
        )
        print(
            f"   {loc}: Bleaching risk={risk:.2f}, Acidification={stress:.2f} {status}"
        )

    # 4. Biodiversity Analysis
    print("\n4. Analyzing Reef Biodiversity...")

    species_counts = {
        "Amphiprion ocellaris": 150,
        "Chromis viridis": 500,
        "Acanthurus leucosternon": 80,
        "Pomacanthus imperator": 25,
        "Dascyllus trimaculatus": 200,
        "Halichoeres hortulanus": 120,
        "Chaetodon lunula": 45,
    }

    biodiversity = modeler.calculate_biodiversity_indices(species_counts, area_km2=0.5)

    print(f"   Species richness: {biodiversity['species_richness']}")
    print(f"   Shannon diversity: {biodiversity['shannon_diversity']:.3f}")
    print(f"   Simpson diversity: {biodiversity['simpson_diversity']:.3f}")
    print(f"   Evenness: {biodiversity['evenness']:.3f}")
    print(f"   Species density: {biodiversity['species_density']:.1f} species/km²")

    # 5. Marine Protected Area Planning
    print("\n5. Creating Marine Protected Area...")

    # Define MPA boundary (Great Barrier Reef-like area)
    mpa_boundary = [
        (145.0, -16.0),
        (147.0, -16.0),
        (147.5, -18.0),
        (146.5, -20.0),
        (145.0, -20.0),
        (144.5, -18.0),
    ]

    mpa = modeler.create_marine_protected_area(
        mpa_id="GBR_ZONE_A",
        name="Great Barrier Reef Protection Zone A",
        boundary=mpa_boundary,
        protection_level="full",
        target_species=["clownfish", "sea_turtle"],
    )

    print(f"   Name: {mpa['name']}")
    print(f"   Area: {mpa['area_km2']:,.0f} km²")
    print(f"   Protection level: {mpa['protection_level']}")

    # 6. MPA Effectiveness Assessment
    print("\n6. Assessing MPA Effectiveness...")

    # Survey data inside and outside MPA
    inside_counts = {
        "clownfish": 150,
        "sea_turtle": 28,
        "reef_shark": 15,
        "grouper": 45,
    }

    outside_counts = {
        "clownfish": 80,
        "sea_turtle": 12,
        "reef_shark": 5,
        "grouper": 20,
    }

    effectiveness = modeler.assess_mpa_effectiveness(
        mpa_id="GBR_ZONE_A",
        species_counts_inside=inside_counts,
        species_counts_outside=outside_counts,
        time_since_establishment_years=10.0,
    )

    print(f"   Abundance ratio: {effectiveness['abundance_ratio']:.2f}x")
    print(f"   Richness ratio: {effectiveness['richness_ratio']:.2f}")
    print(f"   Effectiveness score: {effectiveness['effectiveness_score']:.1f}/100")
    print(f"   Recommendation: {effectiveness['recommendation']}")

    # 7. Climate Change Impact Assessment
    print("\n7. Assessing Climate Change Impacts...")

    # RCP 4.5 scenario
    climate_impact = modeler.assess_climate_change_impact(
        temperature_change=1.5,
        sea_level_rise_cm=40,
        ph_change=-0.15,
        time_horizon_years=50,
    )

    print("   Coral Reef Impacts:")
    print(
        f"   - Bleaching risk: {climate_impact['coral_reef_impacts']['bleaching_risk']:.1%}"
    )
    print(
        f"   - Survival probability: {climate_impact['coral_reef_impacts']['survival_probability']:.1%}"
    )

    print("   Habitat Impacts:")
    print(
        f"   - Coastal habitat loss: {climate_impact['habitat_impacts']['coastal_habitat_loss_pct']:.1f}%"
    )

    print("   Fisheries Impacts:")
    print(
        f"   - Productivity change: {climate_impact['fisheries_impacts']['productivity_change_pct']:.1f}%"
    )

    print(f"   Overall Vulnerability: {climate_impact['overall_vulnerability']:.2f}")
    print(f"   Adaptation Priority: {climate_impact['adaptation_priority']}")

    # 8. Blue Carbon Estimation
    print("\n8. Estimating Blue Carbon Storage...")

    habitat_areas = {
        "mangrove": 150.0,
        "seagrass": 300.0,
        "salt_marsh": 50.0,
        "coral_reef": 200.0,
    }

    blue_carbon = modeler.estimate_blue_carbon(
        habitat_area_km2=habitat_areas, condition="healthy"
    )

    print(f"   Total habitat area: {blue_carbon['total_area_km2']:.0f} km²")
    print(
        f"   Annual CO2 storage: {blue_carbon['total_annual_storage_tonnes']:,.0f} tonnes"
    )
    print(f"   Annual carbon value: ${blue_carbon['carbon_value_usd_annual']:,.0f}")
    print(f"   30-year carbon value: ${blue_carbon['carbon_value_usd_30yr']:,.0f}")

    print("\n   Storage by habitat:")
    for habitat, data in blue_carbon["storage_by_habitat"].items():
        print(f"   - {habitat}: {data['annual_storage_tonnes']:,.0f} tonnes/year")

    # 9. Species Distribution Modeling
    print("\n9. Modeling Species Distribution...")

    # Create environmental grids
    temp_grid = xr.DataArray(
        data=[[25, 26, 27, 28], [26, 27, 28, 29], [27, 28, 29, 30]], dims=["lat", "lon"]
    )
    depth_grid = xr.DataArray(
        data=[[5, 10, 15, 20], [8, 12, 18, 25], [10, 15, 22, 30]], dims=["lat", "lon"]
    )

    distribution = modeler.model_species_distribution(
        species_id="clownfish", temperature=temp_grid, depth=depth_grid
    )

    avg_suitability = float(distribution["suitability"].mean())
    avg_probability = float(distribution["occurrence_probability"].mean())

    print("   Clownfish Distribution Analysis:")
    print(f"   - Average suitability: {avg_suitability:.3f}")
    print(f"   - Average occurrence probability: {avg_probability:.3f}")

    print("\n" + "=" * 60)
    print("Marine Ecosystem Analysis Complete!")
    print("=" * 60)

    # Summary
    print("\nKey Findings:")
    print(
        f"  - Coral reef: {sum(1 for r in coral_health['bleaching_risk'].values if float(r) < 0.5)}/{len(locations)} sites healthy"
    )
    print(f"  - Biodiversity (Shannon): {biodiversity['shannon_diversity']:.2f}")
    print(f"  - MPA effectiveness: {effectiveness['effectiveness_score']:.0f}/100")
    print(f"  - Climate vulnerability: {climate_impact['overall_vulnerability']:.1%}")
    print(
        f"  - Blue carbon: {blue_carbon['total_annual_storage_tonnes']:,.0f} tonnes CO2/year"
    )


if __name__ == "__main__":
    main()
