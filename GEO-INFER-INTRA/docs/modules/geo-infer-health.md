--
- module: GEO-INFER-HEALTH type: Domain-Specific Module category: Health Systems status: Beta maturity: High framework_version: 1.0.0 dependencies:
- GEO-INFER-DATA - GEO-INFER-SPACE - GEO-INFER-TIME - GEO-INFER-AI optional_dependencies:
- GEO-INFER-RISK - GEO-INFER-BIO - GEO-INFER-SPM license: CC BY-ND-SA 4.0 maintainer: GEO-INFER Framework Team contact: https://github.com/ActiveInferenceInstitute/GEO_INFER/issues last_updated: 2025-01-19 --
-

# GEO-INFER-HEALTH: Geospatial Health Analytics

> **Illustrative example notice.** This page contains historical or
> conceptual integration sketches. Names such as `SpatialAnalyzer` and
> domain-specific facade classes are not public GEO-INFER exports in the
> current checkout; verify imports against each module's `src/` package
> and use the module README/tests for executable examples.


> **Explanation**: geospatial health analytics framework implementing Active Inference principles for public health surveillance, disease modeling, and healthcare accessibility analysis.

## 🎯 What is GEO-INFER-HEALTH?
GEO-INFER-HEALTH is a geospatial health analytics framework that implements Active Inference principles for public health surveillance, epidemiological modeling, and healthcare accessibility analysis. The module provides tools for spatial health analysis with probabilistic reasoning and uncertainty quantification.

### Key Capabilities
- **Active Inference Disease Surveillance**

- Probabilistic outbreak detection with belief updating - **Healthcare Accessibility**
- Multi-modal transport analysis with equity assessment - **Environmental Health Risk Assessment**
- Multi-pollutant exposure modeling - **Spatial Epidemiology Toolkit**
- Rigorous statistical analysis with uncertainty quantification - **Real-time Health Intelligence**
- Automated early warning systems - **Health Data Standards Integration**
- HL7 FHIR and OMOP CDM support

### Links
- Module README: [Module README](../../../GEO-INFER-HEALTH/README.md) - API Documentation: ../../GEO-INFER-HEALTH/docs/api_schema.yaml - Examples: ../../GEO-INFER-HEALTH/examples/ - Tests: ../../GEO-INFER-HEALTH/tests/ The module integrates seamlessly with the GEO-INFER framework, leveraging:

- **GEO-INFER-DATA**: Population demographics, environmental data, and health indicators - **GEO-INFER-SPACE**: spatial analysis, clustering, and accessibility modeling - **GEO-INFER-TIME**: Temporal trend analysis and forecasting - **GEO-INFER-AI**: Machine learning for predictive modeling and pattern recognition - **GEO-INFER-RISK**: Hazard assessment and vulnerability analysis - **GEO-INFER-SPM**: statistical methods for epidemiological analysis

### Core Architecture
The module follows a layered architecture:

```
 GEO-INFER-HEALTH/ ├── API Layer (FastAPI) - REST endpoints and data validation ├── Service Layer - Business logic and orchestration ├── Core Engines - Specialized analysis components ├── Data Models - Pydantic schemas and validation └── Utilities - Geospatial functions and configuration
```
 ### Key Concepts #### Active Inference Disease Surveillance Implements probabilistic reasoning for outbreak detection:
```
python from geo_infer_health.core.enhanced_disease_surveillance import ActiveInferenceDiseaseAnalyzer from geo_infer_health.models import DiseaseReport, Location # Create disease surveillance analyzer analyzer = ActiveInferenceDiseaseAnalyzer( reports=disease_reports, population_data=population_data ) # Perform Active Inference analysis results = analyzer.analyze_with_active_inference(time_window_days=7) # Access probabilistic results print(f"Disease Activity Belief: {results['belief_states']['disease_activity']:.3f}") print(f"Risk Level: {results['risk_assessment']['risk_level']}") print(f"Hotspots: {len(results['enhanced_hotspots'])}")
```
 #### Healthcare Accessibility Multi-modal accessibility analysis with equity considerations:
```python
 from geo_infer_health.core.healthcare_accessibility import HealthcareAccessibilityAnalyzer from geo_infer_health.models import HealthFacility # Create accessibility analyzer analyzer = HealthcareAccessibilityAnalyzer( facilities=healthcare_facilities, population_data=population_data ) # Find nearest facility with service filtering nearest = analyzer.get_nearest_facility( loc=target_location, required_services=["Emergency"] ) # Calculate facility-to-population ratios ratios = analyzer.calculate_facility_to_population_ratio( area_id="study_area" )
```
 #### Environmental Health Risk Assessment Multi-pollutant exposure modeling with temporal analysis:
```python
 from geo_infer_health.core.environmental_health import EnvironmentalHealthAnalyzer from geo_infer_health.models import EnvironmentalData # Create environmental analyzer analyzer = EnvironmentalHealthAnalyzer(environmental_readings=readings) # Calculate average exposure exposure = analyzer.calculate_average_exposure( target_locations=target_locations, radius_km=2.0, parameter_name="PM2.5", time_window_days=7 ) # Get readings near location with time filtering nearby_readings = analyzer.get_environmental_readings_near_location( center_loc=location, radius_km=5.0, parameter_name="PM2.5", start_time=datetime.now() - timedelta(hours=24) )
```
 ## 📚 Core Features ### 1. Active Inference Disease Surveillance Engine **Purpose**: disease surveillance using probabilistic reasoning and belief updating.
```python
 from geo_infer_health.core.enhanced_disease_surveillance import ActiveInferenceDiseaseAnalyzer # Initialize Active Inference analyzer analyzer = ActiveInferenceDiseaseAnalyzer( reports=disease_reports, population_data=population_data ) # Configure Active Inference parameters analyzer.precision_parameter = 1.0 analyzer.learning_rate = 0.01 analyzer.free_energy_threshold = 0.1 # Perform analysis results = analyzer.analyze_with_active_inference(time_window_days=14) # Access probabilistic outputs beliefs = results['belief_states'] print(f"Disease activity belief: {beliefs['disease_activity']:.3f}") print(f"Transmission belief: {beliefs['transmission_rate']:.3f}") print(f"Spatial clustering belief: {beliefs['spatial_clustering']:.3f}")
```
 ### 2. Healthcare Accessibility Engine **Purpose**: Multi-modal healthcare accessibility analysis with equity assessment.
```
python from geo_infer_health.core.healthcare_accessibility import HealthcareAccessibilityAnalyzer # Initialize accessibility analyzer analyzer = HealthcareAccessibilityAnalyzer( facilities=healthcare_facilities, population_data=population_data ) # Perform accessibility analysis nearby_facilities = analyzer.find_facilities_in_radius( center_loc=target_location, radius_km=10.0, facility_type="Hospital", required_services=["Emergency", "Surgery"] ) # Get nearest facility with distance nearest = analyzer.get_nearest_facility( loc=target_location, facility_type="Hospital", required_services=["Emergency"] ) if nearest: facility, distance = nearest print(f"Nearest emergency hospital: {facility.name} ({distance:.1f} km)")
```
 ### 3. Environmental Health Risk Assessment Engine **Purpose**: Multi-pollutant exposure assessment with temporal analysis and health impact quantification.
```
python from geo_infer_health.core.environmental_health import EnvironmentalHealthAnalyzer # Initialize environmental analyzer analyzer = EnvironmentalHealthAnalyzer(environmental_readings=readings) # Calculate exposure for multiple pollutants pollutants = ["PM2.5", "NO2", "O3"] for pollutant in pollutants: exposure = analyzer.calculate_average_exposure( target_locations=target_locations, radius_km=5.0, parameter_name=pollutant, time_window_days=30 ) avg_exposure = sum(exposure.values()) / len(exposure) print(f"Average {pollutant} exposure: {avg_exposure:.2f}") # Get temporal readings recent_readings = analyzer.get_environmental_readings_near_location( center_loc=location, radius_km=10.0, parameter_name="PM2.5", start_time=datetime.now() - timedelta(hours=24) )
```
 ### 4. Geospatial Analysis Utilities **Purpose**: geospatial analysis tools for health applications.
```
python from geo_infer_health.utils.advanced_geospatial import ( spatial_clustering, calculate_spatial_statistics, calculate_spatial_autocorrelation, validate_geographic_bounds ) # Perform spatial clustering analysis locations = [report.location for report in disease_reports] clusters = spatial_clustering(locations, eps_km=1.0, min_samples=3) print(f"Identified {len(clusters)} spatial clusters") # Calculate spatial statistics stats = calculate_spatial_statistics(locations) print(f"Mean distance from centroid: {stats['mean_distance_from_centroid']:.3f} km") # Validate geographic bounds validation = validate_geographic_bounds(locations) if not validation['valid']: print("Geographic bounds validation failed:") for issue in validation['invalid_locations']: print(f" Location {issue['index']}: {issue['issues']}")
```
 ### 5. Configuration and Logging System **Purpose**: configuration management and logging utilities.
```
python from geo_infer_health.utils.config import load_config, HealthConfig from geo_infer_health.utils.logging import setup_logging, get_logger # Load configuration config = load_config("config/health_config.yaml") # Setup logging with configuration setup_logging( level=config.logging.get('level', 'INFO'), file_path=config.logging.get('file', {}).get('path') ) logger = get_logger("health_analysis") logger.info("Health analysis initialized") # Access configuration values api_host = config.api['host'] analysis_params = config.analysis['disease_surveillance'] logger.info(f"API configured for {api_host}")
```
 ## 🔧 API Reference ### ActiveInferenceDiseaseAnalyzer Core class for Active Inference-based disease surveillance.
```
python class ActiveInferenceDiseaseAnalyzer: def __init__(self, reports: list[DiseaseReport], population_data: list[PopulationData] | None = None): """ Initialize Active Inference disease analyzer. Args: reports: List of disease reports population_data: Optional population data for analysis """ def analyze_with_active_inference(self, time_window_days: int | None = None) -> dict[str, Any]: """ Perform Active Inference analysis. Args: time_window_days: Optional time window for analysis Returns: Dictionary containing analysis results with belief states, hotspots, predictions, and recommendations """ def get_cases_in_radius(self, center_loc: Location, radius_km: float) -> list[DiseaseReport]: """Get disease reports within radius of location.""" def calculate_local_incidence_rate(self, center_loc: Location, radius_km: float, time_window_days: int | None = None) -> tuple[float, int, int]: """Calculate local incidence rate with confidence intervals.""" def identify_simple_hotspots(self, threshold_case_count: int = 5, scan_radius_km: float = 1.0) -> list[Dict]: """Identify disease hotspots using traditional methods."""
```
 ### HealthcareAccessibilityAnalyzer Engine for healthcare accessibility analysis.
```
python class HealthcareAccessibilityAnalyzer: def __init__(self, facilities: list[HealthFacility], population_data: list[PopulationData] | None = None): """ Initialize healthcare accessibility analyzer. Args: facilities: List of healthcare facilities population_data: Optional population data """ def find_facilities_in_radius(self, center_loc: Location, radius_km: float, facility_type: str | None = None, required_services: list[str] | None = None) -> list[HealthFacility]: """Find facilities within radius with optional filtering.""" def get_nearest_facility(self, loc: Location, facility_type: str | None = None, required_services: list[str] | None = None) -> tuple[HealthFacility, float] | None: """Find nearest facility with optional filtering.""" def calculate_facility_to_population_ratio(self, area_id: str, facility_type: str | None = None) -> dict[str, Any] | None: """Calculate facility-to-population ratios."""
```
 ### EnvironmentalHealthAnalyzer Engine for environmental health risk assessment.
```
python class EnvironmentalHealthAnalyzer: def __init__(self, environmental_readings: list[EnvironmentalData]): """ Initialize environmental health analyzer. Args: environmental_readings: List of environmental data readings """ def get_environmental_readings_near_location(self, center_loc: Location, radius_km: float, parameter_name: str | None = None, start_time: datetime | None = None, end_time: datetime | None = None) -> list[EnvironmentalData]: """Get environmental readings near location with temporal filtering.""" def calculate_average_exposure(self, target_locations: list[Location], radius_km: float, parameter_name: str, time_window_days: int) -> dict[str, float | None]: """Calculate average environmental exposure for locations."""
```
 ### Configuration Management Configuration and logging utilities.
```
python from geo_infer_health.utils.config import load_config, HealthConfig, validate_config from geo_infer_health.utils.logging import setup_logging, get_logger, PerformanceLogger # Configuration management config = load_config("config/health_config.yaml") # Load YAML/JSON config validated_config = validate_config(config_dict) # Validate configuration # Logging setup setup_logging(level="INFO", file_path="logs/health.log") logger = get_logger("health_module") # Performance monitoring with PerformanceLogger("analysis_operation"): # Your analysis code here results = perform_analysis()
```
 ## 🎯 Use Cases ### 1. Disease Outbreak Modeling **Problem**: Model and predict disease outbreaks for public health planning. **Solution**: Use epidemiological modeling framework.
```
python from geo_infer_health import DiseaseOutbreakFramework # Initialize disease outbreak framework outbreak_model = DiseaseOutbreakFramework() # Define outbreak modeling parameters outbreak_config = outbreak_model.configure_outbreak_modeling({ 'transmission_modeling': 'comprehensive', 'population_mobility': 'spatial', 'environmental_factors': 'detailed', 'intervention_effects': 'modeled', 'prediction_accuracy': 'high' }) # Model disease outbreaks outbreak_result = outbreak_model.model_disease_outbreaks( outbreak_system=disease_system, outbreak_config=outbreak_config, disease_data=disease_characteristics )
```
 ### 2. Healthcare Resource Planning **Problem**: Optimize healthcare facility locations and resource allocation. **Solution**: Use healthcare resource optimization framework.
```
python from geo_infer_health.healthcare import HealthcarePlanningFramework # Initialize healthcare planning framework healthcare_planning = HealthcarePlanningFramework() # Define healthcare planning parameters planning_config = healthcare_planning.configure_healthcare_planning({ 'facility_location': 'optimal', 'resource_allocation': 'efficient', 'capacity_planning': 'strategic', 'accessibility_analysis': 'spatial', 'cost_optimization': 'comprehensive' }) # Plan healthcare resources planning_result = healthcare_planning.plan_healthcare_resources( healthcare_system=healthcare_system, planning_config=planning_config, demand_data=healthcare_demand )
```
 ### 3. Environmental Health Impact Assessment **Problem**: Assess environmental impacts on public health. **Solution**: Use health impact assessment framework.
```
python from geo_infer_health.environmental import EnvironmentalHealthFramework # Initialize environmental health framework env_health = EnvironmentalHealthFramework() # Define environmental health parameters env_config = env_health.configure_environmental_health({ 'pollution_impacts': 'detailed', 'climate_health_effects': 'comprehensive', 'vulnerability_mapping': 'spatial', 'equity_analysis': 'inclusive', 'mitigation_strategies': True }) # Assess environmental health impacts env_result = env_health.assess_environmental_health_impacts( environmental_system=environmental_system, env_config=env_config, health_data=health_indicators )
```
 ## 🔗 Integration with Other Modules ### GEO-INFER-SPACE Integration
```
python from geo_infer_health import HealthFramework from geo_infer_space import SpatialAnalysisEngine # Combine health systems with spatial analysis health_framework = HealthFramework(health_parameters) spatial_engine = SpatialAnalysisEngine() # Integrate health systems with spatial analysis spatial_health_system = health_framework.integrate_with_spatial_analysis( spatial_engine=spatial_engine, health_config=health_config )
```
 ### GEO-INFER-TIME Integration
```
python from geo_infer_health import TemporalHealthEngine from geo_infer_time import TemporalAnalysisEngine # Combine health systems with temporal analysis temporal_health_engine = TemporalHealthEngine() temporal_engine = TemporalAnalysisEngine() # Integrate health systems with temporal analysis temporal_health_system = temporal_health_engine.integrate_with_temporal_analysis( temporal_engine=temporal_engine, temporal_config=temporal_config )
```
 ### GEO-INFER-BIO Integration
```
python from geo_infer_health import BiologicalHealthEngine from geo_infer_bio import BiologicalFramework # Combine health systems with biological analysis bio_health_engine = BiologicalHealthEngine() bio_framework = BiologicalFramework() # Integrate health systems with biological analysis bio_health_system = bio_health_engine.integrate_with_biological_analysis( bio_framework=bio_framework, biological_config=biological_config )
```
 ## 🚨 Troubleshooting ### Common Issues **Epidemiological modeling problems:**
```
python # Improve epidemiological modeling epi_engine.configure_epidemiological_modeling({ 'disease_transmission': 'comprehensive', 'population_mobility': 'detailed', 'environmental_factors': 'advanced', 'intervention_effects': 'modeled', 'outbreak_prediction': 'accurate' }) # Add epidemiological modeling diagnostics epi_engine.enable_epidemiological_modeling_diagnostics( diagnostics=['transmission_accuracy', 'prediction_validation', 'model_uncertainty'] )
```
 **Public health surveillance issues:**
```
python # Improve public health surveillance surveillance_engine.configure_public_health_surveillance({ 'disease_monitoring': 'comprehensive', 'outbreak_detection': 'real_time', 'case_tracking': 'detailed', 'contact_tracing': 'automated', 'risk_assessment': 'prioritized' }) # Enable surveillance monitoring surveillance_engine.enable_surveillance_monitoring( monitoring=['disease_trends', 'outbreak_alerts', 'case_patterns'] )
```
 **Healthcare resource optimization issues:**
```
python # Improve healthcare resource optimization healthcare_engine.configure_healthcare_resources({ 'facility_location': 'optimal', 'resource_allocation': 'efficient', 'capacity_planning': 'strategic', 'accessibility_analysis': 'spatial', 'cost_optimization': 'comprehensive' }) # Enable healthcare monitoring healthcare_engine.enable_healthcare_monitoring( monitoring=['facility_utilization', 'resource_efficiency', 'accessibility_metrics'] )
```
 ## 📊 Performance Optimization ### Efficient Health Processing
```
python # Enable parallel health processing health_framework.enable_parallel_processing(n_workers=8) # Enable health caching health_framework.enable_health_caching( cache_size=10000, cache_ttl=1800 ) # Enable adaptive health systems health_framework.enable_adaptive_health_systems( adaptation_rate=0.1, adaptation_threshold=0.05 )
```
 ### Epidemiological Forecasting Optimization
```
python # Enable efficient epidemiological forecasting epi_engine.enable_efficient_epidemiological_forecasting( forecasting_strategy='ensemble_models', outbreak_prediction=True, uncertainty_quantification=True ) # Enable health intelligence epi_engine.enable_health_intelligence( intelligence_sources=['disease_data', 'population_mobility', 'environmental_factors'], update_frequency='real_time' )
```
 ## 🔗 Related Documentation ### Tutorials - **[Health Systems Basics](geo-infer-health.md)** - Learn health systems fundamentals - **[Epidemiological Modeling Tutorial](../getting_started/index.md)** - Build your first epidemiological model ### How-to Guides - **[Disease Outbreak Modeling](../examples_gallery.md)** - Implement disease outbreak modeling - **[Healthcare Resource Planning](../examples_gallery.md)** - Plan healthcare resource optimization ### Technical Reference - **[Health Systems API Reference](../api/reference.md)** - health systems API documentation - **[Epidemiological Modeling Patterns](../api/reference.md)** - Epidemiological modeling patterns and best practices ### Explanations - **Health Systems Theory** - Deep dive into health concepts - **Epidemiological Principles** - Understanding epidemiological foundations ### Related Modules - **[GEO-INFER-SPACE](../modules/geo-infer-space.md)** - Spatial analysis capabilities - **[GEO-INFER-TIME](../modules/geo-infer-time.md)** - Temporal analysis capabilities - **[GEO-INFER-BIO](../modules/geo-infer-bio.md)** - Biological analysis capabilities - **[GEO-INFER-DATA](../modules/geo-infer-data.md)** - Data management capabilities --- **Ready to get started?** Check out the **[Health Systems Basics Tutorial](geo-infer-health.md)** or explore **[Disease Outbreak Modeling Examples](../examples_gallery.md)**!

## 🗺️ Interactive Spatial Preview

Pre-rendered spatial snapshot for **GEO-INFER-HEALTH** (*Geospatial Health Analytics*). Reproducible preview cards are generated by `geo_infer_intra.core.documentation.visual_preview`.

| Preview | Widget |
| --- | --- |
| ![GEO-INFER-HEALTH Leaflet Preview](previews/geo-infer-health_preview.svg) | [Interactive map](previews/geo-infer-health_preview.html) · [PNG](previews/geo-infer-health_preview.png) |

> **Reproducible contract:** each map ships as `geo-infer-health_preview.html`, `geo-infer-health_preview.svg`, `geo-infer-health_preview.png`, and `geo-infer-health_preview.manifest.json` beneath `previews/`. The receipt records geometry provenance and artifact SHA-256 hashes. Values are illustrative, not observations.