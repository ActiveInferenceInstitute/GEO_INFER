## GEO-INFER-TRANSPORT — Transportation Planning and Traffic Analysis

GEO-INFER-TRANSPORT provides transportation planning and traffic analysis for geospatial systems (per its `README.md`). Its package `geo_infer_transport` contains `core/` and `__init__.py` for transport modeling and analysis.

The public interface, verified from `__init__.py`, exports transport classes: `TransportNetwork` for representing network topology (nodes, edges, modes) that the other engines operate on; `RoutingEngine` for path computation across that network; `TrafficAnalyzer` for flow and congestion analysis; `AccessibilityAnalyzer` for measuring how reachable destinations are from locations — a canonical geospatial-transport question; and `TransitOptimizer` for improving public-transit service configurations.

The unit and scenario tests exercise transport engine behavior and its geographic inputs. These fixtures do not establish traffic forecast accuracy or network capacity under field conditions.

Under the root README's Module Themes, TRANSPORT belongs to Spatial & Place-based together with SPACE, PLACE, TIME, MARINE, and WATER. Its role there is the mobility layer: PLACE analyzes what a region contains, SPACE indexes where things are, and TRANSPORT models how people and goods move between them. Its network structures also provide the corridors along which other modules — RISK's exposure models, TIME's event detection — can be spatially conditioned.
