# GEO-INFER documentation index

This is the top-level, modular index for the GEO-INFER repository. It points at
the per-module documentation instead of duplicating it: every module owns its
`README.md` signpost, its `GEO-INFER-*/docs/` tree, and its entry in the
central [documentation hub](../GEO-INFER-INTRA/docs/index.md).

> The [INTRA documentation hub](../GEO-INFER-INTRA/docs/index.md) is the
> authoritative curated guide (concepts, tutorials, integration, releases).
> This page is only a navigation index across modules. If a page here conflicts
> with importable code or a passing contract validator, treat the source and
> validator as authoritative.

## Repository guidance

- [Repository overview and quick start](../README.md)
- [Agent operating contract](../AGENTS.md)
- [Open task and backlog ledger](../TODO.md)
- [Contribution rules](../CONTRIBUTING.md)
- [Security reporting](../SECURITY.md)
- [Release history](../CHANGELOG.md)

## Central documentation hub

- [Documentation hub](../GEO-INFER-INTRA/docs/index.md)
- [Installation and first workflow](../GEO-INFER-INTRA/docs/getting_started/index.md)
- [Framework architecture](../GEO-INFER-INTRA/docs/overview.md)
- [Active Inference reference](../GEO-INFER-INTRA/docs/active_inference_guide.md)
- [Bayesian inference reference](../GEO-INFER-INTRA/docs/bayesian_inference_guide.md)
- [Module catalog](../GEO-INFER-INTRA/docs/modules/index.md)
- [Developer workflow](../GEO-INFER-INTRA/docs/developer_guide/index.md)
- [Integration guide](../GEO-INFER-INTRA/docs/integration/index.md)
- [Spatial / H3 reference](../GEO-INFER-INTRA/docs/geospatial/data_formats/h3/index.md)
- [Test and release gates](../GEO-INFER-TEST/docs/index.md)
- [Local gate arms and measured wall time](../GEO-INFER-TEST/docs/gate.md)
- [Manuscript generation and evidence](../manuscript/README.md)

## Modules by theme

Each row links the module's generated signpost and its page in the central
module catalog. Modules with a `docs/` tree also expose it from their own
`README.md`.

### Spatial & place-based analysis

| Module | Package | Documentation |
| --- | --- | --- |
| `GEO-INFER-SPACE` | `geo_infer_space` | [README](../GEO-INFER-SPACE/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-space.md) |
| `GEO-INFER-PLACE` | `geo_infer_place` | [README](../GEO-INFER-PLACE/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-place.md) |
| `GEO-INFER-TIME` | `geo_infer_time` | [README](../GEO-INFER-TIME/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-time.md) |
| `GEO-INFER-MARINE` | `geo_infer_marine` | [README](../GEO-INFER-MARINE/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-marine.md) |
| `GEO-INFER-WATER` | `geo_infer_water` | [README](../GEO-INFER-WATER/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-water.md) |
| `GEO-INFER-FOREST` | `geo_infer_forest` | [README](../GEO-INFER-FOREST/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-forest.md) |
| `GEO-INFER-CLIMATE` | `geo_infer_climate` | [README](../GEO-INFER-CLIMATE/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-climate.md) |
| `GEO-INFER-ENERGY` | `geo_infer_energy` | [README](../GEO-INFER-ENERGY/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-energy.md) |
| `GEO-INFER-TRANSPORT` | `geo_infer_transport` | [README](../GEO-INFER-TRANSPORT/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-transport.md) |
| `GEO-INFER-EMERGENCY` | `geo_infer_emergency` | [README](../GEO-INFER-EMERGENCY/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-emergency.md) |

### Bayesian & Active Inference

| Module | Package | Documentation |
| --- | --- | --- |
| `GEO-INFER-BAYES` | `geo_infer_bayes` | [README](../GEO-INFER-BAYES/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-bayes.md) |
| `GEO-INFER-SIM` | `geo_infer_sim` | [README](../GEO-INFER-SIM/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-sim.md) |
| `GEO-INFER-SPM` | `geo_infer_spm` | [README](../GEO-INFER-SPM/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-spm.md) |
| `GEO-INFER-COG` | `geo_infer_cog` | [README](../GEO-INFER-COG/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-cog.md) |
| `GEO-INFER-ACT` | `geo_infer_act` | [README](../GEO-INFER-ACT/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-act.md) |
| `GEO-INFER-MATH` | `geo_infer_math` | [README](../GEO-INFER-MATH/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-math.md) |

### Agents & AI orchestration

| Module | Package | Documentation |
| --- | --- | --- |
| `GEO-INFER-AGENT` | `geo_infer_agent` | [README](../GEO-INFER-AGENT/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-agent.md) |
| `GEO-INFER-AG` | `geo_infer_ag` | [README](../GEO-INFER-AG/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-ag.md) |
| `GEO-INFER-AI` | `geo_infer_ai` | [README](../GEO-INFER-AI/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-ai.md) |
| `GEO-INFER-ANT` | `geo_infer_ant` | [README](../GEO-INFER-ANT/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-ant.md) |
| `GEO-INFER-OPS` | `geo_infer_ops` | [README](../GEO-INFER-OPS/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-ops.md) |
| `GEO-INFER-COMMS` | `geo_infer_comms` | [README](../GEO-INFER-COMMS/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-comms.md) |

### Governance, risk & domain modeling

| Module | Package | Documentation |
| --- | --- | --- |
| `GEO-INFER-RISK` | `geo_infer_risk` | [README](../GEO-INFER-RISK/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-risk.md) |
| `GEO-INFER-INSURANCE` | `geo_infer_insurance` | [README](../GEO-INFER-INSURANCE/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-insurance.md) |
| `GEO-INFER-METAGOV` | `geo_infer_metagov` | [README](../GEO-INFER-METAGOV/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-metagov.md) |
| `GEO-INFER-NORMS` | `geo_infer_norms` | [README](../GEO-INFER-NORMS/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-norms.md) |
| `GEO-INFER-ECON` | `geo_infer_econ` | [README](../GEO-INFER-ECON/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-econ.md) |
| `GEO-INFER-PEP` | `geo_infer_pep` | [README](../GEO-INFER-PEP/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-pep.md) |
| `GEO-INFER-REQ` | `geo_infer_req` | [README](../GEO-INFER-REQ/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-req.md) |
| `GEO-INFER-SEC` | `geo_infer_sec` | [README](../GEO-INFER-SEC/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-sec.md) |
| `GEO-INFER-CIV` | `geo_infer_civ` | [README](../GEO-INFER-CIV/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-civ.md) |
| `GEO-INFER-HEALTH` | `geo_infer_health` | [README](../GEO-INFER-HEALTH/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-health.md) |
| `GEO-INFER-ORG` | `geo_infer_org` | [README](../GEO-INFER-ORG/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-org.md) |

### Data, API & applications

| Module | Package | Documentation |
| --- | --- | --- |
| `GEO-INFER-API` | `geo_infer_api` | [README](../GEO-INFER-API/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-api.md) |
| `GEO-INFER-APP` | `geo_infer_app` | [README](../GEO-INFER-APP/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-app.md) |
| `GEO-INFER-DATA` | `geo_infer_data` | [README](../GEO-INFER-DATA/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-data.md) |
| `GEO-INFER-IOT` | `geo_infer_iot` | [README](../GEO-INFER-IOT/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-iot.md) |
| `GEO-INFER-ART` | `geo_infer_art` | [README](../GEO-INFER-ART/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-art.md) |
| `GEO-INFER-EDU` | `geo_infer_edu` | [README](../GEO-INFER-EDU/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-edu.md) |

### Infrastructure & validation

| Module | Package | Documentation |
| --- | --- | --- |
| `GEO-INFER-INTRA` | `geo_infer_intra` | [README](../GEO-INFER-INTRA/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-intra.md) |
| `GEO-INFER-TEST` | `geo_infer_test` | [README](../GEO-INFER-TEST/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-test.md) |
| `GEO-INFER-LOG` | `geo_infer_log` | [README](../GEO-INFER-LOG/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-log.md) |
| `GEO-INFER-GIT` | `geo_infer_git` | [README](../GEO-INFER-GIT/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-git.md) |
| `GEO-INFER-EXAMPLES` | `geo_infer_examples` | [README](../GEO-INFER-EXAMPLES/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-examples.md) |
| `GEO-INFER-BIO` | `geo_infer_bio` | [README](../GEO-INFER-BIO/README.md) · [catalog](../GEO-INFER-INTRA/docs/modules/geo-infer-bio.md) |
