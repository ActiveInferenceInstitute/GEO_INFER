"""
Endpoint map for an externally deployed GEO-INFER-ACT model service.
"""


def create_endpoints() -> dict[str, str]:
    """Create the endpoint map for the external ACT ``/models`` service.

    The returned paths describe a deployment **outside** this module —
    GEO-INFER-ACT ships no HTTP server (see :mod:`geo_infer_act.api.client`);
    the map exists so client code and the :class:`Client <geo_infer_act.api.client.Client>`
    reference one shared contract for the external deployment.
    """
    return {
        "models": "/models",
        "beliefs": "/models/{model_id}/beliefs",
        "policies": "/models/{model_id}/policies",
    }
