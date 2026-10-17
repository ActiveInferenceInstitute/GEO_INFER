"""Run the same native scientific workflow used by the JSON CLI."""

import json
from pathlib import Path

from geo_infer_api.agent_runtime import AgentSession


def main() -> int:
    """Execute the adjacent workflow through the script API."""
    request = json.loads(Path(__file__).with_name("agent_workflow.json").read_text())
    result = AgentSession().execute(request)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
