"""
File: mllm/utils/get_coagent_id.py
Summary: Helper for deriving co-agent identifiers from rollout metadata.
"""


def get_coagent_id(ids: list[str], agent_id: str) -> str | None:
    for id in ids:
        if id != agent_id:
            return id
