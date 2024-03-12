import os
import tomllib
import uuid
from typing import Any

UUID_NAMESPACE = uuid.UUID("bab08d37-ac12-40c4-847a-20ca337742fd")


def paper_url_to_uuid(paper_url: str) -> "uuid.UUID":
    return uuid.uuid5(UUID_NAMESPACE, paper_url)


def read_toml(toml_file: str) -> dict[str, Any]:
    if not os.path.isfile(toml_file):
        raise FileNotFoundError(f"Not Found: {toml_file}")

    with open(toml_file, "rb") as f:
        return tomllib.load(f)
