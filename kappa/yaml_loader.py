from __future__ import annotations

import yaml


class NoBoolSafeLoader(yaml.SafeLoader):
    """YAML loader that avoids implicit boolean conversion (e.g., 'NO')."""


# Strip the bool resolver so plain scalars like "NO" stay as strings.
for first, mappings in list(NoBoolSafeLoader.yaml_implicit_resolvers.items()):
    NoBoolSafeLoader.yaml_implicit_resolvers[first] = [
        (tag, regexp) for tag, regexp in mappings if tag != "tag:yaml.org,2002:bool"
    ]


def safe_load_no_bool(text: str):
    """Load YAML text without implicit bool conversions."""
    return yaml.load(text, Loader=NoBoolSafeLoader)
