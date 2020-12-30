import os
from pathlib import Path
from typing import Any, MutableMapping, Optional

import toml

CORONA_APP_CONFIG = os.environ.get("CORONA_APP_CONFIG")


def app_config(toml_config: Optional[str] = None) -> MutableMapping[str, Any]:
    if toml_config is None:
        toml_config = CORONA_APP_CONFIG
    if toml_config is None:
        raise Exception(
            "CORONA_APP_CONFIG cannot be empty, make sure to "
            "export CORONA_APP_CONFIG='path/to/my_config.toml'")
    config_fp = Path(toml_config).absolute()
    if not config_fp.is_file():
        raise Exception(
            f"Expected valid TOML config file, got: {config_fp.as_posix()}")
    config_dict = toml.load(config_fp.as_posix())
    return config_dict
