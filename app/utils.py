
from pathlib import Path
from typing import Any, Dict

import toml


def app_config(toml_config: str = 'config.toml') -> Dict[str, Any]:
    from app import __path__ as app_path
    if not isinstance(app_path, list):
        if hasattr(app_path, "_path"):
            app_path = app_path._path[0]
    else:
        app_path = app_path[0]
    app_home = Path(app_path).parent
    config_file = app_home.joinpath(toml_config)
    config_dict = toml.load(config_file.as_posix())
    return config_dict
