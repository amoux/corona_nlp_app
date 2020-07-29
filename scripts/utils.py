from pathlib import Path
import toml

CONFIG_FILE = './config.toml'
CONFIG_DICT = toml.load(Path(CONFIG_FILE).absolute().as_posix())
