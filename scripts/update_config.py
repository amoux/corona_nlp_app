from pathlib import Path

import plac
import toml
from corona_nlp.datatypes import Papers

DEFAULT_CONFIG = './config.toml'


@plac.annotations(
    streamlit=("Update streamlit settings.", "option", "streamlit", bool),
    backend_port=("Update FastAPI backend port.", "option", "b-port", int),
    frontend_port=("Update Streamlit frontend port.", "option", "f-port", int),
    enable_tts=("Enable text-to-speech for webapp.", "option", "tts", bool),
    config_file=("Name of the config file, if None it will overide existing",
                 "option", "config", str)
)
def main(streamlit: bool = False, enable_tts: bool = None,
         backend_port: int = None, frontend_port: int = None,
         config_file: str = None):
    """Update the main configuration toml file in the root directory."""
    default_config_file = Path(DEFAULT_CONFIG).absolute()
    config = toml.load(default_config_file.as_posix())

    if streamlit:
        papers = Papers.from_disk(config['engine']['papers'])
        config['streamlit'].update({
            'num_papers': '{:,.0f}'.format(papers.num_papers),
            'num_sentences': '{:,.0f}'.format(papers.num_sents)
        })
    if enable_tts is not None:
        config['streamlit'].update({'enable_tts': enable_tts})
    if backend_port is not None and isinstance(backend_port, int):
        config['fastapi']['port'].update({'port': backend_port})
    if frontend_port is not None and isinstance(frontend_port, int):
        config['streamlit']['port'].update({'port': frontend_port})

    file_path = DEFAULT_CONFIG if config_file is None else config_file
    with open(file_path, 'w') as f:
        toml.dump(config, f)

    print(f"configuration file has been updated, file: {file_path}")


if __name__ == '__main__':
    plac.call(main)
