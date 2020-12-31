from typing import Any, MutableMapping, Optional

from app.utils import app_config


def engine_config(toml_config: Optional[str] = None
                  ) -> MutableMapping[str, Any]:

    config = app_config(toml_config=toml_config)
    encoder = config['models']['sentence_encoder']
    model = config['models']['question_answering']
    encoder = None if encoder == model else encoder

    engine_kwargs = config['engine']
    engine_kwargs.update(
        {
            'sents': config['stores']['sents'],
            'index': config['stores']['index'],
            'encoder': encoder,
            'cord19': False,  # Disable initialization.
            'model': model,
            'nlp_model': config['models']['spacy_nlp']
        }
    )
    compressor_kwargs = engine_kwargs.pop('compressor')
    engine_kwargs.update(compressor_kwargs)
    return engine_kwargs
