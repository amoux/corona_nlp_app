from app.utils import app_config


def engine_config(toml_config: str = 'cfg.toml') -> dict:
    config = app_config(toml_config=toml_config)

    encoder = config['models']['sentence_encoder']
    model = config['models']['question_answering']
    encoder = None if encoder == model else encoder

    kwargs = config['engine']
    kwargs.update(
        {
            'sents': config['stores']['sents'],
            'index': config['stores']['index'],
            'encoder': encoder,
            'model': model,
            'nlp_model': config['models']['spacy_nlp']
        }
    )
    return kwargs
