from app.utils import app_config


def engine_config(toml_config: str = 'config.toml') -> dict:
    config = app_config(toml_config=toml_config)
    engine_config = config['engine']
    engine_config.update({
        'source': config['cord']['source'],
        'encoder': config['models']['sentence_transformer'],
        'model': config['models']['question_answering'],
        'nlp_model': config['models']['spacy_nlp']
    })
    engine_config.update(config['cord']['init'])
    return engine_config
