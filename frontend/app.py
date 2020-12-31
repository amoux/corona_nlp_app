import random
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from coronanlp.utils import DataIO

from api import EngineAPI
from info import (main_app_body, main_app_head, render_about, render_answer,
                  render_similar, render_titles_urls)
from utils import MetadataReader, app_config, count_words

config = app_config()

# Application specific parameters:
QKNN_FILE = config['streamlit']['qknn_file']
FRONTEND_PORT = config['streamlit']['port']
# sliders (min, max, default-value)
TOP_K_PARAMS = config['streamlit']['top_k_params']
TOP_P_PARAMS = config['streamlit']['top_p_params']
# minimum number of words to consider valid
MIN_VALID_WORDS = config['streamlit']['min_valid_words']
# topic index to user at welcome screen
TOPIC_INDEX = config['streamlit']['topic_index']
BACKEND_PORT = config['fastapi']['port']

# Text to speech specific configuration.
TTS_PORT: Optional[int] = None
if config['streamlit']['enable_tts']:
    unique_tts_port = config['tts']['port']
    TTS_PORT = unique_tts_port if unique_tts_port else BACKEND_PORT

engine_api = EngineAPI(BACKEND_PORT)
meta_reader = MetadataReader(
    config['cord']['metadata'],
    config['cord']['source'],
)


@st.cache(allow_output_mutation=True)
def cache_qknn_topics(file_name=QKNN_FILE):
    tasks = DataIO.load_data(file_name)
    return tasks


@st.cache
def cache_topic_data(
    cat: str, knnq: Dict[str, str],
) -> Tuple[List[str], str, int]:

    questions = knnq[cat]['questions']
    key_words = knnq[cat]['key_words']
    key_words = ', '.join(key_words)
    rand_index = random.choice(list(range(len(questions))))
    return (questions, key_words, rand_index)


@st.cache(allow_output_mutation=True)
def cache_api_answer(question, topk, top_p, mode):
    """Cache and return the output answer for the selected topic question."""
    output = engine_api.answer(question, topk=topk, top_p=top_p, mode=mode)
    return output


def init(
    session: str,
    text: str,
    topk: int,
    top_p: int,
    mode: str,
    return_text_to_speech=False,
    return_titles_and_links=False,
) -> Tuple[Any, ...]:

    output_fn = None
    about_fn = None
    titles_urls_fn = None
    pids: Optional[List[int]] = None
    context: Optional[str] = None

    if session == 'SentenceSimilarity':
        output = engine_api.similar(text, top_p=top_p)
        if output is not None:
            # render similar sentences line by line.
            pids = output.pids.squeeze(0).tolist()
            sentences = output.sentences
            output_fn = render_similar(st, sentences)
            nsids, npids = output.sids.size, len(set(pids))
            about_fn = render_about(st, nsids, npids)

    elif session == 'QuestionAnswering':
        num_words = count_words(text, min_word_length=2)
        if num_words < MIN_VALID_WORDS:
            e = 'Text needs to be at least {} words long, and not {}'
            st.sidebar.error(e.format(MIN_VALID_WORDS, num_words))
        else:
            output = engine_api.answer(text, topk=topk, top_p=top_p, mode=mode)
            if output is not None:
                with st.spinner('Fetching results...'):
                    pids = output.pids.squeeze(0).tolist()
                    context = output.context
                    answer = output.a[output.topk(0)]
                    nsids, npids = output.sids.size, len(set(pids))
                    # Do not cache outputs from user's questions.
                    output_fn = render_answer(st, text, answer, context)
                    about_fn = render_about(st, nsids, npids)
            else:
                e = 'There was an ⚠ issue in trying to answer your question.'
                st.sidebar.error(e)

    elif session == 'Demo':
        # Cache the outputs from the demo questions.
        output = cache_api_answer(text, topk, top_p, mode)
        pids = output.pids.squeeze(0).tolist()
        context = output.context
        answer = output.a[output.topk(0)]
        output_fn = render_answer(st, text, answer, context)
        nsids, npids = output.sids.size, len(set(pids))
        about_fn = render_about(st, nsids, npids)

    if return_titles_and_links and pids is not None:
        try:
            titles_urls = meta_reader.load_urls(pids)
        except Exception as e:
            print(f'Loading titles and urls raised an exception {e}')
        else:
            titles_urls_fn = render_titles_urls(st, titles_urls)

    # Both `Demo` and `QuestionAnswering` sessions have TTS enabled.
    if return_text_to_speech and TTS_PORT is not None and context is not None:
        try:
            audio = engine_api.tts(context, prob=0.99, port=TTS_PORT)
        except Exception as e:
            print(f'Loading audio for text-to-speech raised an exception, {e}')
        else:
            st.audio(audio['audio_file_path'], format='audio/wav')

    return output_fn, about_fn, titles_urls_fn


def onUserSession() -> Optional[Dict[str, Any]]:

    # OPTION 1: Select check-box to enable rendering of titles with outputs.
    is_render_titles_checked = st.sidebar.checkbox(
        "Render Article Titles and Links", value=False,
    )

    # OPTION 2: Select check-box to enable text-to-speech for the context.
    tts_bool_value = False
    info_tts = "Enable text to speech for context outputs."
    if TTS_PORT is None:  # If not enabled let the user know that.
        tts_bool_value = True
        info_tts = "Text to speech feature is currently not enabled."

    is_tts_checked = st.sidebar.checkbox(info_tts, value=tts_bool_value)

    # OPTION 3: Select compression mode.
    st.sidebar.subheader('Context Compression')
    info_mode = (
        '⚗ BERT - (Easy-to-read Context: 🥇🥇🥇); Allows for correct '
        'generalization of terms, also known as "semantic-compression". 📊 '
        'Frequency - (Easy-to-read Context: 🥇); Topmost sentences scored '
        'based on basic word frequency.'
    )
    compression_modes = ('BERT', 'Frequency')
    mode = st.sidebar.selectbox(info_mode, compression_modes).lower()
    mode_selected = 'freq' if mode != 'bert' else 'bert'

    # OPTION 4: Select number of k-nearest neighbors from slider.
    st.sidebar.header('TopK & TopP')
    k, p = TOP_K_PARAMS, TOP_P_PARAMS
    topk = st.sidebar.slider('number of answers', k[0], k[1], k[2])
    top_p = st.sidebar.slider('size of context', p[0], p[1], p[2])

    # Welcome screen (Displays the topics for the demo).
    qknn = cache_qknn_topics()
    topics = list(qknn.keys())
    topic = st.selectbox('🧬 Select a Topic', topics, index=TOPIC_INDEX)
    questions, key_words, rand_idx = cache_topic_data(topic.lower(), qknn)
    st.subheader("{cat} Related Questions".format(cat=topic.lower().title()))
    info_entities = (f"🏷 this group of '{topic}' questions has a "
                     f"relationship to the subsequent entities: {key_words}")
    demo_input = st.selectbox(info_entities, questions, index=rand_idx)

    st.sidebar.subheader('Session Mode')
    info_session = 'Select a session style from the drop-down list below:'
    session_modes = ('Demo', 'SentenceSimilarity', 'QuestionAnswering',)
    session_choice = st.sidebar.selectbox(info_session, session_modes)
    user_input = st.sidebar.text_area('💬 Type your question or sentence')

    text = demo_input if session_choice == 'Demo' else user_input
    if text is not None:
        return {
            'session': session_choice,
            'text': text,
            'topk': topk,
            'top_p': top_p,
            'mode': mode_selected,
            'return_text_to_speech': is_tts_checked,
            'return_titles_and_links': is_render_titles_checked,
        }


def main():
    main_app_head(st)

    session = onUserSession()
    if session is not None:
        renderOnOutput, renderOnAbout, renderOnTitlesUrls = init(**session)

        if renderOnOutput is not None:
            renderOnOutput()
            renderOnAbout()
        if renderOnTitlesUrls is not None:
            renderOnTitlesUrls()

    main_app_body(st)


if __name__ == '__main__':
    main()
