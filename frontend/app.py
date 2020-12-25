import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st
from coronanlp.utils import DataIO, clean_tokenization, normalize_whitespace

from info import main_app_body, main_app_head
from utils import MetadataReader, ModelAPI, app_config, count_words

# Application specific parameters:
# slider (min, max, default)
TOPK_CONFIG = (1, 10, 5)
TOP_P_CONFIG = (5, 100, 25)
# minimum number of words to consider valid
MIN_VALID_WORDS = 4
# topic index to user at user welcome screen
TOPIC_INDEX = 4

config = app_config()
# Information related to the dataset and number of samples.
NUM_PAPERS = config['streamlit']['num_papers']
NUM_SENTS = config['streamlit']['num_sentences']
SUBSETS = ', '.join([f'`{s}`' for s in config['streamlit']['subsets']])
DATASET_VERSION = config['streamlit']['dataset_version']
TEXT_KEYS = config['streamlit']['text_source']
# Data sources for the application's content.
CORD19_SOURCE = config['cord']['source']
CORD19_METADATA = config['cord']['metadata']
QKNN_FILE = config['streamlit']['qknn_file']
# Ports for connecting the backend services
FRONTEND_PORT = config['streamlit']['port']
BACKEND_PORT = config['fastapi']['port']
# Text to speech specific configuration.
TTS_PORT: Optional[int] = None
if config['streamlit']['enable_tts']:
    unique_tts_port = config['tts']['port']
    TTS_PORT = unique_tts_port if unique_tts_port \
        else config['fastapi']['port']


api = ModelAPI(port=BACKEND_PORT)
meta_reader = MetadataReader(
    metadata_path=CORD19_METADATA,
    source=CORD19_SOURCE,
)


@st.cache(allow_output_mutation=True)
def cache_qknn_topics(file_name=QKNN_FILE):
    tasks = DataIO.load_data(file_name)
    return tasks


@st.cache
def cache_topic_data(cat: str, knnq: Dict[str, str],
                     ) -> Tuple[List[str], str, int]:
    questions = knnq[cat]['questions']
    key_words = knnq[cat]['key_words']
    key_words = ', '.join(key_words)
    rand_index = random.choice(list(range(len(questions))))
    return (questions, key_words, rand_index)


@st.cache(allow_output_mutation=True)
def cache_api_answer(question, topk, top_p, mode):
    """Cache and return the output answer for the selected topic question."""
    output = api.answer(question, topk=topk, top_p=top_p, mode=mode)
    return output


def render_answer(question: str, output: Dict[str, Any]) -> Callable:
    def function(question=question, output=output):
        st.success("Done!")
        question = normalize_whitespace(question)
        answer = clean_tokenization(output['answer'])
        context = clean_tokenization(output['context'])

        # markdown template formats:
        highlight, bold = '`{}`', '**{}**'
        answer_title_md = '### 💡 Answer'
        context_title_md = '### ⚗ Context'
        summary_title_md = '### 📃 Summary'

        if len(answer) == 0:
            st.markdown(summary_title_md)
            question_md = bold.format(question)
            st.write('> ', context.replace(question, question_md, 1))
        else:
            try:
                match = re.search(answer, context)
                match.span()
            except Exception as e:
                print(f'Match generated an exception for {answer}: {e}')
                pass
            finally:
                st.markdown(answer_title_md)
                st.write('> ', answer.capitalize())
                st.markdown(context_title_md)
                context = context.replace(answer, highlight.format(answer), -1)
                if not question.endswith('?'):
                    question = f'{question}?'
                question = bold.format(question)
                context = question.strip() + "  " + context.strip()
                st.write('> ', context)
    return function


def render_about(output: Dict[str, Any]) -> Callable:
    def function(output=output):
        nsents, npapers = output['num_sents'], output['num_papers']
        out1 = f'Answer based on {NUM_SENTS}/***{nsents}*** sentences '
        out2 = f'obtained from {NUM_PAPERS}/***{npapers}*** papers:'
        st.markdown('---')
        st.markdown(out1 + out2)
    return function


def render_similar(sents: List[str]) -> Callable:
    def function(sents=sents):
        st.markdown('### Similar Sentences')
        st.markdown('---')
        for sequence in sents:
            sequence = normalize_whitespace(sequence)
            st.markdown(f'- {sequence}')
    return function


def render_titles_urls(titles_urls) -> Callable:
    def function(titles_urls=titles_urls):
        for key in titles_urls:
            title, url = key['title'], key['url']
            st.markdown(f"- [{title}]({url})")
            st.sidebar.markdown('---')
    return function


def init(session, text, topk, top_p, mode, with_text_to_speech=False):
    output_fn, about_fn, titles_urls_fn = None, None, None

    if session == 'SentenceSimilarity':
        output = api.similar(text, top_p=top_p)
        if output is not None:
            # render similar sentences line by line.
            output_fn = render_similar(output['sents'])
            titles_urls = meta_reader.load_urls(output['paper_ids'])
            output.update({'num_papers': len(titles_urls)})
            about_fn = render_about(output)
            titles_urls_fn = render_titles_urls(titles_urls)

    elif session == 'QuestionAnswering':
        output = api.answer(text, topk=topk, top_p=top_p, mode=mode)
        num_words = count_words(text, min_word_length=2)
        if output is not None and num_words >= MIN_VALID_WORDS:
            with st.spinner("'Fetching results..."):
                output_fn = render_answer(text, output)
                titles_urls = meta_reader.load_urls(output['paper_ids'])
                output.update({'num_papers': len(titles_urls)})
                about_fn = render_about(output)
                titles_urls_fn = render_titles_urls(titles_urls)
        else:
            st.sidebar.error(
                f'Text needs to be at least {MIN_VALID_WORDS}'
                f' words long, and not {num_words}')

    elif session == 'Demo':
        output = cache_api_answer(text, topk, top_p, mode)
        output_fn = render_answer(text, output)
        titles_urls = meta_reader.load_urls(output['paper_ids'])
        output.update({'num_papers': len(titles_urls)})
        about_fn = render_about(output)
        titles_urls_fn = render_titles_urls(titles_urls)

        if TTS_PORT is not None and with_text_to_speech:
            audio = api.tts(output['context'], prob=0.99, port=TTS_PORT)
            if audio is not None:
                fp = audio['audio_file_path']
                st.audio(fp, format='audio/wav')

    return output_fn, about_fn, titles_urls_fn


def main():
    main_app_head(st)

    TEXT = st.empty()
    renderOnOutput: Callable = st.empty()
    renderOnAbout: Callable = st.empty()
    renderOnTitlesUrls: Callable = st.empty()

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
    k, p = TOPK_CONFIG, TOP_P_CONFIG
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

    TEXT = demo_input if session_choice == 'Demo' else user_input
    if TEXT is not None:
        renderOnOutput, renderOnAbout, renderOnTitlesUrls = init(
            session=session_choice,
            text=TEXT,
            topk=topk,
            top_p=top_p,
            mode=mode_selected,
            with_text_to_speech=is_tts_checked,
        )
    if renderOnOutput is not None:
        renderOnOutput()
        renderOnAbout()
    if is_render_titles_checked:
        renderOnTitlesUrls()

    main_app_body(
        st=st,
        ds_version=DATASET_VERSION,
        text_keys=TEXT_KEYS,
        subsets=SUBSETS,
        num_papers=NUM_PAPERS,
        num_sents=NUM_SENTS,
    )


if __name__ == '__main__':
    main()
