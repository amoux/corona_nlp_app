import random
import re
from string import punctuation
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st
from corona_nlp.utils import DataIO, clean_tokenization, normalize_whitespace

from .utils import MetadataReader, ModelAPI, app_config

config = app_config()
FRONTEND_PORT = config['streamlit']['port']
BACKEND_PORT = config['fastapi']['port']
CORD19_SOURCE = config['cord']['source']
CORD19_METADATA = config['cord']['metadata']
KNNQ_FILE = config['streamlit']['knnq_file']
ENABLE_TTS = config['streamlit']['enable_tts']
TTS_PORT = config['tts']['port']
if not TTS_PORT:  # Set to default port if text-to-speech enabled.
    TTS_PORT = config['fastapi']['port']

api = ModelAPI(port=BACKEND_PORT)
meta_reader = MetadataReader(metadata_path=CORD19_METADATA,
                             source=CORD19_SOURCE)


@st.cache(allow_output_mutation=True)
def load_task_clusters(file_name=KNNQ_FILE):
    tasks = DataIO.load_data(file_name)
    return tasks


@st.cache
def set_info_state(titles_and_urls):
    titles_and_urls = titles_and_urls
    return titles_and_urls


@st.cache
def load_questions(category: str,
                   knnq: Dict[str, str]) -> Tuple[List[str], str, int]:
    questions = knnq[category]['questions']
    key_words = knnq[category]['key_words']
    key_words = ', '.join(key_words)
    random_id = random.choice(list(range(len(questions))))
    return (questions, key_words, random_id)


def main():
    st.title('⚕ COVID-19')
    st.header('Semantic Question Answering System')
    st.sidebar.title('Built with state-of-the-art Transformer models 🤗')
    st.sidebar.markdown("> The CORD-19 dataset represents the most extensive "
                        "machine-readable coronavirus literature collection "
                        "available for data mining to date")
    st.markdown('> Use this as a tool to search and explore `COVID-19` '
                'research literature with natural language.')
    st.markdown('- Open the sidebar **>** (↖ *top-left*) to enter a'
                ' question or choose a question based on one of the topics'
                ' below to get started.')
    st.markdown('---')
    st.markdown('*Questions extracted from the literature, sorted'
                ' based on similarity and grouped via clustering*.')
    st.sidebar.markdown('---')
    st.sidebar.title('Say Something to Coronavirus Literature')
    st.sidebar.markdown('- Articulating to it in sentences will '
                        'usually produce better results than keywords.')
    st.sidebar.markdown('- The model is case sensitive. Indicating that '
                        'upper/lower case letters affect the "meaning".')
    st.sidebar.header('Settings')

    SUBSET_INFO_STATE = None
    render_titles = st.sidebar.checkbox(
        "Render Article Titles and Links", value=False)

    # NOTE: Text to speech is optional. To enable uncomment
    # the code below and set ``ENABLE_TTS = True`` and set the server port.
    # with_text_to_speech = st.sidebar.checkbox("Turn ON/OFF Text-to-Speech",
    #                                           value=False)

    st.sidebar.subheader('Context Compression')
    desc_compressor = (
        '⚗ Embedding - (Easy-to-read Context: 🥇🥇🥇); Allows for correct'
        ' generalization of terms, also known as "semantic-compression". 📊'
        ' Frequency - (Easy-to-read Context: 🥇); Topmost sentences scored '
        'based on basic word frequency.'
    )
    compressor = st.sidebar.selectbox(
        desc_compressor,
        options=("Embedding", "Frequency",),
    )
    compressor = 'freq' if compressor == "Frequency" else "bert"
    st.sidebar.header('K-Nearest Neighbors (K-NN)')
    desc_context = (
        'Explore how the answers improve based on the context '
        'size. (Some questions might require less or more similar'
        ' sentences to be answered by the question answering model)'
    )

    MIN_SENTS = st.sidebar.slider(desc_context, 5, 100, 25)
    MAX_SENTS = powerup(MIN_SENTS, rate=3.7)
    KNNQ = load_task_clusters()
    KNNQ_INDEX = 4

    k_categories = list(KNNQ.keys())
    selected_cat = st.selectbox('🧬 Select a Topic',
                                k_categories,
                                index=KNNQ_INDEX).lower()
    # @st.cache
    # def load_questions(category: str) -> Tuple[List[str], str, int]:
    #     questions = KNNQ[category]
    #     key_words = ', '.join(list(KNNQ.keys()))
    #     random_id = random.choice(list(range(len(questions))))
    #     return questions, key_words, random_id

    st.subheader(f"{selected_cat.title()} Related Questions")
    desc_entities = (
        f"🏷 this group of '{selected_cat}' questions"
        " has a relationship to the subsequent entities"
    )
    questions, key_words, random_id = load_questions(selected_cat, knnq=KNNQ)
    chosen_question = st.selectbox(f"{desc_entities}: {key_words}",
                                   questions,
                                   index=random_id)
    if chosen_question:
        output = load_answer(chosen_question,
                             mink=MIN_SENTS,
                             maxk=MAX_SENTS,
                             mode=compressor)

        render_answer(chosen_question, output)
        titles_and_urls = meta_reader.load_urls(output)
        render_output_info(n_sents=output["n_sents"],
                           n_papers=len(titles_and_urls))
        SUBSET_INFO_STATE = set_info_state(titles_and_urls)

        if ENABLE_TTS:  # to enable this feature see line [67]
            audiofile = api.text_to_speech(context=output['context'],
                                           port=TTS_PORT)
            if audiofile:
                st.audio(audiofile['file_path'], format='audio/wav')

    sentence_mode = 'Sentence Similarity'
    st.sidebar.subheader('Search Mode')
    desc_search_mode = (
        "Sentence-similarity works only when"
        " the text entered in the box below."
    )
    search_mode = st.sidebar.selectbox(
        desc_search_mode,
        ('Question Answering', sentence_mode,),
    )
    question = st.sidebar.text_area('💬 Type your question or sentence')
    if question:
        if search_mode == sentence_mode:
            output = api.sentence_similarity(question, topk=MIN_SENTS)
            if output:
                for sent in output['sents']:
                    sent = normalize_whitespace(sent)
                    st.markdown(f'- {sent}')

                titles_and_urls = meta_reader.load_urls(output)
                render_output_info(n_sents=output['n_sents'],
                                   n_papers=len(titles_and_urls))
                SUBSET_INFO_STATE = set_info_state(titles_and_urls)
        else:
            max_valid_words = 4
            num_valid_words = count_words(question, min_word_length=2)

            if num_valid_words >= max_valid_words:
                with st.spinner("'Fetching results..."):
                    output = load_answer(question,
                                         mink=MIN_SENTS,
                                         maxk=MAX_SENTS,
                                         mode=compressor)

                    render_answer(question, output)
                    titles_and_urls = meta_reader.load_urls(output)
                    render_output_info(n_sents=output['n_sents'],
                                       n_papers=len(titles_and_urls))
                    SUBSET_INFO_STATE = set_info_state(titles_and_urls)
            else:
                st.sidebar.error(
                    f'A question needs to be at least {max_valid_words}'
                    f' words long, not {num_valid_words}'
                )

    if render_titles:
        if SUBSET_INFO_STATE is not None:
            for key in SUBSET_INFO_STATE:
                title, url = key['title'], key['url']
                st.markdown(f"- [{title}]({url})")
            st.sidebar.markdown('---')

    # SIDEBAR BOTTOM INFO:
    st.sidebar.markdown('🗃 *The data for all outputs, questions, and links '
                        'to the articles in this application is entirely from '
                        'the CORD-19 dataset.*')
    st.sidebar.markdown('---')

    # Content displayed at the bottom of the page:
    st.markdown('---')
    st.markdown('#### Outputs based on the following:')
    st.markdown('- dataset: `2020-04-24`')
    st.markdown(
        '- subsets: `comm_use_subset`, `noncomm_use_subset`, `biorxiv_medrxiv`')
    st.markdown(f'- papers: `{N_PAPERS}`')
    st.markdown('- text-source: `body_text`')
    st.markdown(f'- embeddings/sentences: `{N_SENTS}` ')
    st.markdown('Tool created by *Carlos Segura* for the '
                '[COVID-19 Open Research Dataset Challenge]'
                '(https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)')
    st.markdown('- download the question answering model used in this app:'
                ' [scibert_nli_squad](https://huggingface.co/amoux/scibert_nli_squad)')
    st.markdown('Github page: [amoux](https://github.com/amoux)')
    st.markdown('Source code: [corona](https://github.com/amoux/corona)')
    st.text(r'''
        @inproceedings{Wang2020CORD19TC,
      title={CORD-19: The Covid-19 Open Research Dataset},
      author={Lucy Lu Wang and Kyle Lo and Yoganand Chandrasekhar and
        Russell Reas and Jiangjiang Yang and Darrin Eide and Kathryn Funk
        and Rodney Kinney and Ziyang Liu and William. Merrill and Paul
        Mooney and Dewey A. Murdick and Devvret Rishi and Jerry Sheehan
        and Zhihong Shen and Brandon Stilson and Alex D. Wade and Kuansan
        Wang and Christopher Wilhelm and Boya Xie and Douglas M. Raymond
        and Daniel S. Weld and Oren Etzioni and Sebastian Kohlmeier},
      year={2020}
    }
    ''')


def count_words(string, min_word_length=2) -> int:
    tokens = clean_tokenization(normalize_whitespace(string)).split()
    words = ["".join([char for char in token if char not in punctuation])
             for token in tokens if len(token) > min_word_length]
    return len(words)


def powerup(k: int, rate=2.5) -> int:
    return round(((k * rate) / 2) + k)


def load_answer(question, mink=15, maxk=30, mode='bert') -> Dict[str, Any]:
    output = api.question_answering(question, mink=mink, maxk=maxk, mode=mode)
    if output:
        if len(output['answer'].strip()) == 0:
            maxk = powerup(mink)
            output = api.question_answering(
                question, mink=mink, maxk=maxk, mode=mode)
    return output


def render_output_info(n_sents: int, n_papers: int):
    st.markdown('---')
    total_sents = int(N_SENTS.replace(',', ''))
    total_papers = int(N_PAPERS.replace(',', ''))
    about_output = (
        f"ℹ Answer based on ***{n_sents}***/{total_sents} sentences "
        f"obtained from ***{n_papers}***/{total_papers} unique papers:"
    )
    st.markdown(about_output)


def render_answer(question: str, output: dict) -> None:
    st.success("Done!")
    question = normalize_whitespace(question)
    answer = clean_tokenization(output["answer"])
    context = clean_tokenization(output["context"])

    # Main markdown labels for the outputs.
    Q = f'**{question}**'
    A = "### 💡 Answer"
    C = "### ⚗ Context"

    if len(answer) == 0:
        st.markdown('### 📃 Summary')
        st.write("> ", context.replace(question, Q, 1))
    else:
        try:
            # Search the answer in the context for highlighting.
            match = re.search(answer, context)
            start, end = match.span()
        except Exception:
            # Failed to highlight the answer in the context.
            st.markdown(A)
            st.write('> ', answer.capitalize())
            st.markdown(C)
            context = context.replace(question, Q, 1)
            st.write('> ' + context.replace(answer, f'`{answer}`', -1))
        else:
            # Highlight the answer found in the context.
            st.markdown(A)
            st.write('>', answer.capitalize())
            st.markdown(C)
            context = context.replace(question, Q, 1)
            context = context.replace(answer, f'`{answer}`', -1)
            st.write('> ' + context)


if __name__ == '__main__':
    main()
