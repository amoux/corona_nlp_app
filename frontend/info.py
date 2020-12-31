import re
from typing import Callable, List

from coronanlp.utils import clean_tokenization, normalize_whitespace
from utils import app_config

config = app_config()

NUM_PAPERS = config['cord']['num_papers']
NUM_SENTS = config['cord']['num_sents']
VERSION = config['cord']['version']
TEXT_SOURCE = config['cord']['text_source']
SUBSETS = config['cord']['subsets']
if isinstance(SUBSETS, (list, tuple, set)):
    SUBSETS = ", ".join([f'`{subset}`' for subset in SUBSETS])


def main_app_head(st):
    # Content displayed at the top of the page (Welcome screen).
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


def main_app_body(st):
    # SIDEBAR BOTTOM INFO:
    st.sidebar.markdown('🗃 *The data for all outputs, questions, and links '
                        'to the articles in this application is entirely from '
                        'the CORD-19 dataset.*')
    st.sidebar.markdown('---')
    # Content displayed at the bottom of the page:
    st.markdown('---')
    st.markdown('#### Outputs based on the following:')
    st.markdown(f'- dataset             : `{VERSION}`')
    st.markdown(f'- subsets             : `{SUBSETS}`')
    st.markdown(f'- papers              : `{NUM_PAPERS:,.0f}`')
    st.markdown(f'- text-source         : `{TEXT_SOURCE}`')
    st.markdown(f'- embeddings/sentences: `{NUM_SENTS:,.0f}`')
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


def render_answer(st, question, answer, context) -> Callable:
    def function(st=st, question=question, answer=answer, context=context):
        st.success("Done!")
        question = normalize_whitespace(question)
        answer = clean_tokenization(answer)
        context = clean_tokenization(context)
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


def render_about(st, nsids: int, npids: int) -> Callable:
    def function(st=st, nsids=nsids, npids=npids):
        out1 = f'Answer based on {NUM_SENTS}/***{nsids}*** sentences '
        out2 = f'obtained from {NUM_PAPERS}/***{npids}*** papers:'
        st.markdown('---')
        st.markdown(out1 + out2)
    return function


def render_similar(st, sents: List[str]) -> Callable:
    def function(st=st, sents=sents):
        st.markdown('### Similar Sentences')
        st.markdown('---')
        for sequence in sents:
            sequence = normalize_whitespace(sequence)
            st.markdown(f'- {sequence}')
    return function


def render_titles_urls(st, titles_urls) -> Callable:
    def function(st=st, titles_urls=titles_urls):
        for key in titles_urls:
            title, url = key['title'], key['url']
            st.markdown(f"- [{title}]({url})")
            st.sidebar.markdown('---')
    return function
