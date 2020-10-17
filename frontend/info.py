

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


def main_app_body(st, ds_version, text_keys, subsets, num_papers, num_sents):
    # SIDEBAR BOTTOM INFO:
    st.sidebar.markdown('🗃 *The data for all outputs, questions, and links '
                        'to the articles in this application is entirely from '
                        'the CORD-19 dataset.*')
    st.sidebar.markdown('---')
    # Content displayed at the bottom of the page:
    st.markdown('---')
    st.markdown('#### Outputs based on the following:')
    st.markdown(f'- dataset             : `{ds_version}`')
    st.markdown(f'- subsets             : `{subsets}`')
    st.markdown(f'- papers              : `{num_papers:,.0f}`')
    st.markdown(f'- text-source         : `{text_keys}`')
    st.markdown(f'- embeddings/sentences: `{num_sents:,.0f}`')
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
