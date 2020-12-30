# CORD-19 Semantic Question Answering APP

TODO: finish documentation

## Getting started

Before starting the application we need to build the data stores. To build all the required sources simply run the following script (Make sure the paths pointing to the CORD-19 Dataset are absolute)

- Download a dataset if you haven't already or skip and configure the file below if you have an existing dataset.

> I highly recommend downloading the "smallest" release `2020-03-20` as shown below:

```python
from coronanlp.allenai import DownloadManager

dm = DownloadManager()
first_release = -1
date = dm.all_release_dates()[first_release]
arch = dm.download(date)
...
```

Make your copy of the `config.toml` file holding the app's configuration. Where `my_config.toml` can be any name you want.

```bash
cp config.toml my_config.toml
```

You can configure the arguments within the file if building a datastore within the file or from the script below.

- If you have an existing CORD19-Dataset
  - **source**: *A single string or a list of path(s) pointing to the directory holding the `*.json` files.*

```toml
[cord]
version = "2020-03-20"  # Optional
num_papers = 6641
num_sents = 1003614
index_start = 1
sort_first = true
text_source = "body_text"
subsets = [ "biorxiv_medrxiv", "comm_use_subset", "noncomm_use_subset", "custom_license",]
metadata = "/home/user/.cache/coronanlp/semanticscholar/hr/2020-03-20/metadata.csv"
source = [
    "/home/user/.cache/coronanlp/semanticscholar/hr/2020-03-20/biorxiv_medrxiv",
    "/home/user/.cache/coronanlp/semanticscholar/hr/2020-03-20/comm_use_subset",
    "/home/user/.cache/coronanlp/semanticscholar/hr/2020-03-20/noncomm_use_subset",
    "/home/user/.cache/coronanlp/semanticscholar/hr/2020-03-20/custom_license",
]
```

Finally export the environment variable pointing the config file.

> Or add the it to your `.bashrc` file

```bash
export CORONA_NLP_APP=/path/to/my_config.toml
```

Build a datastore from scratch, make sure to pass the `-arch_date 2020-03-20` argument along with the date of the dataset downloaded. `-type kaggle` builds a small sample tuned to answer questions from the challenge, it is also a small and friendly sample to start. All options are `none | server | kaggle`. If none, then `-sample <num_samples>` argument is required `-1` for all papers or `1000` for a random sample.

> To see all available arguments execute the command: `python scripts/build_datastore.py -h`

- store:
  - sents : `coronanlp.SentenceStore`
  - embed : `numpy.ndarray`
  - index : `faiss.Index`

```bash
python scripts/build_datastore.py \
    -type kaggle \
    -arch_date 2020-03-20 \
    -store_name webapp \
    -encoder amoux/scibert_nli_squad \
    -sort_first True
```

Output based on `2,159,779` million sentences/vectors

- Pre-processing + tokenizing `13,202` documents, time: `~25` minutes from **IO** on a **HDD**

- Encoding sentences to embeddings, time: `~4` hours with a basic **GTX-1050ti 4GB** card

```bash
Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) solvers for sklearn enabled: https://intelpython.github.io/daal4py/sklearn.html

papers: 100%|████████████████████████████████████████████████████████████████████| 13202/13202 [25:07<00:00,  8.76it/s]

batches: 100%|███████████████████████████████████████████████████████████████| 269973/269973 [4:08:17<00:00, 18.12it/s]

Training level-1 quantizer
Training level-1 quantizer on 2159779 vectors in 768D
Training IVF residual
IndexIVF: no residual training
IndexIVFFlat::add_core: added 2159779 / 2159779 vectors

Done: index and papers saved in path: /home/carlos/corona_nlp_app/src/data
Updating configuration file with the updated settings.
configuration file has been updated, file: ./config.toml
```

Run the application

```bash
./run_application.sh
```
