# CORD-19 Semantic Question Answering APP

> Getting started

Before starting the application we need to build the data stores. To build all the required sources simply run the following script (Make sure the paths pointing to the CORD-19 Dataset are absolute)

```bash
python scripts/build_server_data.py \
    -sample -1 \
    -minlen 25 \
    -index_start 1 \
    -sort_first True \
    -nlp_model en_core_sci_sm \
    -config_file ./config.toml \
    -data_dir ./src/data/
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
