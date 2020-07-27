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

> Run the application

```bash
./run_application.sh
```
