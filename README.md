# Data Factory for GenGO

## Requirements

- Python 3.11
- pdm
- Docker

## Setup

Setup this repository

```bash
git clone git@github.com:gengo-proj/data-factory.git
cd data-factory
pdm install
```

Start Grobid Docker container

```sh
docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.0
```

Start local Qdrant server

```sh
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

## Usage

### Data preparation

The following command will

1. Download papers
2. Extract fulltext using Grobid
3. Save fulltext with metadata

```bash
# We need the meta data from ACL Anthology
git clone git@github.com:acl-org/acl-anthology.git

# Run preparation pipeline
pdm run python -m data_factory.paper2json.main \
  --base-output-dir <path/to/save/raw-paper.json> \
  --pdf-output-dir <path/to/save/downloaded/paper.pdf>
  --anthology-data-dir ./acl-anthology/data/
```

### Paper Enrichment

The following command will

1. Take a raw paper json file
2. Generate summaries on various aspects
3. Extract named entities 
4. Extract feature vectors
5. Save as a new enriched json file

```bash
pdm run python -m data_factory.enrichjsons.runner \
  --config ./src/data_factory/enrichjsons/configs/basic.toml \
  --raw-paper-base-dir <dir/of/raw-paper.json files>  \
  --output-base-dir <dir/to/save/enriched-paper.json files> \
  --device -1 (number of gpus to use, if you have, otherwise set -1)
```

### Upload to Qdrant

The following command will

1. Load a paper from an enriched json file
2. Index features vectors with metadata as payload

```bash
pdm run python -m data_factory.paperuploader.runner \
  --paper-base-dir <dir/of/enriched-paper.json files> \
  --collection-name <collection name for Qdrant> \
  --vector-size <vector size of feature vectors, e.g., 384> \
  --host <host name of Qdrant, e.g., localhost> \
  --port <port number of Qdrant, e.g., 6333>
```
