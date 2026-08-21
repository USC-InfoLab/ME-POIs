# Mobility Embedded POIs (ME-POIs)

Official Implementation of `Mobility-Embedded POIs: Learning What A Place Is and How It Is Used from Human Movement`

## Setup

The code was developed with Python 3.13.2. From the repository root, install the
dependencies:

```bash
python -m pip install -r requirements.txt
```

Before running either script, update `config.json` with paths for your data and
embeddings. In particular, check `data_path`, `file_name`, `emb_path`,
`anchor_path`, `city`, and `save_dir`. The configured paths may need to be
changed from the example paths in this repository.

## Pretraining

Pretraining loads the visit dataset, text embeddings, and anchor data specified
in `config.json`. Run it from the repository root:

```bash
python run_pretrain.py
```

Set `pretraining_strategy` to `CL` or `MLM` in `config.json`. When
`save_pretrained_model` is `true`, the resulting POI embeddings are written to
`emb_path/poi_embeds.pt`.

## Evaluation

Evaluation fine-tunes and evaluates POI embeddings for the downstream task in
`config.json`. Run it after pretraining (or after providing a compatible
`emb_path/poi_embeds.pt`):

```bash
python run_eval.py
```

The supported values for `downstream_task` are `open_hours`, and
`is_closed`. All three tasks use labels from the processed SafeGraph
dataset.

Evaluation also loads the text embedding files for the model names defined in
`run_eval.py` (`e5`, `gtr-t5`, `gemini`, `nomic`, `mpnet`, `openai-large`, and
`openai-small`) from `emb_path`.
