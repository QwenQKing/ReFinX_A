# ReFinX

ReFinX is a benchmark and evaluation framework for financial agents. This repository includes 1,141 tasks, annotations, and evaluation code.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests
```

## Run

See `python -m refinx.run --help`. Configure the model and tool endpoints with the required `REFINX_*` environment variables.
