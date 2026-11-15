# PyroSense SI

Super-Intelligence (SI) and Physics-Informed Neural Network (PINN) platform for NASA's Physical Science Informatics (PSI) microgravity combustion datasets.

## Directory Structure

```text
pyrosense-si/
├── configs/
│   └── default_config.json
├── data/
│   ├── raw/
│   │   ├── psi_burn_runs.csv
│   │   └── psi_investigations.json
│   └── processed/
├── models/
│   └── checkpoints/
├── notebooks/
│   └── colab_gpu_pipeline.ipynb
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── pyrosense/
│       ├── api/
│       ├── data/
│       ├── engine/
│       ├── models/
│       ├── vision/
│       └── config.py
├── tests/
│   └── test_relevance.py
├── web/
│   ├── assets/frames/
│   ├── css/styles.css
│   ├── js/app.js
│   └── index.html
├── pyproject.toml
└── requirements.txt
```

## Running the GPU Pipeline on Google Colab

Open `notebooks/colab_gpu_pipeline.ipynb` in Google Colab or execute:

```bash
pip install -e .
python scripts/run_pipeline.py
```

## Running the Web Server Locally

```bash
pip install -e .
uvicorn pyrosense.api.main:app --reload --port 8000
```
