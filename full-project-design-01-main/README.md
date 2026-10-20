# Multi-Building Cooling Load Forecasting

A production-oriented project based on the **Global AI Challenge for Building E&M Facilities 2025**: continuous cooling-load forecasting for multiple buildings, with a default forecast horizon of 3 hours.

Notebooks orchestrate and present the experiments. All reusable, testable and API-serving logic lives in `src/cooling_load`.

## Architecture

```text
S3/local raw data
      │
      ▼
ingestion + manifest + checksum
      │
      ▼
schema validation → time-aware preprocessing
      │
      ▼
lag/rolling/HVAC features
      │
      ├── K-Means operating regimes (fit on train only)
      ▼
temporal CV → pick the best model family → tune that same family
      │
      ▼
versioned model bundle
      │
      ├── FastAPI / Docker
      └── drift and performance monitoring
```

## Directory structure

```text
configs/                 Data source and pipeline configuration
data/                    raw, interim, processed and ingestion manifests
notebooks/               Entry points that run the pipeline in order, 00–06
src/cooling_load/        Production logic
  ingestion/             S3/local adapters, checksums and manifest
  api/                   HTTP schemas, inference service and FastAPI app
analysis_functions/      Reusable EDA strategy classes
processing_functions/    Reusable preprocessing strategy classes
models/                  Deployable model bundle
reports/                 Metrics and figures
tests/                   Unit and integration tests
```

## Run from a fresh clone, no AWS needed

Requirements: Git, GNU Make and Python 3.11+.

```bash
git clone <repository-url>
cd <repository-directory>
make setup
make verify
```

`make setup` creates `.venv` and installs all dependencies. `make verify` runs the tests, then executes the seven notebooks in order. The default configuration is `ingestion.mode: local`, so the first notebook generates synthetic data when `data/raw/cooling_load.csv` does not exist. No AWS account is needed to verify the repository.

Common commands:

```bash
make test       # unit and integration tests
make pipeline   # run notebooks 00–06 and save the executed copies to reports/
make notebooks  # open JupyterLab
make api        # run FastAPI once a model has been trained
```

The notebooks run in this order:

1. `00_ingestion.ipynb`
2. `01_eda.ipynb`
3. `02_preprocessing.ipynb`
4. `03_clustering_features.ipynb`
5. `04_training_tuning.ipynb`
6. `05_evaluation_inference.ipynb`
7. `06_monitoring.ipynb`

`01_eda.ipynb` always saves the standard chart set to `reports/figures/eda/` and a JSON summary to `reports/eda_summary.json`. The notebook then displays those saved files, so the charts you see in the notebook and the charts used in reports are the same artifacts.

## Ingesting from S3

Set `ingestion.mode` to `s3` in `configs/base.yaml`, then configure:

```yaml
ingestion:
  mode: s3
  s3:
    bucket: your-ml-data-bucket
    prefix: cooling-load/raw/
    profile: default
```

Credentials are never stored in the repository. Locally, use an AWS profile; in production, use an IAM role with the minimum permissions `s3:ListBucket` and `s3:GetObject` on the relevant prefix.

Ingestion keeps the raw data unchanged. It downloads through a `.partial` file, verifies the size, compares the ETag, computes a SHA-256 and writes `data/manifests/ingestion_manifest.json`. Re-running skips local objects that have the same ETag and size.

## Data contract

The canonical dataset contains:

```text
timestamp, building_id, cooling_load_kwh,
outdoor_temperature_c, relative_humidity_pct,
chilled_water_supply_c, chilled_water_return_c,
chilled_water_flow_m3h, active_chillers, occupancy_proxy
```

If the challenge data uses different column names, map them in the ingestion adapter; do not change the model logic to follow source column names.

## Leakage prevention

- Splits are by timestamp, never random.
- Target and sensor features are shifted by at least the forecast horizon.
- Rolling features are computed after the shift.
- K-Means is fitted on the training set only, then used to transform validation and test.
- The test set is evaluated only after the model has been selected and tuned.
- The API reuses the exact `CoolingLoadFeatureBuilder` that was used for training.

## API

After the training notebook creates `models/champion.joblib`:

```bash
make api
```

Endpoints:

- `GET /health/live`
- `GET /health/ready`
- `GET /model-info`
- `POST /forecast`
- `GET /docs`

`POST /forecast` takes the history of exactly one building. Send at least 168 hours so that all lag and rolling features are complete; the model still has an imputer to cope with shorter histories.

## Storing the model bundle on S3

With scikit-learn, the deployable artifact is more than the weights. `champion.joblib` contains the estimator, imputer, K-Means clusterer, feature schema, config snapshot and metrics. Enable the model registry in `configs/base.yaml`:

```yaml
model_registry:
  enabled: true
  bucket: your-model-bucket
  prefix: cooling-load/models
  profile: cooling-load-dev
  server_side_encryption: AES256
```

The training notebook always writes the local model first with an atomic replace, then publishes:

```text
s3://your-model-bucket/cooling-load/models/
├── <created-at>-<model-name>/
│   ├── champion.joblib
│   └── metadata.json
└── latest.json
```

Every version key is immutable; `latest.json` is a mutable pointer holding the URI, SHA-256, size, metrics and creation time. Production should deploy a specific version URI rather than read `latest.json` directly.

FastAPI can download the model into a local cache at startup:

```bash
export MODEL_S3_URI=s3://your-model-bucket/cooling-load/models/<version>/champion.joblib
export MODEL_SHA256=<sha256-from-metadata>
export MODEL_PATH=models/champion.joblib
make api
```

Locally, use an AWS profile. On ECS, EKS or EC2, leave `AWS_PROFILE` empty and attach an IAM role with `s3:GetObject` on the model prefix. The training role also needs `s3:PutObject`; enable versioning on the bucket and block public access.

## Docker

```bash
docker compose up --build
curl http://localhost:8000/health/ready
```

The container runs as a non-root user. The model is mounted read-only so the artifact can be replaced without rebuilding the source image.

## Model maintenance

- Track feature PSI and the distribution of `operating_regime`.
- When ground truth becomes available, compute MAE/RMSE/WAPE per building and regime.
- Alert when PSI ≥ 0.1; consider retraining when PSI ≥ 0.25.
- Store `model_name`, `created_at`, metrics and the config snapshot in the model bundle.
- Promote a new model only if it beats both the seasonal baseline and the current champion on a temporal holdout.

## Note on the data

The file created by `make sample` is synthetic data so the whole repository runs out of the box. It is not the official competition data. When using real data, keep the canonical schema and change the source in the configuration.
