# Public Notebook And Model Release Notes

Public Colab URL:

https://colab.research.google.com/github/FPSica/BeyondBackscatter/blob/main/notebooks/back2coh_grd_gee_colab.ipynb

Expected Hugging Face model repository:

https://huggingface.co/FPSica/beyond-backscatter-grd-gee

## Notebook Workflow

The notebook:

1. clones this GitHub repository when opened from Colab;
2. installs `requirements-colab.txt`;
3. authenticates Google Earth Engine;
4. lets the user draw or define an ROI;
5. searches Sentinel-1 GRD acquisitions for two date windows;
6. downloads linear sigma0 GeoTIFFs from GEE;
7. loads the GRD/GEE model from Hugging Face Hub, Google Drive, or a local directory;
8. runs tiled inference with `model.eval()` and `torch.no_grad()`;
9. exports predicted coherence and SAR/coherence RGB visualizations.

The notebook uses Sentinel-1 GRD data from GEE. It does not use the SLC inference path.

## Model Files Expected On Hugging Face

The default notebook settings expect:

```text
checkpoint.pth
config.yaml
README.md
model.py or equivalent model source file
normalization/statistics file if required
```

`config.yaml` should define the exact GRD/GEE model class and preprocessing. Example:

```yaml
model:
  module: model
  class_name: Back2CohGRDModel
  kwargs: {}
strict_load: true
preprocessing:
  db_min: -20
  db_max: 0
  channel_order: [t1, t2]
tiling:
  patch_size: 128
  stride: 32
  batch_size: 8
  aggregation: kaiser
```

If the checkpoint is a complete serialized `torch.nn.Module`, the model class fields are optional. State-dict checkpoints should include source code and model class information in `config.yaml`.

## Upload Model Files To Hugging Face

Install the Hub client and authenticate outside the notebook:

```bash
pip install huggingface_hub
hf auth login
```

Then upload a model directory:

```bash
python scripts/upload_model_to_hf.py \
  --repo-id FPSica/beyond-backscatter-grd-gee \
  --model-dir /path/to/model_dir \
  --private false
```

The helper uploads only model-related files such as `.pth`, `.pt`, `.ckpt`, `.yaml`, `.yml`, `.json`, `README.md`, config files, and normalization/statistics files. It excludes outputs, GeoTIFFs, NumPy arrays, notebooks, hidden environment files, token files, credentials, and caches.

The script does not hardcode tokens. It relies on `hf auth login`, environment variables, or the existing Hugging Face cache.

## Running The Notebook

1. Open the public Colab URL.
2. Use a GPU runtime.
3. Set `GEE_PROJECT_ID`.
4. Start with a small ROI.
5. Configure two Sentinel-1 dates and filters.
6. Keep `MODEL_SOURCE = "huggingface"` once the public model repo is released.
7. Run all cells.

Outputs are saved under `outputs/`.

## Credentials And Privacy

Do not hardcode:

- Hugging Face tokens;
- Earth Engine credentials;
- Google service-account keys;
- Google Drive private paths;
- `.env` values.

For private Hugging Face repositories, set `HF_TOKEN` through Colab secrets or an environment variable. Public repositories do not require a token.

## Visualization Note

The pseudo-natural RGB output is a SAR/coherence visualization derived from backscatter amplitudes and predicted coherence. It is not a true optical image and should not be interpreted as optical reflectance.
