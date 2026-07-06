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
6. lets the user select exactly one image per period;
7. verifies shared valid coverage and downloads linear sigma0 GeoTIFFs over the common area;
8. loads the GRD/GEE model from Hugging Face Hub, Google Drive, or a local directory;
9. runs tiled TensorFlow/Keras inference with the released GRD/GEE Keras weights;
10. exports grayscale predicted coherence and SAR/coherence RGB visualizations.

The notebook uses Sentinel-1 GRD data from GEE. It does not use the SLC inference path.
It does not compute or recommend temporal baselines.

## Model Files Expected On Hugging Face

The default notebook settings expect:

```text
model.weights.h5
config.yaml
README.md
normalization/statistics file if required
```

The TensorFlow/Keras ResUNet architecture used by the notebook lives in the GitHub repository at `src/colab_grd_gee/tf_model.py`, so the default Hugging Face model package does not need a `model.py` file. The loader first tries standard Keras weight loading and falls back to legacy Keras H5 by-name loading for newer Keras runtimes.

`config.yaml` should define the exact GRD/GEE model and preprocessing. Example:

```yaml
framework: tensorflow
weights_filename: model.weights.h5
architecture:
  name: resunet
  input_shape: [128, 128, 2]
  output_channels: 2
preprocessing:
  db_min: -20
  db_max: 0
  channel_order: [t1, t2]
  input_scale: linear_sigma0_from_gee
  model_scale: db_clipped_normalized
tiling:
  patch_size: 128
  stride: 32
  batch_size: 8
  aggregation: kaiser
output:
  name: coherence
  range: [0, 1]
  channel: 0
```

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

The helper uploads only model-related files such as `.h5`, `.keras`, `.yaml`, `.yml`, `.json`, `README.md`, config files, and normalization/statistics files. It excludes outputs, GeoTIFFs, NumPy arrays, notebooks, hidden environment files, token files, credentials, and caches.

The script does not hardcode tokens. It relies on `hf auth login`, environment variables, or the existing Hugging Face cache.

## Running The Notebook

1. Open the public Colab URL.
2. Use a GPU runtime.
3. Set `GEE_PROJECT_ID`.
4. Start with a small ROI.
5. Configure two Sentinel-1 dates and filters. Leave `RELATIVE_ORBIT` empty if you do not want to filter by relative orbit.
6. Keep `MODEL_SOURCE = "huggingface"` once the public model repo is released.
7. Review the acquisition tables, choose one image in each dropdown, and run the remaining cells.

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

Predicted coherence is displayed in grayscale with values clipped to `[0, 1]`.

The pseudo-natural RGB output is a SAR/coherence visualization derived from backscatter amplitudes and predicted coherence. It is not a true optical image and should not be interpreted as optical reflectance.
