# Google Colab Setup

This guide explains how to run the public Beyond Backscatter GRD/GEE notebook:

https://colab.research.google.com/github/FPSica/BeyondBackscatter/blob/main/notebooks/back2coh_grd_gee_colab.ipynb

The notebook predicts InSAR coherence from two detected Sentinel-1 GRD backscatter images downloaded from Google Earth Engine. It does not use the SLC inference path.

## Run In Colab

1. Open the notebook from the Colab link above.
2. Select `Runtime` > `Change runtime type`.
3. Choose a GPU runtime.
4. Run the setup cell. It clones `https://github.com/FPSica/BeyondBackscatter.git`, checks out `main`, installs `requirements-colab.txt`, and adds the repository helpers to `sys.path`.

For testing an unmerged branch, set `GITHUB_BRANCH` in the setup cell before running it.

## Earth Engine

You need a Google account with Earth Engine access and a Google Cloud project enabled for Earth Engine.

Set this notebook variable:

```python
GEE_PROJECT_ID = "your-gee-project-id"
```

The authentication cell runs:

```python
ee.Authenticate()
ee.Initialize(project=GEE_PROJECT_ID)
```

Do not hardcode credentials or service-account keys in the notebook.

## ROI And Dates

Start with a small ROI. The notebook supports:

- drawing a polygon or rectangle with `geemap`;
- manual lon/lat polygon coordinates;
- manual GeoJSON.

Configure two date windows:

```python
DATE1_START = "2025-06-18"
DATE1_END   = "2025-06-19"
DATE2_START = "2025-06-24"
DATE2_END   = "2025-06-25"
```

The notebook searches `COPERNICUS/S1_GRD` for matching acquisitions and prints image index, image ID, date/time, orbit pass, relative orbit, platform, polarization list, instrument mode, and approximate ROI coverage. The user then selects exactly one image for period 1 and exactly one image for period 2. `RELATIVE_ORBIT` can stay empty; when it is empty, no relative-orbit filter is applied during search. The final selected image pair must still have the same relative orbit number.

Earth Engine direct GeoTIFF downloads are limited to roughly 48 MiB per request. The notebook checks the estimated download size before requesting the GeoTIFF URLs. If the check fails, use a smaller ROI, increase `SCALE_METERS`, or split the area into several smaller runs.

## Sentinel-1 GRD Preprocessing

The Earth Engine preprocessing mirrors the GRD/GEE workflow:

- collection: `COPERNICUS/S1_GRD`;
- ROI filter;
- `instrumentMode == "IW"` by default;
- `orbitProperties_pass == "ASCENDING"` by default;
- optional relative orbit filter;
- date-window filter;
- one selected single Sentinel-1 GRD image per period;
- common valid area from ROI, image 1 footprint/mask, and image 2 footprint/mask;
- dB to linear conversion with `pow(10, db / 10)`;
- selected polarization, usually `VV`;
- clip/export to the common area;
- direct-download size check;
- download at 10 m and `EPSG:4326` by default.

After download, the notebook verifies that the two GeoTIFFs have the same CRS, affine transform, width, height, valid-mask shape, and nodata handling.

Downloaded inputs are saved under:

```text
outputs/gee_inputs/
```

## Model Weights

Model checkpoints are not committed to GitHub. The notebook supports three sources.

### Hugging Face Hub

Default:

```python
MODEL_SOURCE = "huggingface"
MODEL_FRAMEWORK = "tensorflow"
HF_REPO_ID = "FPSica/beyond-backscatter-grd-gee"
HF_REVISION = "main"
HF_WEIGHTS_FILENAME = "model.weights.h5"
HF_CONFIG_FILENAME = "config.yaml"
```

Public model repositories do not need a token. For private repositories, use Colab secrets or environment variables such as `HF_TOKEN`; do not paste tokens into the notebook.

### Google Drive

Mount Drive and point to the model directory:

```python
MODEL_SOURCE = "google_drive"
GDRIVE_MODEL_DIR = "/content/drive/MyDrive/back2coh_grd_gee_model"
```

### Local Path

Use a model directory already present in the Colab runtime:

```python
MODEL_SOURCE = "local"
LOCAL_MODEL_DIR = "/content/BeyondBackscatter/model"
```

## Expected Model Directory

The default model package expects:

```text
model.weights.h5
config.yaml
README.md
normalization/statistics file if required
```

The TensorFlow/Keras ResUNet architecture is included in the GitHub repository under `src/colab_grd_gee/tf_model.py`; the Hugging Face repo does not need a `model.py` file for the default workflow. The loader first tries standard Keras weight loading and falls back to legacy Keras H5 by-name loading for newer Keras runtimes.

`config.yaml` should describe the real GRD/GEE model and preprocessing. Example:

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

## Outputs

The notebook writes all products under:

```text
outputs/
```

Important files include:

- `outputs/gee_inputs/sigma0_<POL>_t1_linear.tif`
- `outputs/gee_inputs/sigma0_<POL>_t2_linear.tif`
- `outputs/gee_inputs/metadata.json`
- `outputs/coherence_pred.tif`
- `outputs/coherence_pred.png`
- `outputs/coherence_pred.npy`
- `outputs/rgb_diagnostic.png`
- `outputs/rgb_pseudo_natural.png`
- `outputs/rgb_pseudo_natural.tif`
- `outputs/beyond_backscatter_outputs.zip`

The RGB files are SAR/coherence visualization products. They are not true optical images.

Predicted coherence is displayed with a grayscale color map from 0 to 1.

## Security Notes

- Do not commit Earth Engine credentials, Google credentials, Hugging Face tokens, `.env` files, or private paths.
- Keep model checkpoints out of GitHub.
- Use small ROIs first to reduce processing time and export size.
