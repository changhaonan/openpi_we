# Convert the world engine's data into pi-0 format.

## Install ffmpeg
In order to install ffmpeg with `libsvtav1`, we need to build ffmpeg from source. However, there is a recent change in SVT-AV1's API, which results in that if we still follows the old instruction. There will be a bug in compiling. So we need to install a old-version of SVT-AV1. (Checkout to branch r2.3.0)

## Convert the data into LeRoboDataset

```python
python examples/world_engine/convert_mcap_data_to_lerobot.py
```

## Compute norm & stats
```python
uv run scripts/compute_norm_stats_we.py --config-name pi0_plate_collect
```

## Fine-tune
```python
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_we.py
```