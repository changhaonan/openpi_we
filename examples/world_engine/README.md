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

## NATS address
```
/mnt/scratch/datasets/
```

## Intrinsic information

1. We-01:
1.1. Top: 218622278781. [1280x720]
 Intrinsic of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8}
  Width:      	1280
  Height:     	720
  PPX:        	651.172546386719
  PPY:        	363.466522216797
  Fx:         	645.511596679688
  Fy:         	644.690490722656
  Distortion: 	Inverse Brown Conrady
  Coeffs:     	-0.0509800985455513  	0.0607213973999023  	-0.000438050570664927  	0.00146730127744377  	-0.019759276881814  
  FOV (deg):  	89.5 x 58.36
1.2. Left: 218622279087
Intrinsic of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8}
  Width:      	1280
  Height:     	720
  PPX:        	651.146911621094
  PPY:        	361.052551269531
  Fx:         	649.895751953125
  Fy:         	648.952453613281
  Distortion: 	Inverse Brown Conrady
  Coeffs:     	-0.0537844635546207  	0.0645119920372963  	0.000249037868343294  	0.000429995881859213  	-0.0220546256750822  
  FOV (deg):  	89.11 x 58.04
1.3. Right: 218622274594
 Intrinsic of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8}
  Width:      	1280
  Height:     	720
  PPX:        	647.578002929688
  PPY:        	355.128265380859
  Fx:         	652.365600585938
  Fy:         	651.405822753906
  Distortion: 	Inverse Brown Conrady
  Coeffs:     	-0.0516580604016781  	0.0572841465473175  	1.91315448319074e-05  	0.00131414690986276  	-0.0189359113574028  
  FOV (deg):  	88.9 x 57.85

2. We-02:
2.1. Top:
Intrinsic of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8}
  Width:      	1280
  Height:     	720
  PPX:        	645.988403320312
  PPY:        	362.89111328125
  Fx:         	653.672790527344
  Fy:         	652.861572265625
  Distortion: 	Inverse Brown Conrady
  Coeffs:     	-0.0537487268447876  	0.0605481714010239  	-5.2034647524124e-05  	0.00103097432292998  	-0.020617974922061  
  FOV (deg):  	88.79 x 57.75