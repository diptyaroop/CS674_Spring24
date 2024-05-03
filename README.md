# CS674_Spring24
Course project for CS 674

## Data
Generated a new *bottle* dataset from a video using using [this](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md) link. Frames are in the data folder

## Outputs
All outputs are in the `logs` folder.
1. Lego (original synthetic dataset): Reproducing results for lego using DVGO code (unmodified)<br>
Testing psnr 34.67910163402557 (avg)<br>
Generated images & video are in `logs/nerf_synthetic/dvgo_lego/render_test_fine_last_new/`<br>
These images can be used for style transfer.<br>

https://github.com/diptyaroop/CS674_Spring24/assets/48976139/591e5120-f3fc-4a00-ad76-dcc8c12502d0

**Lego using NeRF (vanilla DVGO)**

2. Lego: Pipeline: Nerf (using DVGO), then Style transfer. <br>
Images & video after style transfer are in `logs/nerf_synthetic/lego_nerf_sty/`<br>

https://github.com/diptyaroop/CS674_Spring24/assets/48976139/6722d105-0011-4b41-812d-5246e7665549

**Lego using NeRF followed by ST**

3. Lego: Pipeline: Style transfer, then NeRF (using DVGO). <br>
We first used style transfer on vanilla NeRF synthetic images. Then, we ran DVGO.<br>
Testing psnr 19.06837311387062 (avg)<br>
NeRF generated images & video are in `logs/nerf_synthetic/lego_sty_nerf/`<br>

https://github.com/diptyaroop/CS674_Spring24/assets/48976139/91a89e75-48d5-4204-a5a1-4f14ea87f8b9

**Lego using ST followed by NeRF**

4. Bottle (Generated. **Work in progress**): Checking how DVGO works on a new dataset<br>
Testing PSNR around 28. (Taking a few hours if the fine grid resolution is 160x160x160. Changed this to 128x128x128 to speed up. Now taking around an hour. Lower PSNR maybe due to this reason)<br>
Generated images & video are in `logs/nerf_synthetic/dvgo_bottle/render_test_fine_last_new/`<br>

## Scripts
1. Script to remove background from generated images. Images are generated using [this](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md) link & colmap2NeRF.py
2. Script to generate train/val/test images & respective transforms.json file. Images & original transforms.json file generated using above link.
3. Command to check power usage while DVGO/application is running: 
`nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=1000`


