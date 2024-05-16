# CS674_Spring24
Course project for CS 674: Generating aesthetic and novel scenes from images using style transfer on NeRF

## Authors:
Caleb Teeter<br>
Chirag Totla<br>
Jerry Zeng<br>
Diptyaroop Maji<br>

## Contributions:

Caleb Teeter: Worked on the style transfer component. Specifically, updated StyTR2 to work with newest versions of libraries, & wrote 147-155, 162-165, 184-188, 206-230, 268-270, 275-297 in eval_metrics.py

Chirag Totla: Worked on the style transfer component. Completely wrote transformer.py of StyTR(Lines 1-228) and modified StyTR.py(Lines 8-16)

Jerry Zeng: Worked on the NeRF component & automating the process. Completely wrote e2e.py (lines 1-152) & also helped Diptyaroop with hash-encoding wrappers used in tiny-cuda-nn.

Diptyaroop Maji: Worked on the NeRF component. Wrote/updated lines 12-13, 45-47 72-82, 95-109, 181-189, 347-353, 381-385 in nerf/lib/dvgo.py based on Jerry's suggestions to replace dense voxel grids with hash encoding. Also modified several lines in run.py. Wrote stitchImagesToVideo.py. Also, created a new bottle dataset.


## Instructions:

### Installation requirements
1. Nerf: Follow [DVGO installation instructions](https://github.com/sunset1995/DirectVoxGO/tree/main) and [tiny-cuda-nn installation instructions](https://github.com/NVlabs/tiny-cuda-nn/tree/master) for installing all the dependecies.
For tiny-cuda-nn, also install the PyTorch extension specified in their webpage.
2. ST: python, pytorch, PIL, numpy, scipy tqdm, torchvision, tensorboardX

There may be installation issues with CUDA with correct PyTorch or mmcv. Pleae refer to the existing solutions on the web or their respective source pages to resolve those.

### Running our project:

#### To run the NeRF -- ST design (generate images using NeRF followed by style transfer):
1. Add the desired style in ```st/style``` directory.
2. ```python3 e2e.py --dataset chair --first_method nerf --dump_images --render_only --render_test```

#### To run the ST -- NeRF design (generate images using NeRF followed by style transfer):

We do not have an automated script for this due to time constraints. Please do the following for this design (assuming dataset is "lego"):

1. Copy images from ```lego/train``` to ```st/content```
2. ```cd st/```
3. Run ```python test.py```
4. Copy the stylized images from ```output/``` to ```../data/nerf_synthetic/lego_stylized/train```
4. ```cd ..```
5. Repeat steps 1-4 for val & test images
6. Run ```python run.py --config configs/nerf/lego.py --render_test --i_print 500 --dump_images --stylized```

#### For quantitative evaluation of stylized views:

Please create and fill the following folders accordingly: <br>
Content basis placed in ./eval/content_basis/ (This paper used [nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)) <br>
Images to be evaluated placed in ./eval/model_output/
Style image of the evaluation images placed in ./eval/style/
Pretrained vgg placed in ./eval/ (dowloaded [here](https://www.dropbox.com/s/xc78chba9ffs82a/vgg_conv.pth?e=1&dl=0))
```
python eval_metrics.py
```
The output will be in the file ./eval/E_base.txt/ with the calibrated overall <b>E</b> score being <br>
E: 0.7468E1+0.2557E2+2.3768*E3



### Trained models
Available at ```nerf/logs/nerf_synthetic/dvgo_<dataset>/fine_last```

### Generated images & video
Available at ```nerf/logs/nerf_synthetic/dvgo_<dataset>/hash_render_test_fine_last_new```

### New dataset
Generated a new *bottle* dataset from a video using using [this](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md) link. Frames are in the data folder. <br>
Available at ```data/bottle```

<!-- 
## NeRF
The NeRF part is bason on [DirectVoxelGO](https://arxiv.org/pdf/2111.11215). The code used is mostly from [here](https://github.com/sunset1995/DirectVoxGO), with minor modifications. We refer to the term "vanilla" when we use the existing code without any modifications & "hash" when we use our modifications.

To run the NeRF part, follow the installation requirements specified [here](https://github.com/sunset1995/DirectVoxGO). Then use the following command in ```DVGO/```:

``` python run.py --config configs/nerf/chair.py --render_test --dump_images```

## Style Transfer
To be updated

## Outputs
All outputs are in the `logs` folder.
1. Lego (original synthetic dataset): Reproducing results for lego using DVGO code (vanilla)<br>
Testing psnr 34.67910163402557 (avg)<br>
Generated images & video are in `logs/nerf_synthetic/dvgo_lego/render_test_fine_last_new/`<br>
These images can be used for style transfer.<br>

https://github.com/diptyaroop/CS674_Spring24/assets/48976139/591e5120-f3fc-4a00-ad76-dcc8c12502d0

**Lego using NeRF (vanilla DVGO)**

2. Lego: Pipeline: Nerf (using DVGO), then Style transfer. <br>
Images & video after style transfer are in `logs/nerf_synthetic/lego_nerf_sty/`<br>

https://github.com/diptyaroop/CS674_Spring24/assets/48976139/6722d105-0011-4b41-812d-5246e7665549

**Lego using NeRF followed by ST**

3. Lego: Pipeline: Style transfer, then NeRF (using DVGO, vanilla). <br>
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
2. Script to generate train/val/test images & respective transforms.json file. Images & original transforms.json file generated using above link. -->

## Power Usage
To check the GPU power usage while running the application, use the following command on another terminal:<br>
`nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=1000`

<!-- On NVIDIA RTX 3050 Laptop GPU, power usage is arounf 25W when the code is running. Multiply with total time
elapsed to get the total power usage. -->



