# CS674_Spring24
Course project for CS 674

### Requirements
* python 3.11
* pytorch 2.2.0
* PIL, numpy, scipy
* tqdm  <br> 

### Testing Style Transfer
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [vit_embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing), [decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing), [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing)   <br> 
Please download them and put them into the folder  ./experiments/  <br> 
<br>
Place content images in ./input/content/   <br> 
Place style images in ./input/style/  <br> 
```
python test.py
```

### Training Style Transfer
Style dataset is [WIKIART](https://www.wikiart.org/) <br>
Place the Images folder so that you have the path ./datasets/Images/ <br>
<br>
Content dataset is [COCO2014](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3) <br>
Place the train2014 folder so that you have the path ./datasets/train2014/ <br> 
```
python train.py
```

### Evaluating Style Transfer
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