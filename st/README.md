# NeRFStyle vs StyleNeRF

Course project for CS674


## Installation

* python
* pytorch
* PIL, numpy, scipy
* tqdm
    
## Training Locally
To replicate our results follow the following steps-
* Clone the project

  ```bash
  git clone https://link-to-project
  ```

* Go to the project directory

  ```bash
  cd my-project
  ```

* Download [MS-COCO](http://images.cocodataset.org/zips/val2017.zip) images and unzip them in ./content
* Download the [Style Images](https://drive.google.com/file/d/1rLhs9hEEfuRXd_4FyCQgD2jKDVYMLa1B/view?usp=sharing) and place Style training Images in ./style

* Run Training file
  
```bash
  python train.py
```

## Inference

* Place Content Images in ./content
* Place Style Images in ./style
* Download checkpoints [transformer](https://drive.google.com/file/d/1piKfMau1bGwzjZQNI9BM3nUccacUFOsv/view?usp=sharing), [decoder](https://drive.google.com/file/d/1ZW9SuSBcS7COdsRyK-9ywKZ9y-WPaX6X/view?usp=sharing), [embedding](https://drive.google.com/file/d/1mx8u8aiqgPtg7YtV3Q7WRjMJz8wJQzS6/view?usp=sharing) and place them in ./experiments directory
* Run the Test file
    ```bash
    python test.py
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
